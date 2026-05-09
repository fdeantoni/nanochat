#!/usr/bin/env python3
"""
Unified web chat server - serves both UI and API from a single FastAPI instance.

Uses data parallelism to distribute requests across multiple GPUs. Each GPU loads
a full copy of the model, and incoming requests are distributed to available workers.

Launch examples:

- single available GPU (default)
python -m scripts.chat_web

- 4 GPUs
python -m scripts.chat_web --num-gpus 4

To chat, open the URL printed in the console. (If on cloud box, make sure to use public IP)

Endpoints:
  GET  /           - Chat UI
  POST /chat/completions - Chat API (streaming only)
  GET  /health     - Health check with worker pool status
  GET  /stats      - Worker pool statistics and GPU utilization

Abuse Prevention:
  - Maximum 500 messages per request
  - Maximum 8000 characters per message
  - Maximum 32000 characters total conversation length
  - Temperature clamped to 0.0-2.0
  - Top-k clamped to 0-200 (0 disables top-k filtering, using full vocabulary)
  - Max tokens clamped to 1-4096
"""

import argparse
import json
import os
import torch
import asyncio
import logging
import random
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse, FileResponse
from pydantic import BaseModel
from typing import List, Optional, AsyncGenerator
from dataclasses import dataclass
from nanochat.common import compute_init, autodetect_device_type
from nanochat.checkpoint_manager import load_model
from nanochat.engine import Engine

# Abuse prevention limits
MAX_MESSAGES_PER_REQUEST = 500
MAX_MESSAGE_LENGTH = 8000
MAX_TOTAL_CONVERSATION_LENGTH = 32000
MIN_TEMPERATURE = 0.0
MAX_TEMPERATURE = 2.0
MIN_TOP_K = 0 # 0 disables top-k filtering, using full vocabulary
MAX_TOP_K = 200
MIN_MAX_TOKENS = 1
MAX_MAX_TOKENS = 4096

parser = argparse.ArgumentParser(description='NanoChat Web Server')
parser.add_argument('-n', '--num-gpus', type=int, default=1, help='Number of GPUs to use (default: 1)')
parser.add_argument('-i', '--source', type=str, default="sft", help="Source of the model: sft|rl")
parser.add_argument('-t', '--temperature', type=float, default=0.8, help='Default temperature for generation')
parser.add_argument('-k', '--top-k', type=int, default=50, help='Default top-k sampling parameter')
parser.add_argument('--no-repeat-ngram-size', type=int, default=3, help='Ban n-grams already seen; 0 disables (default: 3)')
parser.add_argument(
    '--system-prompt',
    type=str,
    default="",
    help=(
        "System prompt prepended to the first user message of every conversation. "
        "Empty by default (model was not trained with system messages, so adding "
        "one is an experimental inference-time steer rather than a default behaviour). "
        "When set, the content is merged into the first user turn (matching "
        "tokenizer.render_conversation behaviour). Per-request system messages in "
        "/chat/completions take precedence over this default. Example: "
        "--system-prompt 'You are Max Babbelaar, born 1830. The date is 1 January 1880.'"
    ),
)
parser.add_argument('-m', '--max-tokens', type=int, default=1024, help='Default max tokens for generation')
parser.add_argument('-g', '--model-tag', type=str, default=None, help='Model tag to load')
parser.add_argument('-s', '--step', type=int, default=None, help='Step to load')
parser.add_argument('-p', '--port', type=int, default=8000, help='Port to run the server on')
parser.add_argument('--device-type', type=str, default='', choices=['cuda', 'cpu', 'mps'], help='Device type for evaluation: cuda|cpu|mps. empty => autodetect')
parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to bind the server to')
parser.add_argument('--log-level', type=str, default='INFO',
                    choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                    help='Logging level (default: INFO)')
args = parser.parse_args()

# common.py calls basicConfig at import time, so basicConfig here would be a no-op.
# Set the root level directly instead so --log-level is always honoured.
logging.root.setLevel(getattr(logging, args.log_level))
logger = logging.getLogger(__name__)

device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)

@dataclass
class Worker:
    """A worker with a model loaded on a specific GPU."""
    gpu_id: int
    device: torch.device
    engine: Engine
    tokenizer: object

class WorkerPool:
    """Pool of workers, each with a model replica on a different GPU."""

    def __init__(self, num_gpus: Optional[int] = None):
        if num_gpus is None:
            if device_type == "cuda":
                num_gpus = torch.cuda.device_count()
            else:
                num_gpus = 1 # e.g. cpu|mps
        self.num_gpus = num_gpus
        self.workers: List[Worker] = []
        self.available_workers: asyncio.Queue = asyncio.Queue()

    async def initialize(self, source: str, model_tag: Optional[str] = None, step: Optional[int] = None):
        """Load model on each GPU."""
        print(f"Initializing worker pool with {self.num_gpus} GPUs...")
        if self.num_gpus > 1:
            assert device_type == "cuda", "Only CUDA supports multiple workers/GPUs. cpu|mps does not."

        for gpu_id in range(self.num_gpus):

            if device_type == "cuda":
                device = torch.device(f"cuda:{gpu_id}")
                print(f"Loading model on GPU {gpu_id}...")
            else:
                device = torch.device(device_type) # e.g. cpu|mps
                print(f"Loading model on {device_type}...")

            model, tokenizer, _ = load_model(source, device, phase="eval", model_tag=model_tag, step=step)
            engine = Engine(model, tokenizer)
            worker = Worker(
                gpu_id=gpu_id,
                device=device,
                engine=engine,
                tokenizer=tokenizer,
            )
            self.workers.append(worker)
            await self.available_workers.put(worker)

        print(f"All {self.num_gpus} workers initialized!")

    async def acquire_worker(self) -> Worker:
        """Get an available worker from the pool."""
        return await self.available_workers.get()

    async def release_worker(self, worker: Worker):
        """Return a worker to the pool."""
        await self.available_workers.put(worker)

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    top_k: Optional[int] = None

def validate_chat_request(request: ChatRequest):
    """Validate chat request to prevent abuse."""
    # Check number of messages
    if len(request.messages) == 0:
        raise HTTPException(status_code=400, detail="At least one message is required")
    if len(request.messages) > MAX_MESSAGES_PER_REQUEST:
        raise HTTPException(
            status_code=400,
            detail=f"Too many messages. Maximum {MAX_MESSAGES_PER_REQUEST} messages allowed per request"
        )

    # Check individual message lengths and total conversation length
    total_length = 0
    for i, message in enumerate(request.messages):
        if not message.content:
            raise HTTPException(status_code=400, detail=f"Message {i} has empty content")

        msg_length = len(message.content)
        if msg_length > MAX_MESSAGE_LENGTH:
            raise HTTPException(
                status_code=400,
                detail=f"Message {i} is too long. Maximum {MAX_MESSAGE_LENGTH} characters allowed per message"
            )
        total_length += msg_length

    if total_length > MAX_TOTAL_CONVERSATION_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=f"Total conversation is too long. Maximum {MAX_TOTAL_CONVERSATION_LENGTH} characters allowed"
        )

    # Validate role values
    for i, message in enumerate(request.messages):
        if message.role not in ["user", "assistant"]:
            raise HTTPException(
                status_code=400,
                detail=f"Message {i} has invalid role. Must be 'user', 'assistant', or 'system'"
            )

    # Validate temperature
    if request.temperature is not None:
        if not (MIN_TEMPERATURE <= request.temperature <= MAX_TEMPERATURE):
            raise HTTPException(
                status_code=400,
                detail=f"Temperature must be between {MIN_TEMPERATURE} and {MAX_TEMPERATURE}"
            )

    # Validate top_k
    if request.top_k is not None:
        if not (MIN_TOP_K <= request.top_k <= MAX_TOP_K):
            raise HTTPException(
                status_code=400,
                detail=f"top_k must be between {MIN_TOP_K} and {MAX_TOP_K}"
            )

    # Validate max_tokens
    if request.max_tokens is not None:
        if not (MIN_MAX_TOKENS <= request.max_tokens <= MAX_MAX_TOKENS):
            raise HTTPException(
                status_code=400,
                detail=f"max_tokens must be between {MIN_MAX_TOKENS} and {MAX_MAX_TOKENS}"
            )

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on all GPUs on startup."""
    print("Loading nanochat models across GPUs...")
    app.state.worker_pool = WorkerPool(num_gpus=args.num_gpus)
    await app.state.worker_pool.initialize(args.source, model_tag=args.model_tag, step=args.step)
    # Pre-warm the BM25 search index so the first search request doesn't pay
    # the cold-load cost (~1 s to parse the 125 MB corpus from disk).
    try:
        from nanochat.babbelaar_search import preload_index
        if preload_index() is not None:
            print("  Search index pre-warmed.")
        else:
            print("  No search index found; search will be disabled.")
    except Exception as e:
        print(f"  Search index pre-warm skipped: {e}")
    print(f"Server ready at http://localhost:{args.port}")
    yield

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Serve the chat UI."""
    ui_html_path = os.path.join("nanochat", "ui.html")
    with open(ui_html_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    # Replace the API_URL to use the same origin
    html_content = html_content.replace(
        "const API_URL = `http://${window.location.hostname}:8000`;",
        "const API_URL = '';"
    )
    return HTMLResponse(content=html_content)


@app.get("/logo.png")
async def logo():
    """Serve the Max Babbelaar logo for the header and favicon."""
    logo_path = os.path.join("nanochat", "logo.png")
    return FileResponse(logo_path, media_type="image/png")


async def generate_stream(
    worker: Worker,
    tokens,
    temperature=None,
    max_new_tokens=None,
    top_k=None
) -> AsyncGenerator[str, None]:
    """Generate assistant response with streaming."""
    temperature = temperature if temperature is not None else args.temperature
    max_new_tokens = max_new_tokens if max_new_tokens is not None else args.max_tokens
    top_k = top_k if top_k is not None else args.top_k

    assistant_end = worker.tokenizer.encode_special("<|assistant_end|>")
    bos = worker.tokenizer.get_bos_token_id()
    python_start = worker.tokenizer.encode_special("<|python_start|>")
    output_end   = worker.tokenizer.encode_special("<|output_end|>")

    # Suppress tool-use blocks (python call + search output) from the text
    # stream; emit structured events instead so the UI can show a search
    # indicator rather than a wall of raw JSON.
    in_tool_block = False

    # Accumulate tokens to properly handle multi-byte UTF-8 characters (like emojis)
    accumulated_tokens = []
    # Track the last complete UTF-8 string (without replacement characters)
    last_clean_text = ""

    for token_column, token_masks in worker.engine.generate(
        tokens,
        num_samples=1,
        max_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        seed=random.randint(0, 2**31 - 1),
        no_repeat_ngram_size=args.no_repeat_ngram_size,
    ):
        token = token_column[0]

        # Stopping criteria
        if token == assistant_end or token == bos:
            break

        # --- Tool-use gate ---
        if token == python_start:
            in_tool_block = True
            yield f"data: {json.dumps({'searching': True})}\n\n"
            continue
        if token == output_end:
            in_tool_block = False
            # Reset the text buffer so output-block text can't bleed into the
            # narration delta on the next iteration.
            accumulated_tokens = []
            last_clean_text = ""
            yield f"data: {json.dumps({'search_done': True})}\n\n"
            continue
        if in_tool_block:
            continue  # suppress python call tokens + output_start + JSON tokens

        # Append the token to sequence
        accumulated_tokens.append(token)
        # Decode all accumulated tokens to get proper UTF-8 handling
        # Note that decode is a quite efficient operation, basically table lookup and string concat
        current_text = worker.tokenizer.decode(accumulated_tokens)
        # Only emit text if it doesn't end with a replacement character
        # This ensures we don't emit incomplete UTF-8 sequences
        if not current_text.endswith('�'):
            # Extract only the new text since last clean decode
            new_text = current_text[len(last_clean_text):]
            if new_text:  # Only yield if there's new content
                yield f"data: {json.dumps({'token': new_text, 'gpu': worker.gpu_id}, ensure_ascii=False)}\n\n"
                last_clean_text = current_text

    yield f"data: {json.dumps({'done': True})}\n\n"

@app.post("/chat/completions")
async def chat_completions(request: ChatRequest):
    """Chat completion endpoint (streaming only) - uses worker pool for multi-GPU."""

    # Basic validation to prevent abuse
    validate_chat_request(request)

    # Log incoming conversation to console
    logger.info("="*20)
    for i, message in enumerate(request.messages):
        logger.info(f"[{message.role.upper()}]: {message.content}")
    logger.info("-"*20)

    # Acquire a worker from the pool (will wait if all are busy)
    worker_pool = app.state.worker_pool
    worker = await worker_pool.acquire_worker()

    try:
        # Build conversation tokens
        bos = worker.tokenizer.get_bos_token_id()
        user_start = worker.tokenizer.encode_special("<|user_start|>")
        user_end = worker.tokenizer.encode_special("<|user_end|>")
        assistant_start = worker.tokenizer.encode_special("<|assistant_start|>")
        assistant_end = worker.tokenizer.encode_special("<|assistant_end|>")

        # Resolve system-prompt prefix.
        # Precedence: request system message > server default (--system-prompt).
        # The model was not trained with system messages, so we merge the
        # system content into the first user turn — same convention as
        # tokenizer.render_conversation. This keeps the token sequence
        # in-distribution with SFT.
        request_system = next(
            (m.content for m in request.messages if m.role == "system"), None,
        )
        system_prefix = request_system if request_system is not None else args.system_prompt
        first_user_seen = False

        conversation_tokens = [bos]
        for message in request.messages:
            if message.role == "system":
                continue  # already captured above; never emit as its own turn
            if message.role == "user":
                content = message.content
                if not first_user_seen and system_prefix:
                    content = system_prefix.strip() + "\n\n" + content
                first_user_seen = True
                conversation_tokens.append(user_start)
                conversation_tokens.extend(worker.tokenizer.encode(content))
                conversation_tokens.append(user_end)
            elif message.role == "assistant":
                conversation_tokens.append(assistant_start)
                conversation_tokens.extend(worker.tokenizer.encode(message.content))
                conversation_tokens.append(assistant_end)

        conversation_tokens.append(assistant_start)

        # Streaming response with worker release after completion
        response_tokens = []
        async def stream_and_release():
            try:
                async for chunk in generate_stream(
                    worker,
                    conversation_tokens,
                    temperature=request.temperature,
                    max_new_tokens=request.max_tokens,
                    top_k=request.top_k
                ):
                    # Accumulate response for logging
                    chunk_data = json.loads(chunk.replace("data: ", "").strip())
                    if "token" in chunk_data:
                        response_tokens.append(chunk_data["token"])
                    yield chunk
            finally:
                # Log the assistant response to console
                full_response = "".join(response_tokens)
                logger.info(f"[ASSISTANT] (GPU {worker.gpu_id}): {full_response}")
                logger.info("="*20)
                # Release worker back to pool after streaming is done
                await worker_pool.release_worker(worker)

        return StreamingResponse(
            stream_and_release(),
            media_type="text/event-stream"
        )
    except Exception as e:
        # Make sure to release worker even on error
        await worker_pool.release_worker(worker)
        raise e

@app.get("/health")
async def health():
    """Health check endpoint."""
    worker_pool = getattr(app.state, 'worker_pool', None)
    num_params = None
    if worker_pool and worker_pool.workers:
        num_params = sum(p.numel() for p in worker_pool.workers[0].engine.model.parameters())
    return {
        "status": "ok",
        "ready": worker_pool is not None and len(worker_pool.workers) > 0,
        "num_gpus": worker_pool.num_gpus if worker_pool else 0,
        "available_workers": worker_pool.available_workers.qsize() if worker_pool else 0,
        "num_params": num_params
    }

@app.get("/stats")
async def stats():
    """Get worker pool statistics."""
    worker_pool = app.state.worker_pool
    return {
        "total_workers": len(worker_pool.workers),
        "available_workers": worker_pool.available_workers.qsize(),
        "busy_workers": len(worker_pool.workers) - worker_pool.available_workers.qsize(),
        "workers": [
            {
                "gpu_id": w.gpu_id,
                "device": str(w.device)
            } for w in worker_pool.workers
        ]
    }

if __name__ == "__main__":
    import uvicorn
    print(f"Starting NanoChat Web Server")
    print(f"Temperature: {args.temperature}, Top-k: {args.top_k}, Max tokens: {args.max_tokens}")
    if args.system_prompt:
        sp_preview = args.system_prompt[:120] + ("…" if len(args.system_prompt) > 120 else "")
        print(f"System prompt ({len(args.system_prompt)} chars): {sp_preview}")
    else:
        print("System prompt: (disabled)")
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level.lower())
