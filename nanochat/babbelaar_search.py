"""BM25 search over the Babbelaar Delpher kranten archive.

Single source of truth for the ``search()`` function used by:

- **Step 142** (``142_build_search_index.py``) — builds the index.
- **Step 143** (``143_generate_tool_use_sft.py``) — calls the loaded index
  while generating SFT examples, so the data the model sees during
  training matches byte-for-byte what the runtime produces.
- **The nanochat inference runtime** — registers ``search()`` in the
  python sandbox so the model can invoke it via
  ``<|python_start|>search(...)<|python_end|>`` blocks.

Output format (printed JSON list of dicts) is defined here exactly once.
Step 143 captures the printed output to embed in ``<|output_start|>``
blocks; the runtime captures stdout from the sandbox in the same way.
The two paths cannot drift.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Iterable

import bm25s

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Index location resolution
# -----------------------------------------------------------------------------

_DEFAULT_RUNTIME_SUBDIR = "babbelaar_index"


def default_index_dir() -> Path:
    """Resolve the index directory.

    Order of precedence:
      1. ``BABBELAAR_INDEX_DIR`` env var (explicit override)
      2. ``$NANOCHAT_BASE_DIR/babbelaar_index/`` (runtime default — same
         pattern as model checkpoints, persists on the network volume)
      3. ``{project_root}/data/index/delpher_bm25/`` (Babbelaar dev layout:
         this file lives at ``nanochat/nanochat/babbelaar_search.py`` inside
         the babbelaar project; three ``parent`` steps reach the project root
         where step 142 writes the index)
      4. ``~/.cache/nanochat/babbelaar_index/`` (final fallback)
    """
    if explicit := os.environ.get("BABBELAAR_INDEX_DIR"):
        return Path(explicit)
    base = os.environ.get("NANOCHAT_BASE_DIR")
    if base:
        primary = Path(base) / _DEFAULT_RUNTIME_SUBDIR
        if primary.exists():
            return primary
    # Babbelaar dev: babbelaar_search.py is at <project>/nanochat/nanochat/;
    # three parents up is the project root, where step 142 writes the index.
    project_index = Path(__file__).resolve().parent.parent.parent / "data" / "index" / "delpher_bm25"
    if project_index.exists():
        return project_index
    if base:
        return Path(base) / _DEFAULT_RUNTIME_SUBDIR  # let caller report missing index
    return Path.home() / ".cache" / "nanochat" / _DEFAULT_RUNTIME_SUBDIR


# -----------------------------------------------------------------------------
# Source metadata helpers
# -----------------------------------------------------------------------------

_SOURCE_ID_RE = re.compile(r"DDD_ddd_(\d+)_mpeg21")


def delpher_url(source_id: str) -> str:
    """Build a Delpher resolver URL from a source_id like
    ``DDD_ddd_010713355_mpeg21`` → ``https://resolver.kb.nl/resolve?urn=ddd:010713355``.

    Returns an empty string for inputs that don't match the expected pattern.
    Used as a fallback when an enriched record's ``url`` field is empty
    (legacy data; current step 80 populates it inline). The KRANTEN_KBDDD02
    archive uses a different URN namespace not handled here — those records
    keep their empty url and the model simply omits the link.
    """
    if not source_id:
        return ""
    m = _SOURCE_ID_RE.match(source_id)
    if not m:
        return ""
    return f"https://resolver.kb.nl/resolve?urn=ddd:{m.group(1)}"


# -----------------------------------------------------------------------------
# Index build (used by step 142)
# -----------------------------------------------------------------------------

def _indexable_text(rec: dict) -> str:
    """Concatenate the fields BM25 should match against.

    Summary first (longest, most informative), then topics (categorical
    boost terms), then title (newspaper name — lets queries that mention
    a paper find the right hits).
    """
    summary = rec.get("summary") or ""
    topics = rec.get("topics") or []
    title = rec.get("title") or ""
    return " ".join([summary, " ".join(topics), title]).strip()


def _hit_metadata(rec: dict) -> dict:
    """Per-doc metadata stored in the index. This is the exact dict
    returned by ``search()`` — what the model sees in the tool output.

    ``url`` is taken from the record (populated by step 80 going forward
    and by ``backfill_delpher_urls.py`` for legacy data). If still empty,
    fall back to constructing it from ``source_id`` so we never silently
    drop the link for a DDD record that just slipped through. ``source_id``
    itself is **not** exposed to the model — it's a machine-readable
    archive identifier that's only useful for URL construction.
    """
    url = (rec.get("url") or "").strip() or delpher_url(rec.get("source_id") or "")
    return {
        "date": rec.get("date_raw") or (str(rec["date"]) if rec.get("date") else ""),
        "title": rec.get("title") or "",
        "summary": rec.get("summary") or "",
        "url": url,
    }


def build_index(records: Iterable[dict], output_dir: Path) -> dict:
    """Build a BM25 index from an iterable of enriched delpher records.

    Records without a ``summary`` field (typically OCR-junk articles the
    enrichment skipped) are dropped — they are unsearchable in practice.

    Writes the bm25s index files plus a ``manifest.json`` to
    ``output_dir``. The metadata is stored as the bm25s "corpus" so
    ``BabbelaarSearchIndex`` can return it directly from ``retrieve()``.

    Returns a small build summary dict.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata: list[dict] = []
    texts: list[str] = []
    skipped_no_summary = 0
    for rec in records:
        if not rec.get("summary"):
            skipped_no_summary += 1
            continue
        metadata.append(_hit_metadata(rec))
        texts.append(_indexable_text(rec))

    if not texts:
        raise RuntimeError("No indexable records found — check input source.")

    # Tokenize. ``stopwords=None`` keeps Dutch function words; BM25 weights
    # them low naturally and removing them risks dropping useful signal in
    # 19th-century Dutch where function-word frequencies differ from modern.
    tokenized = bm25s.tokenize(texts, stopwords=None, show_progress=True)

    bm25 = bm25s.BM25()
    bm25.index(tokenized, show_progress=True)
    bm25.save(str(output_dir), corpus=metadata)

    # Manifest captures build-time facts that aren't in the bm25s files.
    manifest = {
        "n_documents": len(metadata),
        "skipped_no_summary": skipped_no_summary,
        "indexable_fields": ["summary", "topics", "title"],
        "tokenizer": "bm25s.tokenize(stopwords=None)",
        "version": 1,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest


# -----------------------------------------------------------------------------
# Index query (used by step 143 and the inference runtime)
# -----------------------------------------------------------------------------

class BabbelaarSearchIndex:
    """Loaded BM25 index over Delpher kranten. Construct once, reuse.

    Thread-safety: bm25s' retrieve is read-only and safe to call from
    multiple workers, but a single instance per process is the intended
    pattern (see module-level ``search()``).
    """

    def __init__(self, index_dir: Path | str | None = None):
        self.index_dir = Path(index_dir) if index_dir else default_index_dir()
        if not (self.index_dir / "params.index.json").exists():
            raise FileNotFoundError(
                f"No bm25s index at {self.index_dir}. "
                f"Run step 142 (or set BABBELAAR_INDEX_DIR)."
            )
        self.bm25 = bm25s.BM25.load(str(self.index_dir), load_corpus=True)
        # bm25s loads each corpus row as a JSON-decoded dict (we wrote
        # plain dicts in build_index). Verify the shape so misconfigured
        # indexes fail loudly here rather than at query time.
        sample = self.bm25.corpus[0] if len(self.bm25.corpus) else {}
        if isinstance(sample, dict) and "text" in sample and "metadata" in sample:
            # bm25s wraps dicts in {id, text, metadata} when it serializes.
            # Unwrap so downstream sees the original metadata dict.
            self._corpus = [row["metadata"] for row in self.bm25.corpus]
        else:
            self._corpus = list(self.bm25.corpus)

    # ---- query ---------------------------------------------------------

    def search(
        self,
        query: str,
        year_from: int | None = None,
        year_to: int | None = None,
        limit: int = 10,
    ) -> list[dict]:
        """Return up to ``limit`` matching records.

        Each hit is a dict with keys ``date``, ``title``, ``summary``, ``url``.

        If ``year_from`` or ``year_to`` is given, results are filtered
        to that range and **sorted ascending by date** (timeline mode).
        Otherwise results are sorted by descending BM25 score.
        """
        logger.debug("Running search with query=%r, year_from=%r, year_to=%r, limit=%r",
                    query, year_from, year_to, limit)
        if not query or not query.strip():
            # Year-only mode: no BM25 scoring, just scan corpus by date range.
            # Teaches the model that "Wat schreven de kranten in 1859?" →
            # search("", year_from=1859, year_to=1859) is a valid call.
            if year_from is None and year_to is None:
                return []
            lo = year_from or -10**9
            hi = year_to or 10**9
            hits = [h for h in self._corpus if lo <= _year_of(h) <= hi]
            hits.sort(key=lambda h: h.get("date") or "")
            return [{k: v for k, v in h.items() if not k.startswith("_")}
                    for h in hits[:limit]]
        # Over-fetch when filtering so the year filter does not starve the
        # result list. BM25 ranking can put in-range hits at positions 10+
        # when an out-of-range hit happens to share more keywords with the
        # summary text — without enough headroom, narrow year ranges silently
        # drop relevant hits. Larger over-fetch is still very cheap (a few ms
        # over 320k docs).
        if year_from is not None or year_to is not None:
            k_fetch = max(limit * 20, 200)
        else:
            k_fetch = limit
        k_fetch = min(k_fetch, len(self._corpus))
        if k_fetch <= 0:
            return []

        qtok = bm25s.tokenize([query], stopwords=None, show_progress=False)
        docs, scores = self.bm25.retrieve(
            qtok, corpus=self._corpus, k=k_fetch, show_progress=False
        )
        # docs/scores are 2D arrays shaped (1, k); flatten to first query.
        hits = [
            {**doc, "_score": float(score)}
            for doc, score in zip(docs[0].tolist(), scores[0].tolist())
        ]

        if year_from or year_to:
            lo = year_from or -10**9
            hi = year_to or 10**9
            hits = [h for h in hits if lo <= _year_of(h) <= hi]
            hits.sort(key=lambda h: h["date"])

        # Strip internal score field; the model never sees it.
        out = [{k: v for k, v in h.items() if not k.startswith("_")} for h in hits[:limit]]
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("Search results: %s", json.dumps(out, ensure_ascii=False, indent=2))
        return out

def _year_of(hit: dict) -> int:
    d = hit.get("date") or ""
    if not d:
        return 0
    try:
        return int(d[:4])
    except ValueError:
        return 0


# -----------------------------------------------------------------------------
# Module-level singleton (used by the inference sandbox)
# -----------------------------------------------------------------------------

_singleton: BabbelaarSearchIndex | None = None


def _ensure_loaded() -> BabbelaarSearchIndex:
    global _singleton
    if _singleton is None:
        _singleton = BabbelaarSearchIndex()
    return _singleton


def preload_index() -> BabbelaarSearchIndex | None:
    """Pre-warm the singleton index.

    Call once at server startup so the first search request doesn't pay the
    cold-load cost (~1 s to parse the 125 MB corpus from disk). Safe to call
    multiple times. Returns the loaded index, or None if the index is missing.
    """
    try:
        return _ensure_loaded()
    except Exception as exc:
        logger.warning("preload_index failed: %s", exc)
        return None


def search(
    query: str,
    year_from: int | None = None,
    year_to: int | None = None,
    limit: int = 10,
) -> None:
    """Run a search and **print the JSON-encoded hits** to stdout.

    The print-not-return convention is deliberate: this function is
    registered in the nanochat python sandbox at inference time, where
    the model emits ``<|python_start|>search(...)<|python_end|>`` and
    the runtime captures stdout into ``<|output_start|>...<|output_end|>``.
    Step 143 produces SFT examples in the same shape so the model sees a
    consistent format at training time and inference time.
    """
    idx = _ensure_loaded()
    hits = idx.search(query, year_from=year_from, year_to=year_to, limit=limit)
    sys.stdout.write(json.dumps(hits, ensure_ascii=False))
    sys.stdout.write("\n")
    sys.stdout.flush()


# -----------------------------------------------------------------------------
# Sandbox registration helper (used by nanochat inference)
# -----------------------------------------------------------------------------

def register_in_sandbox(namespace: dict) -> None:
    """Register ``search`` (and nothing else) in a Python sandbox namespace.

    The nanochat inference loop calls this when preparing the namespace
    that runs ``<|python_start|>``-block code. Keeping this as a single
    explicit function (rather than scattering ``namespace['search'] = ...``
    across the codebase) keeps the runtime contract auditable.
    """
    namespace["search"] = search
