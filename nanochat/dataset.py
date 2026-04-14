"""
Parquet dataset utilities for Babbelaar pretraining.

Parquet shards live in the directory pointed to by NANOCHAT_DATA_DIR.
Files must use split-prefixed naming: train-*.parquet / validation-*.parquet.

By default files are stored in ~/.cache/nanochat/base_data_babbelaar (same
convention as upstream nanochat). Override with NANOCHAT_DATA_DIR.
SFT files default to ~/.cache/nanochat/sft_data_babbelaar. Override with NANOCHAT_SFT_DIR.

To download a subset of the pretraining corpus from HuggingFace:

    python -m nanochat.dataset --num-train-shards 3   # smoke test: ~600 MB
    python -m nanochat.dataset                         # full corpus: ~14 GB

To download the SFT files:

    python -m nanochat.dataset --sft

All validation shards are always downloaded. Files already present are skipped.
"""

import os
import argparse
import pyarrow.parquet as pq

from nanochat.common import get_base_dir

HF_REPO_ID = "fdeantoni/max-babbelaar-corpus"
HF_SUBDIR = "data/all"
DEFAULT_LOCAL_DIR = os.path.join(get_base_dir(), "base_data_babbelaar")

NUM_TRAIN_SHARDS = 66
NUM_VAL_SHARDS = 4

HF_SFT_REPO_ID = "fdeantoni/max-babbelaar-sft"
SFT_FILES = ["sft_train.jsonl", "sft_val.jsonl"]
DEFAULT_SFT_DIR = os.path.join(get_base_dir(), "sft_data_babbelaar")

# Default to ~/.cache/nanochat/base_data_babbelaar; override with NANOCHAT_DATA_DIR.
DATA_DIR = os.environ.get("NANOCHAT_DATA_DIR", DEFAULT_LOCAL_DIR)
# Default to ~/.cache/nanochat/sft_data_babbelaar; override with NANOCHAT_SFT_DIR.
SFT_DIR = os.environ.get("NANOCHAT_SFT_DIR", DEFAULT_SFT_DIR)


def list_parquet_files(data_dir=None, split=None, **_kwargs):
    """Return sorted full paths to all Parquet files in *data_dir*.

    When *split* is "train" or "val" only files with the matching prefix
    (``train-*.parquet`` / ``validation-*.parquet``) are returned.
    """
    data_dir = data_dir or DATA_DIR
    assert os.path.isdir(data_dir), (
        f"Dataset directory does not exist: {data_dir}\n"
        "Run `python -m nanochat.dataset` to download the dataset."
    )

    all_files = sorted(
        f for f in os.listdir(data_dir)
        if f.endswith(".parquet") and not f.endswith(".tmp")
    )

    if split is not None:
        assert split in ("train", "val"), f"split must be 'train' or 'val', got {split!r}"
        prefix = "train-" if split == "train" else "validation-"
        all_files = [f for f in all_files if f.startswith(prefix)]

    assert all_files, (
        f"No {'train-' if split == 'train' else 'validation-' if split else ''}*.parquet "
        f"files found in {data_dir}"
    )
    return [os.path.join(data_dir, f) for f in all_files]


def parquets_iter_batched(split, start=0, step=1):
    """Iterate through the dataset in row-group batches.

    Yields lists of document strings.
    *split* is "train" or "val".
    *start* / *step* support DDP sharding (start=rank, step=world_size).
    """
    assert split in ("train", "val"), f"split must be 'train' or 'val', got {split!r}"
    for filepath in list_parquet_files(split=split):
        pf = pq.ParquetFile(filepath)
        for rg_idx in range(start, pf.num_row_groups, step):
            rg = pf.read_row_group(rg_idx)
            yield rg.column("text").to_pylist()


def _shard_filename(split, index, total):
    """Return the HuggingFace filename for a shard, e.g. train-00003-of-00066.parquet"""
    prefix = "train" if split == "train" else "validation"
    return f"{prefix}-{index:05d}-of-{total:05d}.parquet"


def download(local_dir=DEFAULT_LOCAL_DIR, num_train_shards=None):
    """Download Babbelaar Parquet shards from HuggingFace Hub into *local_dir*.

    If *local_dir* already contains Parquet files those are counted first;
    only missing files are fetched. All validation shards are always downloaded.

    *num_train_shards* caps how many train shards are downloaded (default: all).
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise SystemExit(
            "huggingface_hub is not installed.\n"
            "Install it with:  pip install huggingface_hub"
        )

    os.makedirs(local_dir, exist_ok=True)

    n_train = num_train_shards if num_train_shards is not None else NUM_TRAIN_SHARDS

    # Build the list of files we want
    wanted = []
    for i in range(n_train):
        wanted.append(("train", i, NUM_TRAIN_SHARDS))
    for i in range(NUM_VAL_SHARDS):
        wanted.append(("val", i, NUM_VAL_SHARDS))

    # Filter out files already present
    already = set(os.listdir(local_dir))
    missing = [
        (split, idx, total) for split, idx, total in wanted
        if _shard_filename(split, idx, total) not in already
    ]

    if not missing:
        print(f"All {len(wanted)} shards already present in {local_dir} — nothing to download.")
        _print_export(local_dir)
        return

    print(f"{len(already)} shards already present; downloading {len(missing)} missing shards from {HF_REPO_ID}...")

    for split, idx, total in missing:
        filename = _shard_filename(split, idx, total)
        hf_path = f"{HF_SUBDIR}/{filename}"
        dest = os.path.join(local_dir, filename)
        print(f"  {hf_path}")
        hf_hub_download(
            repo_id=HF_REPO_ID,
            repo_type="dataset",
            filename=hf_path,
            local_dir=os.path.dirname(local_dir),  # parent: ./data (mirrors data/all/ structure)
        )

    total_present = len(os.listdir(local_dir))
    print(f"\nDone — {total_present} Parquet files in {local_dir}")
    _print_export(local_dir)


def download_sft(sft_dir=None):
    """Download SFT JSONL files from HuggingFace Hub into *sft_dir*.

    Downloads ``sft_train.jsonl`` and ``sft_val.jsonl`` from
    ``fdeantoni/max-babbelaar-sft``. Files already present are skipped.
    """
    sft_dir = sft_dir or SFT_DIR
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise SystemExit(
            "huggingface_hub is not installed.\n"
            "Install it with:  pip install huggingface_hub"
        )

    os.makedirs(sft_dir, exist_ok=True)

    already = set(os.listdir(sft_dir))
    missing = [f for f in SFT_FILES if f not in already]

    if not missing:
        print(f"All SFT files already present in {sft_dir} — nothing to download.")
        _print_sft_export(sft_dir)
        return

    present = len(SFT_FILES) - len(missing)
    print(f"{present} SFT file(s) already present; downloading {len(missing)} missing file(s) from {HF_SFT_REPO_ID}...")

    for filename in missing:
        dest = os.path.join(sft_dir, filename)
        print(f"  {filename}")
        hf_hub_download(
            repo_id=HF_SFT_REPO_ID,
            repo_type="dataset",
            filename=filename,
            local_dir=sft_dir,
        )

    print(f"\nDone — SFT files in {sft_dir}")
    _print_sft_export(sft_dir)


def _print_sft_export(sft_dir):
    abs_dir = os.path.abspath(sft_dir)
    print(f"\nSet the SFT file paths before fine-tuning:")
    print(f"  export SFT_TRAIN_FILE={abs_dir}/sft_train.jsonl")
    print(f"  export SFT_VAL_FILE={abs_dir}/sft_val.jsonl")


def _print_export(local_dir):
    abs_dir = os.path.abspath(local_dir)
    print(f"\nSet the data directory before training:")
    print(f"  export NANOCHAT_DATA_DIR={abs_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=f"Download the Babbelaar pretraining corpus or SFT files from HuggingFace"
    )
    parser.add_argument(
        "--local-dir",
        default=DEFAULT_LOCAL_DIR,
        help=f"destination directory for Parquet shards (default: {DEFAULT_LOCAL_DIR})",
    )
    parser.add_argument(
        "--num-train-shards",
        type=int,
        default=None,
        help=f"number of train shards to download (default: all {NUM_TRAIN_SHARDS}); "
             "all validation shards are always downloaded",
    )
    parser.add_argument(
        "--sft",
        action="store_true",
        help=f"download SFT JSONL files from {HF_SFT_REPO_ID} instead of Parquet shards",
    )
    parser.add_argument(
        "--sft-dir",
        default=None,
        help=f"destination directory for SFT files (default: NANOCHAT_SFT_DIR env var, or {DEFAULT_SFT_DIR})",
    )
    args = parser.parse_args()
    if args.sft:
        download_sft(args.sft_dir)
    else:
        download(args.local_dir, args.num_train_shards)
