"""
Parquet dataset utilities for Babbelaar pretraining.

Parquet shards live in the directory pointed to by NANOCHAT_DATA_DIR.
Files must use split-prefixed naming: train-*.parquet / validation-*.parquet.

To download the dataset from HuggingFace before training:

    python -m nanochat.dataset

This downloads fdeantoni/max-babbelaar-corpus into ./data/all/ and prints
the export command to set NANOCHAT_DATA_DIR.
"""

import os
import argparse
import pyarrow.parquet as pq

HF_REPO_ID = "fdeantoni/max-babbelaar-corpus"
HF_SUBDIR = "data/all"          # subdirectory within the repo that holds the shards
DEFAULT_LOCAL_DIR = "./data/all"

# NANOCHAT_DATA_DIR must be set to the directory containing the Parquet shards.
DATA_DIR = os.environ.get("NANOCHAT_DATA_DIR")


def list_parquet_files(data_dir=None, split=None, **_kwargs):
    """Return sorted full paths to all Parquet files in *data_dir*.

    When *split* is "train" or "val" only files with the matching prefix
    (``train-*.parquet`` / ``validation-*.parquet``) are returned.
    """
    data_dir = data_dir or DATA_DIR
    assert data_dir, (
        "NANOCHAT_DATA_DIR is not set.\n"
        "Download the dataset first:\n"
        "  python -m nanochat.dataset\n"
        "Then set the env var as printed by that command."
    )
    assert os.path.isdir(data_dir), (
        f"NANOCHAT_DATA_DIR does not exist: {data_dir}\n"
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


def download(local_dir=DEFAULT_LOCAL_DIR):
    """Download the Babbelaar Parquet shards from HuggingFace Hub.

    Only the ``data/all/`` subdirectory is downloaded — ~14 GB of train and
    validation shards. Files that already exist are skipped.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise SystemExit(
            "huggingface_hub is not installed.\n"
            "Install it with:  pip install huggingface_hub"
        )

    print(f"Downloading {HF_REPO_ID}/{HF_SUBDIR} → {local_dir}")
    print("(files already present will be skipped)\n")

    # snapshot_download fetches into a cache dir then symlinks; local_dir=... +
    # local_dir_use_symlinks=False gives us a plain copy we can point DATA_DIR at.
    snapshot_download(
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        allow_patterns=f"{HF_SUBDIR}/*.parquet",
        local_dir=os.path.dirname(local_dir),   # parent: ./data
        local_dir_use_symlinks=False,
    )

    # Confirm we got something
    parquet_count = len([
        f for f in os.listdir(local_dir)
        if f.endswith(".parquet")
    ])
    print(f"\nDone — {parquet_count} Parquet files in {local_dir}")
    abs_dir = os.path.abspath(local_dir)
    print(f"\nSet the data directory before training:")
    print(f"  export NANOCHAT_DATA_DIR={abs_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=f"Download the Babbelaar pretraining corpus from {HF_REPO_ID}"
    )
    parser.add_argument(
        "--local-dir",
        default=DEFAULT_LOCAL_DIR,
        help=f"destination directory (default: {DEFAULT_LOCAL_DIR})",
    )
    args = parser.parse_args()
    download(args.local_dir)
