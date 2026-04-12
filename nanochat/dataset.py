"""
Parquet dataset utilities for Babbelaar pretraining.

Parquet shards live in the directory pointed to by NANOCHAT_DATA_DIR (required).
Files must use split-prefixed naming: train-*.parquet / validation-*.parquet.
"""

import os
import pyarrow.parquet as pq

# NANOCHAT_DATA_DIR must be set to the directory containing the Parquet shards.
DATA_DIR = os.environ.get("NANOCHAT_DATA_DIR")


def list_parquet_files(data_dir=None, split=None, **_kwargs):
    """Return sorted full paths to all Parquet files in *data_dir*.

    When *split* is "train" or "val" only files with the matching prefix
    (``train-*.parquet`` / ``validation-*.parquet``) are returned.
    """
    data_dir = data_dir or DATA_DIR
    assert data_dir, (
        "NANOCHAT_DATA_DIR is not set. "
        "Point it at the directory containing Babbelaar's Parquet shards, e.g.:\n"
        "  export NANOCHAT_DATA_DIR=/path/to/babbelaar/data/final/data/all"
    )
    assert os.path.isdir(data_dir), f"NANOCHAT_DATA_DIR does not exist: {data_dir}"

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
