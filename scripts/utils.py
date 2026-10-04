"""
Shared utility functions for the Benchmark Suite pipeline.
"""
from __future__ import annotations

import re
from pathlib import Path

import xxhash

# Separator used to append the dataset source identifier to a sanitized name.
# sanitize_name() collapses every run of underscores into a single one, so a
# double underscore can never be produced from a dataset name itself and stays
# unambiguous when the name is parsed back.
SOURCE_ID_SEPARATOR = "__"

# Chunk size used when streaming file contents into a checksum.
_CHECKSUM_CHUNK_SIZE = 1 << 20


def file_checksum(path: Path | str) -> str:
    """Returns the deterministic xxh64 hex digest of a file's raw bytes.

    Used by the incremental catalog: cheap to compute, stable across runs and
    platforms, and precise enough to tell whether a raw CSV changed since the
    previous build.
    """
    hasher = xxhash.xxh64()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(_CHECKSUM_CHUNK_SIZE), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def sanitize_name(name: str) -> str:
    """
    Transforms a raw dataset name into a safe, snake_case format for file systems.
    Matches exact folder sanitization logic required by the OpenML API records.

    Note:
        This function is not injective: different raw names ("Dataset A",
        "Dataset-A" and "Dataset_A") collapse into the same output. Always use
        dataset_storage_name() when the result identifies a directory or file
        on disk.
    """
    s = str(name).lower().strip()
    s = re.sub(r'[\s\-\(\)]+', '_', s)
    s = re.sub(r'[^\w]', '', s)
    s = re.sub(r'_+', '_', s)
    return s.strip('_')


def dataset_storage_name(name: str, source_id: int | str | None = None) -> str:
    """
    Builds a deterministic, collision-free and human-readable directory name for
    a single dataset.

    The sanitized name alone is ambiguous ("Dataset A", "Dataset-A" and
    "Dataset_A" all become "dataset_a"), so the dataset identifier in its source
    repository (the OpenML dataset id used by the ingestion layer) is appended
    when it is known, e.g. "dataset_a__61".
    """
    base = sanitize_name(name) or "dataset"
    if source_id is None or str(source_id).strip() == "":
        return base
    return f"{base}{SOURCE_ID_SEPARATOR}{source_id}"


def extract_source_id(storage_name: str) -> int | None:
    """
    Recovers the source identifier from a name produced by dataset_storage_name().

    Returns None for directories that were added manually (no identifier
    suffix), so callers can fall back to name based lookups.
    """
    match = re.search(rf"{re.escape(SOURCE_ID_SEPARATOR)}(\d+)$", storage_name)
    return int(match.group(1)) if match else None
