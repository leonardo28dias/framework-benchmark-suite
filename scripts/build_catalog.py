#!/usr/bin/env python3
from __future__ import annotations

import sys
import urllib.error
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from config import RAW_DIR, DATASETS_CATALOG_DIR, CATALOG_CSV_PATH, ROOT_DIR
from utils import extract_source_id, sanitize_name, file_checksum
from logging_utils import configure_logging, log_event, stage_context

import logging

try:
    import openml
except ImportError:
    print("[ERROR] The 'openml' package is missing. Please run: pip install openml")
    sys.exit(1)

# Failures raised by the OpenML API while enriching the catalog with academic
# metadata. Lookups are best effort, so they are logged and never block the
# cataloging of local files.
API_ACCESS_ERRORS = (openml.exceptions.PyOpenMLError, urllib.error.URLError, TimeoutError)

# A raw CSV that cannot be read or parsed is dataset scoped: it is reported and
# skipped instead of aborting the whole catalog run.
CSV_READ_ERRORS = (OSError, UnicodeDecodeError, pd.errors.ParserError, pd.errors.EmptyDataError)

# Keys every reusable catalog entry must provide. A metadata file that carries
# a matching checksum but is missing these is treated as absent and rebuilt.
METADATA_REQUIRED_KEYS = ("task", "structure", "feature_types", "data_quality", "academic_meta")

def infer_task_type(df: pd.DataFrame, target_col: str) -> str:
    """Infers classification or regression based on data type and cardinality."""
    if target_col not in df.columns:
        return "unknown"

    col_type = df[target_col].dtype
    nunique = df[target_col].nunique()

    if (
        pd.api.types.is_bool_dtype(col_type)
        or pd.api.types.is_object_dtype(col_type)
        or pd.api.types.is_string_dtype(col_type)
        or isinstance(col_type, pd.CategoricalDtype)
    ):
        return "classification"
    if pd.api.types.is_float_dtype(col_type):
        return "regression"
    if pd.api.types.is_integer_dtype(col_type):
        if nunique <= 15 or nunique < (len(df) * 0.05):
            return "classification"
        return "regression"
    return "unknown"

def analyze_dataframe(df: pd.DataFrame) -> dict:
    """Performs a structural x-ray of the dataframe to feed the YAML schema."""
    n_instances = len(df)
    n_features = len(df.columns)

    missing_counts = df.isnull().sum()
    missing_rate = float(missing_counts.sum() / (n_instances * n_features)) if n_features > 0 else 0.0
    cols_with_missing = missing_counts[missing_counts > 0].index.tolist()

    duplicate_rows = int(df.duplicated().sum())
    memory_mb = float(round(df.memory_usage(deep=True).sum() / (1024 * 1024), 3))

    numeric_cols = df.select_dtypes(include=['number']).columns
    datetime_cols = df.select_dtypes(include=['datetime']).columns
    categorical_cols = df.select_dtypes(include=['object', 'category', 'str']).columns

    nunique = df.nunique()
    constant_columns = nunique[nunique == 1].index.tolist()
    high_cardinality = nunique[(nunique == n_instances) & (n_instances > 1)].index.tolist()
    binary_cols = nunique[nunique == 2].index.tolist()

    target_col = df.columns[-1] if n_features > 0 else "unknown"
    inferred_task = infer_task_type(df, target_col)

    return {
        "inferred_task": inferred_task,
        "target_variable": target_col,
        "structure": {
            "n_instances": int(n_instances),
            "n_features": int(n_features),
            "missing_rate": round(missing_rate, 4),
            "duplicate_rows": duplicate_rows,
            "memory_usage_mb": memory_mb
        },
        "feature_types": {
            "numeric": len(numeric_cols),
            "categorical": len(categorical_cols),
            "binary": len(binary_cols),
            "datetime": len(datetime_cols)
        },
        "data_quality": {
            "constant_columns": constant_columns,
            "high_cardinality_columns": high_cardinality,
            "columns_with_missing": cols_with_missing
        }
    }

def load_reusable_metadata(yaml_path: Path, checksum: str) -> dict[str, Any] | None:
    """Returns preserved catalog metadata when the raw file is byte-identical.

    The previous per-dataset YAML is reused only when it stores the same
    checksum AND carries all required sections. Missing files, unreadable
    content and absent/stale checksums all mean the dataset is rebuilt.
    """
    try:
        with yaml_path.open("r", encoding="utf-8") as f:
            metadata = yaml.safe_load(f)
    except (OSError, yaml.YAMLError, UnicodeDecodeError):
        return None

    if not isinstance(metadata, dict):
        return None
    if metadata.get("raw_checksum") != checksum:
        return None
    if any(key not in metadata for key in METADATA_REQUIRED_KEYS):
        return None
    return metadata


def build_flat_entry(
    collection_id: str,
    dataset_id: str,
    csv_path: Path,
    yaml_path: Path,
    metadata: dict[str, Any],
    checksum: str,
) -> dict[str, Any]:
    """Builds one master-catalog row from stored (possibly reused) metadata."""
    structure = metadata["structure"]
    data_quality = metadata["data_quality"]
    academic_meta = metadata.get("academic_meta", {})
    citation_info = academic_meta.get("citation", "unknown")
    return {
        "collection_id": collection_id,
        "dataset_id": dataset_id,
        "n_instances": structure["n_instances"],
        "n_features": structure["n_features"],
        "missing_rate": structure["missing_rate"],
        "has_constants": len(data_quality["constant_columns"]) > 0,
        "has_high_cardinality": len(data_quality["high_cardinality_columns"]) > 0,
        "has_citation": citation_info != "unknown",
        "license": academic_meta.get("license", "unknown"),
        "status": "cataloged",
        "checksum": checksum,
        "csv_path": str(csv_path.relative_to(ROOT_DIR)),
        "yaml_path": str(yaml_path.relative_to(ROOT_DIR)),
    }


def main(argv: list[str] | None = None) -> int:
    """Catalogs raw CSVs into per-dataset YAML metadata and catalog.csv.

    The run is incremental: datasets whose raw checksum matches the previous
    build are skipped (their preserved metadata is reused), everything else is
    fully reprocessed. Returns an exit code (0 = success).
    """
    configure_logging()
    logging.getLogger("openml").setLevel(logging.ERROR)

    with stage_context("catalog") as stage:
        if not RAW_DIR.exists():
            print(f"[ERROR] Raw data directory not found at: {RAW_DIR}")
            stage.fail("raw directory not found", reason=str(RAW_DIR))
            return 1

        skipped_count = 0
        DATASETS_CATALOG_DIR.mkdir(parents=True, exist_ok=True)

        print("[INFO] Connecting to OpenML API to cache academic metadata tracking...")
        openml_lookup = {}
        try:
            datasets_df = openml.datasets.list_datasets(
                number_instances="500..10000",
                number_features="5..100",
                status="active",
                output_format="dataframe"
            )
            if isinstance(datasets_df, pd.DataFrame):
                for _, row in datasets_df.iterrows():
                    sanitized = sanitize_name(row["name"])
                    openml_lookup[sanitized] = int(row["did"])
            else:
                print(f"[WARNING] OpenML returned {type(datasets_df).__name__} instead of a dataset listing; "
                      "academic metadata will stay 'unknown'.")
        except API_ACCESS_ERRORS as e:
            print(f"[WARNING] Failed to pull master OpenML lookup list ({type(e).__name__}: {e}). "
                  "Academic metadata will stay 'unknown' for datasets without an OpenML id.")

        catalog_entries = []

        # Sorted traversal keeps the generated catalog reproducible across runs.
        for collection_path in sorted(RAW_DIR.iterdir()):
            if not collection_path.is_dir():
                continue

            collection_id = collection_path.name

            for dataset_path in sorted(collection_path.iterdir()):
                if not dataset_path.is_dir():
                    continue

                dataset_id = dataset_path.name
                csv_path = dataset_path / "data.csv"
                yaml_path = DATASETS_CATALOG_DIR / f"{dataset_id}.yaml"

                if not csv_path.exists():
                    print(f"[WARNING] Skipping [{collection_id}] -> {dataset_id}: no data.csv found.")
                    log_event(
                        "dataset_skipped", level=30, collection_id=collection_id, dataset_id=dataset_id,
                        operation="catalog", outcome="skipped", reason="no data.csv found",
                    )
                    continue

                try:
                    checksum = file_checksum(csv_path)
                except OSError as e:
                    print(f"[ERROR] Failed to catalog [{collection_id}] -> {dataset_id} "
                          f"({type(e).__name__}: {e}). The dataset is not part of the catalog.")
                    log_event(
                        "dataset_failed", level=40, collection_id=collection_id, dataset_id=dataset_id,
                        operation="checksum", outcome="failure", error=f"{type(e).__name__}: {e}",
                    )
                    continue

                reusable = load_reusable_metadata(yaml_path, checksum)
                if reusable is not None:
                    # Incremental path: the raw bytes did not change since the previous
                    # build, so the expensive CSV read, analysis and metadata lookup
                    # are skipped and the preserved catalog information is reused.
                    print(f"[INFO] Unchanged: [{collection_id}] -> {dataset_id} (checksum match, reusing metadata)")
                    catalog_entries.append(build_flat_entry(
                        collection_id, dataset_id, csv_path, yaml_path, reusable, checksum,
                    ))
                    skipped_count += 1
                    log_event(
                        "dataset_skipped", collection_id=collection_id, dataset_id=dataset_id,
                        operation="catalog", outcome="skipped", reason="raw checksum unchanged",
                        checksum=checksum,
                    )
                    continue

                entry = process_one_dataset(
                    collection_id, dataset_id, csv_path, yaml_path, checksum, openml_lookup,
                )
                if entry is not None:
                    catalog_entries.append(entry)

        if catalog_entries:
            pd.DataFrame(catalog_entries).to_csv(CATALOG_CSV_PATH, index=False)
            print(f"\n[SUCCESS] Master catalog generated with {len(catalog_entries)} datasets at: {CATALOG_CSV_PATH}")
            log_event(
                "catalog_completed", operation="catalog", outcome="success",
                datasets=len(catalog_entries), skipped=skipped_count,
            )
        else:
            print("\n[WARNING] No valid data.csv files found to catalog.")
            log_event("catalog_completed", operation="catalog", outcome="empty", datasets=0)

        return 0


def process_one_dataset(
    collection_id: str,
    dataset_id: str,
    csv_path: Path,
    yaml_path: Path,
    checksum: str,
    openml_lookup: dict[str, int],
) -> dict[str, Any] | None:
    """Catalogs one dataset from scratch. Returns the flat catalog entry, or None on failure."""
    print(f"[INFO] Cataloging: [{collection_id}] -> {dataset_id}")

    citation_info = "unknown"
    paper_url_info = "unknown"
    license_info = "unknown"

    # Downloaded datasets carry their OpenML id in the folder name;
    # manually added datasets fall back to a name based lookup.
    openml_id = extract_source_id(dataset_id)
    if openml_id is None:
        openml_id = openml_lookup.get(dataset_id)

    if openml_id is not None:
        try:
            ds_meta = openml.datasets.get_dataset(openml_id, download_data=False)
            citation_info = getattr(ds_meta, "citation", None) or "unknown"
            paper_url_info = getattr(ds_meta, "paper_url", None) or "unknown"
            license_info = getattr(ds_meta, "licence", None) or getattr(ds_meta, "license", None) or "unknown"
        except API_ACCESS_ERRORS as e:
            print(f"[WARNING] Academic metadata lookup failed for {dataset_id} (OpenML id {openml_id}): "
                  f"{type(e).__name__}: {e}")
            log_event(
                "metadata_lookup_failed", level=30, collection_id=collection_id, dataset_id=dataset_id,
                operation="metadata_lookup", outcome="degraded",
                error=f"{type(e).__name__}: {e}", openml_id=openml_id,
            )

    try:
        df = pd.read_csv(csv_path, low_memory=False)
    except CSV_READ_ERRORS as e:
        print(f"[ERROR] Failed to catalog [{collection_id}] -> {dataset_id} "
              f"({type(e).__name__}: {e}). The dataset is not part of the catalog.")
        log_event(
            "dataset_failed", level=40, collection_id=collection_id, dataset_id=dataset_id,
            operation="catalog", outcome="failure", error=f"{type(e).__name__}: {e}",
            checksum=checksum,
        )
        return None

    analysis = analyze_dataframe(df)

    yaml_data = {
        "dataset_id": dataset_id,
        "collection_id": collection_id,
        "name": dataset_id.replace("_", " ").title(),
        "raw_checksum": checksum,
        "academic_meta": {
            "citation": citation_info,
            "paper_url": paper_url_info,
            "license": license_info
        },
        "task": {
            "type": analysis["inferred_task"],
            "target_variable": analysis["target_variable"]
        },
        "structure": analysis["structure"],
        "feature_types": analysis["feature_types"],
        "data_quality": analysis["data_quality"],
        "status": "cataloged"
    }

    with open(yaml_path, 'w', encoding="utf-8") as f:
        yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False)

    log_event(
        "dataset_cataloged", collection_id=collection_id, dataset_id=dataset_id,
        operation="catalog", outcome="success",
        rows=analysis["structure"]["n_instances"], task=analysis["inferred_task"],
    )
    return build_flat_entry(collection_id, dataset_id, csv_path, yaml_path, yaml_data, checksum)

if __name__ == "__main__":
    raise SystemExit(main())
