#!/usr/bin/env python3
from __future__ import annotations

import datetime
from typing import Any, Sequence

import pandas as pd
import yaml

from config import (
    ROOT_DIR,
    CATALOG_CSV_PATH,
    DATASETS_CATALOG_DIR,
    PROCESSED_DIR,
    MANIFEST_DIR,
    MIN_INSTANCES,
    MAX_INSTANCES,
    MIN_FEATURES,
    MAX_FEATURES,
    MAX_MISSING_RATE,
    MIN_MINORITY_CLASS_COUNT,
    MAX_CLASSIFICATION_CLASSES,
    GLOBAL_MIN_CLASS_SIZE
)
from logging_utils import StageOutcome, configure_logging, log_event, stage_context

# Failures that are scoped to a single dataset: unreadable or corrupted CSVs and
# malformed per-dataset metadata. They are reported, the dataset is flagged as
# "error" in its catalog YAML and the run continues with the next dataset.
# Anything else is a defect in the pipeline and is left to propagate.
DATASET_PROCESSING_ERRORS = (
    OSError,
    UnicodeDecodeError,
    yaml.YAMLError,
    pd.errors.ParserError,
    pd.errors.EmptyDataError,
)

def load_dataset_target(dataset_id: str) -> str:
    """Reads the inferred target variable from the catalog YAML."""
    yaml_path = DATASETS_CATALOG_DIR / f"{dataset_id}.yaml"
    if yaml_path.exists():
        with open(yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        task = data.get("task", {}) if isinstance(data, dict) else {}
        return task.get("target_variable", "unknown") if isinstance(task, dict) else "unknown"
    return "unknown"

def update_dataset_yaml_status(dataset_id: str, status: str):
    """Updates the individual dataset YAML status."""
    yaml_path = DATASETS_CATALOG_DIR / f"{dataset_id}.yaml"
    if yaml_path.exists():
        with open(yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict):
            print(f"    [WARNING] Cannot flag '{yaml_path.name}' as '{status}': the file is empty or not a mapping.")
            return
        data["status"] = status
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

def main(argv: Sequence[str] | None = None) -> int:
    """Runs the validation/cleaning gate. Returns an exit code (0 = success)."""
    configure_logging()
    with stage_context("validate") as stage:
        return run_validation_stage(stage)


def reject_dataset(collection_id: str, dataset_id: str, reason: str) -> None:
    """Flags a dataset as rejected and emits one structured record."""
    update_dataset_yaml_status(dataset_id, "rejected")
    log_event(
        "dataset_rejected", collection_id=collection_id, dataset_id=dataset_id,
        operation="validate", outcome="rejected", reason=reason,
    )


def run_validation_stage(stage: StageOutcome) -> int:
    if not CATALOG_CSV_PATH.exists():
        print(f"[ERROR] Master catalog not found at: {CATALOG_CSV_PATH}")
        stage.fail("master catalog not found", reason=str(CATALOG_CSV_PATH))
        return 1

    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    master_catalog = pd.read_csv(CATALOG_CSV_PATH)
    print(f"[INFO] Loaded master catalog. Processing {len(master_catalog)} datasets for sanitization...\n")

    collection_metrics: dict[str, dict[str, Any]] = {}
    rejected_count = 0
    validated_count = 0
    failed_count = 0

    for _, row in master_catalog.iterrows():
        collection_id = str(row["collection_id"])
        dataset_id = str(row["dataset_id"])
        raw_csv_relative = str(row["csv_path"])
        csv_path = ROOT_DIR / raw_csv_relative

        if collection_id not in collection_metrics:
            collection_metrics[collection_id] = {
                "total_matched_before": 0,
                "validated_datasets": []
            }

        collection_metrics[collection_id]["total_matched_before"] += 1
        print(f"[INFO] Sanitizing: [{collection_id}] -> {dataset_id}")

        if not csv_path.exists():
            reason = f"Raw CSV file missing at {raw_csv_relative}"
            print(f"    [REJECTED] {reason}")
            reject_dataset(collection_id, dataset_id, reason)
            rejected_count += 1
            continue

        try:
            df = pd.read_csv(csv_path, low_memory=False)
            target_col = load_dataset_target(dataset_id)
            yaml_file_path = DATASETS_CATALOG_DIR / f"{dataset_id}.yaml"

            if not yaml_file_path.exists():
                reason = f"Metadata YAML missing at {yaml_file_path.name}"
                print(f"    [REJECTED] {reason}")
                reject_dataset(collection_id, dataset_id, reason)
                rejected_count += 1
                continue

            with open(yaml_file_path, 'r', encoding='utf-8') as f:
                current_yaml = yaml.safe_load(f)

            if not isinstance(current_yaml, dict):
                reason = f"Metadata file {yaml_file_path.name} is empty or not a mapping"
                print(f"    [REJECTED] {reason}.")
                reject_dataset(collection_id, dataset_id, reason)
                rejected_count += 1
                continue

            task_info = current_yaml.get("task", {})
            task_type = task_info.get("type", "unknown")

            if task_type not in ["classification", "regression"]:
                reason = f"Unsupported task type: '{task_type}'"
                print(f"    [REJECTED] {reason}.")
                reject_dataset(collection_id, dataset_id, reason)
                rejected_count += 1
                continue

            if target_col not in df.columns:
                reason = f"Target variable '{target_col}' not found"
                print(f"    [REJECTED] {reason}.")
                reject_dataset(collection_id, dataset_id, reason)
                rejected_count += 1
                continue

            # Drop rows with missing targets
            initial_rows_count = len(df)
            df = df.dropna(subset=[target_col])
            if (dropped := initial_rows_count - len(df)) > 0:
                print(f"    -> Dropped {dropped} rows missing target values.")

            # Drop duplicated rows
            initial_rows = len(df)
            df = df.drop_duplicates()
            if (dropped := initial_rows - len(df)) > 0:
                print(f"    -> Removed {dropped} duplicate rows.")

            unique_target_count = df[target_col].nunique()
            if unique_target_count <= 1:
                reason = f"Target variable '{target_col}' is constant"
                print(f"    [REJECTED] {reason}.")
                reject_dataset(collection_id, dataset_id, reason)
                rejected_count += 1
                continue

            if task_type == "classification":
                if unique_target_count > MAX_CLASSIFICATION_CLASSES:
                    reason = (f"Target cardinality too high ({unique_target_count}). "
                              "Suspected continuous feature")
                    print(f"    [REJECTED] {reason}.")
                    reject_dataset(collection_id, dataset_id, reason)
                    rejected_count += 1
                    continue

                min_class_size = df[target_col].value_counts().min()
                if min_class_size < MIN_MINORITY_CLASS_COUNT:
                    reason = (f"Minority class size ({min_class_size}) below threshold "
                              f"({MIN_MINORITY_CLASS_COUNT})")
                    print(f"    [REJECTED] {reason}.")
                    reject_dataset(collection_id, dataset_id, reason)
                    rejected_count += 1
                    continue

            global_min_class_size = df[target_col].value_counts().min()
            if global_min_class_size < GLOBAL_MIN_CLASS_SIZE:
                reason = "Dataset contains singleton targets"
                print(f"    [REJECTED] {reason}.")
                reject_dataset(collection_id, dataset_id, reason)
                rejected_count += 1
                continue

            # Column pruning
            cols_to_drop_missing = [c for c in df.columns if df[c].isnull().mean() > MAX_MISSING_RATE and c != target_col]
            if cols_to_drop_missing:
                df = df.drop(columns=cols_to_drop_missing)
                print(f"    -> Dropped {len(cols_to_drop_missing)} columns exceeding missing threshold.")

            nunique = df.nunique()
            constant_cols = [c for c in nunique[nunique == 1].index if c != target_col]
            if constant_cols:
                df = df.drop(columns=constant_cols)
                print(f"    -> Dropped {len(constant_cols)} constant columns.")

            high_cardinality_cols = [
                c for c in nunique[(nunique == len(df)) & (len(df) > 1)].index
                if c != target_col and not pd.api.types.is_float_dtype(df[c])
            ]
            if high_cardinality_cols:
                df = df.drop(columns=high_cardinality_cols)
                print(f"    -> Dropped {len(high_cardinality_cols)} leakage ID columns.")

            # Validate final structure against configured constraints
            final_instances = len(df)
            if final_instances < MIN_INSTANCES or final_instances > MAX_INSTANCES:
                reason = f"Instance count ({final_instances}) out of bounds"
                print(f"    [REJECTED] {reason}.")
                reject_dataset(collection_id, dataset_id, reason)
                rejected_count += 1
                continue

            final_features = len(df.columns) - 1
            if final_features < MIN_FEATURES or final_features > MAX_FEATURES:
                reason = f"Feature count ({final_features}) out of bounds"
                print(f"    [REJECTED] {reason}.")
                reject_dataset(collection_id, dataset_id, reason)
                rejected_count += 1
                continue

            processed_dir = PROCESSED_DIR / collection_id / dataset_id
            processed_dir.mkdir(parents=True, exist_ok=True)
            processed_csv_path = processed_dir / "data.csv"
            df.to_csv(processed_csv_path, index=False)

            update_dataset_yaml_status(dataset_id, "validated")
            validated_count += 1
            print("    [VALIDATED] Cleaned matrix saved successfully.")
            log_event(
                "dataset_validated", collection_id=collection_id, dataset_id=dataset_id,
                operation="validate", outcome="success", rows=int(final_instances),
            )

            df_features_only = df.drop(columns=[target_col])
            collection_metrics[collection_id]["validated_datasets"].append({
                "dataset_id": dataset_id,
                "frozen_metadata": {
                    "name": current_yaml.get("name", dataset_id.replace("_", " ").title()),
                    "task": task_info,
                    "structure": {
                        "n_instances": int(final_instances),
                        "n_features": int(final_features),
                        "missing_rate": round(float(df.isnull().sum().sum() / (final_instances * len(df.columns))), 4)
                    },
                    "feature_types": {
                        "numeric": len(df_features_only.select_dtypes(include=['number']).columns),
                        "categorical": len(df_features_only.select_dtypes(include=['object', 'category', 'str']).columns)
                    },
                    "status": "validated"
                }
            })

        except DATASET_PROCESSING_ERRORS as e:
            print(f"    [FAILED] {collection_id}/{dataset_id}: {type(e).__name__}: {e}")
            update_dataset_yaml_status(dataset_id, "error")
            failed_count += 1
            log_event(
                "dataset_failed", level=40, collection_id=collection_id, dataset_id=dataset_id,
                operation="validate", outcome="failure", error=f"{type(e).__name__}: {e}",
            )

    print("\n--- Generating Collection Manifests ---")
    current_time_iso = datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')

    for collection_id, metrics in collection_metrics.items():
        manifest_path = MANIFEST_DIR / f"{collection_id}.resolved.yaml"
        final_count = len(metrics["validated_datasets"])

        manifest_data = {
            "manifest_id": f"{collection_id}_resolved_v1",
            "source_collection": collection_id,
            "resolved_at": current_time_iso,
            "source_files": {"catalog_file": "catalog/catalog.csv"},
            "filters_applied": {
                "all_of": [
                    {"field": "status", "op": "==", "value": "validated"},
                    {"field": "n_instances", "op": "between", "value": [MIN_INSTANCES, MAX_INSTANCES]},
                    {"field": "n_features", "op": "between", "value": [MIN_FEATURES, MAX_FEATURES]},
                    {"field": "missing_rate_per_column", "op": "<=", "value": MAX_MISSING_RATE}
                ]
            },
            "materialization": {
                "total_matched_before_clean": metrics["total_matched_before"],
                "final_dataset_count": final_count
            },
            "selected_datasets": metrics["validated_datasets"]
        }

        with open(manifest_path, 'w', encoding='utf-8') as f:
            yaml.dump(manifest_data, f, default_flow_style=False, sort_keys=False)

        print(f"[SUCCESS] Manifest Materialized: {manifest_path.name} ({final_count} verified datasets)")
        log_event(
            "manifest_written", collection_id=collection_id, operation="validate",
            outcome="success", selected=final_count,
        )

    log_event(
        "validate_completed", operation="validate", outcome="success",
        validated=validated_count, rejected=rejected_count, failed=failed_count,
    )
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
