#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import sys
import urllib.error
from pathlib import Path
from typing import Any, Sequence

import yaml
import pandas as pd
from pydantic import ValidationError

from config import RAW_DIR, COLLECTIONS_DIR, MIN_INSTANCES, MAX_INSTANCES, MIN_FEATURES, MAX_FEATURES
from utils import dataset_storage_name
from logging_utils import configure_logging, log_event, stage_context
from schemas import CollectionBlueprint
from validate_collections import compute_dataset_metrics, verify_against_criteria


def configure_openml_logging() -> None:
    """Quiets the verbose OpenML client; called from entry points, not on import."""
    logging.getLogger("openml").setLevel(logging.ERROR)

try:
    import openml
except ImportError:
    print("[ERROR] The 'openml' package is missing. Please run: pip install openml")
    sys.exit(1)

# Failures the external API can raise while searching, fetching metadata or
# downloading a single dataset. They are dataset scoped: the batch keeps going
# and the failure is logged together with the affected dataset id.
API_ACCESS_ERRORS = (openml.exceptions.PyOpenMLError, urllib.error.URLError, TimeoutError)

# A payload that downloads but cannot be parsed is dataset scoped as well.
DATA_FETCH_ERRORS = API_ACCESS_ERRORS + (ValueError,)

# Problems that make a collection blueprint on disk unusable.
COLLECTION_FILE_ERRORS = (OSError, yaml.YAMLError, ValueError)


def ensure_dirs():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    COLLECTIONS_DIR.mkdir(parents=True, exist_ok=True)


def load_collection_config(yaml_path: Path) -> CollectionBlueprint:
    """Loads and validates one collection blueprint.

    Flow: YAML file -> raw mapping -> CollectionBlueprint (Pydantic model).
    Pydantic ValidationError is wrapped as ValueError for a consistent error
    surface. Invalid configurations fail before any OpenML work starts.

    Raises:
        OSError: the file cannot be read.
        yaml.YAMLError: the file is not valid YAML.
        ValueError: the YAML does not describe a usable collection.
    """
    with yaml_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError("expected a YAML mapping with 'selection_criteria'")

    selection_criteria = raw.get("selection_criteria")
    if not isinstance(selection_criteria, dict):
        raise ValueError("missing a 'selection_criteria' mapping")

    conditions = selection_criteria.get("all_of")
    if not isinstance(conditions, list) or not conditions:
        raise ValueError("missing a non-empty 'selection_criteria.all_of' list")

    for index, condition in enumerate(conditions):
        if not isinstance(condition, dict) or not {"field", "op", "value"} <= set(condition):
            raise ValueError(f"invalid condition at index {index}: {condition!r}")

    try:
        return CollectionBlueprint.model_validate(raw)
    except ValidationError as e:
        raise ValueError(f"Invalid collection blueprint: {e}") from e
def parse_collection_bounds(all_of_conditions: list) -> tuple[int, int, int, int, str | None, int | None]:
    """Parses generic YAML all_of conditions to extract API optimization bounds."""
    inst_min, inst_max = MIN_INSTANCES, MAX_INSTANCES
    feat_min, feat_max = MIN_FEATURES, MAX_FEATURES
    task_type = None
    num_classes = None

    for cond in all_of_conditions:
        field, op, val = cond.get("field"), cond.get("op"), cond.get("value")
        try:
            if field == "n_instances":
                val_i = int(val)
                if op == "==":
                    inst_min = inst_max = val_i
                elif op == ">=":
                    inst_min = max(inst_min, val_i)
                elif op == ">":
                    inst_min = max(inst_min, val_i + 1)
                elif op == "<=":
                    inst_max = min(inst_max, val_i)
                elif op == "<":
                    inst_max = min(inst_max, val_i - 1)
            elif field == "n_features":
                val_i = int(val)
                if op == "==":
                    feat_min = feat_max = val_i
                elif op == ">=":
                    feat_min = max(feat_min, val_i)
                elif op == ">":
                    feat_min = max(feat_min, val_i + 1)
                elif op == "<=":
                    feat_max = min(feat_max, val_i)
                elif op == "<":
                    feat_max = min(feat_max, val_i - 1)
            elif field == "task" and op == "==":
                task_type = str(val).lower()
            elif field == "task_subtype" and op == "==" and str(val).lower() == "binary":
                num_classes = 2
        except (ValueError, TypeError) as e:
            print(f"[WARNING] Ignoring unusable bound condition {cond!r}: {type(e).__name__}: {e}")
            continue
    return inst_min, inst_max, feat_min, feat_max, task_type, num_classes

def download_openml_collection(collection_id: str, conditions: list, limit: int = 30):
    print(f"\n[INFO] Searching OpenML API for Collection: '{collection_id}'")
    inst_min, inst_max, feat_min, feat_max, _, num_classes = parse_collection_bounds(conditions)

    try:
        kwargs: dict[str, Any] = {
            "output_format": "dataframe",
            "number_instances": f"{inst_min}..{inst_max}",
            "number_features": f"{feat_min}..{feat_max}",
            "status": "active"
        }
        if num_classes is not None:
            kwargs["number_classes"] = str(num_classes)

        datasets_df = openml.datasets.list_datasets(**kwargs)
    except API_ACCESS_ERRORS as e:
        print(f"[ERROR] Failed to search OpenML for collection '{collection_id}': {type(e).__name__}: {e}")
        log_event(
            "search_failed", level=40, collection_id=collection_id,
            operation="search", outcome="failure", error=f"{type(e).__name__}: {e}",
        )
        return

    if not isinstance(datasets_df, pd.DataFrame):
        print(f"[ERROR] OpenML returned {type(datasets_df).__name__} instead of a dataset listing for '{collection_id}'.")
        return

    if datasets_df.empty:
        print("[INFO] No datasets found matching the initial collection criteria bounds.")
        return

    sort_col = "number_downloads" if "number_downloads" in datasets_df.columns else "NumberOfDownloads"
    if sort_col in datasets_df.columns:
        datasets_df = datasets_df.sort_values(by=sort_col, ascending=False)

    downloaded_count = 0
    for _, row in datasets_df.iterrows():
        if downloaded_count >= limit:
            break

        did = int(row["did"])

        try:
            ds = openml.datasets.get_dataset(did, download_data=False)
        except API_ACCESS_ERRORS as e:
            print(f"[WARNING] Skipping dataset {did}: metadata lookup failed ({type(e).__name__}: {e})")
            log_event(
                "dataset_skipped", level=30, collection_id=collection_id, dataset_id=str(did),
                operation="metadata_lookup", outcome="skipped", error=f"{type(e).__name__}: {e}",
            )
            continue

        license_info = getattr(ds, "licence", None) or getattr(ds, "license", None)
        citation_info = getattr(ds, "citation", None) or getattr(ds, "paper_url", None)
        if not license_info or not citation_info:
            print(f"[SKIPPED] ID {did} ('{ds.name}'): no license or citation metadata published.")
            log_event(
                "dataset_skipped", collection_id=collection_id, dataset_id=str(did),
                operation="license_check", outcome="skipped", reason="missing license or citation metadata",
            )
            continue

        try:
            X, _, _, _ = ds.get_data(target=None, dataset_format="dataframe")
        except DATA_FETCH_ERRORS as e:
            print(f"[WARNING] Skipping dataset {did} ('{ds.name}'): download failed ({type(e).__name__}: {e})")
            log_event(
                "dataset_failed", level=40, collection_id=collection_id, dataset_id=str(did),
                operation="download", outcome="failure", error=f"{type(e).__name__}: {e}",
            )
            continue

        if not isinstance(X, pd.DataFrame) or X.empty:
            print(f"[SKIPPED] ID {did} ('{ds.name}'): download returned no usable dataframe.")
            log_event(
                "dataset_skipped", collection_id=collection_id, dataset_id=str(did),
                operation="download", outcome="skipped", reason="download returned no usable dataframe",
            )
            continue

        df_final = X.copy()
        target_attr = getattr(ds, "default_target_attribute", None)

        metrics = compute_dataset_metrics(df_final, target_col=target_attr)
        metrics["temporal"] = any(cond.get("field") == "temporal" and cond.get("value") is True for cond in conditions)
        metrics["grouped"] = any(cond.get("field") == "grouped" and cond.get("value") is True for cond in conditions)

        is_valid, reason = verify_against_criteria(metrics, conditions)
        if not is_valid:
            print(f"[REJECTED] ID {did} ('{ds.name}'): {reason}")
            log_event(
                "dataset_rejected", collection_id=collection_id, dataset_id=str(did),
                operation="criteria_check", outcome="rejected", reason=reason,
            )
            continue

        # The sanitized name alone can collide (e.g. "Dataset A" vs "Dataset-A"),
        # so the OpenML id is part of the folder name.
        storage_name = dataset_storage_name(ds.name, did)
        raw_dir = RAW_DIR / collection_id / storage_name
        raw_dir.mkdir(parents=True, exist_ok=True)

        csv_path = raw_dir / "data.csv"
        df_final.to_csv(csv_path, index=False, encoding='utf-8')

        print(f"[SUCCESS] ID {did}: '{ds.name}' passed strict verification. Saved to {csv_path}")
        log_event(
            "dataset_downloaded", collection_id=collection_id, dataset_id=str(did),
            operation="download", outcome="success", rows=len(df_final),
        )
        downloaded_count += 1

    log_event(
        "collection_completed", collection_id=collection_id,
        operation="download", outcome="success", downloaded=downloaded_count,
    )

def main(argv: Sequence[str] | None = None) -> int:
    """Runs the ingestion stage only: download raw CSVs, nothing downstream.

    Downstream stages are NOT executed here. Run them explicitly through
    scripts/run_pipeline.py so each stage can be imported, tested and executed
    on its own. Returns an exit code (0 = success).
    """
    configure_logging()
    configure_openml_logging()
    with stage_context("download") as stage:
        parser = argparse.ArgumentParser(description="Download OpenML datasets using YAML profiles.")
        parser.add_argument("--collection", type=str, default=None, help="Target collection name.")
        parser.add_argument("--limit", type=int, default=30, help="Max datasets to pull per collection.")
        args = parser.parse_args(argv)

        ensure_dirs()

        if args.collection:
            target_name = args.collection if args.collection.endswith(".yaml") else f"{args.collection}.yaml"
            target_path = COLLECTIONS_DIR / target_name
            if not target_path.exists():
                print(f"[ERROR] Collection file not found at: {target_path}")
                stage.fail("collection file not found", reason=str(target_path))
                return 1
            yaml_files = [target_path]
        else:
            yaml_files = list(COLLECTIONS_DIR.glob("*.yaml"))
            if not yaml_files:
                print(f"[WARNING] No YAML configuration files found inside '{COLLECTIONS_DIR.resolve()}'.")
                return 0

        for yaml_path in yaml_files:
            try:
                blueprint = load_collection_config(yaml_path)
            except COLLECTION_FILE_ERRORS as e:
                print(f"[WARNING] Skipping collection file '{yaml_path.name}': {type(e).__name__}: {e}")
                log_event(
                    "collection_skipped", level=30, collection_id=yaml_path.stem,
                    operation="blueprint_load", outcome="skipped", error=f"{type(e).__name__}: {e}",
                )
                continue

            collection_id = blueprint.collection_id
            conditions = [cond.model_dump() for cond in blueprint.selection_criteria.all_of]

            print(f"\n--- STARTING INGESTION BATCH: {str(collection_id).upper()} ---")
            log_event("collection_started", collection_id=collection_id, operation="download")
            download_openml_collection(collection_id, conditions, limit=args.limit)

    print("\n[INFO] Ingestion step finished. Configure downstream stages via scripts/run_pipeline.py.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
