#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
import pandas as pd
import yaml
import re

logging.getLogger("openml").setLevel(logging.ERROR)

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.append(str(SCRIPT_DIR))

from validate_collections import compute_dataset_metrics, verify_against_criteria

try:
    import openml
except ImportError:
    print("[Error] The 'openml' package is missing. Please run: pip install openml")
    sys.exit(1)

ROOT = Path(__file__).resolve().parents[1]
RAW_BASE = ROOT / "data" / "raw"
COLLECTIONS_DIR = ROOT / "collections"

MIN_INSTANCES, MAX_INSTANCES = 500, 10000
MIN_FEATURES, MAX_FEATURES = 5, 100


def ensure_dirs():
    """Ensures the existence of all baseline data directories."""
    RAW_BASE.mkdir(parents=True, exist_ok=True)
    COLLECTIONS_DIR.mkdir(parents=True, exist_ok=True)


def sanitize_name(name: str) -> str:
    """Transforms the dataset name into a safe format for folders (snake_case)."""
    s = str(name).lower().strip()
    s = re.sub(r'[\s\-\(\)]+', '_', s)
    s = re.sub(r'[^\w]', '', s)
    s = re.sub(r'_+', '_', s)
    return s.strip('_')


def parse_collection_bounds(all_of_conditions: list) -> tuple[int, int, int, int, str | None, int | None]:
    """Parses generic YAML all_of conditions to extract API optimization bounds."""
    inst_min, inst_max = MIN_INSTANCES, MAX_INSTANCES
    feat_min, feat_max = MIN_FEATURES, MAX_FEATURES
    task_type = None
    num_classes = None

    for cond in all_of_conditions:
        field = cond.get("field")
        op = cond.get("op")
        val = cond.get("value")
        
        try:
            if field == "n_instances":
                val_i = int(val)
                if op == "==": inst_min = inst_max = val_i
                elif op == ">=": inst_min = max(inst_min, val_i)
                elif op == ">": inst_min = max(inst_min, val_i + 1)
                elif op == "<=": inst_max = min(inst_max, val_i)
                elif op == "<": inst_max = min(inst_max, val_i - 1)
            elif field == "n_features":
                val_i = int(val)
                if op == "==": feat_min = feat_max = val_i
                elif op == ">=": feat_min = max(feat_min, val_i)
                elif op == ">": feat_min = max(feat_min, val_i + 1)
                elif op == "<=": feat_max = min(feat_max, val_i)
                elif op == "<": feat_max = min(feat_max, val_i - 1)
            elif field == "task" and op == "==":
                task_type = str(val).lower()
            elif field == "task_subtype" and op == "==" and str(val).lower() == "binary":
                num_classes = 2
        except (ValueError, TypeError):
            continue

    return inst_min, inst_max, feat_min, feat_max, task_type, num_classes


def download_openml_collection(collection_id: str, conditions: list, limit: int = 30):
    """Searches OpenML, sorts by popularity, and saves clean DataFrames as amlb-compatible CSVs."""
    print(f"\n=== Searching OpenML API for Collection: '{collection_id}' ===")
    inst_min, inst_max, feat_min, feat_max, _, num_classes = parse_collection_bounds(conditions)

    try:
        kwargs = {
            "output_format": "dataframe",
            "number_instances": f"{inst_min}..{inst_max}",
            "number_features": f"{feat_min}..{feat_max}",
            "status": "active"
        }
        if num_classes is not None:
            kwargs["number_classes"] = str(num_classes)

        datasets_df = openml.datasets.list_datasets(**kwargs)
    except Exception as e:
        print(f"[Error] Failed to search OpenML: {e}")
        return

    if datasets_df.empty:
        print(" -> No datasets found matching the initial collection criteria bounds.")
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
            
            license_info = getattr(ds, "licence", None) or getattr(ds, "license", None)
            citation_info = getattr(ds, "citation", None) or getattr(ds, "paper_url", None)
            
            if not license_info or not citation_info:
                continue

            X, _, _, _ = ds.get_data(target=None, dataset_format="dataframe")
            if X is None or X.empty:
                continue
                
            df_final = X.copy()
            target_attr = getattr(ds, "default_target_attribute", None)

            # --- RUNTIME VALIDATION STEP ---
            metrics = compute_dataset_metrics(df_final, target_col=target_attr)
            metrics["temporal"] = any(cond.get("field") == "temporal" and cond.get("value") is True for cond in conditions)
            metrics["grouped"] = any(cond.get("field") == "grouped" and cond.get("value") is True for cond in conditions)
            
            is_valid, reason = verify_against_criteria(metrics, conditions)
            if not is_valid:
                continue

            print(f"ID {did}: '{ds.name}' passed strict verification matching. Saving...")
            
            folder_name = sanitize_name(ds.name)
            
            # --- SINGLE SOURCE OF TRUTH: Save clean DataFrame directly to /data/raw ---
            # NO /cached/ or /extracted/ folders, NO zip extraction overhead
            raw_dir = RAW_BASE / "collections" / collection_id / folder_name
            raw_dir.mkdir(parents=True, exist_ok=True)
            
            # amlb-compatible CSV: no index, UTF-8 encoding, comma separator
            csv_path = raw_dir / "data.csv"
            df_final.to_csv(csv_path, index=False, encoding='utf-8')
            
            print(f" -> Success: Saved to {csv_path}")
            downloaded_count += 1
            
        except Exception as e:
            print(f"  [Warning] Failed to process dataset {did}: {e}")
            continue


def main():
    parser = argparse.ArgumentParser(description="Download and inspect OpenML datasets using criteria YAML profiles.")
    parser.add_argument("--collection", type=str, default=None, help="Target a specific collection name without .yaml extension.")
    parser.add_argument("--limit", type=int, default=30, help="Max datasets to pull per collection.")
    args = parser.parse_args()

    ensure_dirs()

    if args.collection:
        target_name = args.collection if args.collection.endswith(".yaml") else f"{args.collection}.yaml"
        target_path = COLLECTIONS_DIR / target_name
        if not target_path.exists():
            print(f"[Error] Specified collection file not found at: {target_path}")
            return
        yaml_files = [target_path]
    else:
        yaml_files = list(COLLECTIONS_DIR.glob("*.yaml"))
        if not yaml_files:
            print(f"[Warning] No YAML configuration files found inside '{COLLECTIONS_DIR.resolve()}'.")
            return

    print(f"Loaded {len(yaml_files)} collection blueprint profiles.")

    for yaml_path in yaml_files:
        try:
            with open(yaml_path, 'r') as f:
                config = yaml.safe_load(f)
            if not config:
                continue
        except Exception:
            continue

        collection_id = config.get("collection_id", yaml_path.stem)
        selection_criteria = config.get("selection_criteria", {})
        conditions = selection_criteria.get("all_of", [])

        print(f"\n" + "="*70)
        print(f"🚀 STARTING INGESTION BATCH: {collection_id.upper()}")
        print("="*70)

        download_openml_collection(collection_id, conditions, limit=args.limit)

    print("\n[Completed] Ingestion step finished successfully.")

    print("\n" + "="*70)
    print("🔄 AUTOMATION: Launching Catalog Generation Layer")
    print("="*70)
    try:
        import build_catalog
        build_catalog.main()
    except Exception as e:
        print(f"💥 [Error] Automated execution of build_catalog.py failed: {e}")

    print("\n" + "="*70)
    print("🔄 AUTOMATION: Launching CDataset Validation & Manifest Matrix")
    print("="*70)
    try:
        import validate_datasets
        validate_datasets.main()
    except Exception as e:
        print(f"💥 [Error] Automated execution of validate_datasets.py failed: {e}")

    print("\n🏁 [Finished] Complete end-to-end processing pipeline has concluded.")


if __name__ == "__main__":
    main()