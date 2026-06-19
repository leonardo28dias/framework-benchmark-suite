#!/usr/bin/env python3
from __future__ import annotations

import datetime
from pathlib import Path
import pandas as pd
import yaml

# Configuration of structured directories
ROOT = Path(__file__).resolve().parents[1]
CATALOG_BASE = ROOT / "catalog"
DATASETS_CATALOG_DIR = CATALOG_BASE / "datasets"
COLLECTIONS_MANIFEST_DIR = ROOT / "manifests"
CATALOG_CSV_PATH = CATALOG_BASE / "catalog.csv"
PROCESSED_BASE = ROOT / "data" / "processed" / "collections"

def load_dataset_target(dataset_id: str) -> str:
    """Reads the inferred target variable from the catalog YAML."""
    yaml_path = DATASETS_CATALOG_DIR / f"{dataset_id}.yaml"
    if yaml_path.exists():
        with open(yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
            return data.get("task", {}).get("target_variable", "unknown")
    return "unknown"

def update_dataset_yaml_status(dataset_id: str, status: str):
    """Updates the individual dataset YAML status to validated/rejected."""
    yaml_path = DATASETS_CATALOG_DIR / f"{dataset_id}.yaml"
    if yaml_path.exists():
        with open(yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        data["status"] = status
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

def main():
    if not CATALOG_CSV_PATH.exists():
        print(f"[Error] Master catalog not found at: {CATALOG_CSV_PATH}")
        return

    # Ensure directories exist
    COLLECTIONS_MANIFEST_DIR.mkdir(parents=True, exist_ok=True)

    master_catalog = pd.read_csv(CATALOG_CSV_PATH)
    print(f"Loaded master catalog. Processing {len(master_catalog)} datasets for sanitization...\n")

    # Group trackers for the final manifests
    collection_metrics = {}
    
    for _, row in master_catalog.iterrows():
        collection_id = str(row["collection_id"])
        dataset_id = str(row["dataset_id"])
        raw_csv_relative = str(row["csv_path"])
        csv_path = ROOT / raw_csv_relative

        if collection_id not in collection_metrics:
            collection_metrics[collection_id] = {
                "total_matched_before": 0,
                "validated_datasets": []
            }
        
        collection_metrics[collection_id]["total_matched_before"] += 1
        print(f"🔬 Sanitizing: [{collection_id}] -> {dataset_id}")

        if not csv_path.exists():
            print(f"    ❌ REJECTED: Raw CSV file missing at {raw_csv_relative}")
            update_dataset_yaml_status(dataset_id, "rejected")
            continue

        try:
            df = pd.read_csv(csv_path, low_memory=False)
            target_col = load_dataset_target(dataset_id)

            # Load individual YAML early to extract and validate task type configuration
            yaml_file_path = DATASETS_CATALOG_DIR / f"{dataset_id}.yaml"
            if not yaml_file_path.exists():
                print(f"    ❌ REJECTED: Dataset metadata YAML file missing at {yaml_file_path.name}")
                update_dataset_yaml_status(dataset_id, "rejected")
                continue

            with open(yaml_file_path, 'r', encoding='utf-8') as f:
                current_yaml = yaml.safe_load(f)
            
            task_info = current_yaml.get("task", {})
            task_type = task_info.get("type", "unknown")

            # --- CC-18 TASK TYPE GUARD ---
            if task_type not in ["classification", "regression"]:
                print(f"    ❌ REJECTED: Task type '{task_type}' is unknown or unsupported (must be classification or regression).")
                update_dataset_yaml_status(dataset_id, "rejected")
                continue

            # --- CC-18 TARGET INTEGRITY GUARD ---
            if target_col not in df.columns:
                print(f"    ❌ REJECTED: Target variable '{target_col}' not found in columns.")
                update_dataset_yaml_status(dataset_id, "rejected")
                continue

            # Clean target rows before applying rules (fixes dgf_test crash)
            initial_rows_count = len(df)
            df = df.dropna(subset=[target_col])
            dropped_target_nas = initial_rows_count - len(df)
            if dropped_target_nas > 0:
                print(f"    -> Dropped {dropped_target_nas} rows due to missing target values.")

            # 1. Drop duplicate rows EARLY (Crucial: removing duplicates can alter class counts)
            initial_rows = len(df)
            df = df.drop_duplicates()
            dropped_duplicates = initial_rows - len(df)
            if dropped_duplicates > 0:
                print(f"    -> Removed {dropped_duplicates} duplicate rows.")

            # Ensure target is not empty or completely constant post row drop
            unique_target_count = df[target_col].nunique()
            if unique_target_count <= 1:
                print(f"    ❌ REJECTED: Target variable '{target_col}' is constant or completely empty.")
                update_dataset_yaml_status(dataset_id, "rejected")
                continue

            # --- CC-18 CLASSIFICATION-SPECIFIC ESTRANGEMENT GUARDS ---
            if task_type == "classification":
                # Cardinality heuristic protection (fixes is_fraud issue)
                if unique_target_count > 20:
                    print(f"    ❌ REJECTED: Target cardinality too high ({unique_target_count} classes) for classification. Suspected continuous feature or leakage ID.")
                    update_dataset_yaml_status(dataset_id, "rejected")
                    continue
                
                # Rare class protection (Bans singletons that break stratified splits)
                class_counts = df[target_col].value_counts()
                min_class_size = class_counts.min()
                if min_class_size < 5:
                    print(f"    ❌ REJECTED: Minority class has only {min_class_size} instance(s). Minimum required for robust stratification is 5.")
                    update_dataset_yaml_status(dataset_id, "rejected")
                    continue
            
            # --- GLOBAL ANTI-SINGLETON SANITY CHECK ---
            # Catches regression datasets being forced into classification pipelines with singletons
            global_class_counts = df[target_col].value_counts()
            global_min_class_size = global_class_counts.min()
            if global_min_class_size < 2:
                print(f"    ❌ REJECTED: Dataset contains singleton targets (minimum class count is {global_min_class_size}). Stratified splits will fail.")
                update_dataset_yaml_status(dataset_id, "rejected")
                continue

            # 2. Drop columns with > 50% missing values (excluding target)
            missing_threshold = 0.5
            cols_to_drop_missing = [col for col in df.columns if df[col].isnull().mean() > missing_threshold and col != target_col]
            if cols_to_drop_missing:
                df = df.drop(columns=cols_to_drop_missing)
                for col in cols_to_drop_missing:
                    print(f"    -> Dropped column '{col}' (>{int(missing_threshold*100)}% missing values).")

            # 3. Drop constant columns (excluding target)
            nunique = df.nunique()
            constant_cols = [col for col in nunique[nunique == 1].index if col != target_col]
            if constant_cols:
                df = df.drop(columns=constant_cols)
                print(f"    -> Dropped {len(constant_cols)} constant column(s): {constant_cols}")

            # 4. Drop high-cardinality leakage ID columns (excluding target)
            high_cardinality_cols = [
                col for col in nunique[(nunique == len(df)) & (len(df) > 1)].index 
                if col != target_col and not pd.api.types.is_float_dtype(df[col])
            ]
            if high_cardinality_cols:
                df = df.drop(columns=high_cardinality_cols)
                print(f"    -> Dropped {len(high_cardinality_cols)} high-cardinality leakage ID column(s).")

            # --- FRAMEWORK INSTANCE & FEATURE LIMITS VALIDATION ---
            final_instances = len(df)
            if final_instances < 500 or final_instances > 10000:
                print(f"    ❌ REJECTED: Sample size ({final_instances}) violates framework design rules (500-10000).")
                update_dataset_yaml_status(dataset_id, "rejected")
                continue

            # Feature footprint excludes the target column to match OpenML guidelines perfectly
            final_features = len(df.columns) - 1
            if final_features < 5 or final_features > 100:
                print(f"    ❌ REJECTED: Feature footprint ({final_features} features) violates framework design rules (5-100).")
                update_dataset_yaml_status(dataset_id, "rejected")
                continue

            # --- VALIDATED ---
            processed_dir = PROCESSED_BASE / collection_id / dataset_id
            processed_dir.mkdir(parents=True, exist_ok=True)
            processed_csv_path = processed_dir / "data.csv"
            df.to_csv(processed_csv_path, index=False)
            
            update_dataset_yaml_status(dataset_id, "validated")
            print(f"    ✅ VALIDATED: Cleaned matrix saved to {processed_csv_path.relative_to(ROOT)}")

            # Profile final feature data types excluding target
            df_features_only = df.drop(columns=[target_col])
            num_numeric = len(df_features_only.select_dtypes(include=['number']).columns)
            num_categorical = len(df_features_only.select_dtypes(include=['object', 'category', 'str']).columns)

            # Append metadata to collection tracking
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
                        "numeric": num_numeric,
                        "categorical": num_categorical
                    },
                    "status": "validated"
                }
            })

        except Exception as e:
            print(f"    💥 CRITICAL ERROR processing {dataset_id}: {e}")
            update_dataset_yaml_status(dataset_id, "error")

    print("\n" + "="*50 + "\n⚙️ Generating Collection Manifests (.resolved.yaml)...\n" + "="*50)

    # 5. Materialize the .resolved.yaml files for each collection
    current_time_iso = datetime.datetime.utcnow().replace(tzinfo=datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')

    for collection_id, metrics in collection_metrics.items():
        manifest_path = COLLECTIONS_MANIFEST_DIR / f"{collection_id}.resolved.yaml"
        final_count = len(metrics["validated_datasets"])

        manifest_data = {
            "manifest_id": f"{collection_id}_resolved_v1",
            "source_collection": collection_id,
            "resolved_at": current_time_iso,
            "source_files": {
                "catalog_file": "catalog/catalog.csv"
            },
            "filters_applied": {
                "all_of": [
                    {"field": "status", "op": "==", "value": "validated"},
                    {"field": "n_instances", "op": "between", "value": [500, 10000]},
                    {"field": "n_features", "op": "between", "value": [5, 100]},
                    {"field": "missing_rate_per_column", "op": "<=", "value": 0.5}
                ]
            },
            "sorting_applied": [
                {"field": "dataset_id", "order": "asc"}
            ],
            "materialization": {
                "total_matched_before_clean": metrics["total_matched_before"],
                "top_k": None,
                "final_dataset_count": final_count
            },
            "selected_datasets": metrics["validated_datasets"]
        }

        with open(manifest_path, 'w', encoding='utf-8') as f:
            yaml.dump(manifest_data, f, default_flow_style=False, sort_keys=False)
        
        print(f"📦 Manifest Materialized: manifests/{collection_id}.resolved.yaml ({final_count} datasets verified)")

if __name__ == "__main__":
    main()