#!/usr/bin/env python3
from __future__ import annotations

import os
import re
import logging
from pathlib import Path
import pandas as pd
import yaml

# Suppress harmless OpenML warnings
logging.getLogger("openml").setLevel(logging.ERROR)

try:
    import openml
except ImportError:
    print("[Error] The 'openml' package is missing. Please run: pip install openml")
    sys.exit(1)

# Configuration of structured directories
ROOT = Path(__file__).resolve().parents[1]
RAW_BASE = ROOT / "data" / "raw" / "collections"

# New Centralized Catalog Directories
CATALOG_BASE = ROOT / "catalog"
DATASETS_CATALOG_DIR = CATALOG_BASE / "datasets"
CATALOG_CSV_PATH = CATALOG_BASE / "catalog.csv"


def sanitize_name(name: str) -> str:
    """Replicates the exact folder name sanitization logic to match API records."""
    s = str(name).lower().strip()
    s = re.sub(r'[\s\-\(\)]+', '_', s)
    s = re.sub(r'[^\w]', '', s)
    s = re.sub(r'_+', '_', s)
    return s.strip('_')


def infer_task_type(df: pd.DataFrame, target_col: str) -> str:
    """
    Infers whether a dataset is for classification or regression 
    based on the data type and cardinality of the target column.
    """
    if target_col not in df.columns:
        return "unknown"
    
    col_type = df[target_col].dtype
    nunique = df[target_col].nunique()
    
    if col_type == 'object' or col_type == 'bool' or str(col_type) == 'category':
        return "classification"
    
    if 'float' in str(col_type):
        return "regression"
    
    if 'int' in str(col_type):
        if nunique <= 15 or nunique < (len(df) * 0.05):
            return "classification"
        else:
            return "regression"
            
    return "unknown"


def analyze_dataframe(df: pd.DataFrame) -> dict:
    """
    Performs a structural x-ray of the dataframe to feed the YAML schema.
    Flags data quality issues for the future validate_datasets.py script.
    """
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


def main():
    if not RAW_BASE.exists():
        print(f"[Error] Raw data directory not found at: {RAW_BASE}")
        return

    DATASETS_CATALOG_DIR.mkdir(parents=True, exist_ok=True)

    # Pre-load all active OpenML datasets within framework bounds to map folder names back to IDs
    print("🔄 Connecting to OpenML API to cache academic metadata tracking...")
    openml_lookup = {}
    try:
        datasets_df = openml.datasets.list_datasets(
            number_instances="500..10000",
            number_features="5..100",
            status="active",
            output_format="dataframe"
        )
        for _, row in datasets_df.iterrows():
            sanitized = sanitize_name(row["name"])
            openml_lookup[sanitized] = int(row["did"])
    except Exception as e:
        print(f"[Warning] Failed to pull master OpenML lookup list. Citation matching may fall back: {e}")

    catalog_entries = []
    
    for collection_path in RAW_BASE.iterdir():
        if not collection_path.is_dir():
            continue
            
        collection_id = collection_path.name
        
        for dataset_path in collection_path.iterdir():
            if not dataset_path.is_dir():
                continue
                
            dataset_id = dataset_path.name
            csv_path = dataset_path / "data.csv"
            yaml_path = DATASETS_CATALOG_DIR / f"{dataset_id}.yaml"
            
            if not csv_path.exists():
                continue

            print(f"Cataloging: [{collection_id}] -> {dataset_id}")
            
            # Default academic attributes
            citation_info = "unknown"
            paper_url_info = "unknown"
            license_info = "unknown"

            # Pull academic data tracking from cache if matched
            if dataset_id in openml_lookup:
                try:
                    ds_meta = openml.datasets.get_dataset(openml_lookup[dataset_id], download_data=False)
                    citation_info = getattr(ds_meta, "citation", None) or "unknown"
                    paper_url_info = getattr(ds_meta, "paper_url", None) or "unknown"
                    license_info = getattr(ds_meta, "licence", None) or getattr(ds_meta, "license", None) or "unknown"
                except Exception:
                    pass

            try:
                df = pd.read_csv(csv_path, low_memory=False)
                analysis = analyze_dataframe(df)
                
                # Construct YAML Schema with explicit academic tracking properties
                yaml_data = {
                    "dataset_id": dataset_id,
                    "collection_id": collection_id,
                    "name": dataset_id.replace("_", " ").title(),
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
                
                flat_entry = {
                    "collection_id": collection_id,
                    "dataset_id": dataset_id,
                    "n_instances": analysis["structure"]["n_instances"],
                    "n_features": analysis["structure"]["n_features"],
                    "missing_rate": analysis["structure"]["missing_rate"],
                    "has_constants": len(analysis["data_quality"]["constant_columns"]) > 0,
                    "has_high_cardinality": len(analysis["data_quality"]["high_cardinality_columns"]) > 0,
                    "has_citation": citation_info != "unknown",
                    "license": license_info,
                    "status": "cataloged",
                    "csv_path": str(csv_path.relative_to(ROOT)),
                    "yaml_path": str(yaml_path.relative_to(ROOT))
                }
                catalog_entries.append(flat_entry)
                
            except Exception as e:
                print(f" -> [Error] Processing {dataset_id}: {e}")

    if catalog_entries:
        catalog_df = pd.DataFrame(catalog_entries)
        catalog_df.to_csv(CATALOG_CSV_PATH, index=False)
        print(f"\n[Success] Master catalog generated with {len(catalog_entries)} datasets at: {CATALOG_CSV_PATH}")
    else:
        print("\n[Warning] No valid data.csv files found to catalog.")


if __name__ == "__main__":
    main()