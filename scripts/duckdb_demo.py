#!/usr/bin/env python3
"""
DuckDB demonstration: query the Benchmark Suite catalog data.

This utility reads the exported Parquet matrices (or CSV as a fallback) and the
resolved manifests, loading them into an in-memory DuckDB database for ad-hoc
analytics. It shows how to answer questions like:

  * How many datasets are in the catalog?
  * What are the dataset names and their properties?
  * How many rows and features per dataset?
  * What is the validation status distribution?

Usage:
    python scripts/duckdb_demo.py
    python scripts/duckdb_demo.py --collection imbalanced_classification
    python scripts/duckdb_demo.py --format parquet --collection imbalanced_classification

The command searches data/processed/ for the cleaned matrices and manifests/
for resolved collection manifests. When Parquet files exist they are read
directly; otherwise CSV files are used.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

try:
    import duckdb
except ImportError:
    print("[ERROR] The 'duckdb' package is missing. Please run: pip install duckdb")
    sys.exit(1)

try:
    import pandas as pd
except ImportError:
    print("[ERROR] The 'pandas' package is missing. Please run: pip install pandas")
    sys.exit(1)


def get_processed_dir() -> Path:
    """Returns the current processed data directory (respects test patches)."""
    import config
    return config.PROCESSED_DIR


def get_manifest_dir() -> Path:
    """Returns the current manifest directory (respects test patches)."""
    import config
    return config.MANIFEST_DIR


def get_catalog_dir() -> Path:
    """Returns the current catalog directory (respects test patches)."""
    import config
    return config.CATALOG_DIR


def discover_parquet_datasets() -> list[Path]:
    """Finds all processed Parquet matrices in the data tree."""
    processed_dir = get_processed_dir()
    if not processed_dir.exists():
        return []
    return sorted(processed_dir.rglob("data.parquet"))


def discover_csv_datasets() -> list[Path]:
    """Finds all processed CSV matrices in the data tree."""
    processed_dir = get_processed_dir()
    if not processed_dir.exists():
        return []
    return sorted(processed_dir.rglob("data.csv"))


def build_dataset_catalog(parquet_paths: list[Path], csv_paths: list[Path]) -> pd.DataFrame:
    """Builds a summary table with one row per dataset matrix."""
    from utils import file_checksum

    rows: list[dict[str, Any]] = []
    for path in parquet_paths:
        df = pd.read_parquet(path)
        collection_id = path.parts[-3] if len(path.parts) >= 3 else "unknown"
        dataset_id = path.parts[-2] if len(path.parts) >= 2 else path.stem
        rows.append({
            "collection_id": collection_id,
            "dataset_id": dataset_id,
            "n_rows": len(df),
            "n_features": len(df.columns) - 1,
            "format": "parquet",
            "checksum": file_checksum(path),
        })
    for path in csv_paths:
        df = pd.read_csv(path)
        collection_id = path.parts[-3] if len(path.parts) >= 3 else "unknown"
        dataset_id = path.parts[-2] if len(path.parts) >= 2 else path.stem
        rows.append({
            "collection_id": collection_id,
            "dataset_id": dataset_id,
            "n_rows": len(df),
            "n_features": len(df.columns) - 1,
            "format": "csv",
            "checksum": file_checksum(path),
        })
    if not rows:
        return pd.DataFrame(columns=["collection_id", "dataset_id", "n_rows", "n_features", "format", "checksum"])
    return pd.DataFrame(rows)


def load_manifests() -> list[dict[str, Any]]:
    """Loads all resolved manifest YAML files."""
    import yaml
    manifests: list[dict[str, Any]] = []
    manifest_dir = get_manifest_dir()
    if not manifest_dir.exists():
        return manifests
    for path in sorted(manifest_dir.glob("*.resolved.yaml")):
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if isinstance(data, dict):
            manifests.append(data)
    return manifests


def load_master_catalog() -> pd.DataFrame | None:
    """Loads the master catalog CSV if it exists."""
    catalog_dir = get_catalog_dir()
    catalog_csv = catalog_dir / "catalog.csv"
    if catalog_csv.exists():
        return pd.read_csv(catalog_csv)
    return None


def _filter_query(base_query: str, filter_col: str | None) -> str:
    """Appends a WHERE clause for collection filtering."""
    if filter_col:
        return base_query + " WHERE collection_id = ?"
    return base_query


def run_duckdb_analytics(
    dataset_catalog: pd.DataFrame,
    manifests: list[dict[str, Any]],
    master_catalog: pd.DataFrame | None,
    collection_filter: str | None,
) -> None:
    """Runs ad-hoc DuckDB queries on the collected data."""
    con = duckdb.connect(database=":memory:")

    if not dataset_catalog.empty:
        con.register("dataset_matrix", dataset_catalog)

        print("\n=== Dataset Matrices ===")
        query = """
        SELECT collection_id, dataset_id, n_rows, n_features, format, checksum
        FROM dataset_matrix
        """
        if collection_filter:
            query += f" WHERE collection_id = '{collection_filter}'"
        query += " ORDER BY collection_id, dataset_id"
        result = con.sql(query).df()
        print(result.to_string(index=False))

        print("\n=== Summary by Collection ===")
        summary_query = """
        SELECT collection_id,
               COUNT(*) as dataset_count,
               SUM(n_rows) as total_rows,
               AVG(n_features) as avg_features,
               AVG(n_rows) as avg_rows
        FROM dataset_matrix
        """
        if collection_filter:
            summary_query += f" WHERE collection_id = '{collection_filter}'"
        summary_query += " GROUP BY collection_id ORDER BY collection_id"
        summary = con.sql(summary_query).df()
        print(summary.to_string(index=False))
    else:
        if manifests or master_catalog is not None:
            print("\n=== No processed datasets found ===")
        else:
            print("[INFO] No dataset matrices found in data/processed/")

    if manifests:
        print("\n=== Resolved Manifests ===")
        for manifest in manifests:
            source = manifest.get("source_collection", "unknown")
            if collection_filter and source != collection_filter:
                continue
            mat = manifest.get("materialization", {})
            selected = manifest.get("selected_datasets", [])
            print(f"\n  Collection: {source}")
            print(f"  Resolved at: {manifest.get('resolved_at', 'unknown')}")
            print(f"  Total matched before clean: {mat.get('total_matched_before_clean', 'N/A')}")
            print(f"  Final dataset count: {mat.get('final_dataset_count', len(selected))}")
            for ds in selected:
                ds_id = ds.get("dataset_id", "unknown")
                frozen = ds.get("frozen_metadata", {})
                struct = frozen.get("structure", {})
                print(f"    - {ds_id}: {struct.get('n_instances', 'N/A')} rows, {struct.get('n_features', 'N/A')} features, status={frozen.get('status', 'N/A')}")

    if master_catalog is not None:
        con.register("master_catalog", master_catalog)

        print("\n=== Master Catalog Summary ===")
        status_query = """
        SELECT status, COUNT(*) as count
        FROM master_catalog
        """
        if collection_filter:
            status_query += f" WHERE collection_id = '{collection_filter}'"
        status_query += " GROUP BY status ORDER BY status"
        status = con.sql(status_query).df()
        print(status.to_string(index=False))

        license_query = """
        SELECT license, COUNT(*) as count
        FROM master_catalog
        """
        if collection_filter:
            license_query += f" WHERE collection_id = '{collection_filter}'"
        license_query += " GROUP BY license ORDER BY count DESC"
        print("\n=== By License ===")
        licenses = con.sql(license_query).df()
        print(licenses.to_string(index=False))

    con.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="DuckDB analytics demo for the Benchmark Suite.")
    parser.add_argument(
        "--collection", type=str, default=None,
        help="Filter analytics to a single collection ID.",
    )
    parser.add_argument(
        "--format", type=str, choices=["auto", "parquet", "csv"], default="auto",
        help="Preferred data format to read (default: auto-discover).",
    )
    args = parser.parse_args(argv)

    parquet_paths = discover_parquet_datasets() if args.format in ("auto", "parquet") else []
    csv_paths = discover_csv_datasets() if args.format in ("auto", "csv") else []

    if args.format == "csv" or (args.format == "auto" and not parquet_paths):
        if csv_paths:
            print(f"[INFO] Found {len(csv_paths)} CSV dataset matrices.")
        else:
            print("[INFO] No CSV dataset matrices found either.")
    elif args.format == "parquet" or (args.format == "auto" and parquet_paths):
        if parquet_paths:
            print(f"[INFO] Found {len(parquet_paths)} Parquet dataset matrices.")

    dataset_catalog = build_dataset_catalog(parquet_paths, csv_paths)
    manifests = load_manifests()
    master_catalog = load_master_catalog()

    if dataset_catalog.empty and not manifests and master_catalog is None:
        print("[INFO] No data found. Run the pipeline first:")
        print("  python scripts/run_pipeline.py full --filter \"task == classification\" --id sample")
        return 0

    run_duckdb_analytics(dataset_catalog, manifests, master_catalog, args.collection)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
