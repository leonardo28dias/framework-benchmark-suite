"""
Tests for the DuckDB analytics demonstration.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import yaml

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


def test_duckdb_demo_runs_on_empty_project(project, capsys):
    """Should handle the case where no data exists yet gracefully."""
    import duckdb_demo

    exit_code = duckdb_demo.main([])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "No data found" in output or "No dataset matrices found" in output


def test_duckdb_demo_discovers_parquet_datasets(project):
    from duckdb_demo import discover_parquet_datasets, discover_csv_datasets, build_dataset_catalog

    collection_id = "binary_core"
    dataset_id = "alpha__1"

    processed_dir = project.processed_dir / collection_id / dataset_id
    processed_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"feature_0": [1, 2], "target": ["a", "b"]})
    df.to_parquet(processed_dir / "data.parquet", index=False)

    parquet_paths = discover_parquet_datasets()
    csv_paths = discover_csv_datasets()

    assert len(parquet_paths) == 1
    assert parquet_paths[0].suffix == ".parquet"
    assert len(csv_paths) == 0

    catalog = build_dataset_catalog(parquet_paths, csv_paths)
    assert len(catalog) == 1
    assert catalog.iloc[0]["collection_id"] == collection_id
    assert catalog.iloc[0]["dataset_id"] == dataset_id
    assert catalog.iloc[0]["n_rows"] == 2
    assert catalog.iloc[0]["n_features"] == 1


def test_duckdb_demo_filters_by_collection(project, capsys):
    from duckdb_demo import discover_parquet_datasets, discover_csv_datasets, build_dataset_catalog, load_manifests, load_master_catalog, run_duckdb_analytics

    collection_a = "col_a"
    collection_b = "col_b"
    for coll_id, ds_name, rows in [(collection_a, "alpha__1", 100), (collection_b, "beta__2", 200)]:
        processed_dir = project.processed_dir / coll_id / ds_name
        processed_dir.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame({"feature_0": range(rows), "target": ["a", "b"] * (rows // 2)})
        df.to_csv(processed_dir / "data.csv", index=False)

    parquet_paths = discover_parquet_datasets()
    csv_paths = discover_csv_datasets()
    dataset_catalog = build_dataset_catalog(parquet_paths, csv_paths)
    manifests = load_manifests()
    master_catalog = load_master_catalog()

    run_duckdb_analytics(dataset_catalog, manifests, master_catalog, collection_filter=collection_a)

    output = capsys.readouterr().out
    assert collection_a in output
    assert collection_b not in output


def test_duckdb_demo_queries_manifest_data(project, capsys):
    from duckdb_demo import load_manifests, run_duckdb_analytics

    manifest = {
        "source_collection": "test_col",
        "resolved_at": "2026-01-01T00:00:00Z",
        "materialization": {"total_matched_before_clean": 3, "final_dataset_count": 2},
        "selected_datasets": [
            {"dataset_id": "ds_1", "frozen_metadata": {"structure": {"n_instances": 100, "n_features": 5}, "status": "validated"}},
            {"dataset_id": "ds_2", "frozen_metadata": {"structure": {"n_instances": 200, "n_features": 10}, "status": "validated"}},
        ],
    }
    manifest_path = project.manifest_dir / "test_col.resolved.yaml"
    with manifest_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(manifest, f)

    run_duckdb_analytics(pd.DataFrame(), load_manifests(), None, None)

    output = capsys.readouterr().out
    assert "test_col" in output
    assert "ds_1" in output
    assert "ds_2" in output
    assert "Final dataset count: 2" in output
