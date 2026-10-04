"""Export layer: packaging resolved manifests and processed matrices."""
from __future__ import annotations

import sys
import zipfile
from pathlib import Path

import pytest
import yaml
import pandas as pd

import export_collections

COLLECTION = "binary_core"


def write_resolved_manifest(project, collection_id: str, dataset_ids: list[str]) -> None:
    project.manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "manifest_id": f"{collection_id}_resolved_v1",
        "source_collection": collection_id,
        "selected_datasets": [{"dataset_id": dataset_id} for dataset_id in dataset_ids],
    }
    manifest_path = project.manifest_dir / f"{collection_id}.resolved.yaml"
    with manifest_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(manifest, f, sort_keys=False)


def create_processed_matrix(project, collection_id: str, dataset_id: str) -> Path:
    matrix_path = project.processed_dir / collection_id / dataset_id / "data.csv"
    matrix_path.parent.mkdir(parents=True, exist_ok=True)
    matrix_path.write_text("feature_0,target\n1,class_0\n2,class_1\n", encoding="utf-8")
    return matrix_path


def test_available_collections_lists_resolved_manifests(project) -> None:
    write_resolved_manifest(project, "binary_core", ["alpha__1"])
    write_resolved_manifest(project, "regression_core", ["beta__2"])

    assert sorted(export_collections.get_available_collections()) == ["binary_core", "regression_core"]


def test_export_bundles_manifest_and_processed_matrices(project) -> None:
    write_resolved_manifest(project, COLLECTION, ["alpha__1"])
    create_processed_matrix(project, COLLECTION, "alpha__1")
    staging = project.exports_dir / "staging"

    exported = export_collections.export_single_collection(COLLECTION, staging)

    assert exported is True
    assert (staging / COLLECTION / "manifest.yaml").exists()
    assert (staging / COLLECTION / "alpha__1" / "data.csv").exists()


def test_export_reports_a_missing_processed_matrix(project, capsys: pytest.CaptureFixture) -> None:
    write_resolved_manifest(project, COLLECTION, ["alpha__1", "beta__2"])
    create_processed_matrix(project, COLLECTION, "alpha__1")
    staging = project.exports_dir / "staging"

    exported = export_collections.export_single_collection(COLLECTION, staging)

    output = capsys.readouterr().out
    assert exported is False
    assert "Missing processed CSV matrix for dataset: beta__2" in output
    assert "[FAILED]" in output
    assert (staging / COLLECTION / "alpha__1" / "data.csv").exists()


def test_export_reports_a_missing_manifest(project, capsys: pytest.CaptureFixture) -> None:
    exported = export_collections.export_single_collection("ghost", project.exports_dir / "staging")

    assert exported is False
    assert "[SKIPPED] No resolved manifest found for 'ghost'" in capsys.readouterr().out


def test_export_reports_an_empty_manifest(project, capsys: pytest.CaptureFixture) -> None:
    write_resolved_manifest(project, COLLECTION, [])

    exported = export_collections.export_single_collection(COLLECTION, project.exports_dir / "staging")

    assert exported is False
    assert "zero validated datasets" in capsys.readouterr().out


def test_export_reports_a_malformed_manifest(project, capsys: pytest.CaptureFixture) -> None:
    (project.manifest_dir / f"{COLLECTION}.resolved.yaml").write_text("- not\n- a mapping\n", encoding="utf-8")

    exported = export_collections.export_single_collection(COLLECTION, project.exports_dir / "staging")

    assert exported is False
    assert "[CRITICAL]" in capsys.readouterr().out


def test_export_reports_manifest_entries_without_a_dataset_id(
    project, capsys: pytest.CaptureFixture
) -> None:
    manifest_path = project.manifest_dir / f"{COLLECTION}.resolved.yaml"
    manifest_path.write_text(
        yaml.safe_dump({"selected_datasets": [{"name": "alpha"}]}, sort_keys=False), encoding="utf-8"
    )

    exported = export_collections.export_single_collection(COLLECTION, project.exports_dir / "staging")

    assert exported is False
    assert "invalid dataset entry" in capsys.readouterr().out


def test_export_cli_exits_non_zero_without_manifests(project, monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["export_collections.py"])

    exit_code = export_collections.main()

    assert exit_code == 1


def test_export_cli_rejects_unknown_collections(project, monkeypatch, capsys) -> None:
    write_resolved_manifest(project, COLLECTION, ["alpha__1"])
    monkeypatch.setattr(sys, "argv", ["export_collections.py", "--collection", "ghost"])

    exit_code = export_collections.main()

    assert exit_code == 1
    assert "[ERROR] Invalid target" in capsys.readouterr().out


def test_export_cli_builds_a_folder_tree(project, monkeypatch) -> None:
    write_resolved_manifest(project, COLLECTION, ["alpha__1"])
    create_processed_matrix(project, COLLECTION, "alpha__1")
    monkeypatch.setattr(
        sys,
        "argv",
        ["export_collections.py", "--collection", COLLECTION, "--format", "folder"],
    )

    export_collections.main()

    assert (project.exports_dir / COLLECTION / "manifest.yaml").exists()
    assert (project.exports_dir / COLLECTION / "alpha__1" / "data.csv").exists()


def test_export_cli_builds_a_zip_archive(project, monkeypatch) -> None:
    write_resolved_manifest(project, COLLECTION, ["alpha__1"])
    create_processed_matrix(project, COLLECTION, "alpha__1")
    monkeypatch.setattr(
        sys,
        "argv",
        ["export_collections.py", "--collection", COLLECTION, "--format", "zip"],
    )

    export_collections.main()

    archive = project.exports_dir / f"{COLLECTION}.zip"
    assert archive.exists()
    with zipfile.ZipFile(archive) as zf:
        members = zf.namelist()
    assert f"{COLLECTION}/manifest.yaml" in members
    assert f"{COLLECTION}/alpha__1/data.csv" in members


def test_parquet_export_writes_a_readable_parquet_next_to_csv(project) -> None:
    write_resolved_manifest(project, COLLECTION, ["alpha__1"])
    create_processed_matrix(project, COLLECTION, "alpha__1")
    staging = project.exports_dir / "staging"

    exported = export_collections.export_single_collection(COLLECTION, staging, parquet=True)

    assert exported is True
    assert (staging / COLLECTION / "alpha__1" / "data.csv").exists()
    parquet_file = staging / COLLECTION / "alpha__1" / "data.parquet"
    assert parquet_file.exists()

    df = pd.read_parquet(parquet_file)
    assert list(df.columns) == ["feature_0", "target"]
    assert len(df) == 2
    assert list(df["target"]) == ["class_0", "class_1"]


def test_parquet_export_without_pyarrow_raises_clear_error(project, monkeypatch) -> None:
    write_resolved_manifest(project, COLLECTION, ["alpha__1"])
    create_processed_matrix(project, COLLECTION, "alpha__1")
    staging = project.exports_dir / "staging"

    import pandas as pd

    original_to_parquet = pd.DataFrame.to_parquet

    def broken_to_parquet(self, *args, **kwargs):
        raise ImportError("simulated missing pyarrow")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", broken_to_parquet)

    try:
        exported = export_collections.export_single_collection(COLLECTION, staging, parquet=True)
        assert exported is False
    finally:
        monkeypatch.setattr(pd.DataFrame, "to_parquet", original_to_parquet)


def test_csv_export_still_works_without_parquet(project) -> None:
    write_resolved_manifest(project, COLLECTION, ["alpha__1"])
    create_processed_matrix(project, COLLECTION, "alpha__1")
    staging = project.exports_dir / "staging"

    exported = export_collections.export_single_collection(COLLECTION, staging, parquet=False)

    assert exported is True
    assert (staging / COLLECTION / "alpha__1" / "data.csv").exists()
    parquet_file = staging / COLLECTION / "alpha__1" / "data.parquet"
    assert not parquet_file.exists()


def test_export_cli_with_parquet_flag(project, monkeypatch) -> None:
    write_resolved_manifest(project, COLLECTION, ["alpha__1"])
    create_processed_matrix(project, COLLECTION, "alpha__1")
    monkeypatch.setattr(
        sys, "argv",
        ["export_collections.py", "--collection", COLLECTION, "--format", "folder", "--parquet"],
    )

    export_collections.main()

    assert (project.exports_dir / COLLECTION / "alpha__1" / "data.csv").exists()
    assert (project.exports_dir / COLLECTION / "alpha__1" / "data.parquet").exists()
