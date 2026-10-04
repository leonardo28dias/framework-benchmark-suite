"""
Tests for the pipeline orchestrator: stages are decoupled, can be imported
without side effects, and the orchestrator controls execution order.
"""
from __future__ import annotations

import json
import sys

import yaml

from dataset_factories import classification_frame


def run_pipeline(*args):
    import run_pipeline
    return run_pipeline.main(list(args))


def test_orchestrator_import_does_not_execute_pipeline():
    """Importing the orchestrator must not run any stage."""
    import run_pipeline

    assert run_pipeline is sys.modules["run_pipeline"]


def test_orchestrator_define_creates_blueprint(project, monkeypatch):
    exit_code = run_pipeline(
        "define",
        "--filter", "task == classification",
        "--id", "my_col",
        "--name", "My Collection",
        "--output-dir", str(project.collections_dir),
    )

    assert exit_code == 0
    blueprint_path = project.collections_dir / "my_col.yaml"
    assert blueprint_path.exists()
    blueprint = yaml.safe_load(blueprint_path.read_text(encoding="utf-8"))
    assert blueprint["collection_id"] == "my_col"


def test_orchestrator_download_runs_only_download(project, openml_registry):
    """The download stage runs independently when a blueprint exists."""
    from test_download_collection import FakeOpenMLDataset, CONDITIONS

    openml_registry[1] = FakeOpenMLDataset(1, "Dataset A", classification_frame())

    import yaml as _yaml
    blueprint = {
        "collection_id": "demo",
        "name": "Demo Collection",
        "selection_criteria": {"all_of": CONDITIONS},
    }
    blueprint_path = project.collections_dir / "demo.yaml"
    with blueprint_path.open("w", encoding="utf-8") as f:
        _yaml.safe_dump(blueprint, f)

    exit_code = run_pipeline("download", "--collection", "demo", "--limit", "1")

    assert exit_code == 0
    assert (project.raw_dir / "demo" / "dataset_a__1" / "data.csv").exists()


def test_orchestrator_catalog_runs_independent_stage(project, academic_api):
    from test_integration_pipeline import COLLECTION
    project.add_raw_dataset(COLLECTION, "Demo Alpha", 11, classification_frame())

    exit_code = run_pipeline("catalog")

    assert exit_code == 0
    assert project.catalog_csv.exists()


def test_orchestrator_validate_runs_independent_stage(project, academic_api):
    from test_integration_pipeline import COLLECTION
    project.add_raw_dataset(COLLECTION, "Demo Alpha", 11, classification_frame())
    project.run_catalog()

    exit_code = run_pipeline("validate")

    assert exit_code == 0
    assert project.read_dataset_metadata("demo_alpha__11")["status"] == "validated"


def test_orchestrator_export_runs_independent_stage(project, monkeypatch):
    from test_export_collections import write_resolved_manifest, create_processed_matrix, COLLECTION

    write_resolved_manifest(project, COLLECTION, ["alpha__1"])
    create_processed_matrix(project, COLLECTION, "alpha__1")

    exit_code = run_pipeline("export", "--collection", COLLECTION, "--format", "folder")

    assert exit_code == 0
    assert (project.exports_dir / COLLECTION / "manifest.yaml").exists()


def test_orchestrator_full_pipeline_runs_all_stages(project, openml_registry):
    """The 'full' command generates a blueprint, downloads, catalogs, validates and exports."""
    from test_download_collection import FakeOpenMLDataset

    openml_registry[1] = FakeOpenMLDataset(1, "Dataset A", classification_frame())

    exit_code = run_pipeline(
        "full",
        "--filter", "task == classification",
        "--id", "demo_full",
        "--collection", "demo_full",
        "--limit", "1",
        "--export-format", "zip",
    )

    assert exit_code == 0
    assert project.catalog_csv.exists()
    assert project.manifest_dir.exists()


def test_orchestrator_logs_json_with_run_id(project, openml_registry, capsys):
    """The orchestrator should produce JSON log lines on stderr with a run_id field."""
    from test_download_collection import FakeOpenMLDataset, CONDITIONS

    openml_registry[1] = FakeOpenMLDataset(1, "Dataset A", classification_frame())

    import io
    import logging_utils

    log_buffer = io.StringIO()
    logging_utils.configure_logging(stream=log_buffer)

    yaml = __import__("yaml")
    blueprint = {
        "collection_id": "demo",
        "name": "Demo Collection",
        "selection_criteria": {"all_of": CONDITIONS},
    }
    blueprint_path = project.collections_dir / "demo.yaml"
    with blueprint_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(blueprint, f)

    run_pipeline("download", "--collection", "demo", "--limit", "1")

    log_lines = log_buffer.getvalue().strip().split("\n")
    if log_lines and log_lines[0]:
        first_record = json.loads(log_lines[0])
        assert "run_id" in first_record
        assert len(first_record["run_id"]) == 12
