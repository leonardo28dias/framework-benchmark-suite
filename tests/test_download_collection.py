"""
Ingestion layer tests: collection blueprint loading, bound parsing and the
storage layout produced by a fully mocked OpenML client.
"""
from __future__ import annotations

import sys
import urllib.error
from pathlib import Path

import openml
import pandas as pd
import pytest
import yaml

import download_collection
import config
from dataset_factories import classification_frame

CONDITIONS = [{"field": "task", "op": "==", "value": "classification"}]

VALID_BLUEPRINT = """
collection_id: demo
name: Demo Collection
selection_criteria:
  all_of:
    - field: task
      op: ==
      value: classification
"""


def write_blueprint(path: Path, body: str) -> Path:
    path.write_text(body, encoding="utf-8")
    return path


class FakeOpenMLDataset:
    """Minimal stand-in for ``openml.datasets.OpenMLDataset``."""

    def __init__(
        self,
        did: int,
        name: str,
        frame: pd.DataFrame,
        licence: str | None = "CC0",
        citation: str | None = "Doe et al. (2024)",
    ) -> None:
        self.id = did
        self.name = name
        self.licence = licence
        self.citation = citation
        self.paper_url = "https://example.org/paper"
        self.default_target_attribute = "target"
        self._frame = frame

    def get_data(self, target: object = None, dataset_format: str = "dataframe"):
        return self._frame, None, None, None


class FailingDownloadDataset(FakeOpenMLDataset):
    def get_data(self, target: object = None, dataset_format: str = "dataframe"):
        raise openml.exceptions.PyOpenMLError("dataset archive is corrupt")


def stored_dataset_names(project, collection_id: str) -> list[str]:
    collection_dir = project.raw_dir / collection_id
    if not collection_dir.exists():
        return []
    return sorted(entry.name for entry in collection_dir.iterdir())


def test_load_collection_config_accepts_a_valid_blueprint(tmp_path: Path) -> None:
    blueprint = write_blueprint(tmp_path / "demo.yaml", VALID_BLUEPRINT)

    loaded = download_collection.load_collection_config(blueprint)

    assert loaded.collection_id == "demo"
    dumped = loaded.model_dump()
    assert dumped["selection_criteria"]["all_of"][0] == {"field": "task", "op": "==", "value": "classification"}


def test_load_collection_config_requires_a_yaml_mapping(tmp_path: Path) -> None:
    blueprint = write_blueprint(tmp_path / "demo.yaml", "- just\n- a list\n")

    with pytest.raises(ValueError, match="mapping"):
        download_collection.load_collection_config(blueprint)


def test_load_collection_config_requires_selection_criteria(tmp_path: Path) -> None:
    blueprint = write_blueprint(tmp_path / "demo.yaml", "collection_id: demo\n")

    with pytest.raises(ValueError, match="selection_criteria"):
        download_collection.load_collection_config(blueprint)


def test_load_collection_config_requires_conditions(tmp_path: Path) -> None:
    blueprint = write_blueprint(tmp_path / "demo.yaml", "selection_criteria:\n  all_of: []\n")

    with pytest.raises(ValueError, match="all_of"):
        download_collection.load_collection_config(blueprint)


def test_load_collection_config_rejects_incomplete_conditions(tmp_path: Path) -> None:
    blueprint = write_blueprint(
        tmp_path / "demo.yaml",
        "selection_criteria:\n  all_of:\n    - field: task\n      op: ==\n",
    )

    with pytest.raises(ValueError, match="invalid condition at index 0"):
        download_collection.load_collection_config(blueprint)


def test_load_collection_config_reports_malformed_yaml(tmp_path: Path) -> None:
    blueprint = write_blueprint(tmp_path / "demo.yaml", "selection_criteria: {all_of: [\n")

    with pytest.raises(yaml.YAMLError):
        download_collection.load_collection_config(blueprint)


def test_load_collection_config_reports_missing_files(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        download_collection.load_collection_config(tmp_path / "ghost.yaml")


def test_parse_collection_bounds_extracts_optimization_bounds() -> None:
    conditions = [
        {"field": "n_instances", "op": ">=", "value": 1000},
        {"field": "n_instances", "op": "<=", "value": 5000},
        {"field": "n_features", "op": ">", "value": 9},
        {"field": "task", "op": "==", "value": "classification"},
        {"field": "task_subtype", "op": "==", "value": "binary"},
    ]

    inst_min, inst_max, feat_min, feat_max, task_type, num_classes = (
        download_collection.parse_collection_bounds(conditions)
    )

    assert (inst_min, inst_max) == (1000, 5000)
    assert feat_min == 10
    assert task_type == "classification"
    assert num_classes == 2


def test_parse_collection_bounds_falls_back_to_the_configured_limits() -> None:
    bounds = download_collection.parse_collection_bounds([])

    assert bounds == (
        config.MIN_INSTANCES,
        config.MAX_INSTANCES,
        config.MIN_FEATURES,
        config.MAX_FEATURES,
        None,
        None,
    )


def test_parse_collection_bounds_warns_about_unusable_conditions(capsys: pytest.CaptureFixture) -> None:
    conditions = [{"field": "n_instances", "op": ">=", "value": "lots"}]

    bounds = download_collection.parse_collection_bounds(conditions)

    assert "Ignoring unusable bound condition" in capsys.readouterr().out
    assert bounds[:4] == (
        config.MIN_INSTANCES,
        config.MAX_INSTANCES,
        config.MIN_FEATURES,
        config.MAX_FEATURES,
    )


def test_ingestion_stores_datasets_with_collision_free_names(project, openml_registry) -> None:
    """Two datasets whose names sanitize to the same string never share a folder."""
    openml_registry[1] = FakeOpenMLDataset(1, "Dataset A", classification_frame(n_instances=600))
    openml_registry[2] = FakeOpenMLDataset(2, "Dataset-A", classification_frame(n_instances=520))

    download_collection.download_openml_collection("demo_collection", CONDITIONS)

    assert stored_dataset_names(project, "demo_collection") == ["dataset_a__1", "dataset_a__2"]
    first = pd.read_csv(project.raw_dir / "demo_collection" / "dataset_a__1" / "data.csv")
    second = pd.read_csv(project.raw_dir / "demo_collection" / "dataset_a__2" / "data.csv")
    assert len(first) == 600
    assert len(second) == 520


def test_ingestion_is_deterministic_across_runs(project, openml_registry) -> None:
    openml_registry[1] = FakeOpenMLDataset(1, "Dataset A", classification_frame())

    download_collection.download_openml_collection("demo_collection", CONDITIONS)
    download_collection.download_openml_collection("demo_collection", CONDITIONS)

    assert stored_dataset_names(project, "demo_collection") == ["dataset_a__1"]


def test_ingestion_isolates_dataset_level_download_failures(project, openml_registry, capsys) -> None:
    openml_registry[1] = FailingDownloadDataset(1, "Broken Set", classification_frame())
    openml_registry[2] = FakeOpenMLDataset(2, "Healthy Set", classification_frame())

    download_collection.download_openml_collection("demo_collection", CONDITIONS)

    output = capsys.readouterr().out
    assert "Skipping dataset 1" in output
    assert "download failed" in output
    assert stored_dataset_names(project, "demo_collection") == ["healthy_set__2"]


def test_ingestion_skips_datasets_without_license_or_citation(project, openml_registry, capsys) -> None:
    openml_registry[1] = FakeOpenMLDataset(1, "Dataset A", classification_frame(), licence=None, citation=None)

    download_collection.download_openml_collection("demo_collection", CONDITIONS)

    output = capsys.readouterr().out
    assert "[SKIPPED] ID 1" in output
    assert "license or citation" in output
    assert stored_dataset_names(project, "demo_collection") == []


def test_ingestion_reports_datasets_that_fail_the_collection_criteria(
    project, openml_registry, capsys
) -> None:
    openml_registry[1] = FakeOpenMLDataset(1, "Dataset A", classification_frame())

    download_collection.download_openml_collection(
        "demo_collection", [{"field": "n_instances", "op": ">=", "value": 1000}]
    )

    output = capsys.readouterr().out
    assert "[REJECTED] ID 1" in output
    assert "n_instances" in output
    assert stored_dataset_names(project, "demo_collection") == []


def test_ingestion_reports_search_failures_without_crashing(monkeypatch, capsys) -> None:
    def offline(*args: object, **kwargs: object) -> pd.DataFrame:
        raise urllib.error.URLError("network unreachable")

    monkeypatch.setattr(openml.datasets, "list_datasets", offline)

    download_collection.download_openml_collection("demo_collection", CONDITIONS)

    output = capsys.readouterr().out
    assert "[ERROR] Failed to search OpenML for collection 'demo_collection'" in output
    assert "URLError" in output


def test_ingestion_respects_the_dataset_limit(project, openml_registry) -> None:
    for did, label in enumerate("ABC", start=1):
        openml_registry[did] = FakeOpenMLDataset(did, f"Dataset {label}", classification_frame())

    download_collection.download_openml_collection("demo_collection", CONDITIONS, limit=2)

    assert len(stored_dataset_names(project, "demo_collection")) == 2


def test_main_skips_broken_blueprints_but_processes_the_valid_ones(
    project, monkeypatch, capsys
) -> None:
    write_blueprint(project.collections_dir / "broken_syntax.yaml", "selection_criteria: {all_of: [\n")
    write_blueprint(project.collections_dir / "no_criteria.yaml", "collection_id: no_criteria\n")
    write_blueprint(project.collections_dir / "good.yaml", VALID_BLUEPRINT)

    processed: list[tuple[str, int]] = []

    def fake_download(collection_id: str, conditions: list, limit: int = 30) -> None:
        processed.append((collection_id, len(conditions)))

    monkeypatch.setattr(download_collection, "download_openml_collection", fake_download)
    monkeypatch.setattr(sys, "argv", ["download_collection.py"])

    download_collection.main()

    output = capsys.readouterr().out
    assert "Skipping collection file 'broken_syntax.yaml'" in output
    assert "Skipping collection file 'no_criteria.yaml'" in output
    assert processed == [("demo", 1)]


def test_main_reports_a_missing_collection_file(project, monkeypatch, capsys) -> None:
    monkeypatch.setattr(sys, "argv", ["download_collection.py", "--collection", "ghost"])

    download_collection.main()

    assert "[ERROR] Collection file not found" in capsys.readouterr().out
