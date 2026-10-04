"""
Shared fixtures for the Benchmark Suite test suite.

The pipeline is built from flat scripts that import each other by module name
(``from config import RAW_DIR``), so ``scripts/`` is added to ``sys.path`` here
exactly like it is when the scripts are executed from the command line.

Tests never touch the real project directories and never use the network: every
run works on a temporary project layout and on a blocked/mocked OpenML client.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import openml
import pandas as pd
import pytest
import yaml

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import build_catalog  # noqa: E402
import config  # noqa: E402
import define_collection  # noqa: E402
import download_collection  # noqa: E402
import export_collections  # noqa: E402
import logging_utils  # noqa: E402
import run_pipeline  # noqa: E402
import validate_datasets  # noqa: E402
from utils import dataset_storage_name  # noqa: E402


@pytest.fixture
def openml_registry(monkeypatch: pytest.MonkeyPatch) -> dict[int, Any]:
    """Replaces the OpenML client with an in-memory dataset registry.

    Moved to conftest so orchestrator and download tests share it.
    """
    from test_download_collection import FakeOpenMLDataset
    registry: dict[int, FakeOpenMLDataset] = {}

    def fake_list_datasets(**kwargs: object) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "did": [ds.id for ds in registry.values()],
                "name": [ds.name for ds in registry.values()],
                "number_downloads": list(range(len(registry), 0, -1)),
            }
        )

    def fake_get_dataset(did: int, download_data: bool = False) -> FakeOpenMLDataset:
        try:
            return registry[did]
        except KeyError:
            raise openml.exceptions.PyOpenMLError(f"unknown dataset {did}") from None

    monkeypatch.setattr(openml.datasets, "list_datasets", fake_list_datasets)
    monkeypatch.setattr(openml.datasets, "get_dataset", fake_get_dataset)
    return registry


@pytest.fixture
def academic_api(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mocks OpenML so list_datasets returns empty but get_dataset returns metadata.

    Moved to conftest so orchestrator and integration tests share it.
    """
    monkeypatch.setattr(openml.datasets, "list_datasets", lambda **kwargs: pd.DataFrame())
    monkeypatch.setattr(
        openml.datasets,
        "get_dataset",
        lambda did, download_data=False: _FakeAcademicMetadata(did),
    )


class _FakeAcademicMetadata:
    """Lightweight stand-in for OpenML dataset metadata."""

    def __init__(self, did: int) -> None:
        self.dataset_id = did
        self.licence = "CC BY 4.0"
        self.citation = "Doe, J. et al. (2024). Demo collection data."
        self.paper_url = "https://example.org/paper"


@pytest.fixture(autouse=True)
def reset_structured_logging():
    """Detaches JSON log handlers and clears run/stage context after each test.

    Stages install their own handler when they run; without this, a handler
    created in one test (bound to that test's streams) would leak into the next.
    """
    yield
    logging_utils.reset_logging()


@pytest.fixture(autouse=True)
def offline_openml(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Keeps the suite offline.

    Any OpenML call that was not explicitly mocked by a test fails with the same
    exception the real client raises, so the pipeline code paths that have to
    survive a network problem stay covered instead of silently hitting the API.
    """

    def _no_network(*args: object, **kwargs: object) -> None:
        raise openml.exceptions.PyOpenMLError("network access is disabled in the test suite")

    monkeypatch.setattr(openml.datasets, "list_datasets", _no_network)
    monkeypatch.setattr(openml.datasets, "get_dataset", _no_network)


@dataclass
class PipelineProject:
    """Throw-away copy of the project layout the pipeline steps read and write."""

    root: Path

    @property
    def raw_dir(self) -> Path:
        return self.root / "data" / "raw" / "collections"

    @property
    def processed_dir(self) -> Path:
        return self.root / "data" / "processed" / "collections"

    @property
    def catalog_dir(self) -> Path:
        return self.root / "catalog"

    @property
    def datasets_catalog_dir(self) -> Path:
        return self.catalog_dir / "datasets"

    @property
    def catalog_csv(self) -> Path:
        return self.catalog_dir / "catalog.csv"

    @property
    def manifest_dir(self) -> Path:
        return self.root / "manifests"

    @property
    def collections_dir(self) -> Path:
        return self.root / "collections"

    @property
    def exports_dir(self) -> Path:
        return self.root / "exports"

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Points every pipeline module at this temporary layout."""
        monkeypatch.setattr(config, "ROOT_DIR", self.root)
        monkeypatch.setattr(config, "DATA_DIR", self.root / "data")
        monkeypatch.setattr(config, "RAW_DIR", self.raw_dir)
        monkeypatch.setattr(config, "PROCESSED_DIR", self.processed_dir)
        monkeypatch.setattr(config, "DATASETS_CATALOG_DIR", self.datasets_catalog_dir)
        monkeypatch.setattr(config, "CATALOG_CSV_PATH", self.catalog_csv)
        monkeypatch.setattr(config, "CATALOG_DIR", self.catalog_dir)
        monkeypatch.setattr(config, "MANIFEST_DIR", self.manifest_dir)
        monkeypatch.setattr(config, "COLLECTIONS_DIR", self.collections_dir)
        monkeypatch.setattr(config, "EXPORTS_DIR", self.exports_dir)

        # The scripts import these names into their own namespace ("from config
        # import X"), so each module that actually imported one is patched too.
        for module in (build_catalog, define_collection, download_collection, export_collections, validate_datasets, run_pipeline):
            for attribute, value in (
                ("ROOT_DIR", self.root),
                ("DATA_DIR", self.root / "data"),
                ("RAW_DIR", self.raw_dir),
                ("PROCESSED_DIR", self.processed_dir),
                ("DATASETS_CATALOG_DIR", self.datasets_catalog_dir),
                ("CATALOG_CSV_PATH", self.catalog_csv),
                ("CATALOG_DIR", self.catalog_dir),
                ("MANIFEST_DIR", self.manifest_dir),
                ("COLLECTIONS_DIR", self.collections_dir),
                ("EXPORTS_DIR", self.exports_dir),
            ):
                if hasattr(module, attribute):
                    monkeypatch.setattr(module, attribute, value)

    def add_raw_dataset(self, collection_id: str, name: str, source_id: int, frame: pd.DataFrame) -> str:
        """Writes a raw dataset exactly like the ingestion layer stores it."""
        dataset_id = dataset_storage_name(name, source_id)
        dataset_dir = self.raw_dir / collection_id / dataset_id
        dataset_dir.mkdir(parents=True, exist_ok=True)
        frame.to_csv(dataset_dir / "data.csv", index=False)
        return dataset_id

    def add_broken_raw_dataset(self, collection_id: str, name: str, source_id: int, content: str) -> str:
        """Writes a raw dataset whose CSV cannot be parsed."""
        dataset_id = dataset_storage_name(name, source_id)
        dataset_dir = self.raw_dir / collection_id / dataset_id
        dataset_dir.mkdir(parents=True, exist_ok=True)
        (dataset_dir / "data.csv").write_text(content, encoding="utf-8")
        return dataset_id

    def run_catalog(self) -> None:
        build_catalog.main()

    def run_validation(self) -> None:
        validate_datasets.main()

    def read_catalog(self) -> pd.DataFrame:
        return pd.read_csv(self.catalog_csv)

    def read_dataset_metadata(self, dataset_id: str) -> dict:
        path = self.datasets_catalog_dir / f"{dataset_id}.yaml"
        with path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def update_dataset_metadata(self, dataset_id: str, updates: dict) -> None:
        """Overwrites fields of a cataloged dataset YAML (simulates metadata drift)."""
        path = self.datasets_catalog_dir / f"{dataset_id}.yaml"
        metadata = self.read_dataset_metadata(dataset_id)
        metadata.update(updates)
        with path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(metadata, f, sort_keys=False)

    def read_manifest(self, collection_id: str) -> dict:
        path = self.manifest_dir / f"{collection_id}.resolved.yaml"
        with path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def processed_csv(self, collection_id: str, dataset_id: str) -> Path:
        return self.processed_dir / collection_id / dataset_id / "data.csv"

    def raw_csv(self, collection_id: str, dataset_id: str) -> Path:
        return self.raw_dir / collection_id / dataset_id / "data.csv"


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> PipelineProject:
    """A temporary project layout wired into the real pipeline modules."""
    pipeline_project = PipelineProject(tmp_path)
    for directory in (
        pipeline_project.raw_dir,
        pipeline_project.datasets_catalog_dir,
        pipeline_project.manifest_dir,
        pipeline_project.collections_dir,
        pipeline_project.exports_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)
    pipeline_project.install(monkeypatch)
    return pipeline_project
