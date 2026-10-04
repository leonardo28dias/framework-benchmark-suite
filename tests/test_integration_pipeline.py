"""
Integration smoke test: the real catalog, validation and export layers running
together on a small fixture collection with a mocked OpenML client.
"""
from __future__ import annotations

import sys

import openml
import pandas as pd
import pytest

import export_collections
from dataset_factories import classification_frame

COLLECTION = "demo_collection"
BROKEN_CSV = "feature_a,feature_b,target\n1,2\n3,4,5,6\n"


class FakeAcademicMetadata:
    """Academic metadata served by the mocked OpenML client."""

    def __init__(self, did: int) -> None:
        self.dataset_id = did
        self.licence = "CC BY 4.0"
        self.citation = "Doe, J. et al. (2024). Demo collection data."
        self.paper_url = "https://example.org/paper"


@pytest.fixture
def academic_api(monkeypatch: pytest.MonkeyPatch) -> None:
    """No dataset listing, but metadata lookups succeed for any known id."""
    monkeypatch.setattr(openml.datasets, "list_datasets", lambda **kwargs: pd.DataFrame())
    monkeypatch.setattr(
        openml.datasets,
        "get_dataset",
        lambda did, download_data=False: FakeAcademicMetadata(did),
    )


def test_pipeline_catalogs_validates_and_exports_a_fixture_collection(
    project, academic_api, capsys: pytest.CaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Raw inputs: one healthy dataset, one rejected by the cleaning gate and one
    # whose CSV cannot be parsed at all.
    valid_id = project.add_raw_dataset(COLLECTION, "Demo Alpha", 11, classification_frame())
    rejected_frame = classification_frame()
    rejected_frame["target"] = "only_class"
    rejected_id = project.add_raw_dataset(COLLECTION, "Demo Beta", 12, rejected_frame)
    broken_id = project.add_broken_raw_dataset(COLLECTION, "Demo Gamma", 13, BROKEN_CSV)

    project.run_catalog()
    catalog_output = capsys.readouterr().out

    # Cataloging: two datasets cataloged, the broken matrix is reported loudly.
    assert "Failed to catalog" in catalog_output
    assert broken_id in catalog_output
    catalog = project.read_catalog()
    assert sorted(catalog["dataset_id"]) == sorted([valid_id, rejected_id])

    valid_metadata = project.read_dataset_metadata(valid_id)
    assert valid_metadata["status"] == "cataloged"
    assert valid_metadata["task"] == {"type": "classification", "target_variable": "target"}
    assert valid_metadata["academic_meta"]["license"] == "CC BY 4.0"
    assert valid_metadata["academic_meta"]["citation"] != "unknown"

    project.run_validation()
    validation_output = capsys.readouterr().out

    # Validation: the healthy dataset is published, the rejected one is visible.
    assert "[VALIDATED]" in validation_output
    assert "is constant" in validation_output
    assert project.read_dataset_metadata(valid_id)["status"] == "validated"
    assert project.read_dataset_metadata(rejected_id)["status"] == "rejected"
    assert project.processed_csv(COLLECTION, valid_id).exists()
    assert not project.processed_csv(COLLECTION, rejected_id).exists()

    manifest = project.read_manifest(COLLECTION)
    assert [entry["dataset_id"] for entry in manifest["selected_datasets"]] == [valid_id]
    assert manifest["materialization"] == {"total_matched_before_clean": 2, "final_dataset_count": 1}

    # Export through the real CLI entry point.
    monkeypatch.setattr(
        sys,
        "argv",
        ["export_collections.py", "--collection", COLLECTION, "--format", "folder"],
    )
    export_collections.main()

    export_root = project.exports_dir / COLLECTION
    assert (export_root / "manifest.yaml").exists()
    assert (export_root / valid_id / "data.csv").exists()
    assert not (export_root / rejected_id).exists()
    assert (export_root / valid_id / "data.csv").read_bytes() == project.processed_csv(
        COLLECTION, valid_id
    ).read_bytes()


def test_catalog_output_is_reproducible_across_runs(project, academic_api) -> None:
    project.add_raw_dataset(COLLECTION, "Demo Alpha", 11, classification_frame())
    project.add_raw_dataset(COLLECTION, "Demo Beta", 12, classification_frame(n_instances=550))

    project.run_catalog()
    first_run = list(project.read_catalog()["dataset_id"])

    project.run_catalog()
    second_run = list(project.read_catalog()["dataset_id"])

    assert first_run == second_run
