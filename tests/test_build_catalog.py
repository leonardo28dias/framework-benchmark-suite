"""Structural analysis performed by the cataloging layer."""
from __future__ import annotations

import openml
import pandas as pd
import pytest

from dataset_factories import classification_frame

import build_catalog


@pytest.mark.parametrize(
    ("target_values", "expected_task"),
    [
        (["a", "b", "a", "b"], "classification"),
        ([True, False, True, False], "classification"),
        ([0, 1, 0, 1], "classification"),
        ([1.5, 2.25, 3.75, 4.5], "regression"),
        (list(range(40)), "regression"),
    ],
)
def test_infer_task_type_uses_dtype_and_cardinality(target_values: list, expected_task: str) -> None:
    frame = pd.DataFrame({"feature": list(range(len(target_values))), "target": target_values})

    assert build_catalog.infer_task_type(frame, "target") == expected_task


def test_infer_task_type_returns_unknown_for_missing_target() -> None:
    frame = pd.DataFrame({"feature": [1, 2]})

    assert build_catalog.infer_task_type(frame, "target") == "unknown"


def test_analyze_dataframe_reports_structure_and_quality_flags() -> None:
    frame = pd.DataFrame(
        {
            "numeric": [1, 1, 2, 2, 3],
            "text": ["a", "b", "c", "d", "e"],
            "constant": ["x", "x", "x", "x", "x"],
            "row_id": [101, 102, 103, 104, 105],
            "target": ["no", "yes", "no", "yes", "no"],
        }
    )

    analysis = build_catalog.analyze_dataframe(frame)

    assert analysis["target_variable"] == "target"
    assert analysis["inferred_task"] == "classification"
    assert analysis["structure"]["n_instances"] == 5
    assert analysis["structure"]["n_features"] == 5
    assert analysis["structure"]["duplicate_rows"] == 0
    assert analysis["data_quality"]["constant_columns"] == ["constant"]
    assert analysis["data_quality"]["high_cardinality_columns"] == ["text", "row_id"]
    assert analysis["feature_types"]["numeric"] == 2
    assert analysis["feature_types"]["categorical"] == 3
    assert analysis["feature_types"]["binary"] == 1
    assert analysis["feature_types"]["datetime"] == 0


def test_analyze_dataframe_counts_missing_values_and_duplicate_rows() -> None:
    frame = pd.DataFrame(
        {
            "feature": [1.0, 2.0, None, 4.0],
            "target": ["a", "b", "b", "c"],
        }
    )

    analysis = build_catalog.analyze_dataframe(frame)

    assert analysis["structure"]["missing_rate"] == pytest.approx(0.125)
    assert analysis["data_quality"]["columns_with_missing"] == ["feature"]
    assert analysis["structure"]["duplicate_rows"] == 0


def test_analyze_dataframe_counts_repeated_rows() -> None:
    frame = pd.DataFrame({"feature": [1, 1, 2], "target": ["a", "a", "b"]})

    analysis = build_catalog.analyze_dataframe(frame)

    assert analysis["structure"]["duplicate_rows"] == 1


def test_analyze_dataframe_handles_empty_input() -> None:
    analysis = build_catalog.analyze_dataframe(pd.DataFrame())

    assert analysis["structure"]["n_instances"] == 0
    assert analysis["structure"]["n_features"] == 0
    assert analysis["target_variable"] == "unknown"
    assert analysis["inferred_task"] == "unknown"


def test_catalog_uses_stored_source_id_and_keeps_unknown_metadata_on_api_failure(
    project, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    dataset_id = project.add_raw_dataset("demo_collection", "Demo Alpha", 11, classification_frame())

    def failing_lookup(*args: object, **kwargs: object) -> None:
        raise openml.exceptions.PyOpenMLError("OpenML is unreachable")

    monkeypatch.setattr(openml.datasets, "get_dataset", failing_lookup)

    project.run_catalog()

    output = capsys.readouterr().out
    assert "Academic metadata lookup failed" in output
    assert dataset_id in output
    metadata = project.read_dataset_metadata(dataset_id)
    assert metadata["academic_meta"]["license"] == "unknown"
    assert metadata["status"] == "cataloged"


def test_catalog_warns_about_dataset_directories_without_data(project, capsys: pytest.CaptureFixture) -> None:
    (project.raw_dir / "demo_collection" / "empty_dataset").mkdir(parents=True)

    project.run_catalog()

    output = capsys.readouterr().out
    assert "demo_collection" in output
    assert "empty_dataset" in output
    assert "no data.csv found" in output
    assert not project.catalog_csv.exists()


def test_catalog_reports_missing_raw_directory(project, monkeypatch: pytest.MonkeyPatch,
                                               capsys: pytest.CaptureFixture) -> None:
    monkeypatch.setattr(build_catalog, "RAW_DIR", project.root / "not_created")

    project.run_catalog()

    assert "[ERROR] Raw data directory not found" in capsys.readouterr().out


def test_catalog_reports_unreadable_csv_and_keeps_cataloging(
    project, capsys: pytest.CaptureFixture
) -> None:
    valid_id = project.add_raw_dataset("demo_collection", "Demo Alpha", 11, classification_frame())
    broken_id = project.add_broken_raw_dataset(
        "demo_collection", "Demo Broken", 12, "feature_a,feature_b,target\n1,2\n3,4,5,6\n"
    )

    project.run_catalog()

    output = capsys.readouterr().out
    assert "Failed to catalog" in output
    assert broken_id in output
    catalog = project.read_catalog()
    assert list(catalog["dataset_id"]) == [valid_id]


def test_catalog_does_not_swallow_unexpected_failures(project, monkeypatch: pytest.MonkeyPatch) -> None:
    """Only API and data errors are tolerated; defects must surface."""
    project.add_raw_dataset("demo_collection", "Demo Alpha", 11, classification_frame())

    def unexpected_failure(*args: object, **kwargs: object) -> None:
        raise RuntimeError("unexpected failure")

    monkeypatch.setattr(openml.datasets, "get_dataset", unexpected_failure)

    with pytest.raises(RuntimeError, match="unexpected failure"):
        project.run_catalog()


def test_incremental_catalog_skips_unchanged_datasets(project, monkeypatch) -> None:
    """A second catalog run with the same raw bytes reuses the previous metadata."""
    from utils import file_checksum

    dataset_id = project.add_raw_dataset("demo_collection", "Demo Alpha", 11, classification_frame())
    project.run_catalog()
    first_catalog = project.read_catalog()
    first_checksum = file_checksum(project.raw_csv("demo_collection", dataset_id))

    metadata_after_first = project.read_dataset_metadata(dataset_id)

    project.run_catalog()
    second_catalog = project.read_catalog()

    assert list(second_catalog["dataset_id"]) == list(first_catalog["dataset_id"])
    metadata_after_second = project.read_dataset_metadata(dataset_id)
    assert metadata_after_second["raw_checksum"] == first_checksum

    assert project.read_dataset_metadata(dataset_id) == metadata_after_first


def test_incremental_catalog_rebuilds_changed_datasets(project, capsys) -> None:
    """When raw bytes change the dataset is reprocessed and the checksum updates."""
    from utils import file_checksum

    dataset_id = project.add_raw_dataset("demo_collection", "Demo Alpha", 11, classification_frame())
    project.run_catalog()
    first_checksum = file_checksum(project.raw_csv("demo_collection", dataset_id))
    first_n_instances = project.read_catalog().iloc[0]["n_instances"]

    modified = classification_frame(n_instances=800)
    modified.to_csv(project.raw_csv("demo_collection", dataset_id), index=False)

    project.run_catalog()
    output = capsys.readouterr().out

    assert "Unchanged" not in output
    assert "Cataloging" in output

    second_checksum = file_checksum(project.raw_csv("demo_collection", dataset_id))
    assert second_checksum != first_checksum
    second_n_instances = project.read_catalog().iloc[0]["n_instances"]
    assert second_n_instances == 800
    assert second_n_instances != first_n_instances


def test_incremental_catalog_processes_when_checksum_metadata_missing(project, capsys) -> None:
    """A dataset whose YAML lacks raw_checksum is processed from scratch."""
    dataset_id = project.add_raw_dataset("demo_collection", "Demo Alpha", 11, classification_frame())
    project.run_catalog()

    project.update_dataset_metadata(dataset_id, {"raw_checksum": None})

    project.run_catalog()
    output = capsys.readouterr().out
    assert "Cataloging" in output


def test_incremental_catalog_handles_multiple_datasets(project, capsys) -> None:
    """One unchanged dataset should be skipped while one changed is rebuilt."""
    from utils import file_checksum

    unchanged_id = project.add_raw_dataset("demo_collection", "Alpha", 11, classification_frame())
    changed_id = project.add_raw_dataset("demo_collection", "Beta", 12, classification_frame())

    project.run_catalog()
    unchanged_metadata_before = project.read_dataset_metadata(unchanged_id)

    modified = classification_frame(n_instances=700)
    modified.to_csv(project.raw_csv("demo_collection", changed_id), index=False)

    project.run_catalog()
    output = capsys.readouterr().out

    assert "Unchanged" in output
    unchanged_metadata_after = project.read_dataset_metadata(unchanged_id)
    assert unchanged_metadata_after == unchanged_metadata_before

    changed_metadata = project.read_dataset_metadata(changed_id)
    expected_checksum = file_checksum(project.raw_csv("demo_collection", changed_id))
    assert changed_metadata["raw_checksum"] == expected_checksum
