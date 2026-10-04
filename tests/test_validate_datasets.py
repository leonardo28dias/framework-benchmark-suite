"""
Cleaning and validation gate.

The gate reads the catalog, cleans the raw matrices and publishes the accepted
datasets plus a frozen manifest. These tests exercise the published artefacts
and the reported statuses rather than internal helper functions.
"""
from __future__ import annotations

import pandas as pd
import pytest

import config
from dataset_factories import classification_frame, insert_feature_column, regression_frame

COLLECTION = "demo_collection"
BROKEN_CSV = "feature_a,feature_b,target\n1,2\n3,4,5,6\n"


def ingest(project, frame: pd.DataFrame, name: str = "Demo Alpha", source_id: int = 11) -> str:
    """Stores one raw dataset, catalogs it and runs the cleaning gate."""
    dataset_id = project.add_raw_dataset(COLLECTION, name, source_id, frame)
    project.run_catalog()
    project.run_validation()
    return dataset_id


def test_valid_classification_dataset_is_cleaned_and_published(project) -> None:
    dataset_id = ingest(project, classification_frame())

    assert project.read_dataset_metadata(dataset_id)["status"] == "validated"
    assert project.processed_csv(COLLECTION, dataset_id).exists()


def test_valid_regression_dataset_is_published(project) -> None:
    dataset_id = ingest(project, regression_frame())

    manifest = project.read_manifest(COLLECTION)
    frozen = manifest["selected_datasets"][0]["frozen_metadata"]
    assert project.processed_csv(COLLECTION, dataset_id).exists()
    assert frozen["task"]["type"] == "regression"
    assert frozen["status"] == "validated"


def test_manifest_freezes_structure_and_metadata_of_selected_datasets(project) -> None:
    dataset_id = ingest(project, classification_frame())

    manifest = project.read_manifest(COLLECTION)

    assert manifest["manifest_id"] == f"{COLLECTION}_resolved_v1"
    assert manifest["source_collection"] == COLLECTION
    assert manifest["materialization"] == {"total_matched_before_clean": 1, "final_dataset_count": 1}
    (entry,) = manifest["selected_datasets"]
    assert entry["dataset_id"] == dataset_id
    assert entry["frozen_metadata"]["structure"] == {
        "n_instances": 600,
        "n_features": 5,
        "missing_rate": 0.0,
    }
    assert entry["frozen_metadata"]["feature_types"] == {"numeric": 5, "categorical": 0}


def test_duplicate_rows_are_removed_from_the_published_matrix(project) -> None:
    frame = classification_frame()
    with_duplicates = pd.concat([frame, frame.head(25)], ignore_index=True)

    dataset_id = ingest(project, with_duplicates)

    cleaned = pd.read_csv(project.processed_csv(COLLECTION, dataset_id))
    assert len(cleaned) == 600
    assert project.read_manifest(COLLECTION)["selected_datasets"][0]["frozen_metadata"]["structure"][
        "n_instances"
    ] == 600


def test_rows_without_target_values_are_removed(project) -> None:
    frame = classification_frame()
    frame.loc[frame.index[:40], "target"] = None

    dataset_id = ingest(project, frame)

    cleaned = pd.read_csv(project.processed_csv(COLLECTION, dataset_id))
    assert len(cleaned) == 560
    assert not cleaned["target"].isnull().any()


def test_missing_feature_values_below_the_threshold_are_kept(project) -> None:
    frame = classification_frame()
    frame.loc[frame.index[:100], "feature_0"] = None

    dataset_id = ingest(project, frame)

    cleaned = pd.read_csv(project.processed_csv(COLLECTION, dataset_id))
    assert "feature_0" in cleaned.columns
    assert int(cleaned["feature_0"].isnull().sum()) == 100


def test_columns_above_the_missing_threshold_are_dropped(project) -> None:
    frame = classification_frame(n_features=config.MIN_FEATURES + 1)
    sparse = [None] * (len(frame) - 10) + [1.0] * 10
    frame = insert_feature_column(frame, "sparse_feature", sparse)

    dataset_id = ingest(project, frame)

    cleaned = pd.read_csv(project.processed_csv(COLLECTION, dataset_id))
    assert "sparse_feature" not in cleaned.columns
    assert project.read_dataset_metadata(dataset_id)["status"] == "validated"


def test_constant_columns_are_dropped(project) -> None:
    frame = classification_frame(n_features=config.MIN_FEATURES + 1)
    frame = insert_feature_column(frame, "constant_feature", "same")

    dataset_id = ingest(project, frame)

    cleaned = pd.read_csv(project.processed_csv(COLLECTION, dataset_id))
    assert "constant_feature" not in cleaned.columns
    assert len(cleaned.columns) - 1 == config.MIN_FEATURES + 1


def test_leakage_identifier_columns_are_dropped(project) -> None:
    frame = classification_frame(n_features=config.MIN_FEATURES + 1)
    frame = insert_feature_column(frame, "row_id", list(range(len(frame))))

    dataset_id = ingest(project, frame)

    cleaned = pd.read_csv(project.processed_csv(COLLECTION, dataset_id))
    assert "row_id" not in cleaned.columns
    assert project.read_dataset_metadata(dataset_id)["status"] == "validated"


def test_repeated_validation_runs_produce_identical_output(project) -> None:
    dataset_id = project.add_raw_dataset(COLLECTION, "Demo Alpha", 11, classification_frame())
    project.run_catalog()

    project.run_validation()
    first_run = project.processed_csv(COLLECTION, dataset_id).read_bytes()
    first_manifest = project.read_manifest(COLLECTION)["selected_datasets"]

    project.run_validation()
    second_run = project.processed_csv(COLLECTION, dataset_id).read_bytes()
    second_manifest = project.read_manifest(COLLECTION)["selected_datasets"]

    assert first_run == second_run
    assert first_manifest == second_manifest


def test_validation_reports_missing_master_catalog(project, capsys: pytest.CaptureFixture) -> None:
    project.run_validation()

    assert "[ERROR] Master catalog not found" in capsys.readouterr().out


def test_constant_target_is_rejected(project, capsys: pytest.CaptureFixture) -> None:
    frame = classification_frame()
    frame["target"] = "only_class"

    dataset_id = ingest(project, frame)

    assert project.read_dataset_metadata(dataset_id)["status"] == "rejected"
    assert not project.processed_csv(COLLECTION, dataset_id).exists()
    assert "is constant" in capsys.readouterr().out


def test_minority_class_below_threshold_is_rejected(project, capsys: pytest.CaptureFixture) -> None:
    frame = classification_frame()
    frame["target"] = ["rare"] * 3 + ["common"] * (len(frame) - 3)

    dataset_id = ingest(project, frame)

    assert project.read_dataset_metadata(dataset_id)["status"] == "rejected"
    assert "Minority class size" in capsys.readouterr().out


def test_target_with_too_many_classes_is_rejected(project, capsys: pytest.CaptureFixture) -> None:
    frame = classification_frame(n_classes=config.MAX_CLASSIFICATION_CLASSES + 1)

    dataset_id = ingest(project, frame)

    assert project.read_dataset_metadata(dataset_id)["status"] == "rejected"
    assert "Target cardinality too high" in capsys.readouterr().out


def test_instance_count_below_minimum_is_rejected(project, capsys: pytest.CaptureFixture) -> None:
    frame = classification_frame(n_instances=config.MIN_INSTANCES - 1)

    dataset_id = ingest(project, frame)

    assert project.read_dataset_metadata(dataset_id)["status"] == "rejected"
    assert "Instance count" in capsys.readouterr().out


def test_instance_count_above_maximum_is_rejected(project, capsys: pytest.CaptureFixture) -> None:
    frame = classification_frame(n_instances=config.MAX_INSTANCES + 1)

    dataset_id = ingest(project, frame)

    assert project.read_dataset_metadata(dataset_id)["status"] == "rejected"
    assert "Instance count" in capsys.readouterr().out


def test_feature_count_below_minimum_is_rejected(project, capsys: pytest.CaptureFixture) -> None:
    frame = classification_frame(n_features=config.MIN_FEATURES - 1)

    dataset_id = ingest(project, frame)

    assert project.read_dataset_metadata(dataset_id)["status"] == "rejected"
    assert "Feature count" in capsys.readouterr().out


def test_feature_count_above_maximum_is_rejected(project, capsys: pytest.CaptureFixture) -> None:
    frame = classification_frame(n_features=config.MAX_FEATURES + 1)

    dataset_id = ingest(project, frame)

    assert project.read_dataset_metadata(dataset_id)["status"] == "rejected"
    assert "Feature count" in capsys.readouterr().out


def test_target_that_does_not_exist_in_the_matrix_is_rejected(project, capsys: pytest.CaptureFixture) -> None:
    dataset_id = project.add_raw_dataset(COLLECTION, "Demo Alpha", 11, classification_frame())
    project.run_catalog()
    project.update_dataset_metadata(dataset_id, {"task": {"type": "classification", "target_variable": "ghost"}})

    project.run_validation()

    assert project.read_dataset_metadata(dataset_id)["status"] == "rejected"
    assert "ghost" in capsys.readouterr().out


def test_unsupported_task_type_is_rejected(project, capsys: pytest.CaptureFixture) -> None:
    dataset_id = project.add_raw_dataset(COLLECTION, "Demo Alpha", 11, classification_frame())
    project.run_catalog()
    project.update_dataset_metadata(dataset_id, {"task": {"type": "clustering", "target_variable": "target"}})

    project.run_validation()

    assert project.read_dataset_metadata(dataset_id)["status"] == "rejected"
    assert "Unsupported task type" in capsys.readouterr().out


def test_missing_raw_matrix_is_rejected_on_the_next_run(project, capsys: pytest.CaptureFixture) -> None:
    dataset_id = ingest(project, classification_frame())

    project.raw_csv(COLLECTION, dataset_id).unlink()
    project.run_validation()

    assert project.read_dataset_metadata(dataset_id)["status"] == "rejected"
    assert "Raw CSV file missing" in capsys.readouterr().out


def test_missing_metadata_yaml_is_rejected(project, capsys: pytest.CaptureFixture) -> None:
    dataset_id = project.add_raw_dataset(COLLECTION, "Demo Alpha", 11, classification_frame())
    project.run_catalog()
    (project.datasets_catalog_dir / f"{dataset_id}.yaml").unlink()

    project.run_validation()

    assert "Metadata YAML missing" in capsys.readouterr().out
    assert project.read_manifest(COLLECTION)["selected_datasets"] == []


def test_corrupted_matrix_is_flagged_as_error(project, capsys: pytest.CaptureFixture) -> None:
    dataset_id = project.add_raw_dataset(COLLECTION, "Demo Alpha", 11, classification_frame())
    project.run_catalog()
    project.raw_csv(COLLECTION, dataset_id).write_text(BROKEN_CSV, encoding="utf-8")

    project.run_validation()

    output = capsys.readouterr().out
    assert project.read_dataset_metadata(dataset_id)["status"] == "error"
    assert "FAILED" in output
    assert dataset_id in output


def test_empty_metadata_file_is_reported_and_does_not_abort_the_run(
    project, capsys: pytest.CaptureFixture
) -> None:
    dataset_id = project.add_raw_dataset(COLLECTION, "Demo Alpha", 11, classification_frame())
    project.run_catalog()
    (project.datasets_catalog_dir / f"{dataset_id}.yaml").write_text("", encoding="utf-8")

    project.run_validation()

    output = capsys.readouterr().out
    assert "is empty or not a mapping" in output
    assert "Cannot flag" in output
    assert project.read_manifest(COLLECTION)["selected_datasets"] == []


def test_rejected_datasets_are_excluded_from_the_manifest(project) -> None:
    valid_id = project.add_raw_dataset(COLLECTION, "Demo Alpha", 11, classification_frame())
    rejected_frame = classification_frame()
    rejected_frame["target"] = "only_class"
    rejected_id = project.add_raw_dataset(COLLECTION, "Demo Beta", 12, rejected_frame)
    project.run_catalog()

    project.run_validation()

    manifest = project.read_manifest(COLLECTION)
    assert [entry["dataset_id"] for entry in manifest["selected_datasets"]] == [valid_id]
    assert manifest["materialization"] == {"total_matched_before_clean": 2, "final_dataset_count": 1}
    assert project.read_dataset_metadata(rejected_id)["status"] == "rejected"
    assert project.read_dataset_metadata(valid_id)["status"] == "validated"
