"""Naming helpers, including the collision guarantees of stored dataset names."""
from __future__ import annotations

from pathlib import Path

import pytest

from utils import SOURCE_ID_SEPARATOR, dataset_storage_name, extract_source_id, sanitize_name


@pytest.mark.parametrize(
    ("raw_name", "expected"),
    [
        ("Iris", "iris"),
        ("  Titanic  ", "titanic"),
        ("Dataset A", "dataset_a"),
        ("Dataset-A", "dataset_a"),
        ("Dataset_A", "dataset_a"),
        ("MNIST (784 features)", "mnist_784_features"),
        ("credit;card!", "creditcard"),
        ("", ""),
    ],
)
def test_sanitize_name_normalizes_raw_names(raw_name: str, expected: str) -> None:
    assert sanitize_name(raw_name) == expected


def test_sanitize_name_collapses_names_that_only_differ_in_punctuation() -> None:
    """Documents why the sanitized name alone is not a usable identifier."""
    colliding = {"Dataset A", "Dataset-A", "Dataset_A"}

    assert len({sanitize_name(name) for name in colliding}) == 1


def test_dataset_storage_name_separates_datasets_with_the_same_sanitized_name() -> None:
    first = dataset_storage_name("Dataset A", 61)
    second = dataset_storage_name("Dataset-A", 62)

    assert first != second
    assert first == f"dataset_a{SOURCE_ID_SEPARATOR}61"
    assert second == f"dataset_a{SOURCE_ID_SEPARATOR}62"


def test_dataset_storage_name_is_deterministic() -> None:
    """The same dataset always resolves to the same folder."""
    assert dataset_storage_name("Iris", 61) == dataset_storage_name("Iris", 61)
    assert dataset_storage_name("Iris", "61") == dataset_storage_name("Iris", 61)


def test_dataset_storage_name_stays_human_readable() -> None:
    storage_name = dataset_storage_name("Titanic (survival)", 40945)

    assert storage_name.startswith("titanic_survival")
    assert "40945" in storage_name


@pytest.mark.parametrize("source_id", [1, 61, "40945"])
def test_extract_source_id_round_trips_stored_names(source_id: int | str) -> None:
    assert extract_source_id(dataset_storage_name("Iris", source_id)) == int(source_id)


def test_dataset_storage_name_without_source_id_keeps_the_plain_name() -> None:
    assert dataset_storage_name("Iris") == "iris"
    assert dataset_storage_name("Iris", None) == "iris"
    assert extract_source_id("iris") is None


@pytest.mark.parametrize("manual_folder", ["iris", "covid_data", "my_dataset_v2", "adult-10"])
def test_extract_source_id_ignores_manually_created_folders(manual_folder: str) -> None:
    assert extract_source_id(manual_folder) is None


def test_dataset_storage_name_falls_back_for_unnamed_datasets() -> None:
    assert dataset_storage_name("!!!") == "dataset"
    assert dataset_storage_name("", 7) == f"dataset{SOURCE_ID_SEPARATOR}7"


def test_identical_dataset_names_in_different_collections_do_not_share_a_path(tmp_path: Path) -> None:
    """Sanitized names are only unique together with collection id and source id."""
    raw_dir = tmp_path / "data" / "raw" / "collections"
    paths = {
        raw_dir / "binary_core" / dataset_storage_name("Dataset A", 11),
        raw_dir / "binary_core" / dataset_storage_name("Dataset-A", 12),
        raw_dir / "multiclass_core" / dataset_storage_name("Dataset A", 11),
    }

    assert len(paths) == 3
