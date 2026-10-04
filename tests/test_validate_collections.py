"""
Dataset metrics and criteria evaluation used by the ingestion gate.

These are the values the download layer compares against the collection
blueprint conditions coming from YAML.
"""
from __future__ import annotations

import pandas as pd
import pytest

from validate_collections import compute_dataset_metrics, verify_against_criteria


def classification_metrics() -> dict:
    frame = pd.DataFrame(
        {
            "f1": list(range(10)),
            "f2": list(range(10)),
            "target": ["a"] * 5 + ["b"] * 5,
        }
    )
    return compute_dataset_metrics(frame, target_col="target")


def test_compute_dataset_metrics_profiles_binary_classification() -> None:
    metrics = classification_metrics()

    assert metrics["n_instances"] == 10
    assert metrics["n_features"] == 2
    assert metrics["n_classes"] == 2
    assert metrics["task"] == "classification"
    assert metrics["task_subtype"] == "binary"
    assert metrics["imbalance_ratio"] == 1.0
    assert metrics["missing_rate"] == 0.0


def test_compute_dataset_metrics_reports_imbalance_ratio() -> None:
    frame = pd.DataFrame({"f1": range(10), "target": ["minor"] + ["major"] * 9})

    metrics = compute_dataset_metrics(frame, target_col="target")

    assert metrics["n_classes"] == 2
    assert metrics["imbalance_ratio"] == 9.0


def test_compute_dataset_metrics_detects_multiclass_targets() -> None:
    frame = pd.DataFrame({"f1": range(9), "target": ["a", "b", "c"] * 3})

    metrics = compute_dataset_metrics(frame, target_col="target")

    assert metrics["task"] == "classification"
    assert metrics["task_subtype"] == "multiclass"
    assert metrics["n_classes"] == 3


def test_compute_dataset_metrics_detects_continuous_regression() -> None:
    frame = pd.DataFrame({"f1": range(60), "target": [value * 1.5 for value in range(60)]})

    metrics = compute_dataset_metrics(frame, target_col="target")

    assert metrics["task"] == "regression"
    assert metrics["task_subtype"] == "continuous"
    assert metrics["n_classes"] == 60


def test_compute_dataset_metrics_measures_missing_rate() -> None:
    frame = pd.DataFrame({"f1": [1.0, None], "f2": [None, None], "target": ["a", "b"]})

    metrics = compute_dataset_metrics(frame, target_col="target")

    assert metrics["missing_rate"] == pytest.approx(0.5)


def test_compute_dataset_metrics_counts_every_column_when_target_is_unknown() -> None:
    frame = pd.DataFrame({"f1": [1, 2], "f2": [3, 4]})

    metrics = compute_dataset_metrics(frame, target_col="not_there")

    assert metrics["n_features"] == 2
    assert metrics["n_classes"] == 0
    assert metrics["p_to_n_ratio"] == 1.0


def test_compute_dataset_metrics_handles_empty_frames() -> None:
    metrics = compute_dataset_metrics(pd.DataFrame({"target": []}), target_col="target")

    assert metrics["n_instances"] == 0
    assert metrics["p_to_n_ratio"] == 0.0


@pytest.mark.parametrize(
    "condition",
    [
        {"field": "task", "op": "==", "value": "classification"},
        {"field": "task_subtype", "op": "==", "value": "binary"},
        {"field": "n_instances", "op": ">=", "value": 10},
        {"field": "n_instances", "op": "<=", "value": 10},
        {"field": "n_instances", "op": "<", "value": 11},
        {"field": "missing_rate", "op": "<=", "value": 0.05},
        {"field": "imbalance_ratio", "op": ">=", "value": 1},
    ],
)
def test_verify_against_criteria_accepts_matching_conditions(condition: dict) -> None:
    is_valid, reason = verify_against_criteria(classification_metrics(), [condition])

    assert is_valid, reason


@pytest.mark.parametrize(
    "condition",
    [
        {"field": "task", "op": "==", "value": "regression"},
        {"field": "task", "op": "!=", "value": "classification"},
        {"field": "n_instances", "op": ">", "value": 10},
        {"field": "n_instances", "op": "<", "value": 10},
        {"field": "missing_rate", "op": "<=", "value": -1},
    ],
)
def test_verify_against_criteria_rejects_failing_conditions(condition: dict) -> None:
    is_valid, reason = verify_against_criteria(classification_metrics(), [condition])

    assert not is_valid
    assert condition["field"] in reason


def test_verify_against_criteria_requires_every_condition_to_match() -> None:
    conditions = [
        {"field": "n_instances", "op": ">=", "value": 1},
        {"field": "n_classes", "op": "==", "value": 3},
    ]

    is_valid, reason = verify_against_criteria(classification_metrics(), conditions)

    assert not is_valid
    assert "n_classes" in reason


def test_verify_against_criteria_ignores_conditions_about_unknown_fields() -> None:
    is_valid, reason = verify_against_criteria(
        classification_metrics(), [{"field": "not_a_metric", "op": "==", "value": 1}]
    )

    assert is_valid, reason


def test_verify_against_criteria_reports_conditions_that_cannot_be_compared() -> None:
    is_valid, reason = verify_against_criteria(
        classification_metrics(), [{"field": "n_instances", "op": ">", "value": "many"}]
    )

    assert not is_valid
    assert "n_instances" in reason
