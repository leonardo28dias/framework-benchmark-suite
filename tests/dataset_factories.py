"""
Deterministic synthetic datasets shared by the test modules.

They are generated with a fixed random seed so every run produces byte identical
inputs, and they intentionally stay close to the structural rules enforced by
config.py (500..10000 instances, 5..100 features, at least 2 classes, ...).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def classification_frame(n_instances: int = 600, n_features: int = 5, n_classes: int = 2) -> pd.DataFrame:
    """Synthetic classification frame with a text target column as the last column."""
    rng = np.random.default_rng(20240619)
    data: dict[str, Any] = {
        f"feature_{index}": rng.integers(1, 100, size=n_instances) for index in range(n_features)
    }
    labels = [f"class_{index}" for index in range(n_classes)]
    data["target"] = [labels[index % n_classes] for index in range(n_instances)]
    return pd.DataFrame(data)


def regression_frame(n_instances: int = 600, n_features: int = 5) -> pd.DataFrame:
    """
    Synthetic regression frame with a numeric target column as the last column.

    The target repeats values (60 levels) because the cleaning gate rejects
    datasets whose target values are all singletons.
    """
    rng = np.random.default_rng(20240620)
    data: dict[str, Any] = {f"feature_{index}": rng.normal(size=n_instances) for index in range(n_features)}
    data["target"] = [float((index % 60) + 1) * 0.5 for index in range(n_instances)]
    return pd.DataFrame(data)


def insert_feature_column(frame: pd.DataFrame, name: str, values: Any) -> pd.DataFrame:
    """
    Adds a feature column while keeping the target as the last column.

    The pipeline treats the last column of a raw matrix as the target, so extra
    features always have to be inserted in front of it.
    """
    extended = frame.copy()
    extended.insert(len(extended.columns) - 1, name, values)
    return extended
