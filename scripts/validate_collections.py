#!/usr/bin/env python3
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_dataset_metrics(df: pd.DataFrame, target_col: str | None = None) -> dict:
    """
    Computes exact real-time properties from a pandas DataFrame to match 
    against collection criteria profiles.
    """
    n_instances = len(df)
    total_cols = len(df.columns)
    
    if target_col and target_col in df.columns:
        n_features = total_cols - 1
        target_series = df[target_col].dropna()
    else:
        n_features = total_cols
        target_series = pd.Series(dtype=object)

    missing_rate = float(df.isnull().mean().mean())
    p_to_n_ratio = float(n_features / n_instances) if n_instances > 0 else 0.0

    task = "regression"
    task_subtype = "unknown"
    n_classes = 0
    imbalance_ratio = 1.0

    if not target_series.empty:
        unique_vals = target_series.unique()
        n_classes = len(unique_vals)
        
        is_numeric = pd.api.types.is_numeric_dtype(target_series)
        if not is_numeric or n_classes <= 20:
            task = "classification"
            if n_classes == 2:
                task_subtype = "binary"
            elif n_classes > 2:
                task_subtype = "multiclass"
                
            value_counts = target_series.value_counts()
            if len(value_counts) >= 2:
                min_count = value_counts.min()
                # Safeguard against categories that exist but have 0 instances
                if min_count > 0:
                    imbalance_ratio = float(value_counts.max() / min_count)
                else:
                    imbalance_ratio = float('inf')
        else:
            task = "regression"
            task_subtype = "continuous"

    return {
        "n_instances": n_instances,
        "n_features": n_features,
        "n_classes": n_classes,
        "missing_rate": missing_rate,
        "p_to_n_ratio": p_to_n_ratio,
        "task": task,
        "task_subtype": task_subtype,
        "imbalance_ratio": imbalance_ratio,
        "temporal": False, 
        "grouped": False,
        "academic_usage": True
    }


def verify_against_criteria(metrics: dict, conditions: list) -> tuple[bool, str]:
    """Evaluates computed dataset metrics against all_of conditions from YAML files."""
    for cond in conditions:
        field = cond.get("field")
        op = cond.get("op", "==")
        target_val = cond.get("value")

        if field not in metrics:
            continue

        actual_val = metrics[field]

        try:
            if op in [">", ">=", "<", "<="]:
                act_f, targ_f = float(actual_val), float(target_val)
                if op == ">" and not (act_f > targ_f):
                    return False, f"Failed condition: {field} ({act_f}) > {targ_f}"
                if op == ">=" and not (act_f >= targ_f):
                    return False, f"Failed condition: {field} ({act_f}) >= {targ_f}"
                if op == "<" and not (act_f < targ_f):
                    return False, f"Failed condition: {field} ({act_f}) < {targ_f}"
                if op == "<=" and not (act_f <= targ_f):
                    return False, f"Failed condition: {field} ({act_f}) <= {targ_f}"
            elif op == "==":
                if str(actual_val).lower() != str(target_val).lower():
                    return False, f"Failed condition: {field} ({actual_val}) == {target_val}"
            elif op == "!=":
                if str(actual_val).lower() == str(target_val).lower():
                    return False, f"Failed condition: {field} ({actual_val}) != {target_val}"
        except Exception as e:
            return False, f"Error evaluating condition for {field}: {e}"

    return True, "Passed all collection criteria filters."