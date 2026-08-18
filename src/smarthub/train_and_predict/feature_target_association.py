"""Pre-training feature-to-target association diagnostics for SmartHub.

This module characterizes how strongly each model feature is associated with
SmartHub's binary training target before any train/validation/test splitting.
The diagnostics are descriptive only and must not influence feature selection,
model fitting, hyperparameter selection, or promotion decisions.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif


def _numeric_values(series: pd.Series) -> np.ndarray:
    """Return one numeric feature as a finite two-dimensional array."""
    values = pd.to_numeric(series, errors="coerce")
    median = values.median()
    fill_value = float(median) if pd.notna(median) else 0.0
    return values.fillna(fill_value).to_numpy(dtype=float).reshape(-1, 1)


def _categorical_values(series: pd.Series) -> np.ndarray:
    """Return one categorical feature as integer category codes."""
    values = series.astype("string").fillna("<NA>")
    codes, _ = pd.factorize(values, sort=True)
    return codes.reshape(-1, 1)


def build_feature_target_association(
    frame: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
    target_column: str,
    random_seed: int,
) -> list[dict[str, Any]]:
    """Calculate mutual information for every configured model feature.

    Inputs
    ------
    frame : pandas.DataFrame
        Complete prepared dataset before any HPO data split.
    numeric_features : list[str]
        Numeric model feature names.
    categorical_features : list[str]
        Categorical model feature names.
    target_column : str
        Binary target column.
    random_seed : int
        Random seed used by the continuous mutual-information estimator.

    Returns
    -------
    list[dict[str, Any]]
        Feature association rows sorted by decreasing mutual information.
    """
    target = pd.to_numeric(frame[target_column], errors="coerce")
    valid_target = target.notna()
    target_values = target.loc[valid_target].astype(int).to_numpy()

    rows: list[dict[str, Any]] = []
    categorical_set = set(categorical_features)
    features = list(numeric_features) + list(categorical_features)

    for feature in features:
        if feature not in frame.columns:
            continue

        series = frame.loc[valid_target, feature]
        is_categorical = feature in categorical_set
        values = (
            _categorical_values(series) if is_categorical else _numeric_values(series)
        )

        if np.unique(values).size <= 1:
            mutual_information = 0.0
        else:
            mutual_information = float(
                mutual_info_classif(
                    values,
                    target_values,
                    discrete_features=is_categorical,
                    random_state=random_seed,
                )[0]
            )

        rows.append(
            {
                "feature": feature,
                "feature_type": "categorical" if is_categorical else "numeric",
                "mutual_information": mutual_information,
            }
        )

    rows.sort(
        key=lambda row: (-row["mutual_information"], row["feature"]),
    )
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank

    return rows


def log_feature_target_association(
    diagnostics: list[dict[str, Any]],
    logger,
) -> None:
    """Log a compact feature-target mutual-information ranking."""
    logger.info("Feature-Target Association")
    if not diagnostics:
        logger.info("  No feature-target association diagnostics available.")
        return

    table = pd.DataFrame(diagnostics)[
        ["rank", "feature", "feature_type", "mutual_information"]
    ].copy()
    table["mutual_information"] = table["mutual_information"].round(6)
    logger.info("\n%s", table.to_string(index=False))
