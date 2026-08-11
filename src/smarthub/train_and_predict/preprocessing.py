"""Model input preparation for SmartHub training and serving.

This module loads, validates, and normalizes model-ready data.
"""

from __future__ import annotations

import pandas as pd

from smarthub.core import io
from smarthub.core.logging_utils import get_logger
from smarthub.feature_engineering import features as fe

from . import config

logger = get_logger(__name__)


def normalize_model_frame(df, numeric_features, categorical_features):
    """Normalize numeric and categorical model feature types.

    Inputs
    ------
    df : pandas.DataFrame
        Input dataframe.
    numeric_features : list[str]
        Numeric feature names.
    categorical_features : list[str]
        Categorical feature names.

    Returns
    -------
    pandas.DataFrame
        Copy with normalized model feature types.
    """
    df = df.copy()
    for col in numeric_features:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in categorical_features:
        if col in df.columns:
            df[col] = df[col].astype("string").str.strip()

    fe.apply_registered_missing_values(df)
    return df


def prepare_training_data(lead_type_id, lead_type_name, version=None):
    """Load and prepare a training table for model fitting.

    Inputs
    ------
    lead_type_id : int
        SmartHub lead type identifier.
    lead_type_name : str
        Human-readable lead type name.
    version : str | None
        Optional training-table or model version identifier.

    Returns
    -------
    tuple[pandas.DataFrame, list[str], list[str], dict]
        Prepared dataframe, numeric features, categorical features, and
    preparation summary.
    """
    # Resolve the exact training-table version used for model lineage.
    resolved_version = version
    if resolved_version is None:
        versions = io.training_versions(lead_type_name)
        if not versions:
            raise FileNotFoundError(
                "No training-table versions exist for " f"lead type '{lead_type_name}'."
            )
        resolved_version = versions[-1]

    manifest = io.load_training_metadata(lead_type_name, resolved_version)
    required_metadata = {
        "data_min_created_at",
        "data_max_created_at",
        "row_count",
    }
    missing_metadata = sorted(required_metadata.difference(manifest))
    if missing_metadata:
        raise ValueError(
            "Training metadata is missing required fields for "
            f"'{lead_type_name}' version '{resolved_version}': "
            f"{missing_metadata}"
        )

    table = io.load_training_table(lead_type_name, resolved_version)
    raw_rows = len(table)

    numeric, categorical = config.feature_columns(lead_type_id)
    feature_cols = numeric + categorical

    if config.TARGET_COL not in table.columns:
        raise ValueError(
            f"Training table is missing target column '{config.TARGET_COL}'. "
            "Rebuild features (build-features) with the current pipeline."
        )

    # Ensure every model feature exists (a column may be absent if that field
    # was never pulled); missing ones become all-NA and then receive the
    # registry-defined explicit missing sentinel.
    missing = [c for c in feature_cols if c not in table.columns]
    for col in missing:
        table[col] = pd.NA

    frame = normalize_model_frame(table, numeric, categorical)
    frame[config.TARGET_COL] = pd.to_numeric(frame[config.TARGET_COL], errors="coerce")

    keep = list(feature_cols) + [config.TARGET_COL]
    if config.REVENUE_COL in frame.columns:
        keep.append(config.REVENUE_COL)
    if "created_at" in frame.columns:
        keep.append("created_at")
    for identifier in ("lead_ping_id", "lead_id", "id"):
        if identifier in frame.columns and identifier not in keep:
            keep.append(identifier)

    frame = frame[keep].dropna(subset=[config.TARGET_COL]).copy()
    if "created_at" in frame.columns:
        frame = frame.sort_values("created_at").reset_index(drop=True)

    summary = {
        "raw_rows": int(raw_rows),
        "training_rows": int(len(frame)),
        "dropped_rows": int(raw_rows - len(frame)),
        "missing_feature_columns": missing,
        "win_rate": (float(frame[config.TARGET_COL].mean()) if len(frame) else None),
        # Lineage — what data this training table came from.
        "training_table_version": resolved_version,
        "data_min_created_at": manifest["data_min_created_at"],
        "data_max_created_at": manifest["data_max_created_at"],
        "source_row_count": manifest["row_count"],
    }
    return frame, numeric, categorical, summary


def drop_zero_variance(frame, numeric_features, categorical_features):
    """Remove features with no observed variation.

    Inputs
    ------
    frame : pandas.DataFrame
        Input dataframe.
    numeric_features : list[str]
        Numeric feature names.
    categorical_features : list[str]
        Categorical feature names.

    Returns
    -------
    tuple[list[str], list[str], list[str]]
        Filtered numeric features, filtered categorical features, and
    removed feature names.
    """
    dropped = [
        c
        for c in list(numeric_features) + list(categorical_features)
        if c in frame.columns and frame[c].nunique(dropna=True) <= 1
    ]
    numeric = [c for c in numeric_features if c not in dropped]
    categorical = [c for c in categorical_features if c not in dropped]
    return numeric, categorical, dropped


def assert_trainable(frame, lead_type_name):
    """Validate that the training target contains two classes.

    Inputs
    ------
    frame : pandas.DataFrame
        Input dataframe.
    lead_type_name : str
        Human-readable lead type name.

    Returns
    -------
    list
        Sorted target classes present in the dataframe.

    Raises
    ------
    ValueError
        If the target does not contain both outcome classes.
    """
    classes = sorted(frame[config.TARGET_COL].dropna().unique().tolist())
    if len(classes) < 2:
        only = classes[0] if classes else "none"
        win_rate = float(frame[config.TARGET_COL].mean()) if len(frame) else None
        raise ValueError(
            f"Training data for '{lead_type_name}' has only ONE target class "
            f"({config.TARGET_COL}={only}, win_rate={win_rate}). A classifier "
            "needs both wins and losses. Check the `won` distribution in the "
            "pulled leads: losses may be encoded as blank/NULL (and dropped when "
            "building the training table), or `won` may not be the correct "
            "win/loss label. Confirm the label with the team before training."
        )
    return classes


def serving_frame(records, lead_type_id):
    """Build a model-ready dataframe from serving records.

    Inputs
    ------
    records : list[dict] | dict
        Prediction records or record mappings.
    lead_type_id : int
        SmartHub lead type identifier.

    Returns
    -------
    pandas.DataFrame
        Model-ready serving dataframe.
    """
    raw = records.copy() if isinstance(records, pd.DataFrame) else pd.DataFrame(records)
    derived = fe.derive_serving_features(raw, lead_type_id=lead_type_id)

    numeric, categorical = config.feature_columns(lead_type_id)
    feature_cols = numeric + categorical
    for col in feature_cols:
        if col not in derived.columns:
            derived[col] = pd.NA

    normalized = normalize_model_frame(derived, numeric, categorical)
    return normalized[feature_cols].copy()
