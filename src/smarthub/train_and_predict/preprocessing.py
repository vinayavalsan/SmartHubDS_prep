"""
Data cleaning and feature preparation for Anton model training and prediction.

The same model-feature cleaning logic is used for both historical training data
and future prediction requests. Training-only logic, such as target conversion
and row filtering, is kept in ``clean_training_data``.
"""

import pandas as pd

import config


TEXT_CATEGORICAL_FEATURES = ["state", "gender", "marital_status"]
BOOLEAN_CATEGORICAL_FEATURES = [
    "insured",
    "home_owner",
    "military_affiliation",
    "dui",
]
NUMERIC_DEFAULTS = {
    "num_vehicles": 1,
    "num_drivers": 1,
    "continuous_coverage_months": -1,
    "num_auto_accidents": -1,
    "num_auto_violations": -1,
}


def clean_text_category(value):
    """Clean text category values into model-compatible labels."""
    if pd.isna(value) or value == "":
        return "NAvail"

    return str(value).strip()


def clean_bool_category(value):
    """Clean boolean-like category values into model-compatible labels."""
    if pd.isna(value) or value == "":
        return "Unknown"

    return str(value).strip().title()


def clean_model_features(df):
    """Clean model input features for both training and prediction.

    This function is the shared preprocessing layer used before data reaches
    the sklearn model. It should contain every cleaning rule that must be
    applied consistently to historical training rows and future prediction
    rows.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data containing model feature columns. Missing optional feature
        columns are created and cleaned using the same defaults as training.

    Returns
    -------
    pandas.DataFrame
        Cleaned dataframe containing only ``config.FEATURE_COLS`` in the
        expected order.
    """
    df = df.copy()

    for col in config.FEATURE_COLS:
        if col not in df.columns:
            df[col] = pd.NA

    for col in TEXT_CATEGORICAL_FEATURES:
        if col in df.columns:
            df[col] = df[col].apply(clean_text_category)

    for col in BOOLEAN_CATEGORICAL_FEATURES:
        if col in df.columns:
            df[col] = df[col].apply(clean_bool_category)

    for col, default_value in NUMERIC_DEFAULTS.items():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(default_value)

    for col in config.CONTINUOUS_FEATURES + config.DISCRETE_FEATURES:
        if col in df.columns and col not in NUMERIC_DEFAULTS:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Treat impossible ages as missing. The sklearn transformer will impute median.
    if "age" in df.columns:
        df.loc[(df["age"] < 1) | (df["age"] > 100), "age"] = pd.NA

    return df[config.FEATURE_COLS].copy()


def clean_training_data(df, lead_type_id):
    """Clean raw lead data for model training.

    Rules:
    - Train only on the lead type requested by the training script.
    - Required fields must exist for training.
    - Model features are cleaned using ``clean_model_features``.
    - Target is converted to 0/1 for binary classification.
    - expected_revenue is preserved for optimizer evaluation but not used as a feature.
    """
    raw_rows = len(df)

    df = df[df["lead_type_id"] == lead_type_id].copy()

    required_cols = [
        "bid",
        "campaign_id",
        "lead_type_id",
        "source_type_id",
        "account_id",
        "created_hour",
        "created_dayofweek",
    ]

    df = df.dropna(subset=required_cols)

    df[config.TARGET_COL] = (
        df[config.TARGET_COL]
        .astype(str)
        .str.lower()
        .map(
            {
                "true": 1,
                "false": 0,
                "1": 1,
                "0": 0,
            }
        )
    )

    business_eval_cols = (
        ["expected_revenue"] if "expected_revenue" in df.columns else []
    )

    cleaned_features = clean_model_features(df)
    df = pd.concat(
        [cleaned_features, df[business_eval_cols + [config.TARGET_COL]]],
        axis=1,
    )
    df = df.dropna(subset=[config.TARGET_COL])

    training_rows = len(df)
    dropped_rows = raw_rows - training_rows
    dropped_pct = dropped_rows / raw_rows * 100 if raw_rows else 0

    cleaning_summary = {
        "raw_rows": raw_rows,
        "training_rows": training_rows,
        "dropped_rows": dropped_rows,
        "dropped_pct": dropped_pct,
    }

    return df, cleaning_summary
