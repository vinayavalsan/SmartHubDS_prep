"""
Data cleaning and feature preparation for Anton model training.
"""

import pandas as pd

from config import FEATURE_COLS, TARGET_COL


def clean_training_data(df):
    """Clean raw lead data for model training.

    Rules:
    - Train only on Auto leads for now.
    - Required fields must exist for training.
    - Missing categorical values are filled with NAvail or Unknown.
    - Numeric business defaults are applied where appropriate.
    - Impossible ages are converted to NA and later imputed by the sklearn pipeline.
    - expected_revenue is preserved for optimizer evaluation but not used as a feature.
    """
    raw_rows = len(df)

    df = df[df["lead_type_id"] == 6].copy()

    required_cols = [
        "bid",
        "campaign_id",
        "lead_type_id",
        "created_hour",
        "created_dayofweek",
    ]

    df = df.dropna(subset=required_cols)

    for col in ["state", "gender", "marital_status"]:
        df[col] = df[col].fillna("NAvail").replace("", "NAvail")

    for col in ["insured", "home_owner", "military_affiliation", "dui"]:
        df[col] = (
            df[col]
            .fillna("Unknown")
            .replace("", "Unknown")
            .astype(str)
            .str.strip()
            .str.title()
        )

    df["num_vehicles"] = df["num_vehicles"].fillna(1)
    df["num_drivers"] = df["num_drivers"].fillna(1)

    df["continuous_coverage_months"] = df["continuous_coverage_months"].fillna(-1)
    df["num_auto_accidents"] = df["num_auto_accidents"].fillna(-1)
    df["num_auto_violations"] = df["num_auto_violations"].fillna(-1)

    # Treat impossible ages as missing. The sklearn transformer will impute median.
    df.loc[(df["age"] < 1) | (df["age"] > 100), "age"] = pd.NA

    df[TARGET_COL] = (
        df[TARGET_COL]
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

    business_eval_cols = ["expected_revenue"] if "expected_revenue" in df.columns else []

    df = df[FEATURE_COLS + business_eval_cols + [TARGET_COL]].copy()
    df = df.dropna(subset=[TARGET_COL])

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
