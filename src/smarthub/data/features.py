"""Feature extraction: build the leakage-safe training table.

Turns accumulated `lead_pings` rows into one row per bidding decision:
``(features known at bid time, bid, expected_revenue R, won_flag)``.

Follows MODELING.md: keep only ping-time features, exclude post-bid outcomes
(leakage), keep both wins and losses, drop zero-variance columns. Encoding
(one-hot, binning, scaling) is left to the training step.
"""

from __future__ import annotations

import pandas as pd

# Features known at ping arrival (MODELING.md §3, group ①).
PRE_BID_FEATURES = [
    "state",
    "zip",
    "city",
    "age",
    "gender",
    "marital_status",
    "num_vehicles",
    "num_drivers",
    "num_auto_violations",
    "num_auto_claims",
    "num_auto_accidents",
    "insured",
    "current_carrier",
    "continuous_coverage_months",
    "dui",
    "sr22_required",
    "home_owner",
    "military_affiliation",
    "credit",
    "household_income",
    "pnc_bundle",
    "device_type",
    "traffic_tier",
    "lead_type_id",
    "campaign_id",
    "account_id",
    "source_type_id",
    "total_listings",
    "num_selected_listings",
]

# Time features derived from created_at.
TIME_FEATURES = ["created_hour", "created_dayofweek"]

# Age band edges + labels (competitiveness is non-linear in age — the 25–40
# range is the most contested, per Kiran). Right-open bins. Each band becomes
# its own 0/1 one-hot column named ``age_cohort_<label>``.
AGE_COHORT_BINS = [0, 18, 25, 35, 45, 55, 65, 200]
AGE_COHORT_LABELS = [
    "under_18",
    "18_24",
    "25_34",
    "35_44",
    "45_54",
    "55_64",
    "65_plus",
]
AGE_COHORT_COLUMNS = [f"age_cohort_{label}" for label in AGE_COHORT_LABELS]

# Engineered features derived from raw columns (added by build_training_table).
DERIVED_FEATURES = ["is_married", "multi_vehicle"] + AGE_COHORT_COLUMNS

# Known at bid time, used in the profit objective (kept alongside features).
REVENUE_COLUMN = "expected_revenue"
DECISION_COLUMN = "bid"
TARGET_COLUMN = "won_flag"

# Metadata carried for traceability / time-based splitting (not model inputs).
META_COLUMNS = ["id", "created_at"]

# Post-bid outcomes — never features (leakage). Listed for clarity.
LEAKAGE_COLUMNS = [
    "won",
    "rev",
    "realized_payout",
    "payout",
    "profit",
    "accepted",
    "accepted_listings",
    "erred",
    "error_reason_id",
    "response_ms",
    "bidding_strategy_id",
]

_TRUE = "true"
_FALSE = "false"


def _derive_features(out: pd.DataFrame) -> list[str]:
    """Add engineered features in place; return the names that were added."""
    added = []
    if "marital_status" in out.columns:
        ms = out["marital_status"].astype("string").str.strip().str.lower()
        out["is_married"] = ms.eq("married").fillna(False).astype("int64")
        added.append("is_married")
    if "num_vehicles" in out.columns:
        nv = pd.to_numeric(out["num_vehicles"], errors="coerce")
        out["multi_vehicle"] = (nv > 1).fillna(False).astype("int64")
        added.append("multi_vehicle")
    if "age" in out.columns:
        age = pd.to_numeric(out["age"], errors="coerce")
        cohort = pd.cut(
            age,
            bins=AGE_COHORT_BINS,
            labels=AGE_COHORT_LABELS,
            right=False,
        )
        # One-hot: one 0/1 column per band. Missing/unparseable age -> all 0.
        for label, col in zip(AGE_COHORT_LABELS, AGE_COHORT_COLUMNS):
            out[col] = cohort.eq(label).astype("int64")
            added.append(col)
    return added


def build_training_table(
    df: pd.DataFrame,
    lead_type_id: int | None = None,
    drop_zero_variance: bool = False,
) -> pd.DataFrame:
    """Assemble the leakage-safe training table from raw `lead_pings` rows.

    - Keeps only real bidding decisions (`won` is 'true' or 'false'); blank
      `won` (no bid / no outcome) is dropped. Both wins and losses are kept.
    - Optional `lead_type_id` filter.
    - Selects ping-time features + engineered features + bid + expected_revenue,
      target `won_flag`. **Nothing is dropped by default** — set
      ``drop_zero_variance=True`` to also drop single-valued feature columns.
    - Engineered: ``is_married`` (from marital_status), ``multi_vehicle``
      (from num_vehicles), and one-hot ``age_cohort_<band>`` 0/1 columns
      (banded from age).
    """
    out = df.copy()

    # Target: keep only true/false, map to 1/0.
    won_norm = out["won"].astype("string").str.strip().str.lower()
    out = out[won_norm.isin([_TRUE, _FALSE])].copy()
    won_norm = won_norm[won_norm.isin([_TRUE, _FALSE])]
    out[TARGET_COLUMN] = (won_norm == _TRUE).astype("int64").to_numpy()

    if lead_type_id is not None and "lead_type_id" in out.columns:
        out = out[out["lead_type_id"] == lead_type_id].copy()

    # Time features.
    if "created_at" in out.columns:
        created = pd.to_datetime(out["created_at"], errors="coerce")
        out["created_at"] = created
        out["created_hour"] = created.dt.hour
        out["created_dayofweek"] = created.dt.dayofweek

    # Engineered features (derived from raw columns).
    derived = _derive_features(out)

    feature_cols = [c for c in PRE_BID_FEATURES if c in out.columns]
    feature_cols += [c for c in TIME_FEATURES if c in out.columns]
    feature_cols += derived

    keep = (
        [c for c in META_COLUMNS if c in out.columns]
        + feature_cols
        + [c for c in (DECISION_COLUMN, REVENUE_COLUMN) if c in out.columns]
        + [TARGET_COLUMN]
    )
    table = out[keep].copy()

    if drop_zero_variance:
        constant = [
            c for c in feature_cols if table[c].nunique(dropna=True) <= 1
        ]
        table = table.drop(columns=constant)

    return table.reset_index(drop=True)
