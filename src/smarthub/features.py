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


def build_training_table(
    df: pd.DataFrame,
    lead_type_id: int | None = None,
    drop_zero_variance: bool = True,
) -> pd.DataFrame:
    """Assemble the leakage-safe training table from raw `lead_pings` rows.

    - Keeps only real bidding decisions (`won` is 'true' or 'false'); blank
      `won` (no bid / no outcome) is dropped. Both wins and losses are kept.
    - Optional `lead_type_id` filter.
    - Selects ping-time features + bid + expected_revenue, target `won_flag`.
    - Drops feature columns with no variance (single value / all-null).
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

    feature_cols = [c for c in PRE_BID_FEATURES if c in out.columns]
    feature_cols += [c for c in TIME_FEATURES if c in out.columns]

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
