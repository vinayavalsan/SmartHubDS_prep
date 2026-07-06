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

# Plausible human-age bounds; anything outside is treated as missing (the raw
# warehouse `age` contains garbage like -7648 / 1828). Applied to BOTH the raw
# `age` feature and the age_cohort bands, in training and serving.
AGE_MIN = 1
AGE_MAX = 200

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

# --- Lead types --------------------------------------------------------------
LEAD_TYPE_AUTO = 6
LEAD_TYPE_HOME = 1
LEAD_TYPE_NAMES = {LEAD_TYPE_AUTO: "auto", LEAD_TYPE_HOME: "home"}


def lead_type_name(lead_type_id: int) -> str:
    """Human name for a lead type id (falls back to ``type_<id>``)."""
    return LEAD_TYPE_NAMES.get(lead_type_id, f"type_{lead_type_id}")


# --- Model input schema (SINGLE SOURCE OF TRUTH for train AND serve) ---------
# Curated subset of the training-table columns actually fed to the model, split
# by how the sklearn pipeline should treat them. High-cardinality raw columns
# (zip, city, current_carrier, …) are intentionally excluded here even though
# build_training_table keeps them in the stored table.
_MODEL_NUMERIC = [
    "bid",
    "age",
    "continuous_coverage_months",
    "created_hour",
    "created_dayofweek",
    "num_vehicles",
    "num_drivers",
    "num_auto_violations",
    "num_auto_accidents",
    "is_married",
    "multi_vehicle",
] + AGE_COHORT_COLUMNS
# Note: `source_type_id` is intentionally excluded — ~9k unique values would
# one-hot into thousands of sparse columns the model just memorises (it inflated
# ROC AUC without generalising). `account_id` is excluded as a duplicate of
# `campaign_id` (they map 1:1 in the data). The `insured`/`home_owner`/`dui`/
# `military_affiliation` columns stay listed but are auto-dropped at train time
# when they are constant (see preprocessing.drop_zero_variance), so they start
# contributing automatically if the data ever varies.
_MODEL_CATEGORICAL = [
    "state",
    "gender",
    "marital_status",
    "military_affiliation",
    "insured",
    "home_owner",
    "dui",
    "campaign_id",
]
# Features that only make sense for auto leads (dropped for home).
AUTO_ONLY_FEATURES = {
    "num_vehicles",
    "num_drivers",
    "num_auto_violations",
    "num_auto_accidents",
    "dui",
    "home_owner",
    "multi_vehicle",
}


def model_feature_columns(lead_type_id: int) -> tuple[list[str], list[str]]:
    """Return ``(numeric, categorical)`` model feature names for a lead type.

    Auto-only features are dropped for non-auto lead types so the same code
    path trains a home model without vehicle/driver columns.
    """
    drop = set() if lead_type_id == LEAD_TYPE_AUTO else AUTO_ONLY_FEATURES
    numeric = [c for c in _MODEL_NUMERIC if c not in drop]
    categorical = [c for c in _MODEL_CATEGORICAL if c not in drop]
    return numeric, categorical


def add_time_features(out: pd.DataFrame) -> list[str]:
    """Add ``created_hour``/``created_dayofweek`` from ``created_at`` in place.

    Returns the names added (empty if there is no ``created_at`` column — e.g.
    a live scoring request that already carries the time parts).
    """
    if "created_at" not in out.columns:
        return []
    created = pd.to_datetime(out["created_at"], errors="coerce")
    out["created_at"] = created
    out["created_hour"] = created.dt.hour
    out["created_dayofweek"] = created.dt.dayofweek
    return list(TIME_FEATURES)


def derive_serving_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the SAME engineered features used in training to a scoring frame.

    This is the train/serve parity hook: online prediction and the training
    build both go through the same derivation (time parts + is_married,
    multi_vehicle, age_cohort_*), so a model never sees features computed a
    different way than it was trained on. Returns a new frame; callers select
    ``model_feature_columns(...)`` afterwards.
    """
    out = df.copy()
    add_time_features(out)
    _derive_features(out)
    return out


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
        # Clamp impossible ages to NaN (imputer handles) — fixes the raw `age`
        # feature and keeps out-of-range values out of the cohort bands.
        age = age.where((age >= AGE_MIN) & (age <= AGE_MAX))
        out["age"] = age
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

    Target definition (matches the warehouse: `won` is only ever 'true' or
    NULL — losses are never written as 'false'):

    - A bid is **placed** when ``bid > 0``. A placed bid either **won**
      (``won == 'true'`` -> ``won_flag = 1``) or **lost** (``won`` null/blank/
      'false' -> ``won_flag = 0``).
    - Rows with **no bid placed** (``bid`` <= 0 or missing, and not a win) are
      not bidding decisions and are excluded.
    - Optional `lead_type_id` filter.
    - Selects ping-time features + engineered features + bid + expected_revenue,
      target `won_flag`. **Nothing is dropped by default** — set
      ``drop_zero_variance=True`` to also drop single-valued feature columns.
    - Engineered: ``is_married`` (from marital_status), ``multi_vehicle``
      (from num_vehicles), and one-hot ``age_cohort_<band>`` 0/1 columns
      (banded from age).
    """
    out = df.copy()

    # Target: win = won=='true'; a placed bid is bid > 0. Keep placed bids (and
    # any win, defensively), label win=1 / loss=0. Exclude no-bid pings.
    won_true = out["won"].astype("string").str.strip().str.lower().eq(_TRUE)
    bid_default = pd.Series(float("nan"), index=out.index, dtype="float64")
    bid_num = pd.to_numeric(out.get(DECISION_COLUMN, bid_default), errors="coerce")
    placed = bid_num.gt(0).fillna(False)
    keep_rows = placed | won_true.fillna(False)
    out = out[keep_rows].copy()
    out[TARGET_COLUMN] = won_true[keep_rows].fillna(False).astype("int64").to_numpy()

    if lead_type_id is not None and "lead_type_id" in out.columns:
        out = out[out["lead_type_id"] == lead_type_id].copy()

    # Time features + engineered features (derived from raw columns) — the same
    # helpers used at serving time (see derive_serving_features).
    add_time_features(out)
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
