"""Feature extraction: build the leakage-safe training table.

Turns accumulated `lead_pings` rows into one row per bidding decision:
``(features known at bid time, bid, expected_revenue R, won_flag)``.

Follows MODELING.md: keep only ping-time features, exclude post-bid outcomes
(leakage), keep both wins and losses, drop zero-variance columns. Encoding
(one-hot, binning, scaling) is left to the training step.
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)

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
    "num_home_claims",
    "home_property_type",
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
DERIVED_FEATURES = (
    ["is_married", "multi_vehicle", "age_missing", "is_workday"]
    + AGE_COHORT_COLUMNS
)

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
    """Human name for a lead type id (falls back to ``type_<id>``).

    Inputs
    ------
    lead_type_id : int
        Lead type id (e.g. 6=auto, 1=home).

    Returns
    -------
    str
        The lead type name, or ``type_<id>`` if unknown.
    """
    return LEAD_TYPE_NAMES.get(lead_type_id, f"type_{lead_type_id}")


# --- Model input schema (SINGLE SOURCE OF TRUTH for train AND serve) ---------
# Curated subset of the training-table columns actually fed to the model, split
# by how the sklearn pipeline should treat them. High-cardinality raw columns
# (zip, city, current_carrier, …) are intentionally excluded here even though
# build_training_table keeps them in the stored table.
_MODEL_NUMERIC = [
    "bid",
    "age",
    "age_missing",
    "continuous_coverage_months",
    "created_hour",
    "created_dayofweek",
    "is_workday",
    "num_vehicles",
    "num_drivers",
    "num_auto_violations",
    "num_auto_accidents",
    "num_home_claims",
    "is_married",
    "multi_vehicle",
] + AGE_COHORT_COLUMNS
# `traffic_tier` is included — Kiran: source completeness/quality lives at the
# partner-subsource (traffic tier) level, so this carries competitor-bidding
# signal into the model. `source_type_id` is excluded (~9k unique values → sparse
# one-hot memorisation that inflated AUC without generalising); `account_id` is
# excluded as a duplicate of `campaign_id` (1:1 in the data). The `insured`/
# `home_owner`/`dui`/`military_affiliation` columns stay listed but are
# auto-dropped at train time when constant (preprocessing.drop_zero_variance).
_MODEL_CATEGORICAL = [
    "state",
    "gender",
    "marital_status",
    "military_affiliation",
    "insured",
    "home_owner",
    "dui",
    "sr22_required",
    "campaign_id",
    "traffic_tier",
    "home_property_type",
]
# Lead-type-specific features (Kiran: auto vs home have different fields).
AUTO_ONLY_FEATURES = {
    "num_vehicles",
    "num_drivers",
    "num_auto_violations",
    "num_auto_accidents",
    "dui",
    "sr22_required",
    "home_owner",
    "multi_vehicle",
}
HOME_ONLY_FEATURES = {
    "num_home_claims",
    "home_property_type",
}


# --- Mandatory vs optional model features ------------------------------------
# MANDATORY features are ALWAYS trained on and cannot be switched off via
# config. The auto mandatory core is SmartFinancial's lead-matching criteria
# (the product tiers: home owner, multiple vehicles, currently insured,
# accidents, DUI, SR-22, age) plus ``bid`` — the optimizer's decision variable,
# which the model must always see. Every other model feature for that lead type
# is OPTIONAL and toggled per training run via ``config/smarthub.ini``
# ``[features] <lead_type>_optional`` (see ``_configured_optional``).
#
# Home's mandatory core is not settled yet, so home keeps every feature (only
# ``bid`` is pinned) until we agree it with Kiran.
_MANDATORY_AUTO = {
    "bid",
    "age",
    "age_missing",
    *AGE_COHORT_COLUMNS,
    "num_vehicles",
    "multi_vehicle",
    "num_auto_accidents",
    "home_owner",
    "insured",
    "dui",
    "sr22_required",
}
_MANDATORY_HOME = {"bid"}
MANDATORY_FEATURES = {
    LEAD_TYPE_AUTO: _MANDATORY_AUTO,
    LEAD_TYPE_HOME: _MANDATORY_HOME,
}

# Sentinels for the ``[features] <lead_type>_optional`` config value.
_OPTIONAL_ALL = "all"
_OPTIONAL_NONE = "none"


def mandatory_features(lead_type_id: int) -> set[str]:
    """Model features always trained on for a lead type (never toggled).

    Inputs
    ------
    lead_type_id : int
        Lead type id.

    Returns
    -------
    set[str]
        The mandatory feature names for that lead type.
    """
    return set(MANDATORY_FEATURES.get(lead_type_id, {DECISION_COLUMN}))


def optional_features(lead_type_id: int) -> set[str]:
    """The full set of toggleable (non-mandatory) model features.

    Inputs
    ------
    lead_type_id : int
        Lead type id.

    Returns
    -------
    set[str]
        Optional feature names eligible for that lead type.
    """
    if lead_type_id == LEAD_TYPE_AUTO:
        drop = HOME_ONLY_FEATURES
    elif lead_type_id == LEAD_TYPE_HOME:
        drop = AUTO_ONLY_FEATURES
    else:
        drop = set()
    base = {c for c in _MODEL_NUMERIC + _MODEL_CATEGORICAL if c not in drop}
    return base - mandatory_features(lead_type_id)


def _configured_optional(lead_type_id: int, optional_universe: set[str]) -> set[str]:
    """Which OPTIONAL features are enabled for this lead type, from config.

    Reads ``config/smarthub.ini`` ``[features] <lead_type>_optional``:
    absent / ``"all"`` -> every optional feature (backwards-compatible
    default); ``"none"`` / empty -> no optional features (mandatory core
    only); comma list -> exactly those (unknown names are ignored + warned).

    Inputs
    ------
    lead_type_id : int
        Lead type id whose config key is read.
    optional_universe : set[str]
        Eligible optional feature names for this lead type.

    Returns
    -------
    set[str]
        The enabled optional features.
    """
    from smarthub.core import task_config

    key = f"{lead_type_name(lead_type_id)}_optional"
    raw = (task_config.get("features", key, _OPTIONAL_ALL) or "").strip()
    token = raw.lower()
    if token in ("", _OPTIONAL_ALL):
        return set(optional_universe)
    if token == _OPTIONAL_NONE:
        return set()

    requested = {c.strip() for c in raw.split(",") if c.strip()}
    unknown = requested - optional_universe - set(_MANDATORY_AUTO) - {"bid"}
    if unknown:
        logger.warning(
            "[features] %s lists unknown/ineligible feature(s) %s; ignoring them. "
            "Valid optional features: %s",
            key, sorted(unknown), sorted(optional_universe),
        )
    return requested & optional_universe


def model_feature_columns(
    lead_type_id: int,
    optional_enabled: set[str] | None = None,
) -> tuple[list[str], list[str]]:
    """Return ``(numeric, categorical)`` model feature names for a lead type.

    Auto-only features (vehicles/drivers/violations…) are dropped for home;
    home-only features (home_property_type, num_home_claims) are dropped for
    auto — so one code path trains the right model per lead type. MANDATORY
    features are always kept; OPTIONAL features are included only when enabled.
    The mandatory core can never be dropped here, whatever the config says.
    Training and serving both call this, so they stay identical.

    Inputs
    ------
    lead_type_id : int
        Lead type id to select features for.
    optional_enabled : set[str] | None
        Overrides the config lookup for enabled optional features (tests);
        ``None`` reads the config.

    Returns
    -------
    tuple[list[str], list[str]]
        The ``(numeric, categorical)`` model feature name lists.
    """
    if lead_type_id == LEAD_TYPE_AUTO:
        drop = HOME_ONLY_FEATURES
    elif lead_type_id == LEAD_TYPE_HOME:
        drop = AUTO_ONLY_FEATURES
    else:
        drop = set()
    base_numeric = [c for c in _MODEL_NUMERIC if c not in drop]
    base_categorical = [c for c in _MODEL_CATEGORICAL if c not in drop]

    mandatory = mandatory_features(lead_type_id)
    optional_universe = (set(base_numeric) | set(base_categorical)) - mandatory
    if optional_enabled is None:
        optional_enabled = _configured_optional(lead_type_id, optional_universe)

    keep = mandatory | (set(optional_enabled) & optional_universe)
    numeric = [c for c in base_numeric if c in keep]
    categorical = [c for c in base_categorical if c in keep]
    return numeric, categorical


def add_time_features(out: pd.DataFrame) -> list[str]:
    """Add ``created_hour`` / ``created_dayofweek`` in **Pacific** time.

    Kiran: don't use UTC — the call-centre operational day runs to ~5:30pm PT,
    so UTC would flip the date. Prefers the Pacific columns:
    ``created_dayofweek`` from ``pst_date`` and ``created_hour`` from
    ``pst_hour``. Falls back to ``created_at`` (UTC) if the Pacific columns are
    absent, and leaves values untouched for a live request that already carries
    them.

    Inputs
    ------
    out : pandas.DataFrame
        Frame mutated in place with the time feature columns.

    Returns
    -------
    list[str]
        Names of the time feature columns that were added.
    """
    added = []
    created = (
        pd.to_datetime(out["created_at"], errors="coerce")
        if "created_at" in out.columns
        else None
    )
    if created is not None:
        out["created_at"] = created

    # created_dayofweek: Pacific business day (pst_date) preferred.
    if "pst_date" in out.columns:
        out["created_dayofweek"] = pd.to_datetime(
            out["pst_date"], errors="coerce"
        ).dt.dayofweek
        added.append("created_dayofweek")
    elif created is not None:
        out["created_dayofweek"] = created.dt.dayofweek
        added.append("created_dayofweek")

    # created_hour: Pacific hour (pst_hour) preferred.
    if "pst_hour" in out.columns:
        out["created_hour"] = pd.to_numeric(out["pst_hour"], errors="coerce")
        added.append("created_hour")
    elif created is not None:
        out["created_hour"] = created.dt.hour
        added.append("created_hour")

    return added


def add_is_workday(out: pd.DataFrame) -> list[str]:
    """Add ``is_workday`` (1/0) from ``pst_date`` (else ``created_at``).

    Weekends (Sat/Sun) plus observed holidays (see ``smarthub.core.holidays``)
    are non-workdays. Uses ``pst_date`` — the Pacific business day — so day
    boundaries match how the marketplace runs. Added in place.

    Inputs
    ------
    out : pandas.DataFrame
        Frame mutated in place with the ``is_workday`` column.

    Returns
    -------
    list[str]
        ``["is_workday"]`` if a usable date column was present, else ``[]``.
    """
    from smarthub.core import holidays

    for col in ("pst_date", "created_at"):
        if col in out.columns:
            days = pd.to_datetime(out[col], errors="coerce")
            out["is_workday"] = days.map(
                lambda ts: int(holidays.is_workday(ts.date()))
                if pd.notna(ts)
                else 0
            ).astype("int64")
            return ["is_workday"]
    return []


def derive_serving_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the SAME engineered features used in training to a scoring frame.

    Train/serve parity hook: online prediction and the training build go
    through the same derivation (time parts, is_workday, is_married,
    multi_vehicle, age_missing, age_cohort_*), so a model never sees features
    computed a different way than it was trained on.

    Inputs
    ------
    df : pandas.DataFrame
        Scoring frame with raw ping-time columns.

    Returns
    -------
    pandas.DataFrame
        A new frame with engineered features added; callers then select
        ``model_feature_columns(...)``.
    """
    out = df.copy()
    add_time_features(out)
    add_is_workday(out)
    _derive_features(out)
    return out


def _derive_features(out: pd.DataFrame) -> list[str]:
    """Add engineered features in place; return the names that were added.

    Inputs
    ------
    out : pandas.DataFrame
        Frame mutated in place with ``is_married``, ``multi_vehicle``,
        ``age_missing``, the cleaned ``age`` and one-hot ``age_cohort_*``.

    Returns
    -------
    list[str]
        Names of the engineered columns that were added.
    """
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
        # Missingness as signal (Vinaya): flag null OR implausible/default age,
        # and fill the raw `age` with the -1 sentinel instead of mean-imputing.
        plausible = age.between(AGE_MIN, AGE_MAX)  # False for NaN / out-of-range
        out["age_missing"] = (~plausible).astype("int64")
        added.append("age_missing")
        age_clean = age.where(plausible)           # NaN where implausible
        out["age"] = age_clean.fillna(-1)
        cohort = pd.cut(
            age_clean,
            bins=AGE_COHORT_BINS,
            labels=AGE_COHORT_LABELS,
            right=False,
        )
        # One-hot: one 0/1 column per band. Missing/implausible age -> all 0.
        for label, col in zip(AGE_COHORT_LABELS, AGE_COHORT_COLUMNS):
            out[col] = cohort.eq(label).astype("int64")
            added.append(col)
    return added


def build_training_table(
    df: pd.DataFrame,
    lead_type_id: int | None = None,
    drop_zero_variance: bool = False,
) -> pd.DataFrame:
    """Assemble the leakage-safe training table from raw ``lead_pings`` rows.

    Target definition (matches the warehouse: ``won`` is only ever 'true' or
    NULL — losses are never written as 'false'):

    - A bid is **placed** when ``bid > 0``. A placed bid either **won**
      (``won == 'true'`` -> ``won_flag = 1``) or **lost** (``won`` null/blank/
      'false' -> ``won_flag = 0``).
    - Rows with **no bid placed** (``bid`` <= 0 or missing, and not a win) are
      not bidding decisions and are excluded.
    - Selects ping-time features + engineered features + bid + expected_revenue
      and target ``won_flag``. Nothing is dropped by default.
    - Engineered: ``is_married`` (from marital_status), ``multi_vehicle`` (from
      num_vehicles), and one-hot ``age_cohort_<band>`` 0/1 columns (banded from
      age).
    - When ``lead_type_id`` is given, the table is lead-type-clean: the auto
      table drops home-only columns and the home table drops auto-only columns.

    Inputs
    ------
    df : pandas.DataFrame
        Raw accumulated ``lead_pings`` rows.
    lead_type_id : int | None
        Optional lead type filter; ``None`` keeps all types.
    drop_zero_variance : bool
        Also drop single-valued feature columns when ``True``.

    Returns
    -------
    pandas.DataFrame
        The leakage-safe training table, index reset.
    """
    out = df.copy()

    # Drop errored pings entirely (Kiran: ignore rows where erred is true — they
    # aren't real auction outcomes). Keep all non-errored pings.
    if "erred" in out.columns:
        err = out["erred"].astype("string").str.strip().str.lower()
        out = out[~err.isin(["1", "true", "t", "yes", "y"])].copy()

    # Expected revenue R: prefer the backend `exp_rev` when populated (>0), else
    # fall back to the interim listings-sum. Auto-switches once exp_rev fills in.
    if "exp_rev" in out.columns:
        native = pd.to_numeric(out["exp_rev"], errors="coerce")
        if REVENUE_COLUMN in out.columns:
            listings = pd.to_numeric(out[REVENUE_COLUMN], errors="coerce")
        else:
            listings = pd.Series(float("nan"), index=out.index)
        out[REVENUE_COLUMN] = native.where(native > 0, listings)

    # Target: win = won=='true'; a placed bid is bid > 0. A placed bid with
    # won null/blank = LOST (Kiran). Keep placed bids (+ any win); label
    # win=1 / loss=0. Exclude no-bid pings.
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
    workday = add_is_workday(out)
    derived = _derive_features(out)

    feature_cols = [c for c in PRE_BID_FEATURES if c in out.columns]
    feature_cols += [c for c in TIME_FEATURES if c in out.columns]
    feature_cols += workday + derived

    # Keep the table lead-type-clean: drop the *other* type's exclusive columns
    # so the auto table has no home-only columns and vice-versa.
    if lead_type_id == LEAD_TYPE_AUTO:
        cross_type = HOME_ONLY_FEATURES
    elif lead_type_id == LEAD_TYPE_HOME:
        cross_type = AUTO_ONLY_FEATURES
    else:
        cross_type = set()
    feature_cols = [c for c in feature_cols if c not in cross_type]

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
