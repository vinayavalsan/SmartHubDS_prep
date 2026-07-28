"""Feature extraction and leakage-safe training-table construction.

``feature_registry.py`` is the single source of truth for model-feature
membership, ordering, type, lead-type applicability, mandatory status, raw or
derived status, API input, and derived-feature implementation.

This module applies that registry while building training and serving frames.
It does not maintain separate model-feature lists.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable

import pandas as pd

from smarthub.core.lead_types import lead_type_name
from smarthub.feature_engineering.feature_registry import FEATURES, FeatureSpec

logger = logging.getLogger(__name__)

# Raw columns retained in the stored training table for traceability or future
# analysis. They are not model inputs and therefore do not belong in the model
# feature registry.
RETAINED_NON_MODEL_COLUMNS = [
    "zip",
    "city",
    "num_auto_claims",
    "current_carrier",
    "credit",
    "household_income",
    "pnc_bundle",
    "device_type",
    "lead_type_id",
    "account_id",
    "source_type_id",
    "total_listings",
    "num_selected_listings",
]

REVENUE_COLUMN = "expected_revenue"
DECISION_COLUMN = "bid"
TARGET_COLUMN = "won_flag"
META_COLUMNS = ["id", "created_at"]

# Post-bid outcome columns are intentionally excluded from model inputs.
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
_OPTIONAL_ALL = "all"
_OPTIONAL_NONE = "none"


def _specs_for_lead_type(lead_type: str | None = None) -> list[FeatureSpec]:
    """Return applicable feature specs in registry insertion order."""
    if lead_type is None:
        return list(FEATURES.values())
    return [spec for spec in FEATURES.values() if lead_type in spec.lead_types]


def _lead_type_for_id(lead_type_id: int) -> str:
    """Resolve a registered lead-type id to its canonical name."""
    return lead_type_name(lead_type_id)


def registered_feature_names(lead_type: str | None = None) -> list[str]:
    """Return model-feature names in registry insertion order."""
    return [spec.name for spec in _specs_for_lead_type(lead_type)]


def mandatory_features(lead_type_id: int) -> set[str]:
    """Return model features that cannot be disabled for a lead type."""
    name = _lead_type_for_id(lead_type_id)
    return {
        spec.name for spec in _specs_for_lead_type(name) if name in spec.mandatory_for
    }


def optional_features(lead_type_id: int) -> set[str]:
    """Return toggleable model features for a lead type."""
    name = _lead_type_for_id(lead_type_id)
    return {spec.name for spec in _specs_for_lead_type(name)} - mandatory_features(
        lead_type_id
    )


def _configured_optional(lead_type_id: int, optional_universe: set[str]) -> set[str]:
    """Read enabled optional features for a lead type from runtime config."""
    from smarthub.core import task_config

    lead_type = _lead_type_for_id(lead_type_id)
    key = f"{lead_type}_optional"
    raw = (task_config.get("features", key, _OPTIONAL_ALL) or "").strip()
    token = raw.lower()

    if token in ("", _OPTIONAL_ALL):
        return set(optional_universe)
    if token == _OPTIONAL_NONE:
        return set()

    requested = {column.strip() for column in raw.split(",") if column.strip()}
    unknown = requested - optional_universe - mandatory_features(lead_type_id)
    if unknown:
        logger.warning(
            "features.%s lists unknown/ineligible feature(s) %s; ignoring them. "
            "Valid optional features: %s",
            key,
            sorted(unknown),
            sorted(optional_universe),
        )
    return requested & optional_universe


def model_feature_columns(
    lead_type_id: int,
    optional_enabled: set[str] | None = None,
) -> tuple[list[str], list[str]]:
    """Return numeric and categorical model columns in registry order.

    ``binary`` features are returned with the numeric features, matching the
    existing preprocessing behavior.
    """
    lead_type = _lead_type_for_id(lead_type_id)
    specs = _specs_for_lead_type(lead_type)
    mandatory = mandatory_features(lead_type_id)
    optional_universe = {spec.name for spec in specs} - mandatory

    if optional_enabled is None:
        optional_enabled = _configured_optional(lead_type_id, optional_universe)

    keep = mandatory | (set(optional_enabled) & optional_universe)
    numeric = [
        spec.name
        for spec in specs
        if spec.name in keep and spec.kind in {"numeric", "binary"}
    ]
    categorical = [
        spec.name for spec in specs if spec.name in keep and spec.kind == "categorical"
    ]
    return numeric, categorical


def _apply_registered_derivations(
    frame: pd.DataFrame,
    lead_type: str | None = None,
) -> list[str]:
    """Apply applicable registry-defined derivations in registry order.

    A derivation is skipped when its required raw input is absent. This keeps
    the same function usable for stored training rows and live scoring frames,
    where a derived value may already be present.
    """
    derived: list[str] = []

    for spec in _specs_for_lead_type(lead_type):
        if spec.source != "derived" or spec.derive is None:
            continue

        try:
            frame[spec.name] = spec.derive(frame)
        except KeyError as exc:
            missing = exc.args[0] if exc.args else spec.api_input
            if spec.name not in frame.columns:
                logger.debug(
                    "Skipping derived feature %s because input %r is absent.",
                    spec.name,
                    missing,
                )
            continue

        derived.append(spec.name)

    return derived


def derive_serving_features(
    df: pd.DataFrame,
    lead_type_id: int | None = None,
) -> pd.DataFrame:
    """Apply registry-defined feature engineering to a scoring frame."""
    out = df.copy()
    lead_type = _lead_type_for_id(lead_type_id) if lead_type_id is not None else None
    _apply_registered_derivations(out, lead_type)
    return out


def _unique_columns(columns: Iterable[str]) -> list[str]:
    """Return columns in first-seen order without duplicates."""
    return list(dict.fromkeys(columns))


def build_training_table(
    df: pd.DataFrame,
    lead_type_id: int | None = None,
    drop_zero_variance: bool = False,
) -> pd.DataFrame:
    """Assemble the leakage-safe training table from raw ``lead_pings`` rows.

    A placed bid is ``bid > 0``. Rows with a placed bid are labeled as wins
    when ``won == 'true'`` and losses otherwise. Rows without a placed bid are
    excluded unless the warehouse marks them as wins.

    Model-feature selection and all derived-feature execution come from
    ``FEATURES``. When ``lead_type_id`` is supplied, only registry features
    applicable to that lead type are retained.
    """
    out = df.copy()

    # Errored pings are not valid auction outcomes.
    if "erred" in out.columns:
        err = out["erred"].astype("string").str.strip().str.lower()
        out = out[~err.isin(["1", "true", "t", "yes", "y"])].copy()

    # Prefer the backend expected revenue when populated; otherwise retain the
    # existing expected_revenue value.
    if "exp_rev" in out.columns:
        native = pd.to_numeric(out["exp_rev"], errors="coerce")
        if REVENUE_COLUMN in out.columns:
            fallback = pd.to_numeric(out[REVENUE_COLUMN], errors="coerce")
        else:
            fallback = pd.Series(float("nan"), index=out.index)
        out[REVENUE_COLUMN] = native.where(native > 0, fallback)

    won_true = out["won"].astype("string").str.strip().str.lower().eq(_TRUE)
    bid_default = pd.Series(float("nan"), index=out.index, dtype="float64")
    bid_num = pd.to_numeric(out.get(DECISION_COLUMN, bid_default), errors="coerce")
    placed = bid_num.gt(0).fillna(False)
    keep_rows = placed | won_true.fillna(False)

    out = out[keep_rows].copy()
    out[TARGET_COLUMN] = won_true[keep_rows].fillna(False).astype("int64").to_numpy()

    if lead_type_id is not None and "lead_type_id" in out.columns:
        out = out[out["lead_type_id"] == lead_type_id].copy()

    lead_type = _lead_type_for_id(lead_type_id) if lead_type_id is not None else None
    _apply_registered_derivations(out, lead_type)

    model_columns = [
        spec.name
        for spec in _specs_for_lead_type(lead_type)
        if spec.name in out.columns
    ]
    retained_columns = [
        column
        for column in RETAINED_NON_MODEL_COLUMNS
        if column in out.columns and column not in model_columns
    ]

    keep = _unique_columns(
        [column for column in META_COLUMNS if column in out.columns]
        + model_columns
        + retained_columns
        + [
            column
            for column in (DECISION_COLUMN, REVENUE_COLUMN)
            if column in out.columns
        ]
        + [TARGET_COLUMN]
    )
    table = out[keep].copy()

    if drop_zero_variance:
        constant = [
            column
            for column in model_columns
            if column != DECISION_COLUMN
            and column in table.columns
            and table[column].nunique(dropna=True) <= 1
        ]
        table = table.drop(columns=constant)

    return table.reset_index(drop=True)
