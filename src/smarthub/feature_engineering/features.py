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


def _specs_for_lead_type(lead_type: str | None = None) -> list[FeatureSpec]:
    """Return enabled, applicable feature specs in registry order."""
    specs = [spec for spec in FEATURES.values() if spec.enabled]
    if lead_type is None:
        return specs
    return [spec for spec in specs if lead_type in spec.lead_types]


def _lead_type_for_id(lead_type_id: int) -> str:
    """Resolve a registered lead-type id to its canonical name."""
    return lead_type_name(lead_type_id)


def registered_feature_names(lead_type: str | None = None) -> list[str]:
    """Return model-feature names in registry insertion order."""
    return [spec.name for spec in _specs_for_lead_type(lead_type)]


def model_feature_columns(
    lead_type_id: int,
) -> tuple[list[str], list[str]]:
    """Return enabled model columns for a lead type in registry order.

    ``binary`` features are returned with numeric features, matching the
    existing preprocessing behavior. Feature inclusion is controlled only by
    ``FeatureSpec.enabled`` and ``FeatureSpec.lead_types``.
    """
    lead_type = _lead_type_for_id(lead_type_id)
    specs = _specs_for_lead_type(lead_type)

    numeric = [spec.name for spec in specs if spec.kind in {"numeric", "binary"}]
    categorical = [spec.name for spec in specs if spec.kind == "categorical"]
    return numeric, categorical


def _apply_training_filters(
    frame: pd.DataFrame,
    lead_type: str | None = None,
) -> pd.DataFrame:
    """Apply registry-defined row filters used only for model training."""
    out = frame

    for spec in _specs_for_lead_type(lead_type):
        allowed = spec.training_include_values
        if not allowed or spec.name not in out.columns:
            continue

        values = out[spec.name]
        if all(isinstance(value, int) for value in allowed):
            values = pd.to_numeric(values, errors="coerce")

        out = out[values.isin(allowed)].copy()

    return out


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


def apply_registered_missing_values(
    frame: pd.DataFrame,
    lead_type: str | None = None,
) -> pd.DataFrame:
    """Replace missing model values using registry-defined sentinels.

    Categorical blanks and nulls become ``MISSING_CATEGORY``. Numeric and
    binary nulls become each feature's resolved numeric sentinel. The function
    mutates and returns ``frame`` for convenient pipeline composition.
    """
    for spec in _specs_for_lead_type(lead_type):
        if spec.name not in frame.columns:
            continue

        missing_value = spec.resolved_missing_value()
        values = frame[spec.name]
        if spec.kind == "categorical":
            values = values.astype("string").str.strip()
            values = values.mask(values.isna() | (values == ""))
            frame[spec.name] = values.fillna(str(missing_value))
        else:
            values = pd.to_numeric(values, errors="coerce")
            frame[spec.name] = values.fillna(missing_value)

    return frame


def derive_serving_features(
    df: pd.DataFrame,
    lead_type_id: int | None = None,
) -> pd.DataFrame:
    """Apply registry-defined feature engineering to a scoring frame."""
    out = df.copy()
    lead_type = _lead_type_for_id(lead_type_id) if lead_type_id is not None else None
    _apply_registered_derivations(out, lead_type)
    apply_registered_missing_values(out, lead_type)
    return out


def _unique_columns(columns: Iterable[str]) -> list[str]:
    """Return columns in first-seen order without duplicates."""
    return list(dict.fromkeys(columns))


def build_training_table(
    df: pd.DataFrame,
    lead_type_id: int | None = None,
    drop_zero_variance: bool = False,
    campaign_ids: list[int] | None = None,
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

    # Optional campaign scoping (config: feature_engineering.training_campaign_ids).
    # Empty/None keeps every campaign -- replaces the old hardcoded registry filter.
    if campaign_ids and "campaign_id" in out.columns:
        cid = pd.to_numeric(out["campaign_id"], errors="coerce")
        out = out[cid.isin(campaign_ids)].copy()

    lead_type = _lead_type_for_id(lead_type_id) if lead_type_id is not None else None
    out = _apply_training_filters(out, lead_type)
    _apply_registered_derivations(out, lead_type)
    apply_registered_missing_values(out, lead_type)

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
