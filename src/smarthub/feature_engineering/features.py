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

from smarthub.core import auction
from smarthub.core.lead_types import lead_type_name
from smarthub.feature_engineering.feature_registry import FEATURES, FeatureSpec

logger = logging.getLogger(__name__)

# Raw columns retained in the stored training table for traceability or future
# analysis. They are not model inputs and therefore do not belong in the model
# feature registry.
RETAINED_NON_MODEL_COLUMNS = [
    "lead_type_id",
    "account_id",
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


def _validate_column_contracts() -> None:
    """Fail fast when model and retained non-model column roles overlap."""
    overlap = sorted(set(RETAINED_NON_MODEL_COLUMNS) & set(FEATURES))
    if overlap:
        raise ValueError(
            "Columns cannot be both model features and retained non-model "
            f"columns: {overlap}"
        )


_validate_column_contracts()


def _specs_for_lead_type(lead_type: str | None = None) -> list[FeatureSpec]:
    """Return all applicable feature specs in registry order.

    This intentionally does not filter on ``enabled``. Row-level requirements
    such as ``mandatory`` and ``training_include_values`` are enforced before
    enabled model-feature selection.
    """
    specs = list(FEATURES.values())
    if lead_type is None:
        return specs
    return [spec for spec in specs if lead_type in spec.lead_types]


def _enabled_specs_for_lead_type(
    lead_type: str | None = None,
) -> list[FeatureSpec]:
    """Return enabled, applicable model-feature specs in registry order."""
    return [spec for spec in _specs_for_lead_type(lead_type) if spec.enabled]


def _lead_type_for_id(lead_type_id: int) -> str:
    """Resolve a registered lead-type id to its canonical name."""
    return lead_type_name(lead_type_id)


def registered_feature_names(lead_type: str | None = None) -> list[str]:
    """Return model-feature names in registry insertion order."""
    return [spec.name for spec in _enabled_specs_for_lead_type(lead_type)]


def model_feature_columns(
    lead_type_id: int,
) -> tuple[list[str], list[str]]:
    """Return enabled model columns for a lead type in registry order.

    ``binary`` features are returned with numeric features, matching the
    existing preprocessing behavior. Feature inclusion is controlled only by
    ``FeatureSpec.enabled`` and ``FeatureSpec.lead_types``.
    """
    lead_type = _lead_type_for_id(lead_type_id)
    specs = _enabled_specs_for_lead_type(lead_type)

    numeric = [spec.name for spec in specs if spec.kind in {"numeric", "binary"}]
    categorical = [spec.name for spec in specs if spec.kind == "categorical"]
    return numeric, categorical


def _missing_mask(values: pd.Series) -> pd.Series:
    """Return rows whose value is null, empty, or whitespace-only."""
    if pd.api.types.is_string_dtype(values.dtype) or values.dtype == object:
        normalized = values.astype("string").str.strip()
        return normalized.isna() | normalized.eq("")
    return values.isna()


def _log_drop(
    lead_type: str | None,
    reason: str,
    before: int,
    after: int,
    *,
    examples: list[object] | None = None,
) -> None:
    """Log rows removed by one feature-engineering filter."""
    dropped = before - after
    if dropped <= 0:
        return

    prefix = f"[{lead_type}] " if lead_type else ""
    message = f"{prefix}{reason}: dropped {dropped:,} row(s); " f"{after:,} remaining"
    if examples:
        message += f"; examples={examples}"
    logger.info(message)


def _log_transformation(
    lead_type: str | None,
    message: str,
    *,
    n_rows: int | None = None,
) -> None:
    """Log one feature-engineering transformation.

    Serving processes a single lead per request, so on that hot path these
    per-feature messages are demoted to DEBUG to avoid flooding the logs (and
    adding logging overhead) on every ``/recommend_bid`` call. Batch/training
    runs (many rows) keep logging at INFO, where the per-feature summaries are
    useful. ``n_rows`` is the number of rows being transformed (``1`` = serving).
    """
    prefix = f"[{lead_type}] " if lead_type else ""
    log = logger.debug if n_rows == 1 else logger.info
    log("%s%s", prefix, message)


def _apply_training_filters(
    frame: pd.DataFrame,
    lead_type: str | None = None,
) -> pd.DataFrame:
    """Apply mandatory and registry-defined training row filters.

    Every actual row-removal operation is logged with its reason and the
    remaining row count. Mandatory checks apply to all applicable registry
    entries, including disabled ones. Missing non-mandatory features survive
    for later sentinel replacement.
    """
    out = frame

    for spec in _specs_for_lead_type(lead_type):
        required_column = (
            spec.api_input
            if spec.source == "derived" and spec.api_input is not None
            else spec.name
        )

        if spec.mandatory and required_column in out.columns:
            before = len(out)
            missing = _missing_mask(out[required_column])
            out = out[~missing].copy()
            _log_drop(
                lead_type,
                f"mandatory feature '{required_column}' missing/blank",
                before,
                len(out),
            )

        allowed = spec.training_include_values
        if not allowed or spec.name not in out.columns:
            continue

        raw_values = out[spec.name]
        missing = _missing_mask(raw_values)

        values = raw_values
        if all(isinstance(value, int) for value in allowed):
            values = pd.to_numeric(values, errors="coerce")

        keep = values.isin(allowed)
        if not spec.mandatory:
            keep = keep | missing

        before = len(out)
        dropped_values = raw_values.loc[~keep]
        examples = (
            dropped_values.astype("string")
            .str.strip()
            .replace("", pd.NA)
            .dropna()
            .drop_duplicates()
            .tolist()[:5]
        )
        out = out[keep].copy()
        _log_drop(
            lead_type,
            f"feature '{spec.name}' outside training_include_values",
            before,
            len(out),
            examples=examples,
        )

    return out


def _apply_registered_derivations(
    frame: pd.DataFrame,
    lead_type: str | None = None,
) -> list[str]:
    """Apply and log applicable registry-defined derivations."""
    derived: list[str] = []

    for spec in _enabled_specs_for_lead_type(lead_type):
        if spec.source != "derived" or spec.derive is None:
            continue

        age_missing_count = 0
        age_implausible_count = 0
        age_implausible_examples: list[object] = []
        if spec.name == "age_cohort" and "age" in frame.columns:
            raw_age = pd.to_numeric(frame["age"], errors="coerce")
            raw_missing = _missing_mask(frame["age"]) | raw_age.isna()
            implausible = raw_age.notna() & ~raw_age.between(1, 130)
            age_missing_count = int(raw_missing.sum())
            age_implausible_count = int(implausible.sum())
            age_implausible_examples = (
                frame.loc[implausible, "age"].dropna().drop_duplicates().tolist()[:5]
            )

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

        if age_implausible_count:
            message = (
                f"feature 'age': replaced {age_implausible_count:,} "
                "implausible value(s) outside [1, 130] with -1"
            )
            if age_implausible_examples:
                message += f"; examples={age_implausible_examples}"
            _log_transformation(lead_type, message, n_rows=len(frame))

        if age_missing_count:
            _log_transformation(
                lead_type,
                f"feature 'age': replaced {age_missing_count:,} "
                "missing/non-numeric value(s) with -1",
                n_rows=len(frame),
            )

        source = spec.api_input or "registered inputs"
        _log_transformation(
            lead_type,
            f"derived feature '{spec.name}' from '{source}'; {len(frame):,} row(s)",
            n_rows=len(frame),
        )
        derived.append(spec.name)

    return derived


def apply_registered_missing_values(
    frame: pd.DataFrame,
    lead_type: str | None = None,
) -> pd.DataFrame:
    """Replace and log missing model values using registry-defined sentinels."""
    for spec in _enabled_specs_for_lead_type(lead_type):
        if spec.name not in frame.columns:
            continue

        missing_value = spec.resolved_missing_value()
        values = frame[spec.name]

        if spec.kind == "categorical":
            normalized = values.astype("string").str.strip()
            missing = normalized.isna() | normalized.eq("")
            affected = int(missing.sum())
            normalized = normalized.mask(missing)
            frame[spec.name] = normalized.fillna(str(missing_value))
        else:
            numeric = pd.to_numeric(values, errors="coerce")
            missing = numeric.isna()
            affected = int(missing.sum())
            frame[spec.name] = numeric.fillna(missing_value)

        if affected:
            _log_transformation(
                lead_type,
                f"feature '{spec.name}': replaced {affected:,} "
                f"missing/blank value(s) with {missing_value!r}",
                n_rows=len(frame),
            )

    return frame


def transform_model_features(
    df: pd.DataFrame,
    lead_type_id: int | None = None,
) -> pd.DataFrame:
    """Apply the shared model-input transformations for training and serving.

    This is the single transformation path for model inputs. It applies
    registry-defined derivations, ensures every enabled feature column exists,
    and applies the registry-defined type/missing-value normalization.

    Training-only row filtering and target construction intentionally happen
    outside this function.

    Inputs
    ------
    df : pandas.DataFrame
        Raw or partially engineered rows.
    lead_type_id : int | None
        Lead type used to scope registry features.

    Returns
    -------
    pandas.DataFrame
        Copy containing consistently transformed model features.
    """
    out = df.copy()
    lead_type = _lead_type_for_id(lead_type_id) if lead_type_id is not None else None

    _apply_registered_derivations(out, lead_type)

    # Training and serving must see the same feature schema. If an enabled
    # feature is absent, create it as missing and let the registry-defined
    # missing-value handling assign the correct sentinel.
    for spec in _enabled_specs_for_lead_type(lead_type):
        if spec.name not in out.columns:
            out[spec.name] = pd.NA

    apply_registered_missing_values(out, lead_type)
    return out


def derive_serving_features(
    df: pd.DataFrame,
    lead_type_id: int | None = None,
) -> pd.DataFrame:
    """Apply the shared model-input transformations to a scoring frame.

    Kept as a compatibility wrapper; training and serving now share
    ``transform_model_features``.
    """
    return transform_model_features(df, lead_type_id=lead_type_id)


def _unique_columns(columns: Iterable[str]) -> list[str]:
    """Return columns in first-seen order without duplicates."""
    return list(dict.fromkeys(columns))


def build_training_table(
    df: pd.DataFrame,
    lead_type_id: int | None = None,
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
    lead_type = _lead_type_for_id(lead_type_id) if lead_type_id is not None else None

    if lead_type_id is not None and "lead_type_id" in out.columns:
        before = len(out)
        out = out[out["lead_type_id"] == lead_type_id].copy()
        logger.info(
            "[%s] selected %s rows for lead_type_id=%s from %s stored rows",
            lead_type,
            f"{len(out):,}",
            lead_type_id,
            f"{before:,}",
        )

    input_rows = len(out)
    if lead_type:
        logger.info(
            "[%s] starting feature engineering: %s rows",
            lead_type,
            f"{input_rows:,}",
        )
    else:
        logger.info("starting feature engineering: %s rows", f"{input_rows:,}")

    # Errored pings are not valid auction outcomes.
    before = len(out)
    out = out.loc[~auction.erred_mask(out)].copy()
    _log_drop(
        lead_type,
        "errored ping filter",
        before,
        len(out),
    )

    # Prefer the backend expected revenue when populated; otherwise retain the
    # existing expected_revenue value.
    if "exp_rev" in out.columns:
        native = pd.to_numeric(out["exp_rev"], errors="coerce")
        if REVENUE_COLUMN in out.columns:
            fallback = pd.to_numeric(out[REVENUE_COLUMN], errors="coerce")
        else:
            fallback = pd.Series(float("nan"), index=out.index)
        out[REVENUE_COLUMN] = native.where(native > 0, fallback)

    won_true = auction.won_true_mask(out)
    keep_rows = auction.auction_eligible_mask(out, bid_column=DECISION_COLUMN)

    before = len(out)
    out = out.loc[keep_rows].copy()
    _log_drop(
        lead_type,
        "invalid auction/bid filter",
        before,
        len(out),
    )
    out[TARGET_COLUMN] = won_true.loc[keep_rows].astype("int64").to_numpy()

    # Optional campaign scoping (config: feature_engineering.training_campaign_ids).
    # Empty/None keeps every campaign.
    if campaign_ids and "campaign_id" in out.columns:
        before = len(out)
        cid = pd.to_numeric(out["campaign_id"], errors="coerce")
        out = out[cid.isin(campaign_ids)].copy()
        _log_drop(lead_type, "campaign scope filter", before, len(out))

    out = _apply_training_filters(out, lead_type)

    # All transformations that change model inputs go through the same shared
    # path used by online prediction.
    out = transform_model_features(out, lead_type_id=lead_type_id)

    model_columns = [
        spec.name
        for spec in _enabled_specs_for_lead_type(lead_type)
        if spec.name in out.columns
    ]
    retained_columns = [
        column for column in RETAINED_NON_MODEL_COLUMNS if column in out.columns
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

    output_rows = len(table)
    dropped_rows = input_rows - output_rows
    dropped_pct = (dropped_rows / input_rows) if input_rows else 0.0
    prefix = f"[{lead_type}] " if lead_type else ""
    logger.info(
        "%sfeature-engineering row filtering summary: "
        "input=%s, dropped=%s (%.2f%%), output=%s",
        prefix,
        f"{input_rows:,}",
        f"{dropped_rows:,}",
        dropped_pct * 100,
        f"{output_rows:,}",
    )

    return table.reset_index(drop=True)
