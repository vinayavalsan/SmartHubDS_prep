"""Validation rules: expected schema, per-column domains/ranges, cross-field.

Two layers:
- ``leads_schema()`` builds a **pandera** DataFrameSchema for the per-column
  declarative rules (dtype coercion, numeric ranges, categorical domains,
  ``id`` uniqueness). pandera is imported lazily so this module still imports
  (and the pure-pandas checks below still run) if pandera isn't installed.
- The pure-pandas functions (``missing_rates``, ``cross_field_checks``,
  ``batch_metrics``) cover what pandera doesn't do cleanly: null/blank
  cataloguing, cross-field integrity, and batch quality metrics.

Everything here is detect-only; nothing mutates the data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from numbers import Number

import numpy as np
import pandas as pd

from smarthub.core import auction
from smarthub.core.lead_types import all_lead_types

from . import field_registry

# Registered raw fields must always be present in the pull. The enriched
# expected-revenue query also adds a small set of intentional derived/joined
# columns; those are allowed when present but are not required because callers
# may explicitly disable the expected-revenue join.
EXPECTED_COLUMNS: tuple[str, ...] = tuple(field_registry.field_names())
ALLOWED_DERIVED_COLUMNS: frozenset[str] = frozenset(
    {
        "expected_revenue",
        "realized_payout",
        "num_selected_listings",
    }
)


def leads_schema():
    """Build the pandera schema from the raw-field registry (lazy pandera import).

    The per-column declarative rules are derived from each field's
    ``ValidationSpec`` in ``field_registry`` — the single source of truth — so
    ranges/domains/uniqueness are no longer duplicated here:

    - numeric with min **and** max -> ``in_range``; min only -> ``ge`` (both
      coerce string numerics to float);
    - categorical with ``allowed_values`` -> ``isin``;
    - ``unique`` fields (e.g. ``id``) -> unique + not-null.

    Fields whose only check is a ``custom_rule`` (e.g. ``state``) are handled by
    ``validation_runner`` instead, not pandera. ``required=False`` everywhere (a
    missing column is caught by schema-drift, not here); ``strict=False`` so
    extra columns don't error.

    Returns
    -------
    pandera.pandas.DataFrameSchema
        Schema enforcing dtypes, numeric ranges, categorical domains, and
        ``id`` uniqueness, all sourced from the registry.
    """
    from pandera.pandas import Check, Column, DataFrameSchema

    cols: dict = {}
    for name, spec in field_registry.RAW_FIELD_REGISTRY.items():
        v = spec.validation
        checks: list = []
        dtype = None
        coerce = False
        unique = bool(v.unique)
        nullable = not unique  # id: unique + not-null; everything else nullable

        if v.kind == "numeric":
            if v.min_value is not None and v.max_value is not None:
                dtype, coerce = float, True
                checks.append(Check.in_range(v.min_value, v.max_value))
            elif v.min_value is not None:
                dtype, coerce = float, True
                checks.append(Check.ge(v.min_value))
        elif v.kind == "categorical" and v.allowed_values:
            dtype = str
            checks.append(Check.isin(v.allowed_values))

        if checks or unique:
            cols[name] = Column(
                dtype,
                checks or None,
                nullable=nullable,
                required=False,
                unique=unique,
                coerce=coerce,
            )
    return DataFrameSchema(cols, strict=False, coerce=False)


# --- Pure-pandas helpers ----------------------------------------------------


def _null_or_blank(series: pd.Series) -> pd.Series:
    """Boolean mask: NaN/None, or (for text) empty/whitespace-only strings."""
    if series.dtype == object or pd.api.types.is_string_dtype(series):
        stripped = series.astype("string").str.strip()
        return stripped.isna() | (stripped == "")
    return series.isna()


def _lower(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().str.lower()


_BINARY_VALUES = frozenset({"true", "false"})


@dataclass(frozen=True)
class KindValidationResult:
    """Result of validating one raw column against its declared kind."""

    check: str
    count: int
    examples: list = field(default_factory=list)


def _invalid_numeric_mask(series: pd.Series) -> pd.Series:
    """Return present values that cannot be interpreted as numeric."""
    present = ~_null_or_blank(series)
    parsed = pd.to_numeric(series, errors="coerce")
    invalid_bool = series.map(
        lambda value: isinstance(value, (bool, np.bool_)) if pd.notna(value) else False
    )
    return present & (parsed.isna() | invalid_bool)


def _invalid_categorical_mask(series: pd.Series) -> pd.Series:
    """Return present categorical values that are not strings."""
    present = ~_null_or_blank(series)
    is_string = series.map(
        lambda value: isinstance(value, str) if pd.notna(value) else False
    )
    return present & ~is_string


def _invalid_binary_mask(series: pd.Series) -> pd.Series:
    """Return present values outside the canonical warehouse boolean domain."""
    present = ~_null_or_blank(series)
    normalized = _lower(series)
    return present & ~normalized.isin(_BINARY_VALUES)


def _is_datetime_like(value) -> bool:
    """Return whether one raw value is a supported datetime representation."""
    if pd.isna(value):
        return True
    if isinstance(value, (bool, np.bool_)):
        return False
    if isinstance(value, Number):
        return False
    if isinstance(value, (datetime, date, pd.Timestamp, np.datetime64)):
        return True
    if not isinstance(value, str):
        return False
    stripped = value.strip()
    if not stripped:
        return True
    return not pd.isna(pd.to_datetime(stripped, errors="coerce"))


def _invalid_datetime_mask(series: pd.Series) -> pd.Series:
    """Return present values that cannot be interpreted as datetimes."""
    present = ~_null_or_blank(series)
    valid = series.map(_is_datetime_like)
    return present & ~valid


_KIND_VALIDATORS = {
    "numeric": _invalid_numeric_mask,
    "categorical": _invalid_categorical_mask,
    "binary": _invalid_binary_mask,
    "datetime": _invalid_datetime_mask,
}


def validate_kind(series: pd.Series, kind: str) -> KindValidationResult:
    """Validate present raw values against a registry-declared field kind.

    Missing and blank values are ignored here because missingness is reported
    separately. This check is intentionally pure pandas so fundamental type
    validation still runs when optional Pandera validation is unavailable.

    Inputs
    ------
    series : pandas.Series
        Raw values exactly as returned by the warehouse query.
    kind : str
        Registry kind: numeric, categorical, binary, or datetime.

    Returns
    -------
    KindValidationResult
        Stable check label, violating-row count, and up to five examples.

    Raises
    ------
    ValueError
        If ``kind`` is not one of the supported registry kinds.
    """
    validator = _KIND_VALIDATORS.get(kind)
    if validator is None:
        raise ValueError(f"Unsupported validation kind: {kind!r}.")

    invalid = validator(series)
    examples = series.loc[invalid].dropna().unique().tolist()[:5]
    return KindValidationResult(
        check=f"invalid {kind} value",
        count=int(invalid.sum()),
        examples=examples,
    )


def erred_mask(df: pd.DataFrame) -> pd.Series:
    """Return the shared SmartHub errored-row mask."""
    return auction.erred_mask(df)


def erred_cross_field_checks(df: pd.DataFrame) -> dict[str, int]:
    """Run integrity checks that specifically require errored rows."""
    if df.empty:
        return {}

    out: dict[str, int] = {}
    if "erred" in df.columns and "bid" in df.columns:
        erred = auction.erred_mask(df)
        bid = pd.to_numeric(df["bid"], errors="coerce")
        out["erred_with_bid"] = int((erred & (bid > 0)).sum())
    return out


def auction_eligible_mask(df: pd.DataFrame) -> pd.Series:
    """Return the shared SmartHub auction-eligibility mask."""
    return auction.auction_eligible_mask(df)


def auction_cross_field_checks(df: pd.DataFrame) -> dict[str, int]:
    """Run integrity checks that require rows before auction filtering."""
    if df.empty:
        return {}

    out: dict[str, int] = {}
    if "bid" in df.columns and "won" in df.columns:
        won_true = auction.won_true_mask(df)
        placed_bid = auction.placed_bid_mask(df)
        out["won_true_without_bid"] = int((won_true & ~placed_bid).sum())
    return out


def missing_rates(df: pd.DataFrame) -> dict[str, float]:
    """Compute the null/blank rate per column.

    Inputs
    ------
    df : pd.DataFrame
        Raw batch to profile.

    Returns
    -------
    dict[str, float]
        Column name to missing rate in ``0..1``.
    """
    n = len(df)
    if not n:
        return {c: 0.0 for c in df.columns}
    return {c: float(_null_or_blank(df[c]).mean()) for c in df.columns}


def constant_columns(df: pd.DataFrame) -> dict[str, dict]:
    """Profile columns with zero or one distinct non-missing value.

    Missing values include nulls and blank/whitespace-only strings. A column
    is reported when it is entirely missing or when all populated rows contain
    the same value.

    Returns
    -------
    dict[str, dict]
        Column name to ``kind``, ``value``, and ``missing_rate``.
    """
    n = len(df)
    out: dict[str, dict] = {}
    if not n:
        return out

    for column in df.columns:
        missing = _null_or_blank(df[column])
        populated = df.loc[~missing, column]
        distinct = populated.nunique(dropna=True)

        if distinct > 1:
            continue

        value = None
        if distinct == 1:
            value = populated.iloc[0]

        out[column] = {
            "kind": "all_missing" if distinct == 0 else "single_value",
            "value": value,
            "missing_rate": float(missing.mean()),
        }

    return out


def _completeness_rule_name(type_name: str, col: str) -> str:
    """Rule key for a per-type required-raw-column check.

    ``{type}_missing_{col}``, dropping a leading ``{type}_`` on the column so
    e.g. (home, home_property_type) -> ``home_missing_property_type`` rather
    than the doubled ``home_missing_home_property_type``.
    """
    prefix = f"{type_name}_"
    short = col[len(prefix) :] if col.startswith(prefix) else col
    return f"{type_name}_missing_{short}"


def cross_field_checks(df: pd.DataFrame) -> dict[str, int]:
    """Count rows violating cross-field integrity rules. Detect-only.

    Inputs
    ------
    df : pd.DataFrame
        Raw batch to check.

    Returns
    -------
    dict[str, int]
        Rule name to the number of rows violating it.
    """
    n = len(df)
    out: dict[str, int] = {}
    if not n:
        return out

    def present(col):
        return col in df.columns

    # current_carrier populated while insured is false (Kiran: bad data).
    if present("current_carrier") and present("insured"):
        cc = ~_null_or_blank(df["current_carrier"])
        not_insured = _lower(df["insured"]).isin({"false", "f", "0", "no", "n"})
        out["current_carrier_when_not_insured"] = int((cc & not_insured).sum())

    won = _lower(df["won"]) if present("won") else None

    # accepted == true but won is null/blank.
    if present("accepted") and won is not None:
        accepted = _lower(df["accepted"]).isin({"true", "t", "1", "yes", "y"})
        out["accepted_but_won_null"] = int((accepted & (won == "")).sum())

    # num_vehicles and multi_vehicle must agree when both are present.
    # Missing values are handled separately by the missingness rules.
    if present("num_vehicles") and present("multi_vehicle"):
        num_vehicles = pd.to_numeric(df["num_vehicles"], errors="coerce")
        multi_vehicle = _lower(df["multi_vehicle"])

        known = num_vehicles.notna() & ~_null_or_blank(df["multi_vehicle"])
        expected_multi = num_vehicles.gt(1)
        actual_multi = multi_vehicle.isin({"true", "t", "1", "yes", "y"})

        out["multi_vehicle_inconsistent_with_num_vehicles"] = int(
            (known & actual_multi.ne(expected_multi)).sum()
        )

    # Lead-type completeness belongs to the raw-data validation layer, not
    # the model feature registry. Only fields configured to
    # flag missing values are checked row-by-row for
    # the lead types they apply to.
    if present("lead_type_id"):
        lt = pd.to_numeric(df["lead_type_id"], errors="coerce")

        for lead_type_name in all_lead_types():
            lead_type_id = all_lead_types()[lead_type_name]
            required_raw_columns = {
                spec.name
                for spec in field_registry.fields_for_lead_type(lead_type_name)
                if spec.validation.flag_missing
            }

            for col in sorted(required_raw_columns):
                if not present(col):
                    continue
                miss = _null_or_blank(df[col])
                out[_completeness_rule_name(lead_type_name, col)] = int(
                    ((lt == lead_type_id) & miss).sum()
                )

    return out


def _rate(mask_sum: int, n: int) -> float:
    return float(mask_sum / n) if n else 0.0


def batch_metrics(df: pd.DataFrame) -> dict:
    """Compute headline quality metrics (rates/counts) for the batch.

    Inputs
    ------
    df : pd.DataFrame
        Raw batch to summarize.

    Returns
    -------
    dict
        Metric name to value, always including ``rows``.
    """
    n = len(df)
    m: dict = {"rows": n}
    if not n:
        return m

    if "bid" in df.columns:
        bid = pd.to_numeric(df["bid"], errors="coerce")
        m["bid_zero_rate"] = _rate(int((bid <= 0).sum()), n)
    if "exp_rev" in df.columns:
        er = pd.to_numeric(df["exp_rev"], errors="coerce")
        m["exp_rev_coverage"] = _rate(int((er > 0).sum()), n)
    if "pst_hour" in df.columns:
        m["pst_hour_populated"] = _rate(int((~_null_or_blank(df["pst_hour"])).sum()), n)
    if "age" in df.columns:
        age = pd.to_numeric(df["age"], errors="coerce")
        age_v = field_registry.get("age").validation
        lo, hi = age_v.min_value, age_v.max_value
        implausible = age.notna() & ~age.between(lo, hi)
        m["age_implausible_rate"] = _rate(int(implausible.sum()), n)
    if "won" in df.columns:
        won = _lower(df["won"])
        m["won_false_count"] = int((won == "false").sum())  # should be 0
    if "traffic_tier" in df.columns:
        m["traffic_tier_distinct"] = int(df["traffic_tier"].nunique())
    return m


def schema_drift(df: pd.DataFrame) -> list[str]:
    """Describe schema differences vs the expected pulled columns.

    Inputs
    ------
    df : pd.DataFrame
        Raw batch whose columns are compared against ``EXPECTED_COLUMNS``.

    Returns
    -------
    list[str]
        Human-readable lines for missing and unexpected columns (empty if
        none).
    """
    have = set(df.columns)
    expected = set(EXPECTED_COLUMNS)
    allowed = expected | set(ALLOWED_DERIVED_COLUMNS)
    issues = []
    missing = sorted(expected - have)
    extra = sorted(have - allowed)
    if missing:
        issues.append(f"missing columns: {', '.join(missing)}")
    if extra:
        issues.append(f"unexpected columns: {', '.join(extra)}")
    return issues
