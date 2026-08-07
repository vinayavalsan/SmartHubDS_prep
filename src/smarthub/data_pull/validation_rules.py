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

import pandas as pd

from smarthub.core.lead_types import all_lead_types
from smarthub.feature_engineering.feature_registry import FEATURES

from . import field_registry

# Columns the pull is expected to produce (drives schema-drift detection).
EXPECTED_COLUMNS: tuple[str, ...] = tuple(field_registry.field_names())


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

    bid = pd.to_numeric(df["bid"], errors="coerce") if present("bid") else None
    won = _lower(df["won"]) if present("won") else None

    # won == true but no bid placed.
    if bid is not None and won is not None:
        out["won_true_without_bid"] = int(((won == "true") & (bid <= 0)).sum())

    # erred == true but a bid was placed (should have short-circuited).
    if present("erred") and bid is not None:
        erred = _lower(df["erred"]).isin({"true", "t", "1", "yes", "y"})
        out["erred_with_bid"] = int((erred & (bid > 0)).sum())

    # accepted == true but won is null/blank.
    if present("accepted") and won is not None:
        accepted = _lower(df["accepted"]).isin({"true", "t", "1", "yes", "y"})
        out["accepted_but_won_null"] = int((accepted & (won == "")).sum())

    # Lead-type completeness — driven by the feature registry. A feature is
    # required for a lead type when that name appears in ``mandatory_for``.
    # Raw features use their own name unless ``api_input`` declares a different
    # source column; derived features contribute their declared raw input.
    if present("lead_type_id"):
        lt = pd.to_numeric(df["lead_type_id"], errors="coerce")

        for lead_type_name, lead_type_id in all_lead_types().items():
            required_raw_columns = {
                spec.api_input or spec.name
                for spec in FEATURES.values()
                if lead_type_name in spec.mandatory_for
                and (spec.source == "raw" or spec.api_input is not None)
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

    if "erred" in df.columns:
        erred = _lower(df["erred"]).isin({"true", "t", "1", "yes", "y"})
        m["erred_rate"] = _rate(int(erred.sum()), n)
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
    issues = []
    missing = sorted(expected - have)
    extra = sorted(have - expected)
    if missing:
        issues.append(f"missing columns: {', '.join(missing)}")
    if extra:
        issues.append(f"unexpected columns: {', '.join(extra)}")
    return issues
