"""Orchestrate the leads validation and assemble a ``ValidationReport``.

Warn + report only: ``validate_leads`` never raises and never mutates ``df`` —
it returns a structured report the caller turns into a Prefect artifact / Slack
summary / log line.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd

from . import field_registry
from . import validation_rules as rules

logger = logging.getLogger(__name__)

# Default null/blank rate at/above which a column is called out.
DEFAULT_HIGH_MISSING_THRESHOLD = 0.5


@dataclass
class RuleViolation:
    """One rule failure: column, check, affected row count, and examples."""

    column: str
    check: str
    count: int
    examples: list = field(default_factory=list)


@dataclass
class ValidationReport:
    """Structured result of validating a leads batch (warn + report only)."""

    total_rows: int
    schema_issues: list[str]
    rule_violations: list[RuleViolation]
    cross_field: dict[str, int]
    missing: dict[str, float]
    high_missing: list[str]
    constant_columns: dict[str, dict]
    metrics: dict
    schema_checked: bool  # False if pandera wasn't available
    erred_rows: int = 0
    auction_excluded_rows: int = 0
    validated_rows: int = 0

    @property
    def cross_field_hits(self) -> dict[str, int]:
        """Cross-field rules with a non-zero count."""
        return {k: v for k, v in self.cross_field.items() if v}

    @property
    def total_violation_rows(self) -> int:
        return sum(v.count for v in self.rule_violations)

    @property
    def passed(self) -> bool:
        """Convenience flag (informational — validation is warn-only)."""
        return not (self.schema_issues or self.rule_violations or self.cross_field_hits)


def _friendly_check(check: str) -> str:
    """Map a raw pandera check string to a short, stable label.

    Raw check strings are noisy and unstable — ``isin({...})`` dumps the whole
    domain set with non-deterministic ordering — so collapse the verbose ones.

    Inputs
    ------
    check : str
        The pandera check description.

    Returns
    -------
    str
        A readable, stable label.
    """
    if check.startswith("isin("):
        return "not in allowed values"
    if "field_uniqueness" in check or check == "unique":
        return "duplicate values"
    if check.startswith("in_range("):
        return "out of range " + check[len("in_range") :]
    if check.startswith("greater_than_or_equal_to("):
        return "negative value"
    return check


def _run_schema_checks(df: pd.DataFrame):
    """Run the pandera schema (lazy) and fold failures into RuleViolations.

    Inputs
    ------
    df : pd.DataFrame
        Batch to validate.

    Returns
    -------
    tuple[list[RuleViolation], bool]
        The violations found and whether checks ran (False when pandera is
        not installed — a warn-only degradation; the pandas checks still run).
    """
    try:
        from pandera.errors import SchemaErrors
    except Exception:  # noqa: BLE001 - pandera optional
        logger.warning(
            "pandera not installed; skipping schema/range/domain checks "
            "(install the `validation` extra). Pandas checks still run."
        )
        return [], False

    schema = rules.leads_schema()
    try:
        schema.validate(df, lazy=True)
        return [], True
    except SchemaErrors as err:
        fc = err.failure_cases
        violations = []
        grouped = fc.groupby(["column", "check"], dropna=False)
        for (column, check), grp in grouped:
            examples = grp["failure_case"].dropna().unique().tolist()[:5]
            violations.append(
                RuleViolation(
                    column=str(column),
                    check=_friendly_check(str(check)),
                    count=int(len(grp)),
                    examples=examples,
                )
            )
        return violations, True


def validate_raw_kinds(df: pd.DataFrame) -> list[RuleViolation]:
    """Validate raw values against each registry field's declared kind.

    This check runs before storage dtype coercion so malformed source values
    remain distinguishable from genuinely missing values. Missing and blank
    values are ignored here because missingness is reported separately by the
    ordinary validation pass.

    Inputs
    ------
    df : pandas.DataFrame
        Raw dataframe exactly as returned by the warehouse query.

    Returns
    -------
    list[RuleViolation]
        One violation per field with malformed raw values.
    """
    violations: list[RuleViolation] = []
    for name, spec in field_registry.RAW_FIELD_REGISTRY.items():
        if name not in df.columns:
            continue

        result = rules.validate_kind(
            df[name],
            spec.validation.kind,
        )
        if result.count <= 0:
            continue

        violations.append(
            RuleViolation(
                column=name,
                check=result.check,
                count=result.count,
                examples=list(result.examples),
            )
        )
    return violations


def _run_custom_rules(df: pd.DataFrame) -> list[RuleViolation]:
    """Run each registered field's ``custom_rule`` and fold hits into violations.

    Custom rules are plain pandas (see ``validation_custom``), so — unlike the
    pandera schema — they run even when pandera isn't installed. A rule that
    raises is logged and skipped; validation must never break the pull.

    Inputs
    ------
    df : pd.DataFrame
        Batch to validate.

    Returns
    -------
    list[RuleViolation]
        One violation per field whose custom rule reported a non-zero count.
    """
    violations: list[RuleViolation] = []
    for name, spec in field_registry.RAW_FIELD_REGISTRY.items():
        rule = spec.validation.custom_rule
        if rule is None or name not in df.columns:
            continue
        try:
            result = rule(df[name])
        except Exception as exc:  # noqa: BLE001
            # A custom rule must never break validation of the pull.
            logger.warning("Custom validation rule for '%s' failed: %s", name, exc)
            continue
        if result is not None and int(getattr(result, "count", 0)) > 0:
            violations.append(
                RuleViolation(
                    column=name,
                    check=result.check,
                    count=int(result.count),
                    examples=list(result.examples),
                )
            )
    return violations


def validate_leads(
    df: pd.DataFrame,
    high_missing_threshold: float = DEFAULT_HIGH_MISSING_THRESHOLD,
    lead_type_id: int | None = None,
    raw_kind_violations: list[RuleViolation] | None = None,
) -> ValidationReport:
    """Validate a raw ``lead_pings`` batch. Detect-only; never mutates ``df``.

    Inputs
    ------
    df : pd.DataFrame
        The raw batch to validate.
    high_missing_threshold : float
        Null/blank rate at/above which a column is flagged as high-missing.
    lead_type_id : int | None
        When given (a single-lead-type batch), the high-missing catalogue is
        scoped to the fields that apply to that lead type, so other products'
        columns — legitimately empty for this type (e.g. ``home_*`` on an auto
        pull) — aren't flagged as high-missing. ``None`` (e.g. an all-types
        pull) applies no scoping. The full per-column ``missing`` rates are
        always kept intact regardless.
    raw_kind_violations : list[RuleViolation] | None
        Raw-value kind violations captured before dtype coercion. These are
        merged into the ordinary rule violations for one combined report.

    Returns
    -------
    ValidationReport
        Structured findings for artifact / Slack / log rendering.
    """
    total = len(df)

    # Count errored rows first. These are reported but excluded from all
    # ordinary validation because they are not usable training observations.
    erred = rules.erred_mask(df)
    erred_rows = int(erred.sum())
    non_erred_df = df.loc[~erred].copy()

    # Apply the same auction-eligibility rule used by feature engineering:
    # keep rows where bid > 0 or won == true. Rows excluded here are reported
    # separately and do not distort downstream feature-quality validation.
    auction_eligible = rules.auction_eligible_mask(non_erred_df)
    auction_excluded_rows = int((~auction_eligible).sum())
    validation_df = non_erred_df.loc[auction_eligible].copy()
    validated_rows = len(validation_df)

    # Schema drift is a property of the pulled payload, so keep it on the full
    # batch. Row-level validation runs on the training-eligible population.
    schema_issues = rules.schema_drift(df)
    rule_violations, schema_checked = _run_schema_checks(validation_df)
    rule_violations = (
        list(raw_kind_violations or [])
        + rule_violations
        + _run_custom_rules(validation_df)
    )

    # Checks that depend on rows removed by the pre-validation filters must run
    # before those rows are excluded.
    cross = rules.erred_cross_field_checks(df)
    cross.update(rules.auction_cross_field_checks(non_erred_df))
    cross.update(rules.cross_field_checks(validation_df))

    missing = rules.missing_rates(validation_df)

    # These operational error fields are no longer informative once errored
    # rows have been explicitly counted and removed from the validation
    # population. Keep the top-level errored-row summary, but omit these fields
    # from post-filter missingness reporting.
    for column in ("erred", "error_reason_id"):
        missing.pop(column, None)

    constants = rules.constant_columns(validation_df)
    for column in ("erred", "error_reason_id"):
        constants.pop(column, None)

    if lead_type_id is None:
        out_of_scope = set()
    else:
        out_of_scope = field_registry.columns_not_for_lead_type_id(lead_type_id)
    high_missing = sorted(
        c
        for c, r in missing.items()
        if r >= high_missing_threshold and c not in out_of_scope
    )
    constants = {
        c: profile for c, profile in constants.items() if c not in out_of_scope
    }

    metrics = rules.batch_metrics(validation_df)
    metrics["pulled_rows"] = total
    metrics["erred_rows"] = erred_rows
    metrics["auction_excluded_rows"] = auction_excluded_rows
    metrics["validated_rows"] = validated_rows
    metrics["erred_rate"] = float(erred_rows / total) if total else 0.0
    metrics["auction_excluded_rate_of_non_erred"] = (
        float(auction_excluded_rows / len(non_erred_df)) if len(non_erred_df) else 0.0
    )

    return ValidationReport(
        total_rows=total,
        schema_issues=schema_issues,
        rule_violations=rule_violations,
        cross_field=cross,
        missing=missing,
        high_missing=high_missing,
        constant_columns=constants,
        metrics=metrics,
        schema_checked=schema_checked,
        erred_rows=erred_rows,
        auction_excluded_rows=auction_excluded_rows,
        validated_rows=validated_rows,
    )
