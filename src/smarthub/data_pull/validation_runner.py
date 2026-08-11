"""Orchestrate the leads validation and assemble a ``ValidationReport``.

Warn + report only: ``validate_leads`` never raises and never mutates ``df`` —
it returns a structured report the caller turns into a Prefect artifact / Slack
summary / log line.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd

from smarthub.core.lead_types import lead_type_name

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
    metrics: dict
    schema_checked: bool  # False if pandera wasn't available

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

    Returns
    -------
    ValidationReport
        Structured findings for artifact / Slack / log rendering.
    """
    total = len(df)
    schema_issues = rules.schema_drift(df)
    rule_violations, schema_checked = _run_schema_checks(df)
    rule_violations = rule_violations + _run_custom_rules(df)
    cross = rules.cross_field_checks(df)
    missing = rules.missing_rates(df)
    if lead_type_id is None:
        out_of_scope = set()
    else:
        try:
            resolved_lead_type_name = lead_type_name(lead_type_id)
        except ValueError:
            out_of_scope = set()
        else:
            out_of_scope = field_registry.columns_not_for_lead_type(
                resolved_lead_type_name
            )
    high_missing = sorted(
        c
        for c, r in missing.items()
        if r >= high_missing_threshold and c not in out_of_scope
    )
    metrics = rules.batch_metrics(df)

    return ValidationReport(
        total_rows=total,
        schema_issues=schema_issues,
        rule_violations=rule_violations,
        cross_field=cross,
        missing=missing,
        high_missing=high_missing,
        metrics=metrics,
        schema_checked=schema_checked,
    )
