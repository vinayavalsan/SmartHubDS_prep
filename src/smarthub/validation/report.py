"""Render a ``ValidationReport`` as markdown (artifact), Slack fields, or a log.

Formatting only — no validation logic here.
"""

from __future__ import annotations

import logging

from .validate import ValidationReport


def _pct(value) -> str:
    return f"{value:.1%}" if isinstance(value, (int, float)) else "—"


def _status_line(report: ValidationReport) -> str:
    n_issues = (
        len(report.schema_issues)
        + len(report.rule_violations)
        + len(report.cross_field_hits)
    )
    if n_issues == 0:
        return ":white_check_mark: clean"
    return f":warning: {n_issues} issue type(s) flagged"


def _cross_field_summary(report: ValidationReport) -> str:
    hits = report.cross_field_hits
    if not hits:
        return "none"
    return ", ".join(f"{k}={v:,}" for k, v in sorted(hits.items()))


def _violations_summary(report: ValidationReport) -> str:
    if not report.rule_violations:
        return "none"
    parts = [
        f"{v.column} {v.check} ({v.count:,})"
        for v in report.rule_violations[:6]
    ]
    extra = len(report.rule_violations) - 6
    if extra > 0:
        parts.append(f"+{extra} more")
    return "; ".join(parts)


def slack_group(report: ValidationReport) -> tuple[str, dict]:
    """Build one (title, fields) group for the data-pull notification.

    Inputs
    ------
    report : ValidationReport
        The completed validation report to summarize.

    Returns
    -------
    tuple[str, dict]
        The section title and its Slack field mapping.
    """
    m = report.metrics
    fields = {
        "Status": _status_line(report),
        "Rule violations": (
            f"{report.total_violation_rows:,} rows — {_violations_summary(report)}"
            if report.rule_violations else "none"
        ),
        "Cross-field flags": _cross_field_summary(report),
        "High-missing cols": (
            ", ".join(report.high_missing) if report.high_missing else "none"
        ),
        "exp_rev coverage": _pct(m.get("exp_rev_coverage")),
        "pst_hour populated": _pct(m.get("pst_hour_populated")),
        "age implausible": _pct(m.get("age_implausible_rate")),
        "won=false rows": m.get("won_false_count", 0),
    }
    if not report.schema_checked:
        fields["Schema checks"] = "skipped (pandera not installed)"
    return ("Data quality", fields)


def to_markdown(report: ValidationReport, lead_type_name: str) -> str:
    """Render the full report as a Prefect markdown artifact.

    Inputs
    ------
    report : ValidationReport
        The completed validation report.
    lead_type_name : str
        Human-readable lead type used in the heading.

    Returns
    -------
    str
        The markdown document.
    """
    m = report.metrics
    lines = [
        f"# Data quality — {lead_type_name}",
        "",
        f"**Status:** {_status_line(report)}  ·  **rows:** "
        f"{report.total_rows:,}",
        "",
    ]

    if report.schema_issues:
        lines += ["## Schema issues", ""]
        lines += [f"- {i}" for i in report.schema_issues] + [""]

    lines += ["## Rule violations", ""]
    if report.rule_violations:
        lines += ["| column | check | rows | examples |", "| --- | --- | --- | --- |"]
        for v in report.rule_violations:
            ex = ", ".join(str(e) for e in v.examples)
            lines.append(f"| {v.column} | {v.check} | {v.count:,} | {ex} |")
    else:
        lines.append("none")
    lines.append("")

    lines += ["## Cross-field integrity", ""]
    if report.cross_field:
        lines += ["| rule | rows |", "| --- | --- |"]
        for k, val in sorted(report.cross_field.items()):
            lines.append(f"| {k} | {val:,} |")
    else:
        lines.append("none checked")
    lines.append("")

    lines += [
        "## Batch metrics",
        "",
        "| metric | value |",
        "| --- | --- |",
        f"| erred rate | {_pct(m.get('erred_rate'))} |",
        f"| bid = 0 rate | {_pct(m.get('bid_zero_rate'))} |",
        f"| exp_rev coverage | {_pct(m.get('exp_rev_coverage'))} |",
        f"| pst_hour populated | {_pct(m.get('pst_hour_populated'))} |",
        f"| age implausible | {_pct(m.get('age_implausible_rate'))} |",
        f"| won = false rows | {m.get('won_false_count', 0):,} |",
        f"| traffic_tier distinct | {m.get('traffic_tier_distinct', '—')} |",
        "",
    ]

    high = report.high_missing
    lines += ["## High-missing columns (≥ threshold)", ""]
    if high:
        lines += ["| column | missing |", "| --- | --- |"]
        for c in high:
            lines.append(f"| {c} | {_pct(report.missing.get(c))} |")
    else:
        lines.append("none")
    lines.append("")
    if not report.schema_checked:
        lines += ["_Schema/range/domain checks skipped — pandera not installed._"]
    return "\n".join(lines)


def log_summary(report: ValidationReport, log: logging.Logger) -> None:
    """Log a one-glance summary (used by the Prefect-free CLI).

    Inputs
    ------
    report : ValidationReport
        The completed validation report.
    log : logging.Logger
        Logger to emit the summary to.
    """
    log.info(
        "data-quality: rows=%s, status=%s, rule_violation_rows=%s, "
        "cross_field=%s, high_missing=%s",
        f"{report.total_rows:,}",
        "clean" if report.passed else "issues flagged",
        f"{report.total_violation_rows:,}",
        report.cross_field_hits or "none",
        report.high_missing or "none",
    )
