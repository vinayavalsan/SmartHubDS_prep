"""Render a ``ValidationReport`` as markdown (artifact), Slack fields, or a log.

Formatting only — no validation logic here.
"""

from __future__ import annotations

import logging

from .validation_runner import ValidationReport


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
    parts = [f"{v.column} {v.check} ({v.count:,})" for v in report.rule_violations[:6]]
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
        "Pulled / validated rows": (
            f"{report.total_rows:,} / {report.validated_rows:,}"
        ),
        "Errored rows": (
            f"{report.erred_rows:,} ({_pct(report.metrics.get('erred_rate'))})"
        ),
        "Auction-ineligible rows": (
            f"{report.auction_excluded_rows:,} "
            f"({_pct(report.metrics.get('auction_excluded_rate_of_non_erred'))} "
            "of non-erred)"
        ),
        "Rule violations": (
            f"{report.total_violation_rows:,} rows — {_violations_summary(report)}"
            if report.rule_violations
            else "none"
        ),
        "Cross-field flags": _cross_field_summary(report),
        "Constant / single-value cols": (
            ", ".join(sorted(report.constant_columns))
            if report.constant_columns
            else "none"
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
        f"**Status:** {_status_line(report)}",
        "",
        f"- Pulled rows: **{report.total_rows:,}**",
        (
            f"- Errored rows excluded from ordinary validation: "
            f"**{report.erred_rows:,}** "
            f"({_pct(report.metrics.get('erred_rate'))})"
        ),
        (
            f"- Auction-ineligible rows excluded: "
            f"**{report.auction_excluded_rows:,}** "
            f"({_pct(report.metrics.get('auction_excluded_rate_of_non_erred'))} "
            f"of non-erred)"
        ),
        f"- Rows validated (training-eligible): **{report.validated_rows:,}**",
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

    lines += ["## Constant / single-value columns (training-eligible rows)", ""]
    if report.constant_columns:
        lines += [
            "| column | profile | value | missing |",
            "| --- | --- | --- | --- |",
        ]
        for column, profile in sorted(report.constant_columns.items()):
            kind = profile["kind"]
            if kind == "all_missing":
                label = "all missing"
                value = "—"
            else:
                label = "single non-missing value"
                value = str(profile["value"])
            lines.append(
                f"| {column} | {label} | {value} | "
                f"{_pct(profile['missing_rate'])} |"
            )
    else:
        lines.append("none")
    lines.append("")

    lines += [
        "## Batch metrics (training-eligible rows)",
        "",
        "| metric | value |",
        "| --- | --- |",
        f"| bid = 0 rate | {_pct(m.get('bid_zero_rate'))} |",
        f"| exp_rev coverage | {_pct(m.get('exp_rev_coverage'))} |",
        f"| pst_hour populated | {_pct(m.get('pst_hour_populated'))} |",
        f"| age implausible | {_pct(m.get('age_implausible_rate'))} |",
        f"| won = false rows | {m.get('won_false_count', 0):,} |",
        f"| traffic_tier distinct | {m.get('traffic_tier_distinct', '—')} |",
        "",
    ]

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
        "data-quality: pulled_rows=%s, erred_rows=%s, auction_excluded_rows=%s, "
        "validated_rows=%s, status=%s, rule_violation_rows=%s, "
        "cross_field=%s",
        f"{report.total_rows:,}",
        f"{report.erred_rows:,}",
        f"{report.auction_excluded_rows:,}",
        f"{report.validated_rows:,}",
        "clean" if report.passed else "issues flagged",
        f"{report.total_violation_rows:,}",
        report.cross_field_hits or "none",
    )


def log_detailed(report: ValidationReport, log: logging.Logger) -> None:
    """Log the complete validation report in a CLI-friendly format.

    Inputs
    ------
    report : ValidationReport
        The completed validation report.
    log : logging.Logger
        Logger to emit the detailed report to.
    """
    erred_rate = report.metrics.get("erred_rate", 0.0)
    auction_rate = report.metrics.get("auction_excluded_rate_of_non_erred", 0.0)
    lines = [
        "Data quality validation report",
        f"pulled rows: {report.total_rows:,}",
        f"erred rows: {report.erred_rows:,} ({erred_rate:.1%})",
        (
            f"auction-ineligible rows: {report.auction_excluded_rows:,} "
            f"({auction_rate:.1%} of non-erred)"
        ),
        f"rows validated (training-eligible): {report.validated_rows:,}",
        f"status: {'clean' if report.passed else 'issues flagged'}",
        "",
        "Schema issues:",
    ]

    if report.schema_issues:
        lines.extend(f"  - {issue}" for issue in report.schema_issues)
    else:
        lines.append("  none")

    lines.extend(["", "Rule violations:"])
    if report.rule_violations:
        for violation in report.rule_violations:
            examples = (
                ", ".join(str(value) for value in violation.examples)
                if violation.examples
                else "none"
            )
            lines.append(
                f"  - {violation.column}: {violation.check}; "
                f"rows={violation.count:,}; examples={examples}"
            )
    else:
        lines.append("  none")

    lines.extend(["", "Cross-field integrity:"])
    if report.cross_field_hits:
        for rule, count in sorted(report.cross_field_hits.items()):
            lines.append(f"  - {rule}: {count:,} row(s)")
    else:
        lines.append("  none")

    lines.extend(["", "Missing values (training-eligible rows):"])
    if report.missing:
        reported_missing = False
        for column, rate in sorted(report.missing.items()):
            if rate > 0:
                lines.append(f"  - {column}: {rate:.1%}")
                reported_missing = True
        if not reported_missing:
            lines.append("  none")
    else:
        lines.append("  none")

    lines.extend(["", "Constant / single-value columns (training-eligible rows):"])
    if report.constant_columns:
        for column, profile in sorted(report.constant_columns.items()):
            if profile["kind"] == "all_missing":
                lines.append(f"  - {column}: all missing")
            else:
                lines.append(
                    f"  - {column}: only non-missing value="
                    f"{profile['value']!r}; missing={profile['missing_rate']:.1%}"
                )
    else:
        lines.append("  none")

    lines.extend(["", "Batch metrics (training-eligible rows):"])
    if report.metrics:
        for name, value in sorted(report.metrics.items()):
            if name in {
                "pulled_rows",
                "erred_rows",
                "auction_excluded_rows",
                "validated_rows",
                "erred_rate",
                "auction_excluded_rate_of_non_erred",
            }:
                continue
            if isinstance(value, float):
                rendered = f"{value:.4f}"
            else:
                rendered = str(value)
            lines.append(f"  - {name}: {rendered}")
    else:
        lines.append("  none")

    if not report.schema_checked:
        lines.extend(
            [
                "",
                "Schema checks: skipped (pandera not installed)",
            ]
        )

    log.info("\n%s", "\n".join(lines))
