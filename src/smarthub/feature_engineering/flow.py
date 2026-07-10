"""Prefect wrapper for STEP 2 (build-features) — automation only.

Thin wrapper around the Prefect-free core in ``build.py``: the actual work
(load → build → save) lives in ``build.run_build_features``; this module only
adds what automation needs — a Prefect run, the "run data-pull first" artifact,
a run-summary artifact, the Slack notification, and the failure hook. Manual
runs use ``build.py`` directly (``smarthub-build-features``) and don't touch
Prefect.

Runs on the same work pool as the pull but a **separate queue** (`features`).
"""

from __future__ import annotations

from prefect import flow, get_run_logger, task
from prefect.artifacts import create_markdown_artifact

from smarthub.core import notifications, storage

from . import build
from .build import _DAY_DEFS, _pct, run_build_features


@task(name="build-training-table")
def _build_task(lead_type_id, lead_type_name, window_days):
    """Run the Prefect-free build core inside a tracked Prefect task."""
    return run_build_features(
        lead_type_id=lead_type_id,
        lead_type_name=lead_type_name,
        window_days=window_days,
        log=get_run_logger(),
    )


@flow(name="smarthub-build-features", on_failure=[notifications.flow_failure_hook])
def build_features_flow(
    lead_type_id: int = 6,
    lead_type_name: str = "auto",
    window_days: int | None = None,
) -> dict:
    """Build and save the training table for one lead type (STEP 2).

    ``window_days`` overrides the rolling training window; when ``None`` it falls
    back to ``training_window_days`` in config/smarthub.ini (default 21).
    ``0`` = all data.

    On "no data yet" it publishes a clear "run data-pull first" artifact and
    fails; ``flow_failure_hook`` sends the Slack alert. On success it publishes a
    run-summary artifact and a Slack notification.
    """
    try:
        result = _build_task(lead_type_id, lead_type_name, window_days)
    except storage.StorageError:
        _blocked_artifact()
        raise

    metadata = result["metadata"]
    version = result["version"]
    path = result["path"]
    _report(result["lead_type_name"], version, metadata, path)
    _notify_success(result["lead_type_name"], lead_type_id, version, metadata, path)
    return {
        "lead_type": result["lead_type_name"],
        "version": version,
        "rows": result["rows"],
        "columns": result["columns"],
        "path": path,
    }


def _blocked_artifact() -> None:
    """Publish the 'build-features ran before data-pull' guidance artifact."""
    create_markdown_artifact(
        key="build-features-blocked",
        markdown=(
            "# ⚠️ build-features blocked — run data-pull first\n\n"
            "This is **STEP 2** of the pipeline and found **no lead data** "
            "in storage.\n\n"
            "**Pipeline order:**\n\n"
            "1. `data-pull` — pulls leads from Redshift into storage "
            "(**run this first**)\n"
            "2. `build-features` — builds the training table from that "
            "data (this step)\n\n"
            "**Fix:** run the `smarthub-data-pull/data-pull` deployment "
            "(auto and home), wait for it to finish, then re-run this "
            "deployment.\n"
        ),
        description="build-features ran before data-pull",
    )


def _report(lead_type_name: str, version: str, metadata: dict, path: str) -> None:
    """Publish a Prefect markdown artifact summarising this feature build."""
    feats = metadata.get("feature_columns", [])
    won_rate = metadata.get("won_rate")
    won_rate_str = f"{won_rate:.3f}" if isinstance(won_rate, float) else "-"
    md = f"""# Feature build — {lead_type_name}

| field | value |
| --- | --- |
| version | `{version}` |
| lead_type_id | {metadata.get('lead_type_id')} |
| training window (days) | {metadata.get('training_window_days')} |
| raw rows | {metadata.get('raw_rows')} |
| training rows | {metadata.get('row_count')} |
| dropped (errored/no-bid) | {metadata.get('dropped_rows')} |
| wins / losses | {metadata.get('wins')} / {metadata.get('losses')} |
| win rate | {won_rate_str} |
| data range (created_at) | `{metadata.get('data_min_created_at')}` → \
`{metadata.get('data_max_created_at')}` |
| expected_revenue coverage | {_pct(metadata.get('expected_revenue_coverage'))} |
| age missing | {_pct(metadata.get('age_missing_rate'))} |
| weekday share | {_pct(metadata.get('weekday_share'))} |
| weekend share | {_pct(metadata.get('weekend_share'))} |
| workday share (is_workday) | {_pct(metadata.get('workday_rate'))} |
| traffic_tier distinct | {metadata.get('traffic_tier_distinct')} |
| feature count | {len(feats)} |
| output | `{path}` |

_Day metrics: {_DAY_DEFS}._

**Features ({len(feats)}):** {', '.join(map(str, feats)) if feats else '—'}
"""
    create_markdown_artifact(
        key=f"build-features-{lead_type_name.strip().lower()}",
        markdown=md,
        description=f"Latest {lead_type_name} training table",
    )


def _notify_success(
    lead_type_name, lead_type_id, version, metadata, path
) -> None:
    """Send the Slack 'feature build completed' notification, grouped for reading.

    Leads with a headline (training rows + version + win rate), then groups the
    rest under titled sections (Rows / Coverage / Time mix / Build). Day-metric
    definitions and the output path go in the footer.
    """
    feats = metadata.get("feature_columns", [])
    won_rate = metadata.get("won_rate")
    won_rate_str = f"{won_rate:.3f}" if isinstance(won_rate, float) else "—"
    raw_rows = metadata.get("raw_rows") or 0
    row_count = metadata.get("row_count") or 0
    dropped = metadata.get("dropped_rows") or 0

    headline = (
        f":bar_chart: *{row_count:,} training rows* · `{version}` "
        f"(win rate {won_rate_str})"
    )
    rows_summary = (
        f"{raw_rows:,} → {row_count:,} (dropped {dropped:,} errored/no-bid)"
    )
    groups = [
        ("Rows", {
            "Raw → training": rows_summary,
            "Wins / losses": (
                f"{metadata.get('wins'):,} / {metadata.get('losses'):,}"
            ),
            "Win rate": won_rate_str,
        }),
        ("Coverage", {
            "expected_revenue coverage": _pct(
                metadata.get("expected_revenue_coverage")
            ),
            "age missing": _pct(metadata.get("age_missing_rate")),
            "traffic_tier distinct": metadata.get("traffic_tier_distinct"),
        }),
        ("Time mix", {
            "weekday share": _pct(metadata.get("weekday_share")),
            "weekend share": _pct(metadata.get("weekend_share")),
            "workday share (is_workday)": _pct(metadata.get("workday_rate")),
        }),
        ("Build", {
            "Training window (days)": metadata.get("training_window_days"),
            "Feature count": len(feats),
            "Data range (created_at)": (
                f"`{metadata.get('data_min_created_at')}` → "
                f"`{metadata.get('data_max_created_at')}`"
            ),
        }),
    ]
    notifications.notify_success_grouped(
        "build-features",
        subject=f"{lead_type_name} ({lead_type_id})",
        headline=headline,
        groups=groups,
        footer_extra=f"{_DAY_DEFS} · table `{path}`",
    )


if __name__ == "__main__":
    # Local flow run (automation-style). For a Prefect-free manual run use
    # `python -m smarthub.feature_engineering.build` / `smarthub-build-features`.
    raise SystemExit(build.main())
