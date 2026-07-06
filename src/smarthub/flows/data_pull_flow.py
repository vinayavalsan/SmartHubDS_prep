"""Prefect flow that runs the SmartHub data pull, broken into tasks.

Tasks: resolve_window → fetch → persist → update_watermark.
Reuses the existing pull/storage code (``data_pull.fetch_leads``,
``storage.save_pull``) so the orchestration layer stays thin.

The flow is parametrized by **lead type** (e.g. auto=6, home=1) and pulls each
type separately. Each type keeps its **own** watermark in a Prefect Variable
named ``smarthub_last_pull_timestamp_<lead_type_name>`` (the timestamp of the
last record pulled), so each type resumes from where it left off.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
from prefect import flow, get_run_logger, task
from prefect.artifacts import create_markdown_artifact
from prefect.variables import Variable

from smarthub.data import storage
from smarthub.core.config import PullSettings, StorageSettings
from smarthub.data.data_pull import fetch_leads
from smarthub.flows.windowing import compute_pull_window, format_dt, parse_dt

WATERMARK_PREFIX = "smarthub_last_pull_timestamp"


def watermark_variable(lead_type_name: str) -> str:
    """Per-lead-type watermark Variable name (lowercase, underscore-safe)."""
    return f"{WATERMARK_PREFIX}_{lead_type_name.strip().lower()}"


def _utc_now_naive() -> datetime:
    """Current UTC time as a naive datetime (matches the warehouse columns)."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


@task(retries=2, retry_delay_seconds=30)
def resolve_window(var_name: str, overlap_hours: float, default_lookback_hours: float):
    """Read this lead type's watermark and compute the (min, max) pull window."""
    raw = Variable.get(var_name, default=None)
    last_ts = parse_dt(raw) if raw else None
    min_dt, max_dt = compute_pull_window(
        _utc_now_naive(), last_ts, overlap_hours, default_lookback_hours
    )
    return format_dt(min_dt), format_dt(max_dt)


@task(retries=2, retry_delay_seconds=60)
def fetch(
    min_s: str,
    max_s: str,
    lead_type_id: int,
    with_expected_revenue: bool,
    selected_only: bool,
):
    """Pull lead_pings for one lead type over the window via the ORM pull."""
    return fetch_leads(
        PullSettings.from_env(),
        min_s,
        max_s,
        with_expected_revenue=with_expected_revenue,
        selected_only=selected_only,
        lead_type_id=lead_type_id,
    )


@task
def persist(df: pd.DataFrame) -> dict:
    """Upsert the pulled frame into the configured storage backend(s)."""
    return storage.save_pull(df, StorageSettings.from_env())


@task
def update_watermark(df: pd.DataFrame, var_name: str, window_max: str) -> str:
    """Advance this lead type's watermark to the latest `created_at` pulled.

    If the window had no rows, the previous watermark is kept unchanged (so we
    don't skip a gap) — falling back to the window max only if none was set.
    """
    if df is not None and not df.empty and "created_at" in df.columns:
        latest = pd.to_datetime(df["created_at"]).max()
        new_value = format_dt(latest.to_pydatetime())
    else:
        new_value = Variable.get(var_name, default=window_max)
    Variable.set(var_name, new_value, overwrite=True)
    return new_value


@flow(name="smarthub-data-pull")
def data_pull_flow(
    lead_type_id: int = 6,
    lead_type_name: str = "auto",
    overlap_hours: float = 1.0,
    default_lookback_hours: float = 168.0,
    with_expected_revenue: bool = True,
    selected_only: bool = True,
) -> dict:
    """Scheduled pull for ONE lead type: resolve → fetch → persist → watermark."""
    logger = get_run_logger()
    var_name = watermark_variable(lead_type_name)

    min_s, max_s = resolve_window(var_name, overlap_hours, default_lookback_hours)
    logger.info("[%s] pulling window %s -> %s", lead_type_name, min_s, max_s)

    previous_watermark = Variable.get(var_name, default="(none — first run)")
    df = fetch(min_s, max_s, lead_type_id, with_expected_revenue, selected_only)
    result = persist(df)
    watermark = update_watermark(df, var_name, max_s)

    logger.info(
        "[%s] persisted %s; watermark now %s", lead_type_name, result, watermark
    )
    _report(
        lead_type_name,
        lead_type_id,
        min_s,
        max_s,
        df,
        result,
        previous_watermark,
        watermark,
    )
    return {"lead_type": lead_type_name, "rows": int(len(df)), "watermark": watermark}


def _report(
    lead_type_name, lead_type_id, min_s, max_s, df, result, prev_wm, new_wm
) -> None:
    """Publish a Prefect markdown artifact summarising this pull."""
    rows = int(len(df))
    won = "-"
    if rows and "won" in df.columns:
        flags = df["won"].astype("string").str.strip().str.lower()
        decided = flags.isin(["true", "false"]).sum()
        wins = (flags == "true").sum()
        won = f"{wins}/{decided} decided"
    md = f"""# Data pull — {lead_type_name}

| field | value |
| --- | --- |
| lead_type_id | {lead_type_id} |
| window (UTC) | `{min_s}` → `{max_s}` |
| rows fetched | {rows} |
| won (decided) | {won} |
| duckdb rows (total) | {result.get('duckdb_rows', '-')} |
| parquet rows (written) | {result.get('parquet_rows', '-')} |
| watermark before | `{prev_wm}` |
| watermark after | `{new_wm}` |
"""
    create_markdown_artifact(
        key=f"data-pull-{lead_type_name.strip().lower()}",
        markdown=md,
        description=f"Latest {lead_type_name} data pull",
    )


if __name__ == "__main__":
    data_pull_flow(lead_type_id=6, lead_type_name="auto")
