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

from smarthub.core import notifications, storage, task_config
from smarthub.core.config import PullSettings, StorageSettings
from smarthub.data_pull import validation_report as vreport
from smarthub.data_pull.pull import fetch_leads
from smarthub.data_pull.validation_runner import validate_leads
from smarthub.data_pull.windowing import compute_pull_window, format_dt, parse_dt

WATERMARK_PREFIX = "smarthub_last_pull_timestamp"


def watermark_variable(lead_type_name: str) -> str:
    """Build the watermark Variable name for a lead type.

    Lowercased and underscore-safe, one watermark per lead type.

    Inputs
    ------
    lead_type_name : str
        Lead type name (e.g. ``"auto"``); whitespace/case are normalised.

    Returns
    -------
    str
        Variable name ``smarthub_last_pull_timestamp_<lead_type_name>``.
    """
    return f"{WATERMARK_PREFIX}_{lead_type_name.strip().lower()}"


def _utc_now_naive() -> datetime:
    """Current UTC time as a naive datetime (matches the warehouse columns)."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


@task(retries=2, retry_delay_seconds=30)
def resolve_window(var_name: str, overlap_hours: float, default_lookback_hours: float):
    """Read this lead type's watermark and compute the pull window.

    Inputs
    ------
    var_name : str
        Watermark Variable name for this lead type.
    overlap_hours : float
        Hours to re-pull before the watermark, for late-resolving outcomes.
    default_lookback_hours : float
        Backfill lookback used on the first run (no watermark yet).

    Returns
    -------
    tuple[str, str]
        The ``(min, max)`` window bounds formatted as datetime strings.
    """
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
    """Pull lead_pings for one lead type over the window via the ORM pull.

    Inputs
    ------
    min_s : str
        Inclusive lower bound for ``created_at`` (datetime string).
    max_s : str
        Exclusive upper bound for ``created_at`` (datetime string).
    lead_type_id : int
        Lead type to restrict the pull to (e.g. 6=auto, 1=home).
    with_expected_revenue : bool
        Whether to join per-ping expected revenue from the listings.
    selected_only : bool
        Aggregate expected revenue over selected listings only.

    Returns
    -------
    pandas.DataFrame
        The pulled leads frame.
    """
    return fetch_leads(
        PullSettings.from_env(),
        min_s,
        max_s,
        with_expected_revenue=with_expected_revenue,
        selected_only=selected_only,
        lead_type_ids=[lead_type_id],
    )


@task
def persist(df: pd.DataFrame) -> dict:
    """Upsert the pulled frame into the configured storage backend(s).

    Inputs
    ------
    df : pandas.DataFrame
        The pulled leads frame to store.

    Returns
    -------
    dict
        Storage result (written paths and row counts).
    """
    return storage.save_pull(df, StorageSettings.from_env())


@task(name="validate-leads")
def validate(df: pd.DataFrame, lead_type_name: str, lead_type_id: int):
    """Validate the freshly-pulled batch (warn + report only; never gates).

    Publishes a per-lead-type data-quality Prefect artifact so the success
    notification can carry a summary. Detect-only: it does not drop/fix rows,
    and any error here must not fail the pull.

    Inputs
    ------
    df : pandas.DataFrame
        The freshly-pulled leads batch.
    lead_type_name : str
        Lead type name, used in the artifact key and description.
    lead_type_id : int
        Lead type id; scopes the high-missing catalogue to this type's fields.

    Returns
    -------
    ValidationReport | None
        The validation report, or ``None`` if validation raised.
    """
    logger = get_run_logger()
    threshold = task_config.get_float("validation", "high_missing_threshold", 0.5)
    try:
        rep = validate_leads(
            df, high_missing_threshold=threshold, lead_type_id=lead_type_id
        )
    except Exception as exc:  # noqa: BLE001 - validation must never break a pull
        logger.warning("Data validation skipped (error): %s", exc)
        return None
    vreport.log_summary(rep, logger)
    create_markdown_artifact(
        key=f"data-quality-{lead_type_name.strip().lower()}",
        markdown=vreport.to_markdown(rep, lead_type_name),
        description=f"Data quality for the latest {lead_type_name} pull",
    )
    return rep


@task
def update_watermark(df: pd.DataFrame, var_name: str, window_max: str) -> str:
    """Advance this lead type's watermark to the latest ``created_at`` pulled.

    If the window had no rows, the previous watermark is kept unchanged (so a
    gap is not skipped), falling back to the window max only if none was set.

    Inputs
    ------
    df : pandas.DataFrame
        The pulled frame; its max ``created_at`` becomes the new watermark.
    var_name : str
        Watermark Variable name for this lead type.
    window_max : str
        Window upper bound, used as a fallback when no watermark exists.

    Returns
    -------
    str
        The new watermark value that was stored.
    """
    if df is not None and not df.empty and "created_at" in df.columns:
        latest = pd.to_datetime(df["created_at"]).max()
        new_value = format_dt(latest.to_pydatetime())
    else:
        new_value = Variable.get(var_name, default=window_max)
    Variable.set(var_name, new_value, overwrite=True)
    return new_value


@flow(name="smarthub-data-pull", on_failure=[notifications.flow_failure_hook])
def data_pull_flow(
    lead_type_id: int = 6,
    lead_type_name: str = "auto",
    overlap_hours: float = 1.0,
    default_lookback_hours: float = 168.0,
    with_expected_revenue: bool = True,
    selected_only: bool = True,
) -> dict:
    """Scheduled pull for ONE lead type: resolve, fetch, persist, watermark.

    On any unhandled failure, ``flow_failure_hook`` sends a Slack alert. On
    success, a Slack notification reports the lead type, window, row count and
    stored file paths.

    Inputs
    ------
    lead_type_id : int
        Lead type id to pull (e.g. 6=auto, 1=home).
    lead_type_name : str
        Human name for the lead type, used in watermarks and reports.
    overlap_hours : float
        Hours to re-pull before the watermark, for late-resolving outcomes.
    default_lookback_hours : float
        Backfill lookback used on the first run.
    with_expected_revenue : bool
        Whether to join per-ping expected revenue from the listings.
    selected_only : bool
        Aggregate expected revenue over selected listings only.

    Returns
    -------
    dict
        Summary with ``lead_type``, ``rows`` and ``watermark``.
    """
    logger = get_run_logger()
    started_at = _utc_now_naive()
    var_name = watermark_variable(lead_type_name)

    min_s, max_s = resolve_window(var_name, overlap_hours, default_lookback_hours)
    logger.info("[%s] pulling window %s -> %s", lead_type_name, min_s, max_s)

    previous_watermark = Variable.get(var_name, default="(none — first run)")
    df = fetch(min_s, max_s, lead_type_id, with_expected_revenue, selected_only)
    quality = validate(df, lead_type_name, lead_type_id)
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
    _notify_success(
        lead_type_name,
        lead_type_id,
        min_s,
        max_s,
        df,
        result,
        previous_watermark,
        watermark,
        started_at,
        quality,
    )
    return {"lead_type": lead_type_name, "rows": int(len(df)), "watermark": watermark}


def _notify_success(
    lead_type_name,
    lead_type_id,
    min_s,
    max_s,
    df,
    result,
    prev_wm,
    new_wm,
    started_at,
    quality=None,
) -> None:
    """Send the Slack 'data pull completed' notification, grouped for reading.

    Leads with a headline (rows + window), then groups the rest under titled
    sections (Volume / Watermark / Run); storage paths go in the footer.

    Inputs
    ------
    lead_type_name : str
        Lead type name shown in the subject.
    lead_type_id : int
        Lead type id shown in the subject.
    min_s : str
        Window lower bound (datetime string).
    max_s : str
        Window upper bound (datetime string).
    df : pandas.DataFrame
        The pulled frame, used for the row count.
    result : dict
        Storage result (paths and row counts).
    prev_wm : str
        Watermark value before this pull.
    new_wm : str
        Watermark value after this pull.
    started_at : datetime
        UTC time the run started.
    quality : ValidationReport | None
        Optional validation report to append as a group.
    """
    rows = int(len(df))
    parquet_paths = result.get("parquet_paths") or []
    parquet_txt = ", ".join(f"`{p}`" for p in parquet_paths) if parquet_paths else "—"
    duck = f"`{result['duckdb_path']}`" if result.get("duckdb_path") else "—"

    headline = f":inbox_tray: *{rows:,} rows pulled* · `{min_s}` → `{max_s}`"
    groups = [
        (
            "Volume",
            {
                "Rows fetched": f"{rows:,}",
                "DuckDB rows (total)": result.get("duckdb_rows", "—"),
                "Parquet rows (written)": result.get("parquet_rows", "—"),
            },
        ),
        (
            "Watermark",
            {
                "Before": f"`{prev_wm}`",
                "After": f"`{new_wm}`",
            },
        ),
        (
            "Run (UTC)",
            {
                "Started": started_at.strftime("%Y-%m-%d %H:%M:%S"),
                "Finished": _utc_now_naive().strftime("%Y-%m-%d %H:%M:%S"),
            },
        ),
    ]
    if quality is not None:
        groups.append(vreport.slack_group(quality))
    notifications.notify_success_grouped(
        "data-pull",
        subject=f"{lead_type_name} ({lead_type_id})",
        headline=headline,
        groups=groups,
        footer_extra=f"parquet {parquet_txt} · duckdb {duck}",
    )


def _report(
    lead_type_name, lead_type_id, min_s, max_s, df, result, prev_wm, new_wm
) -> None:
    """Publish a Prefect markdown artifact summarising this pull.

    Inputs
    ------
    lead_type_name : str
        Lead type name for the artifact key and heading.
    lead_type_id : int
        Lead type id shown in the summary.
    min_s : str
        Window lower bound (datetime string).
    max_s : str
        Window upper bound (datetime string).
    df : pandas.DataFrame
        The pulled frame, used for row and win counts.
    result : dict
        Storage result (row counts).
    prev_wm : str
        Watermark value before this pull.
    new_wm : str
        Watermark value after this pull.
    """
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
