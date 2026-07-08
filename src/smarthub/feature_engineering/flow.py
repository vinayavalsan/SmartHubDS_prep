

"""Prefect flow that builds the leakage-safe training table per lead type.

Tasks: load_raw → build_table → save_table. Reads the accumulated lead data
from storage, applies the feature extraction in ``features.build_training_table``
(ping-time features + bid + expected_revenue + won_flag), and writes a per-type
training Parquet for the model step.

Runs on the same work pool as the pull but a **separate queue** (`features`).
"""

from __future__ import annotations

import pandas as pd
from prefect import flow, get_run_logger, task
from prefect.artifacts import create_markdown_artifact

from smarthub.core import io, storage
from smarthub.core import notifications
from smarthub.core.config import StorageSettings, training_window_days
from smarthub.feature_engineering.features import build_training_table

# Shown in the notification/artifact so the three day metrics aren't confused.
_DAY_DEFS = (
    "weekday = Mon–Fri · weekend = Sat/Sun · "
    "is_workday = weekday AND not a holiday — all by Pacific date (pst_date)"
)


@task
def load_raw(training_window_days: int | None) -> pd.DataFrame:
    """Load accumulated leads from storage (full table or recent window).

    build-features is **STEP 2** of the pipeline; it depends on **STEP 1**
    (data-pull). If no data exists yet, surface a clear "run data-pull first"
    message in the run logs and as a Prefect artifact, then fail.
    """
    logger = get_run_logger()
    settings = StorageSettings.from_env()
    try:
        if training_window_days and training_window_days > 0:
            return storage.load_window_raw(settings, training_window_days)
        return storage.load_leads_raw(settings)
    except storage.StorageError as exc:
        logger.error(storage.NO_DATA_MESSAGE)
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
        raise storage.StorageError(storage.NO_DATA_MESSAGE) from exc


@task
def build_table(df: pd.DataFrame, lead_type_id: int) -> pd.DataFrame:
    """Extract the leakage-safe training table for one lead type."""
    return build_training_table(df, lead_type_id=lead_type_id)


def _build_metadata(
    table: pd.DataFrame, lead_type_id: int, window: int, raw_rows: int | None = None
) -> dict:
    """Lineage manifest + build-quality stats for the training table."""
    created = (
        pd.to_datetime(table["created_at"]) if "created_at" in table.columns
        else pd.Series(dtype="datetime64[ns]")
    )
    n = len(table)
    wins = int(pd.to_numeric(table["won_flag"], errors="coerce").sum()) if n else 0

    def _rate(col):
        if col in table.columns and n:
            return float(pd.to_numeric(table[col], errors="coerce").mean())
        return None

    er_coverage = None
    if "expected_revenue" in table.columns and n:
        er = pd.to_numeric(table["expected_revenue"], errors="coerce")
        er_coverage = float((er > 0).mean())

    weekday_share = weekend_share = None
    if "created_dayofweek" in table.columns and n:
        dow = pd.to_numeric(table["created_dayofweek"], errors="coerce")
        weekday_share = float(dow.isin([0, 1, 2, 3, 4]).mean())
        weekend_share = float(dow.isin([5, 6]).mean())

    return {
        "lead_type_id": lead_type_id,
        "training_window_days": window,
        "raw_rows": int(raw_rows) if raw_rows is not None else None,
        "row_count": n,
        "dropped_rows": int(raw_rows - n) if raw_rows is not None else None,
        "wins": wins,
        "losses": n - wins,
        "won_rate": (wins / n if n else None),
        "data_min_created_at": created.min() if len(created) else None,
        "data_max_created_at": created.max() if len(created) else None,
        "expected_revenue_coverage": er_coverage,   # share with R > 0
        "age_missing_rate": _rate("age_missing"),
        "weekday_share": weekday_share,
        "weekend_share": weekend_share,
        "workday_rate": _rate("is_workday"),         # is_workday feature share
        "traffic_tier_distinct": (
            int(table["traffic_tier"].nunique()) if "traffic_tier" in table.columns
            else None
        ),
        "feature_columns": [
            c for c in table.columns
            if c not in ("id", "created_at", "won_flag")
        ],
    }


@task
def save_table(
    table: pd.DataFrame, lead_type_name: str, metadata: dict
) -> str:
    """Persist the per-type training table + lineage manifest."""
    return str(io.save_training_table(table, lead_type_name, metadata=metadata))


@flow(name="smarthub-build-features", on_failure=[notifications.flow_failure_hook])
def build_features_flow(
    lead_type_id: int = 6,
    lead_type_name: str = "auto",
    window_days: int | None = None,
) -> dict:
    """Build and save the training table for one lead type.

    ``window_days`` overrides the rolling training window; when ``None`` it falls
    back to ``training_window_days`` in config/smarthub.ini (default 21).
    ``0`` = all data.

    On any unhandled failure (including "no data — run data-pull first"),
    ``flow_failure_hook`` sends a Slack alert. On success, a Slack notification
    reports the version, row/feature counts and output path.
    """
    logger = get_run_logger()
    window = window_days if window_days is not None else training_window_days()
    raw = load_raw(window)
    raw_rows = int(len(raw))
    table = build_table(raw, lead_type_id)
    metadata = _build_metadata(table, lead_type_id, window, raw_rows=raw_rows)
    path = save_table(table, lead_type_name, metadata)
    version = path.rsplit("/", 1)[-1].removesuffix(".parquet")
    logger.info(
        "[%s] training table %s: %s rows, %s cols -> %s",
        lead_type_name,
        version,
        len(table),
        table.shape[1],
        path,
    )
    _report(lead_type_name, version, table, metadata, path)
    _notify_success(lead_type_name, lead_type_id, version, table, metadata, path)
    return {
        "lead_type": lead_type_name,
        "version": version,
        "rows": int(len(table)),
        "columns": int(table.shape[1]),
        "path": path,
    }


def _report(
    lead_type_name: str, version: str, table: pd.DataFrame, metadata: dict, path: str
) -> None:
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


def _pct(value) -> str:
    return f"{value:.1%}" if isinstance(value, float) else "—"


def _notify_success(
    lead_type_name, lead_type_id, version, table, metadata, path
) -> None:
    """Send the Slack 'feature build completed' notification with full context."""
    feats = metadata.get("feature_columns", [])
    won_rate = metadata.get("won_rate")
    won_rate_str = f"{won_rate:.3f}" if isinstance(won_rate, float) else "—"
    fields = {
        "Lead type": f"{lead_type_name} ({lead_type_id})",
        "Version": f"`{version}`",
        "Rows (raw → training)": (
            f"{metadata.get('raw_rows'):,} → {metadata.get('row_count'):,} "
            f"(dropped {metadata.get('dropped_rows'):,} errored/no-bid)"
        ),
        "Wins / losses": f"{metadata.get('wins'):,} / {metadata.get('losses'):,}",
        "Win rate": won_rate_str,
        "Feature count": len(feats),
        "Training window (days)": metadata.get("training_window_days"),
        "Data range (created_at)": (
            f"`{metadata.get('data_min_created_at')}` → "
            f"`{metadata.get('data_max_created_at')}`"
        ),
        "expected_revenue coverage": _pct(metadata.get("expected_revenue_coverage")),
        "age missing": _pct(metadata.get("age_missing_rate")),
        "weekday share": _pct(metadata.get("weekday_share")),
        "weekend share": _pct(metadata.get("weekend_share")),
        "workday share (is_workday)": _pct(metadata.get("workday_rate")),
        "Day definitions": _DAY_DEFS,
        "traffic_tier distinct": metadata.get("traffic_tier_distinct"),
        "Training table": f"`{path}`",
    }
    notifications.notify_success("build-features", fields)


if __name__ == "__main__":
    build_features_flow(lead_type_id=6, lead_type_name="auto")
