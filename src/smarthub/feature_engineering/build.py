"""Prefect-free core for STEP 2 (build-features).

Runs the feature build directly — load → build → save — with **no Prefect
dependency**, so it can be invoked manually (``smarthub-build-features`` /
``python -m smarthub.feature_engineering.build``) or from tests without a worker
or server. The Prefect deployment (``flow.py``) is a thin wrapper around
``run_build_features`` for automation (it adds the run artifact + Slack
notification + failure hook).

Mirrors the split already used by the other stages: ``data_pull/pull.py`` and
``train_and_predict/train.py`` hold the Prefect-free core + CLI, while their
``flow.py`` wraps it.
"""

from __future__ import annotations

import argparse
import logging

import pandas as pd

from smarthub.core import io, storage
from smarthub.core.config import StorageSettings, training_window_days
from smarthub.core.lead_types import lead_type_name as _lead_type_name
from smarthub.feature_engineering import features as fe
from smarthub.feature_engineering.feature_registry import FEATURES
from smarthub.feature_engineering.features import build_training_table

logger = logging.getLogger(__name__)


def _required_raw_columns() -> list[str]:
    """Raw columns ``build_training_table`` actually consumes.

    Projecting the storage read to just these keeps peak memory down — the wide
    unused columns (naics/sic codes, life_*, health_conditions, …) are dropped
    from the training table anyway, so they never need to enter pandas.

    Returns
    -------
    list[str]
        Ordered, de-duplicated raw column names to read.
    """
    # Registry entries declare every model feature and the raw input used to
    # produce it. Raw features use ``api_input`` when provided (otherwise the
    # feature name); derived features contribute their declared raw dependency.
    registry_inputs = [
        spec.api_input or spec.name
        for spec in FEATURES.values()
        if spec.source == "raw" or spec.api_input is not None
    ]

    cols = (
        registry_inputs
        + list(fe.RETAINED_NON_MODEL_COLUMNS)
        + [
            "id",
            "created_at",
            # Registry time derivations prefer Pacific fields when available.
            "pst_date",
            "pst_hour",
            "won",
            fe.DECISION_COLUMN,
            "erred",
            "exp_rev",
            fe.REVENUE_COLUMN,
        ]
    )
    seen: set[str] = set()
    ordered = []
    for c in cols:
        if c not in seen:
            seen.add(c)
            ordered.append(c)
    return ordered


# Shown in the notification/artifact so the three day metrics aren't confused.
_DAY_DEFS = (
    "weekday = Mon–Fri · weekend = Sat/Sun · "
    "is_workday = weekday AND not a holiday — all by Pacific date (pst_date)"
)


def _pct(value) -> str:
    return f"{value:.1%}" if isinstance(value, float) else "—"


def _load_raw(window: int | None, log: logging.Logger) -> pd.DataFrame:
    """Load accumulated leads from storage (full table or recent window).

    Reads only the columns the training build needs (column projection) to keep
    peak memory low on the wide ``lead_pings`` table.

    Inputs
    ------
    window : int | None
        Rolling window in days; a full-table read when falsy or <= 0.
    log : logging.Logger
        Logger for the "no data" error message.

    Returns
    -------
    pandas.DataFrame
        The loaded raw leads frame.

    Raises
    ------
    storage.StorageError
        When there is no data yet (STEP 1 has not run).
    """
    settings = StorageSettings.from_env()
    columns = _required_raw_columns()
    try:
        if window and window > 0:
            return storage.load_window_raw(settings, window, columns=columns)
        return storage.load_leads_raw(settings, columns=columns)
    except storage.StorageError as exc:
        log.error(storage.NO_DATA_MESSAGE)
        raise storage.StorageError(storage.NO_DATA_MESSAGE) from exc


def build_metadata(
    table: pd.DataFrame, lead_type_id: int, window: int, raw_rows: int | None = None
) -> dict:
    """Lineage manifest + build-quality stats for the training table.

    Inputs
    ------
    table : pandas.DataFrame
        The built training table.
    lead_type_id : int
        Lead type id the table was built for.
    window : int
        Training window in days recorded in the manifest.
    raw_rows : int | None
        Raw row count before filtering, for the dropped-rows stat.

    Returns
    -------
    dict
        Metadata: row counts, win rate, coverage/time-mix stats and the
        feature column list.
    """
    created = (
        pd.to_datetime(table["created_at"])
        if "created_at" in table.columns
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
        "expected_revenue_coverage": er_coverage,  # share with R > 0
        # age_cohort is null whenever age was missing/implausible (see
        # feature_engineering.features._derive_features) -- same signal the
        # old standalone age_missing flag carried, now folded into the
        # single age_cohort column. Key name kept as "age_missing_rate" so
        # existing manifest/Slack/artifact readers (flow.py) don't need to
        # change.
        "age_missing_rate": (
            float(table["age_cohort"].isna().mean())
            if "age_cohort" in table.columns and n
            else None
        ),
        "weekday_share": weekday_share,
        "weekend_share": weekend_share,
        "workday_rate": _rate("is_workday"),  # is_workday feature share
        "traffic_tier_distinct": (
            int(table["traffic_tier"].nunique())
            if "traffic_tier" in table.columns
            else None
        ),
        "feature_columns": [
            c for c in table.columns if c not in ("id", "created_at", "won_flag")
        ],
    }


def run_build_features(
    lead_type_id: int = 6,
    lead_type_name: str | None = None,
    window_days: int | None = None,
    log: logging.Logger | None = None,
) -> dict:
    """Build and save the training table for one lead type. Prefect-free.

    Inputs
    ------
    lead_type_id : int
        Lead type id to build (6=auto, 1=home).
    lead_type_name : str | None
        Override name; derived from the id when ``None``.
    window_days : int | None
        Rolling training window in days; config default when ``None``, 0=all.
    log : logging.Logger | None
        Logger to use; the module logger when ``None``.

    Returns
    -------
    dict
        Result with the built ``table`` + ``metadata`` (so a caller like the
        Prefect flow can report/notify) plus ``version`` / ``path`` / ``rows``
        / ``columns``.

    Raises
    ------
    storage.StorageError
        If STEP 1 (data-pull) has not produced any data.
    """
    log = log or logger
    name = lead_type_name or _lead_type_name(lead_type_id)
    window = window_days if window_days is not None else training_window_days()

    raw = _load_raw(window, log)
    raw_rows = int(len(raw))
    table = build_training_table(raw, lead_type_id=lead_type_id)
    metadata = build_metadata(table, lead_type_id, window, raw_rows=raw_rows)
    path = str(io.save_training_table(table, name, metadata=metadata))
    version = path.rsplit("/", 1)[-1].removesuffix(".parquet")
    log.info(
        "[%s] training table %s: %s rows, %s cols -> %s",
        name,
        version,
        len(table),
        table.shape[1],
        path,
    )
    return {
        "lead_type_name": name,
        "lead_type_id": lead_type_id,
        "window": window,
        "version": version,
        "path": path,
        "table": table,
        "metadata": metadata,
        "rows": int(len(table)),
        "columns": int(table.shape[1]),
    }


def main(argv=None):
    """Run build-features (STEP 2) directly — Prefect-free.

        python -m smarthub.feature_engineering.build --lead-type-id 6  # auto
        python -m smarthub.feature_engineering.build --lead-type-id 1  # home
        python -m smarthub.feature_engineering.build --window-days 0   # all

    Inputs
    ------
    argv : list[str] | None
        Argument vector; defaults to ``sys.argv`` when ``None``.

    Returns
    -------
    int
        Process exit code (0 success, 1 on missing data).
    """
    parser = argparse.ArgumentParser(
        description="Build the SmartHub training table (STEP 2). Prefect-free."
    )
    parser.add_argument(
        "--lead-type-id", type=int, default=6, help="6=auto (default), 1=home"
    )
    parser.add_argument(
        "--lead-type-name", default=None, help="override name (default: from id)"
    )
    parser.add_argument(
        "--window-days",
        type=int,
        default=None,
        help="rolling training window in days; 0=all data; default from ini",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    try:
        result = run_build_features(
            lead_type_id=args.lead_type_id,
            lead_type_name=args.lead_type_name,
            window_days=args.window_days,
        )
    except storage.StorageError as exc:
        logger.error("%s", exc)
        return 1
    print(
        f"Done. {result['rows']:,} rows, {result['columns']} cols "
        f"-> {result['path']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
