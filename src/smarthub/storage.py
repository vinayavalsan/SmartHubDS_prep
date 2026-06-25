"""Persistence for accumulated pulls — DuckDB and/or partitioned Parquet.

Both backends upsert/dedupe on ``id`` because pulls run on *overlapping*
windows so that late-resolving outcomes (``won``, ``rev``, ``accepted_listings``,
listing payouts) get updated in place (CONTEXT.md §4, §7).

- **DuckDB**: a single file, native upsert, SQL window reads.
- **Partitioned Parquet**: one file per calendar day, laid out as
  ``<root>/YYYY/MM/DD-MM-YYYY.parquet``; same-day pulls are merged and deduped.

Which backend(s) are used is controlled by ``StorageSettings`` (env-driven).
"""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path

import duckdb
import pandas as pd

from . import paths
from .config import StorageSettings
from .logging_utils import get_logger

logger = get_logger(__name__)

LEADS_TABLE = "lead_pings"
KEY = "id"
UPDATED_AT = "updated_at"


class StorageError(RuntimeError):
    """Raised on storage schema / state problems."""


def _dedupe(df: pd.DataFrame, key: str = KEY) -> pd.DataFrame:
    """Keep one row per key. Prefer the most recently updated row when an
    ``updated_at`` column is present; otherwise keep the last occurrence
    (callers concat existing-then-incoming so 'last' favours the new pull)."""
    if key not in df.columns:
        return df
    if UPDATED_AT in df.columns:
        df = df.sort_values(UPDATED_AT, na_position="first")
    return df.drop_duplicates(subset=key, keep="last")


# ---------------------------------------------------------------------------
# DuckDB backend
# ---------------------------------------------------------------------------


def duckdb_path(path: str | os.PathLike[str] | None = None) -> Path:
    if path is not None:
        return paths.resolve(path)
    return paths.data_dir() / "smarthub.duckdb"


def _connect(path: str | os.PathLike[str] | None = None) -> duckdb.DuckDBPyConnection:
    resolved = duckdb_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(resolved))


def _table_exists(con: duckdb.DuckDBPyConnection, table: str) -> bool:
    row = con.execute(
        "SELECT count(*) FROM information_schema.tables WHERE table_name = ?",
        [table],
    ).fetchone()
    return bool(row and row[0])


def _table_columns(con: duckdb.DuckDBPyConnection, table: str) -> list[str]:
    return [r[1] for r in con.execute(f'PRAGMA table_info("{table}")').fetchall()]


def append_duckdb(
    df: pd.DataFrame,
    table: str = LEADS_TABLE,
    key: str = KEY,
    path: str | os.PathLike[str] | None = None,
) -> int:
    """Upsert ``df`` into the DuckDB ``table`` keyed on ``key``; return row count."""
    if df.empty:
        logger.info("append_duckdb: empty frame, nothing to write")
        return 0
    if key not in df.columns:
        raise StorageError(f"Key column '{key}' not present in dataframe")

    con = _connect(path)
    try:
        con.register("incoming", df)
        if not _table_exists(con, table):
            con.execute(f'CREATE TABLE "{table}" AS SELECT * FROM incoming')
            logger.info("Created DuckDB table '%s' from first pull", table)
        else:
            # Schema evolution: add any new incoming columns to the table
            # (existing rows get NULL for them). Columns missing from this pull
            # are filled with NULL on insert via BY NAME.
            existing = set(_table_columns(con, table))
            new_cols = [c for c in df.columns if c not in existing]
            if new_cols:
                types = {
                    r[0]: r[1]
                    for r in con.execute("DESCRIBE SELECT * FROM incoming").fetchall()
                }
                for col in new_cols:
                    con.execute(
                        f'ALTER TABLE "{table}" ADD COLUMN "{col}" {types[col]}'
                    )
                logger.info("Added new columns to '%s': %s", table, new_cols)
            con.execute(
                f'DELETE FROM "{table}" WHERE {key} IN (SELECT {key} FROM incoming)'
            )
            con.execute(f'INSERT INTO "{table}" BY NAME SELECT * FROM incoming')
        total = con.execute(f'SELECT count(*) FROM "{table}"').fetchone()[0]
        logger.info("DuckDB '%s': upserted %s rows (now %s)", table, len(df), total)
        return int(total)
    finally:
        con.unregister("incoming")
        con.close()


def read_duckdb_table(
    table: str = LEADS_TABLE, path: str | os.PathLike[str] | None = None
) -> pd.DataFrame:
    con = _connect(path)
    try:
        if not _table_exists(con, table):
            raise StorageError(f"Table '{table}' not found in {duckdb_path(path)}.")
        return con.execute(f'SELECT * FROM "{table}"').df()
    finally:
        con.close()


def read_duckdb_window(
    days: int,
    table: str = LEADS_TABLE,
    time_col: str = "created_at",
    path: str | os.PathLike[str] | None = None,
) -> pd.DataFrame:
    """Rows within the most recent ``days``, anchored on ``max(time_col)``."""
    con = _connect(path)
    try:
        if not _table_exists(con, table):
            raise StorageError(f"Table '{table}' not found in {duckdb_path(path)}.")
        latest = con.execute(f'SELECT max("{time_col}") FROM "{table}"').fetchone()[0]
        if latest is None:
            return con.execute(f'SELECT * FROM "{table}" LIMIT 0').df()
        return con.execute(
            f'SELECT * FROM "{table}" WHERE "{time_col}" >= ? - INTERVAL (?) DAY',
            [latest, days],
        ).df()
    finally:
        con.close()


def duckdb_exists(
    table: str = LEADS_TABLE, path: str | os.PathLike[str] | None = None
) -> bool:
    if not duckdb_path(path).exists():
        return False
    con = _connect(path)
    try:
        return _table_exists(con, table)
    finally:
        con.close()


# ---------------------------------------------------------------------------
# Partitioned Parquet backend  (<root>/YYYY/MM/DD-MM-YYYY.parquet)
# ---------------------------------------------------------------------------


def _partition_path(root: Path, day: date) -> Path:
    return root / f"{day.year:04d}" / f"{day.month:02d}" / f"{day:%d-%m-%Y}.parquet"


def _row_days(df: pd.DataFrame, date_col: str) -> pd.Series:
    """Per-row partition day: ``date_col`` if usable, else ``created_at``."""
    fallback = pd.to_datetime(df.get("created_at"), errors="coerce")
    if date_col in df.columns:
        primary = pd.to_datetime(df[date_col], errors="coerce")
        primary = primary.fillna(fallback)
    else:
        primary = fallback
    return primary.dt.normalize()


def append_parquet(
    df: pd.DataFrame,
    root: str | os.PathLike[str],
    date_col: str = "created_at",
    key: str = KEY,
) -> int:
    """Write ``df`` into per-day Parquet files, merging+deduping same-day data.

    Returns the number of incoming rows written.
    """
    if df.empty:
        logger.info("append_parquet: empty frame, nothing to write")
        return 0

    root_path = paths.resolve(root)
    days = _row_days(df, date_col)
    valid = days.notna()
    if not valid.all():
        logger.warning(
            "append_parquet: dropping %s rows with no resolvable date",
            int((~valid).sum()),
        )
    written = 0
    for day_ts, group in df[valid].groupby(days[valid]):
        day = day_ts.date()
        target = _partition_path(root_path, day)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            combined = pd.concat([pd.read_parquet(target), group], ignore_index=True)
        else:
            combined = group
        _dedupe(combined, key).to_parquet(target, index=False)
        written += len(group)
        logger.info("Parquet partition %s: +%s rows", target.name, len(group))
    return written


def read_parquet_dataset(root: str | os.PathLike[str]) -> pd.DataFrame:
    """Read and concatenate every per-day Parquet file under ``root``."""
    root_path = paths.resolve(root)
    files = sorted(root_path.glob("*/*/*.parquet"))
    if not files:
        raise StorageError(f"No Parquet files found under {root_path}.")
    return pd.concat((pd.read_parquet(f) for f in files), ignore_index=True)


def parquet_exists(root: str | os.PathLike[str]) -> bool:
    return any(paths.resolve(root).glob("*/*/*.parquet"))


# ---------------------------------------------------------------------------
# Settings-driven facade
# ---------------------------------------------------------------------------


def save_pull(df: pd.DataFrame, settings: StorageSettings) -> dict[str, int]:
    """Persist a pull to whichever backend(s) the settings enable."""
    results: dict[str, int] = {}
    if settings.use_duckdb:
        results["duckdb_rows"] = append_duckdb(df, path=settings.duckdb_path)
    if settings.use_parquet:
        results["parquet_rows"] = append_parquet(
            df, settings.parquet_dir, settings.partition_date_col
        )
    return results


def load_leads_raw(settings: StorageSettings) -> pd.DataFrame:
    """Load the full accumulated leads frame from the preferred backend."""
    if settings.use_duckdb and duckdb_exists(path=settings.duckdb_path):
        return read_duckdb_table(path=settings.duckdb_path)
    if settings.use_parquet and parquet_exists(settings.parquet_dir):
        return read_parquet_dataset(settings.parquet_dir)
    raise StorageError(
        "No stored data found. Run the pull first: python -m smarthub.data_pull"
    )


def load_window_raw(settings: StorageSettings, days: int) -> pd.DataFrame:
    """Load the most recent ``days`` of leads from the preferred backend."""
    if settings.use_duckdb and duckdb_exists(path=settings.duckdb_path):
        return read_duckdb_window(days, path=settings.duckdb_path)
    if settings.use_parquet and parquet_exists(settings.parquet_dir):
        df = read_parquet_dataset(settings.parquet_dir)
        ts = pd.to_datetime(df["created_at"], errors="coerce")
        cutoff = ts.max() - pd.Timedelta(days=days)
        return df[ts >= cutoff]
    raise StorageError(
        "No stored data found. Run the pull first: python -m smarthub.data_pull"
    )
