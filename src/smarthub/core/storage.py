"""Persistence for accumulated pulls — DuckDB and/or partitioned Parquet.

Both backends upsert/dedupe on ``id`` because pulls run on overlapping
windows, so late-resolving outcomes get updated in place (CONTEXT.md §4, §7).
DuckDB is a single file with native upsert and SQL window reads; the Parquet
backend writes one file per calendar day
(``<root>/YYYY/MM/DD-MM-YYYY.parquet``), merging and deduping same-day pulls.
The enabled backend(s) come from ``StorageSettings`` (env-driven).
"""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path

import duckdb
import pandas as pd

from smarthub.core import paths
from smarthub.core.config import StorageSettings
from smarthub.core.logging_utils import get_logger

logger = get_logger(__name__)

LEADS_TABLE = "lead_pings"
KEY = "id"
UPDATED_AT = "updated_at"


class StorageError(RuntimeError):
    """Raised on storage schema / state problems."""


# Shown whenever a downstream step (e.g. build-features) is run before any
# data exists. The pipeline order is: **1) data-pull  →  2) build-features**.
NO_DATA_MESSAGE = (
    "No lead data found in storage.\n"
    "The pipeline must run in order: STEP 1 = data-pull, then "
    "STEP 2 = build-features.\n"
    "It looks like build-features ran first. Run the data pull, wait for it "
    "to finish, then re-run build-features.\n"
    "  - Prefect: run deployment 'smarthub-data-pull/data-pull' "
    "(auto and home), or\n"
    "  - CLI:     smarthub-pull --min-created-at <YYYY-MM-DD HH:MM:SS> "
    "--max-created-at <YYYY-MM-DD HH:MM:SS>"
)


def _dedupe(df: pd.DataFrame, key: str = KEY) -> pd.DataFrame:
    """Keep one row per key, preferring the most recent update.

    Uses ``updated_at`` when present; otherwise keeps the last occurrence
    (callers concat existing-then-incoming, so 'last' favours the new pull).

    Inputs
    ------
    df : pandas.DataFrame
        Rows to deduplicate; returned unchanged if ``key`` is absent.
    key : str
        Column identifying a unique row.

    Returns
    -------
    pandas.DataFrame
        The deduplicated frame.
    """
    if key not in df.columns:
        return df
    if UPDATED_AT in df.columns:
        df = df.sort_values(UPDATED_AT, na_position="first")
    return df.drop_duplicates(subset=key, keep="last")


# ---------------------------------------------------------------------------
# DuckDB backend
# ---------------------------------------------------------------------------


def duckdb_path(path: str | os.PathLike[str] | None = None) -> Path:
    """Resolve the DuckDB file path (defaults to ``data/raw_datasets/leads.duckdb``).

    Inputs
    ------
    path : str | os.PathLike[str] | None
        Explicit path; the default location is used when omitted.

    Returns
    -------
    pathlib.Path
        The resolved DuckDB file path.
    """
    if path is not None:
        return paths.resolve(path)
    return paths.data_dir() / "raw_datasets" / "leads.duckdb"


def _connect(path: str | os.PathLike[str] | None = None) -> duckdb.DuckDBPyConnection:
    """Open a DuckDB connection, creating parent dirs as needed."""
    resolved = duckdb_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(resolved))


def _table_exists(con: duckdb.DuckDBPyConnection, table: str) -> bool:
    """Return True when ``table`` exists in the database."""
    row = con.execute(
        "SELECT count(*) FROM information_schema.tables WHERE table_name = ?",
        [table],
    ).fetchone()
    return bool(row and row[0])


def _table_columns(con: duckdb.DuckDBPyConnection, table: str) -> list[str]:
    """Return the column names of ``table``."""
    return [r[1] for r in con.execute(f'PRAGMA table_info("{table}")').fetchall()]


def _table_column_types(con: duckdb.DuckDBPyConnection, table: str) -> dict[str, str]:
    """Map column name -> DuckDB type string for an existing table."""
    return {
        r[1]: str(r[2]).upper()
        for r in con.execute(f'PRAGMA table_info("{table}")').fetchall()
    }


# DuckDB timestamp type -> pandas datetime unit. DuckDB won't downcast timestamp
# precision on INSERT (e.g. ns -> s is "unimplemented"), but pandas can, so we
# align the incoming frame's precision to the existing column before inserting.
_DUCKDB_TS_UNIT = {
    "TIMESTAMP_S": "s",
    "TIMESTAMP_MS": "ms",
    "TIMESTAMP": "us",
    "DATETIME": "us",
    "TIMESTAMP WITH TIME ZONE": "us",
    "TIMESTAMPTZ": "us",
    "TIMESTAMP_NS": "ns",
}


def _align_datetime_precision(
    df: pd.DataFrame, table_types: dict[str, str]
) -> pd.DataFrame:
    """Cast incoming datetime columns to the existing table's precision.

    Prevents a DuckDB "Unimplemented type for cast (TIMESTAMP_NS ->
    TIMESTAMP_S)" error on INSERT when a column was first created at a
    coarser precision than the nanosecond datetimes pandas produces.

    Inputs
    ------
    df : pandas.DataFrame
        Incoming frame to align.
    table_types : dict[str, str]
        Column name to DuckDB type string for the existing table.

    Returns
    -------
    pandas.DataFrame
        The frame with datetime columns cast to the table's precision (a
        copy only when a change was needed).
    """
    out = df
    copied = False
    for col, dtype_str in table_types.items():
        if col not in df.columns or not pd.api.types.is_datetime64_any_dtype(df[col]):
            continue
        unit = _DUCKDB_TS_UNIT.get(dtype_str)
        if unit is None or getattr(df[col].dtype, "unit", None) == unit:
            continue
        if not copied:
            out = df.copy()
            copied = True
        out[col] = out[col].astype(f"datetime64[{unit}]")
    return out


def append_duckdb(
    df: pd.DataFrame,
    table: str = LEADS_TABLE,
    key: str = KEY,
    path: str | os.PathLike[str] | None = None,
) -> int:
    """Upsert ``df`` into a DuckDB table, evolving the schema as needed.

    Creates the table on first write; on later writes aligns datetime
    precision, adds any new columns, then deletes-and-reinserts matching
    keys.

    Inputs
    ------
    df : pandas.DataFrame
        Rows to upsert; an empty frame is a no-op.
    table : str
        Target table name.
    key : str
        Key column used to dedupe/upsert.
    path : str | os.PathLike[str] | None
        DuckDB file path; the default location is used when omitted.

    Returns
    -------
    int
        Total row count in the table after the upsert.

    Raises
    ------
    StorageError
        If ``key`` is not present in ``df``.
    """
    if df.empty:
        logger.info("append_duckdb: empty frame, nothing to write")
        return 0
    if key not in df.columns:
        raise StorageError(f"Key column '{key}' not present in dataframe")

    con = _connect(path)
    try:
        if not _table_exists(con, table):
            con.register("incoming", df)
            con.execute(f'CREATE TABLE "{table}" AS SELECT * FROM incoming')
            logger.info("Created DuckDB table '%s' from first pull", table)
        else:
            # Match the incoming datetime precision to the table's, so DuckDB
            # doesn't hit an unimplemented ns->s cast on INSERT.
            df = _align_datetime_precision(df, _table_column_types(con, table))
            con.register("incoming", df)
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
        try:
            con.unregister("incoming")
        except Exception:  # noqa: BLE001 - nothing registered / already gone
            pass
        con.close()


def _projection(con, table: str, columns) -> str:
    """Build the SQL column list for a projected read.

    Only columns that exist in the table are selected, so a caller can pass a
    superset without erroring on columns that were never pulled.

    Inputs
    ------
    con : duckdb.DuckDBPyConnection
        Open connection used to inspect the table.
    table : str
        Table whose columns constrain the projection.
    columns : list[str] | None
        Requested columns; ``*`` is returned when none are valid.

    Returns
    -------
    str
        A SQL column list, or ``*``.
    """
    if not columns:
        return "*"
    have = set(_table_columns(con, table))
    picked = [c for c in columns if c in have]
    if not picked:
        return "*"
    return ", ".join(f'"{c}"' for c in picked)


def read_duckdb_table(
    table: str = LEADS_TABLE,
    path: str | os.PathLike[str] | None = None,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Read a DuckDB table into a DataFrame, optionally projecting columns.

    Inputs
    ------
    table : str
        Table to read.
    path : str | os.PathLike[str] | None
        DuckDB file path; the default location is used when omitted.
    columns : list[str] | None
        Columns to project; only those present are selected.

    Returns
    -------
    pandas.DataFrame
        The table contents.

    Raises
    ------
    StorageError
        If the table does not exist.
    """
    con = _connect(path)
    try:
        if not _table_exists(con, table):
            raise StorageError(f"Table '{table}' not found in {duckdb_path(path)}.")
        cols_sql = _projection(con, table, columns)
        return con.execute(f'SELECT {cols_sql} FROM "{table}"').df()
    finally:
        con.close()


def read_duckdb_window(
    days: int,
    table: str = LEADS_TABLE,
    time_col: str = "created_at",
    path: str | os.PathLike[str] | None = None,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Read rows within the most recent ``days``, anchored on max(time_col).

    Inputs
    ------
    days : int
        Size of the trailing window in days.
    table : str
        Table to read.
    time_col : str
        Timestamp column used to anchor and filter the window.
    path : str | os.PathLike[str] | None
        DuckDB file path; the default location is used when omitted.
    columns : list[str] | None
        Columns to project; filtering still uses ``time_col`` even when it
        is not selected, keeping peak memory low on wide tables.

    Returns
    -------
    pandas.DataFrame
        Rows within the window (empty when the table has no timestamps).

    Raises
    ------
    StorageError
        If the table does not exist.
    """
    con = _connect(path)
    try:
        if not _table_exists(con, table):
            raise StorageError(f"Table '{table}' not found in {duckdb_path(path)}.")
        cols_sql = _projection(con, table, columns)
        latest = con.execute(f'SELECT max("{time_col}") FROM "{table}"').fetchone()[0]
        if latest is None:
            return con.execute(f'SELECT {cols_sql} FROM "{table}" LIMIT 0').df()
        return con.execute(
            f'SELECT {cols_sql} FROM "{table}" '
            f'WHERE "{time_col}" >= ? - INTERVAL (?) DAY',
            [latest, days],
        ).df()
    finally:
        con.close()


def duckdb_exists(
    table: str = LEADS_TABLE, path: str | os.PathLike[str] | None = None
) -> bool:
    """Return True when the DuckDB file and ``table`` both exist.

    Inputs
    ------
    table : str
        Table to check for.
    path : str | os.PathLike[str] | None
        DuckDB file path; the default location is used when omitted.

    Returns
    -------
    bool
        True when the file exists and contains the table.
    """
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
    """Return the Parquet partition path for ``day`` under ``root``."""
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
    """Write ``df`` into per-day Parquet files, merging same-day data.

    Same-day partitions are read back, concatenated, deduped on ``key`` and
    rewritten. Rows with no resolvable date are dropped.

    Inputs
    ------
    df : pandas.DataFrame
        Rows to write; an empty frame is a no-op.
    root : str | os.PathLike[str]
        Root directory of the partitioned dataset.
    date_col : str
        Column used to choose each row's partition day.
    key : str
        Key column used to dedupe within a partition.

    Returns
    -------
    int
        The number of incoming rows written.
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


def read_parquet_dataset(
    root: str | os.PathLike[str], columns: list[str] | None = None
) -> pd.DataFrame:
    """Read and concatenate every per-day Parquet file under ``root``.

    Inputs
    ------
    root : str | os.PathLike[str]
        Root directory of the partitioned dataset.
    columns : list[str] | None
        Columns to project; only those present in a file are loaded,
        lowering peak memory on wide datasets.

    Returns
    -------
    pandas.DataFrame
        All partitions concatenated into one frame.

    Raises
    ------
    StorageError
        If no Parquet files are found under ``root``.
    """
    root_path = paths.resolve(root)
    files = sorted(root_path.glob("*/*/*.parquet"))
    if not files:
        raise StorageError(f"No Parquet files found under {root_path}.")

    def _read(f):
        if not columns:
            return pd.read_parquet(f)
        available = set(pd.read_parquet(f, columns=[]).columns)
        picked = [c for c in columns if c in available]
        return pd.read_parquet(f, columns=picked or None)

    return pd.concat((_read(f) for f in files), ignore_index=True)


def read_parquet_window(
    root: str | os.PathLike[str], days: int, columns: list[str] | None = None
) -> pd.DataFrame:
    """Read only the most recent ``days`` of a day-partitioned Parquet dataset.

    Each partition file holds one day, so this reads just the last ~``days``
    files (plus a small margin for partial days / ordering) instead of the whole
    dataset, then trims to the exact cutoff. This keeps callers -- notably the
    dashboards -- from loading all accumulated history into memory (which for a
    long-running deployment can be millions of rows and OOM the container).
    """
    root_path = paths.resolve(root)
    files = sorted(root_path.glob("*/*/*.parquet"))
    if not files:
        raise StorageError(f"No Parquet files found under {root_path}.")
    recent = files[-(days + 2) :] if days and days > 0 else files

    def _read(f):
        if not columns:
            return pd.read_parquet(f)
        available = set(pd.read_parquet(f, columns=[]).columns)
        picked = [c for c in columns if c in available]
        return pd.read_parquet(f, columns=picked or None)

    df = pd.concat((_read(f) for f in recent), ignore_index=True)
    if days and days > 0 and "created_at" in df.columns:
        ts = pd.to_datetime(df["created_at"], errors="coerce")
        cutoff = ts.max() - pd.Timedelta(days=days)
        df = df[ts >= cutoff]
    return df


def parquet_exists(root: str | os.PathLike[str]) -> bool:
    """Return True when any Parquet partition exists under ``root``."""
    return any(paths.resolve(root).glob("*/*/*.parquet"))


# ---------------------------------------------------------------------------
# Settings-driven facade
# ---------------------------------------------------------------------------


def parquet_partition_paths(
    df: pd.DataFrame,
    root: str | os.PathLike[str],
    date_col: str = "created_at",
) -> list[str]:
    """Return the per-day Parquet file paths a pull of ``df`` writes into.

    Mirrors ``append_parquet``'s partitioning so callers (e.g.
    notifications) can report which files were touched. Rows with no
    resolvable date are ignored, matching the writer.

    Inputs
    ------
    df : pandas.DataFrame
        The frame that would be written; empty yields an empty list.
    root : str | os.PathLike[str]
        Root directory of the partitioned dataset.
    date_col : str
        Column used to choose each row's partition day.

    Returns
    -------
    list[str]
        The distinct partition file paths, sorted by day.
    """
    if df.empty:
        return []
    root_path = paths.resolve(root)
    days = _row_days(df, date_col)
    unique_days = sorted({d.date() for d in days[days.notna()]})
    return [str(_partition_path(root_path, day)) for day in unique_days]


def save_pull(df: pd.DataFrame, settings: StorageSettings) -> dict[str, object]:
    """Persist a pull to whichever backend(s) the settings enable.

    Inputs
    ------
    df : pandas.DataFrame
        The pulled rows to persist.
    settings : StorageSettings
        Selects the enabled backend(s) and their locations.

    Returns
    -------
    dict[str, object]
        Row counts and the storage locations written: ``duckdb_rows`` /
        ``duckdb_path`` and/or ``parquet_rows`` / ``parquet_paths``.
    """
    results: dict[str, object] = {}
    if settings.use_duckdb:
        results["duckdb_rows"] = append_duckdb(df, path=settings.duckdb_path)
        results["duckdb_path"] = str(duckdb_path(settings.duckdb_path))
    if settings.use_parquet:
        results["parquet_rows"] = append_parquet(
            df, settings.parquet_dir, settings.partition_date_col
        )
        results["parquet_paths"] = parquet_partition_paths(
            df, settings.parquet_dir, settings.partition_date_col
        )
    return results


def load_leads_raw(
    settings: StorageSettings, columns: list[str] | None = None
) -> pd.DataFrame:
    """Load the full accumulated leads frame from the preferred backend.

    Prefers DuckDB, then Parquet.

    Inputs
    ------
    settings : StorageSettings
        Selects the enabled backend(s) and their locations.
    columns : list[str] | None
        Columns to project, to keep peak memory low.

    Returns
    -------
    pandas.DataFrame
        The accumulated leads frame.

    Raises
    ------
    StorageError
        If no data exists in any enabled backend.
    """
    if settings.use_duckdb and duckdb_exists(path=settings.duckdb_path):
        return read_duckdb_table(path=settings.duckdb_path, columns=columns)
    if settings.use_parquet and parquet_exists(settings.parquet_dir):
        return read_parquet_dataset(settings.parquet_dir, columns=columns)
    raise StorageError(NO_DATA_MESSAGE)


def load_window_raw(
    settings: StorageSettings, days: int, columns: list[str] | None = None
) -> pd.DataFrame:
    """Load the most recent ``days`` of leads from the preferred backend.

    Prefers DuckDB, then Parquet.

    Inputs
    ------
    settings : StorageSettings
        Selects the enabled backend(s) and their locations.
    days : int
        Size of the trailing window in days.
    columns : list[str] | None
        Columns to project, to keep peak memory low.

    Returns
    -------
    pandas.DataFrame
        The leads within the window.

    Raises
    ------
    StorageError
        If no data exists in any enabled backend.
    """
    if settings.use_duckdb and duckdb_exists(path=settings.duckdb_path):
        return read_duckdb_window(days, path=settings.duckdb_path, columns=columns)
    if settings.use_parquet and parquet_exists(settings.parquet_dir):
        # Partition-aware: read only the recent day-files, not the whole dataset.
        return read_parquet_window(settings.parquet_dir, days, columns=columns)
    raise StorageError(NO_DATA_MESSAGE)
