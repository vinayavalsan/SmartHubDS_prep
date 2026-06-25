"""Data loading and saving with friendly, actionable errors."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from . import paths, storage, transforms
from .config import StorageSettings

# Default on-disk locations, resolved relative to the project root.
DEFAULT_LEADS_PATH = paths.project_root() / "data" / "leads.parquet"
DEFAULT_MONITORING_PATH = paths.data_dir() / "etl" / "sample_data.csv"


class DataNotFoundError(FileNotFoundError):
    """Raised when an expected data file is missing."""


def _check_exists(path: Path, hint: str) -> None:
    if not path.exists():
        raise DataNotFoundError(f"Expected data file not found: {path}\n{hint}")


def load_leads(path: str | os.PathLike[str] | None = None) -> pd.DataFrame:
    """Load the cleaned, enriched leads frame.

    With no ``path``, reads from the configured storage backend(s) — see
    ``STORAGE_BACKEND`` in the environment. Pass an explicit ``path`` to
    force-load a specific Parquet file instead.
    """
    if path is not None:
        resolved = paths.resolve(path)
        _check_exists(resolved, "Run the data pull first: python -m smarthub.data_pull")
        return transforms.prepare_leads_frame(pd.read_parquet(resolved))
    try:
        raw = storage.load_leads_raw(StorageSettings.from_env())
    except storage.StorageError as exc:
        raise DataNotFoundError(str(exc)) from exc
    return transforms.prepare_leads_frame(raw)


def load_leads_window(days: int) -> pd.DataFrame:
    """Load only the most recent ``days`` of data, cleaned/enriched.

    Use this for rolling-recency training reads (CONTEXT §7)."""
    try:
        raw = storage.load_window_raw(StorageSettings.from_env(), days)
    except storage.StorageError as exc:
        raise DataNotFoundError(str(exc)) from exc
    return transforms.prepare_leads_frame(raw)


def save_leads(df: pd.DataFrame, path: str | os.PathLike[str] | None = None) -> Path:
    """Write the leads frame to parquet, creating parent dirs as needed."""
    resolved = paths.resolve(path) if path is not None else DEFAULT_LEADS_PATH
    resolved.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(resolved, index=False)
    return resolved


def load_monitoring(path: str | os.PathLike[str] | None = None) -> pd.DataFrame:
    """Load the monitoring CSV with parsed datetimes and coerced numerics."""
    resolved = paths.resolve(path) if path is not None else DEFAULT_MONITORING_PATH
    _check_exists(resolved, "Check the data/etl directory for the ETL export.")
    df = pd.read_csv(resolved)
    for col in ("datetime_min", "datetime_max"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
    transforms.coerce_numeric(df, transforms.MONITORING_NUMERIC_COLS)
    return df
