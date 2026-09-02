"""Data loading and saving with friendly, actionable errors."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from smarthub.core import paths, storage, transforms
from smarthub.core.config import StorageSettings

# Default on-disk locations, resolved relative to the project root.
DEFAULT_LEADS_PATH = paths.project_root() / "data" / "leads.parquet"
TRAINING_DIR = paths.data_dir() / "training_datasets"


class DataNotFoundError(FileNotFoundError):
    """Raised when an expected data file is missing."""


def _check_exists(path: Path, hint: str) -> None:
    """Raise :class:`DataNotFoundError` with ``hint`` if ``path`` is missing."""
    if not path.exists():
        raise DataNotFoundError(f"Expected data file not found: {path}\n{hint}")


def load_leads(path: str | os.PathLike[str] | None = None) -> pd.DataFrame:
    """Load the cleaned, enriched leads frame.

    With no ``path``, reads from the configured storage backend(s) (see
    ``STORAGE_BACKEND``). Pass an explicit ``path`` to force-load a specific
    Parquet file instead.

    Inputs
    ------
    path : str | os.PathLike[str] | None
        Optional Parquet file to load instead of the storage backend.

    Returns
    -------
    pandas.DataFrame
        The prepared leads frame.

    Raises
    ------
    DataNotFoundError
        If the given path or the storage backend has no data.
    """
    if path is not None:
        resolved = paths.resolve(path)
        _check_exists(
            resolved,
            "Run the data pull first (STEP 1): "
            "smarthub-pull --min-created-at ... --max-created-at ...",
        )
        return transforms.prepare_leads_frame(pd.read_parquet(resolved))
    try:
        raw = storage.load_leads_raw(StorageSettings.from_env())
    except storage.StorageError as exc:
        raise DataNotFoundError(str(exc)) from exc
    return transforms.prepare_leads_frame(raw)


def load_leads_window(days: int) -> pd.DataFrame:
    """Load only the most recent ``days`` of data, cleaned and enriched.

    Use for rolling-recency training reads (CONTEXT §7).

    Inputs
    ------
    days : int
        Size of the trailing window to load.

    Returns
    -------
    pandas.DataFrame
        The prepared leads frame for the window.

    Raises
    ------
    DataNotFoundError
        If the storage backend has no data.
    """
    try:
        raw = storage.load_window_raw(StorageSettings.from_env(), days)
    except storage.StorageError as exc:
        raise DataNotFoundError(str(exc)) from exc
    return transforms.prepare_leads_frame(raw)


def load_monitoring_window(days: int | None = None) -> pd.DataFrame:
    """Load the persisted prediction-monitoring dataset for ``monitoring_app``.

    This is the joined prediction-log + lead/outcome data written by the
    ``--include-prediction-logs`` pull. Reads straight from storage — no
    prediction-log DB connection required.

    Inputs
    ------
    days : int | None
        Trailing window by ``created_at``; full dataset when ``None``.

    Returns
    -------
    pandas.DataFrame
        Monitoring rows (empty when the pull has not populated it yet).
    """
    return storage.load_monitoring(StorageSettings.from_env(), days=days)


def save_leads(df: pd.DataFrame, path: str | os.PathLike[str] | None = None) -> Path:
    """Write the leads frame to Parquet, creating parent dirs as needed.

    Inputs
    ------
    df : pandas.DataFrame
        The leads frame to write.
    path : str | os.PathLike[str] | None
        Destination path; defaults to ``DEFAULT_LEADS_PATH`` when omitted.

    Returns
    -------
    pathlib.Path
        The resolved path the frame was written to.
    """
    resolved = paths.resolve(path) if path is not None else DEFAULT_LEADS_PATH
    resolved.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(resolved, index=False)
    return resolved


def training_dir(lead_type_name: str) -> Path:
    """Return the per-lead-type training folder ``data/training_datasets/<name>/``.

    Inputs
    ------
    lead_type_name : str
        Lead type name; lower-cased and stripped for the folder name.

    Returns
    -------
    pathlib.Path
        The training folder path for the lead type.
    """
    return TRAINING_DIR / lead_type_name.strip().lower()


def _version_stamp() -> str:
    """UTC, filesystem-safe, lexicographically sortable version id."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")


def save_training_table(
    df: pd.DataFrame,
    lead_type_name: str,
    version: str | None = None,
    metadata: dict | None = None,
) -> Path:
    """Write a versioned training table for a lead type.

    Writes ``data/training_datasets/<name>/<version>.parquet``. Each build is kept
    (never overwritten) so a model can be traced to its exact training
    snapshot. When ``metadata`` is given, a ``<version>.json`` manifest is
    written beside the Parquet describing what data went in.

    Inputs
    ------
    df : pandas.DataFrame
        The training table to write.
    lead_type_name : str
        Lead type the table belongs to.
    version : str | None
        Version id; defaults to a UTC timestamp when omitted.
    metadata : dict | None
        Extra manifest fields; when given, a JSON manifest is written too.

    Returns
    -------
    pathlib.Path
        The path the Parquet table was written to.
    """
    folder = training_dir(lead_type_name)
    folder.mkdir(parents=True, exist_ok=True)
    version = version or _version_stamp()
    target = folder / f"{version}.parquet"
    df.to_parquet(target, index=False)

    if metadata is not None:
        manifest = {
            "version": version,
            "lead_type": lead_type_name,
            "rows": int(len(df)),
            "columns": list(df.columns),
            **metadata,
        }
        (folder / f"{version}.json").write_text(
            json.dumps(manifest, indent=2, default=str)
        )
    return target


def load_training_metadata(lead_type_name: str, version: str | None = None) -> dict:
    """Load the manifest for a training version (defaults to the latest).

    Inputs
    ------
    lead_type_name : str
        Lead type whose manifest to load.
    version : str | None
        Version id; defaults to the latest saved version.

    Returns
    -------
    dict
        The parsed manifest contents.

    Raises
    ------
    DataNotFoundError
        If no training tables or manifest exist for the version.
    """
    folder = training_dir(lead_type_name)
    if version is None:
        versions = training_versions(lead_type_name)
        if not versions:
            raise DataNotFoundError(f"No training tables under {folder}.")
        version = versions[-1]
    manifest = folder / f"{version}.json"
    _check_exists(manifest, "No manifest for that version.")
    return json.loads(manifest.read_text())


def training_versions(lead_type_name: str) -> list[str]:
    """List all saved version ids for a lead type, oldest first.

    Inputs
    ------
    lead_type_name : str
        Lead type to list versions for.

    Returns
    -------
    list[str]
        Sorted version ids (oldest first); empty when none exist.
    """
    return sorted(p.stem for p in training_dir(lead_type_name).glob("*.parquet"))


def load_training_table(
    lead_type_name: str, version: str | None = None
) -> pd.DataFrame:
    """Load a training table, defaulting to the latest version.

    Inputs
    ------
    lead_type_name : str
        Lead type whose table to load.
    version : str | None
        Version id; defaults to the latest saved version.

    Returns
    -------
    pandas.DataFrame
        The loaded training table.

    Raises
    ------
    DataNotFoundError
        If no matching training table exists.
    """
    folder = training_dir(lead_type_name)
    if version is not None:
        target = folder / f"{version}.parquet"
    else:
        files = sorted(folder.glob("*.parquet"))
        if not files:
            raise DataNotFoundError(
                f"No training tables under {folder}. Run the feature build first."
            )
        target = files[-1]
    _check_exists(target, "Run the feature build first (build-features deployment).")
    return pd.read_parquet(target)
