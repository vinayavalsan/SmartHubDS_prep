"""Data loading and saving with friendly, actionable errors."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from smarthub.core import paths
from smarthub.core.config import StorageSettings
from smarthub.data import storage, transforms

# Default on-disk locations, resolved relative to the project root.
DEFAULT_LEADS_PATH = paths.project_root() / "data" / "leads.parquet"
TRAINING_DIR = paths.data_dir() / "training"


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


def training_dir(lead_type_name: str) -> Path:
    """Per-lead-type training folder: data/training/<name>/."""
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
    """Write a **versioned** training table: data/training/<name>/<version>.parquet.

    Each build is kept (not overwritten) so a model can be traced to its exact
    training snapshot. ``version`` defaults to a UTC timestamp. When ``metadata``
    is given, a ``<version>.json`` manifest is written beside the parquet
    describing what data went in (lead type, window, date range, row count, …).
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


def load_training_metadata(
    lead_type_name: str, version: str | None = None
) -> dict:
    """Load the manifest for a training version (defaults to the latest)."""
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
    """All saved version ids for a lead type, oldest first."""
    return sorted(p.stem for p in training_dir(lead_type_name).glob("*.parquet"))


def load_training_table(
    lead_type_name: str, version: str | None = None
) -> pd.DataFrame:
    """Load a training table; defaults to the latest version."""
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
