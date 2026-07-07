"""Project path resolution.

Every path used by the package is resolved relative to the project root rather
than the current working directory, so scripts and dashboards behave the same
no matter where they are launched from.
"""

from __future__ import annotations

import os
from pathlib import Path

# src/smarthub/core/paths.py -> project root is four levels up
# (core -> smarthub -> src -> repo root).
PROJECT_ROOT = Path(__file__).resolve().parents[3]


def project_root() -> Path:
    """Return the repository root.

    Honours the ``SMARTHUB_ROOT`` environment variable when set (useful inside
    containers where the code may live at a different absolute path).
    """
    override = os.getenv("SMARTHUB_ROOT")
    return Path(override).resolve() if override else PROJECT_ROOT


def data_dir() -> Path:
    return project_root() / "data"


def resolve(path: str | os.PathLike[str]) -> Path:
    """Resolve ``path`` against the project root unless it is already absolute."""
    candidate = Path(path)
    return candidate if candidate.is_absolute() else project_root() / candidate
