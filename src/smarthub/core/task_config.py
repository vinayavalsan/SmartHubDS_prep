"""Task-specific (Tier-2 file) configuration.

Non-secret, non-business *task* knobs live in an ini file with per-stage
sections — ``[data_pull]``, ``[feature_engineering]``, ``[training]``,
``[prediction]`` — per the team decision (Kiran/Vinaya): the UI holds business
settings only, secrets live in ``.env``, and task configs live in an editable
ini file.

Defaults: every getter takes a ``default``; if the file, section, or key is
missing (or unparseable), the default is returned — so the ini is optional and
nothing breaks without it. Path: ``$SMARTHUB_TASK_CONFIG`` or
``<project_root>/config/smarthub.ini``.
"""

from __future__ import annotations

import configparser
import logging
import os
from functools import lru_cache

from smarthub.core import paths

logger = logging.getLogger(__name__)

DEFAULT_REL_PATH = "config/smarthub.ini"
_TRUE = {"1", "true", "yes", "y", "on"}
_FALSE = {"0", "false", "no", "n", "off"}


def config_path() -> str:
    """Resolved path to the task config ini (env override wins)."""
    override = os.getenv("SMARTHUB_TASK_CONFIG")
    return override if override else str(paths.resolve(DEFAULT_REL_PATH))


@lru_cache(maxsize=1)
def _parser() -> configparser.ConfigParser:
    cp = configparser.ConfigParser(inline_comment_prefixes=(";", "#"))
    path = config_path()
    if os.path.exists(path):
        cp.read(path)
    else:
        logger.info("Task config not found at %s; using defaults.", path)
    return cp


def reload() -> None:
    """Clear the cached parser (e.g. after editing the file at runtime)."""
    _parser.cache_clear()


def get(section: str, key: str, default=None):
    """Raw string value, or ``default`` if missing."""
    cp = _parser()
    if cp.has_option(section, key):
        return cp.get(section, key)
    return default


def get_int(section: str, key: str, default: int) -> int:
    raw = get(section, key)
    try:
        return int(raw) if raw is not None else default
    except (TypeError, ValueError):
        return default


def get_float(section: str, key: str, default: float) -> float:
    raw = get(section, key)
    try:
        return float(raw) if raw is not None else default
    except (TypeError, ValueError):
        return default


def get_bool(section: str, key: str, default: bool) -> bool:
    raw = get(section, key)
    if raw is None:
        return default
    token = str(raw).strip().lower()
    if token in _TRUE:
        return True
    if token in _FALSE:
        return False
    return default
