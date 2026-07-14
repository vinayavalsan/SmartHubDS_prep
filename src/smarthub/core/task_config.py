"""Task-specific (Tier-2 file) configuration.

Non-secret, non-business task knobs live in an ini file with per-stage
sections (``[data_pull]``, ``[feature_engineering]``, ``[training]``,
``[prediction]``). Every getter takes a ``default`` returned when the file,
section or key is missing or unparseable, so the ini is optional. Path:
``$SMARTHUB_TASK_CONFIG`` or ``<project_root>/config/smarthub.ini``.
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
    """Return the cached parser, reading the ini file when it exists."""
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
    """Return the raw string value for a key, or ``default`` if missing.

    Inputs
    ------
    section : str
        Ini section name.
    key : str
        Option name within the section.
    default : Any
        Value returned when the section or key is absent.

    Returns
    -------
    str | Any
        The raw string value, or ``default``.
    """
    cp = _parser()
    if cp.has_option(section, key):
        return cp.get(section, key)
    return default


def get_int(section: str, key: str, default: int) -> int:
    """Return a key parsed as ``int``, or ``default`` on missing/bad value.

    Inputs
    ------
    section : str
        Ini section name.
    key : str
        Option name within the section.
    default : int
        Value returned when missing or not an integer.

    Returns
    -------
    int
        The parsed integer, or ``default``.
    """
    raw = get(section, key)
    try:
        return int(raw) if raw is not None else default
    except (TypeError, ValueError):
        return default


def get_float(section: str, key: str, default: float) -> float:
    """Return a key parsed as ``float``, or ``default`` on missing/bad value.

    Inputs
    ------
    section : str
        Ini section name.
    key : str
        Option name within the section.
    default : float
        Value returned when missing or not a float.

    Returns
    -------
    float
        The parsed float, or ``default``.
    """
    raw = get(section, key)
    try:
        return float(raw) if raw is not None else default
    except (TypeError, ValueError):
        return default


def get_bool(section: str, key: str, default: bool) -> bool:
    """Return a key parsed as ``bool``, or ``default`` on unknown value.

    Recognises true/false-ish tokens (e.g. 1/0, yes/no, on/off).

    Inputs
    ------
    section : str
        Ini section name.
    key : str
        Option name within the section.
    default : bool
        Value returned when missing or unrecognised.

    Returns
    -------
    bool
        The parsed boolean, or ``default``.
    """
    raw = get(section, key)
    if raw is None:
        return default
    token = str(raw).strip().lower()
    if token in _TRUE:
        return True
    if token in _FALSE:
        return False
    return default
