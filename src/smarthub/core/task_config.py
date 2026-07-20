"""Task-specific (Tier-2 file) configuration.

Non-secret, non-business task knobs live in a YAML file with per-stage
sections (``data_pull``, ``feature_engineering``, ``features``, ``validation``,
``training``, ``prediction``, ``explain``). Every getter takes a ``default``
returned when the file, section or key is missing or unparseable, so the file
is optional. Path: ``$SMARTHUB_TASK_CONFIG`` or
``<project_root>/config/smarthub.yaml``.

The public API (``get`` / ``get_int`` / ``get_float`` / ``get_bool`` /
``reload`` / ``config_path``) is unchanged from the previous INI backend, so
call sites don't care that the store is now YAML.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache

import yaml

from smarthub.core import paths

logger = logging.getLogger(__name__)

DEFAULT_REL_PATH = "config/smarthub.yaml"
_TRUE = {"1", "true", "yes", "y", "on"}
_FALSE = {"0", "false", "no", "n", "off"}


def config_path() -> str:
    """Return the resolved path to the task config YAML (env override wins).

    Returns
    -------
    str
        ``$SMARTHUB_TASK_CONFIG`` when set, else the packaged default.
    """
    override = os.getenv("SMARTHUB_TASK_CONFIG")
    return override if override else str(paths.resolve(DEFAULT_REL_PATH))


@lru_cache(maxsize=1)
def _data() -> dict:
    """Return the cached parsed YAML mapping, or ``{}`` when the file is absent."""
    path = config_path()
    if not os.path.exists(path):
        logger.info("Task config not found at %s; using defaults.", path)
        return {}
    try:
        with open(path, encoding="utf-8") as fh:
            loaded = yaml.safe_load(fh)
    except (OSError, yaml.YAMLError) as exc:
        logger.warning("Could not read task config %s (%s); using defaults.", path, exc)
        return {}
    return loaded if isinstance(loaded, dict) else {}


def reload() -> None:
    """Clear the cached mapping (e.g. after editing the file at runtime)."""
    _data.cache_clear()


def get(section: str, key: str, default=None):
    """Return the value for a key, or ``default`` if missing.

    Inputs
    ------
    section : str
        Top-level YAML section name.
    key : str
        Key within the section.
    default : Any
        Value returned when the section or key is absent.

    Returns
    -------
    Any
        The value (native YAML type), or ``default``.
    """
    section_data = _data().get(section)
    if isinstance(section_data, dict) and key in section_data:
        return section_data[key]
    return default


def get_int(section: str, key: str, default: int) -> int:
    """Return a key coerced to ``int``, or ``default`` on missing/bad value.

    Inputs
    ------
    section : str
        Section name.
    key : str
        Key within the section.
    default : int
        Value returned when missing or not coercible to ``int``.

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
    """Return a key coerced to ``float``, or ``default`` on missing/bad value.

    Inputs
    ------
    section : str
        Section name.
    key : str
        Key within the section.
    default : float
        Value returned when missing or not coercible to ``float``.

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
    """Return a key coerced to ``bool``, or ``default`` on unknown value.

    Accepts native YAML booleans as well as true/false-ish tokens
    (e.g. 1/0, yes/no, on/off).

    Inputs
    ------
    section : str
        Section name.
    key : str
        Key within the section.
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
    if isinstance(raw, bool):
        return raw
    token = str(raw).strip().lower()
    if token in _TRUE:
        return True
    if token in _FALSE:
        return False
    return default
