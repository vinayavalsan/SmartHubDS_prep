"""Holiday calendar for the ``is_workday`` feature.

Weekends (Sat/Sun) are always non-workdays, computed in code. Observed
holidays live in a git-versioned JSON file (``config/holidays.json``, path
overridable via ``SMARTHUB_HOLIDAYS``) so they are transparent and editable
without a code change. The file may be ``{"holidays": [{"date", "label"},
...]}`` or a bare list of ISO date strings.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import date, datetime
from functools import lru_cache

from smarthub.core import paths

logger = logging.getLogger(__name__)

DEFAULT_REL_PATH = "config/holidays.json"


def holidays_path() -> str:
    """Resolved path to the holidays JSON (``SMARTHUB_HOLIDAYS`` overrides)."""
    override = os.getenv("SMARTHUB_HOLIDAYS")
    return override if override else str(paths.resolve(DEFAULT_REL_PATH))


def _parse(entry) -> date | None:
    """Parse a holiday entry (dict or string) to a date, or None."""
    raw = entry.get("date") if isinstance(entry, dict) else entry
    try:
        return datetime.strptime(str(raw).strip()[:10], "%Y-%m-%d").date()
    except (TypeError, ValueError):
        return None


@lru_cache(maxsize=1)
def holiday_dates() -> frozenset[date]:
    """Observed holiday dates loaded from the JSON file.

    Returns
    -------
    frozenset[date]
        The observed holiday dates; empty when the file is absent or
        unreadable. Cached; call :func:`reload` after editing the file.
    """
    path = holidays_path()
    if not os.path.exists(path):
        logger.info("Holiday file not found at %s; weekends-only.", path)
        return frozenset()
    try:
        with open(path, encoding="utf-8") as fh:
            raw = json.load(fh)
    except (OSError, ValueError) as exc:  # unreadable / bad JSON -> weekends-only
        logger.warning("Could not read holidays file %s: %s", path, exc)
        return frozenset()
    items = raw.get("holidays", []) if isinstance(raw, dict) else raw
    return frozenset(d for d in (_parse(x) for x in items) if d is not None)


def reload() -> None:
    """Clear the cached holiday set (after editing the file at runtime)."""
    holiday_dates.cache_clear()


def is_holiday(day: date) -> bool:
    """Return True when ``day`` is an observed holiday."""
    return day in holiday_dates()


def is_workday(day: date) -> bool:
    """True on Mon–Fri that are not observed holidays (Sat/Sun always False)."""
    return day.weekday() < 5 and day not in holiday_dates()
