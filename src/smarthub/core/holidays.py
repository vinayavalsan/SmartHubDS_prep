"""Holiday calendar for the ``is_workday`` feature.

Design (per Kiran/Vinaya):
- **Weekends (Sat/Sun) are always non-workdays — computed in code**, no config.
- **Observed holidays** live in a git-versioned JSON file
  (``config/holidays.json``) so they're transparent (not hardcoded) and editable
  without touching code. Path overridable via ``SMARTHUB_HOLIDAYS``; mount the
  file to edit without a rebuild.

If a non-dev ever needs to edit dates from a UI, swap the JSON backend for a
``smarthub_holidays`` table — ``is_workday`` / ``holiday_dates`` stay the same.

File format (either shape works)::

    {"holidays": [{"date": "2026-01-01", "label": "New Year's Day"}, ...]}
    ["2026-01-01", "2026-12-25", ...]
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
    raw = entry.get("date") if isinstance(entry, dict) else entry
    try:
        return datetime.strptime(str(raw).strip()[:10], "%Y-%m-%d").date()
    except (TypeError, ValueError):
        return None


@lru_cache(maxsize=1)
def holiday_dates() -> frozenset[date]:
    """Observed holiday dates from the JSON file (empty if absent/unreadable)."""
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
    return day in holiday_dates()


def is_workday(day: date) -> bool:
    """True on Mon–Fri that are not observed holidays (Sat/Sun always False)."""
    return day.weekday() < 5 and day not in holiday_dates()
