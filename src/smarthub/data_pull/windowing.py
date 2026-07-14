"""Pure helpers for the data-pull time window (no Prefect import).

The pull window is anchored on a saved watermark (the last pull's timestamp):
each run starts from that watermark minus a small overlap, so late-resolving
outcomes (`won`, `rev`, listing payouts) get re-pulled and upserted. On the very
first run (no watermark) it falls back to a backfill lookback.
"""

from __future__ import annotations

from datetime import datetime, timedelta

# Warehouse `created_at` is naive UTC; we keep the same format throughout.
DT_FORMAT = "%Y-%m-%d %H:%M:%S"


def parse_dt(value: str) -> datetime:
    """Parse a warehouse datetime string into a ``datetime``."""
    return datetime.strptime(value, DT_FORMAT)


def format_dt(value: datetime) -> str:
    """Format a ``datetime`` as a warehouse datetime string."""
    return value.strftime(DT_FORMAT)


def compute_pull_window(
    now: datetime,
    last_ts: datetime | None,
    overlap_hours: float,
    default_lookback_hours: float,
) -> tuple[datetime, datetime]:
    """Return the ``(min, max)`` datetimes for the next pull.

    No watermark yet gives ``[now - default_lookback_hours, now]`` (backfill);
    an existing watermark gives ``[last_ts - overlap_hours, now]`` (overlapping
    window). The lower bound is clamped so it never exceeds ``now``.

    Inputs
    ------
    now : datetime
        Current time (naive UTC), used as the window's upper bound.
    last_ts : datetime | None
        Saved watermark; ``None`` triggers the backfill lookback.
    overlap_hours : float
        Hours to re-pull before the watermark.
    default_lookback_hours : float
        Backfill lookback used when there is no watermark.

    Returns
    -------
    tuple[datetime, datetime]
        The ``(min_dt, max_dt)`` window bounds.
    """
    max_dt = now
    if last_ts is None:
        min_dt = now - timedelta(hours=default_lookback_hours)
    else:
        min_dt = last_ts - timedelta(hours=overlap_hours)
    if min_dt > max_dt:
        min_dt = max_dt
    return min_dt, max_dt
