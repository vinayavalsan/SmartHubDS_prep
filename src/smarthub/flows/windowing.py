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
    return datetime.strptime(value, DT_FORMAT)


def format_dt(value: datetime) -> str:
    return value.strftime(DT_FORMAT)


def compute_pull_window(
    now: datetime,
    last_ts: datetime | None,
    overlap_hours: float,
    default_lookback_hours: float,
) -> tuple[datetime, datetime]:
    """Return the (min, max) datetimes for the next pull.

    - No watermark yet → ``[now - default_lookback_hours, now]`` (backfill).
    - Have a watermark → ``[last_ts - overlap_hours, now]`` (overlapping window).
    The lower bound is clamped so it never exceeds ``now``.
    """
    max_dt = now
    if last_ts is None:
        min_dt = now - timedelta(hours=default_lookback_hours)
    else:
        min_dt = last_ts - timedelta(hours=overlap_hours)
    if min_dt > max_dt:
        min_dt = max_dt
    return min_dt, max_dt
