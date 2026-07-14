"""Tests for the pull-window logic (no Prefect needed)."""

from datetime import datetime

from smarthub.data_pull.windowing import (
    DT_FORMAT,
    compute_pull_window,
    format_dt,
    parse_dt,
)

NOW = datetime(2026, 6, 25, 12, 0, 0)


def test_first_run_uses_backfill_lookback():
    """With no watermark, the window spans the backfill lookback up to now."""
    min_dt, max_dt = compute_pull_window(
        NOW, last_ts=None, overlap_hours=8, default_lookback_hours=168
    )
    assert max_dt == NOW
    assert min_dt == datetime(2026, 6, 18, 12, 0, 0)  # 7 days back


def test_subsequent_run_overlaps_watermark():
    """A subsequent run starts the window at the watermark minus the overlap."""
    last = datetime(2026, 6, 25, 10, 0, 0)
    min_dt, max_dt = compute_pull_window(
        NOW, last_ts=last, overlap_hours=8, default_lookback_hours=168
    )
    assert max_dt == NOW
    assert min_dt == datetime(2026, 6, 25, 2, 0, 0)  # last - 8h


def test_lower_bound_clamped_to_now():
    """A future watermark is clamped so the lower bound never exceeds now."""
    future = datetime(2026, 6, 26, 0, 0, 0)
    min_dt, max_dt = compute_pull_window(
        NOW, last_ts=future, overlap_hours=0, default_lookback_hours=168
    )
    assert min_dt == max_dt == NOW


def test_parse_format_roundtrip():
    """format_dt and parse_dt round-trip through the canonical DT_FORMAT."""
    s = "2026-06-25 12:00:00"
    assert format_dt(parse_dt(s)) == s
    assert DT_FORMAT == "%Y-%m-%d %H:%M:%S"
