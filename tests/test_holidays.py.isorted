"""Tests for the holiday calendar / is_workday logic."""

from datetime import date

from smarthub.core import holidays


def test_weekends_and_holidays(tmp_path, monkeypatch):
    """Weekends and configured holidays are treated as non-workdays."""
    f = tmp_path / "h.json"
    f.write_text('{"holidays": [{"date": "2026-01-01", "label": "New Year"}]}')
    monkeypatch.setenv("SMARTHUB_HOLIDAYS", str(f))
    holidays.reload()
    try:
        assert holidays.is_workday(date(2026, 6, 22)) is True   # Monday
        assert holidays.is_workday(date(2026, 6, 20)) is False  # Saturday
        assert holidays.is_workday(date(2026, 6, 21)) is False  # Sunday
        assert holidays.is_holiday(date(2026, 1, 1)) is True
        assert holidays.is_workday(date(2026, 1, 1)) is False   # holiday (Thu)
    finally:
        holidays.reload()


def test_plain_list_format(tmp_path, monkeypatch):
    """The holiday file may be a plain list of date strings."""
    f = tmp_path / "h.json"
    f.write_text('["2026-12-25"]')
    monkeypatch.setenv("SMARTHUB_HOLIDAYS", str(f))
    holidays.reload()
    try:
        assert holidays.is_holiday(date(2026, 12, 25)) is True
    finally:
        holidays.reload()


def test_missing_file_is_weekends_only(monkeypatch):
    """A missing holiday file falls back to weekends-only."""
    monkeypatch.setenv("SMARTHUB_HOLIDAYS", "/no/such/file.json")
    holidays.reload()
    try:
        assert holidays.holiday_dates() == frozenset()
        assert holidays.is_workday(date(2026, 1, 1)) is True    # no holidays -> Thu
        assert holidays.is_workday(date(2026, 6, 20)) is False  # Saturday
    finally:
        holidays.reload()
