"""Tests for the Prefect-free build-features core (feature_engineering/build)."""

import pathlib

import pandas as pd
import pytest

from smarthub.core import storage
from smarthub.feature_engineering import build


def test_build_module_is_prefect_free():
    """Core build path must not import Prefect (automation-only dependency)."""
    src = pathlib.Path(build.__file__).read_text()
    assert "import prefect" not in src
    assert "from prefect" not in src


def test_build_metadata_counts():
    """build_metadata reports correct row/win/loss counts and coverage."""
    table = pd.DataFrame(
        {
            "won_flag": [1, 0, 1],
            "created_at": pd.to_datetime(["2026-06-22", "2026-06-20", "2026-06-22"]),
            "expected_revenue": [10.0, 0.0, 5.0],
            "created_dayofweek": [0, 5, 0],  # Mon, Sat, Mon
            "is_workday": [1, 0, 1],
            "traffic_tier": ["a", "a", "b"],
            "age_cohort": ["25_34", "35_44", None],
            "state": ["CA", "NY", "TX"],
        }
    )
    m = build.build_metadata(table, 6, 21, raw_rows=5)
    assert m["row_count"] == 3 and m["wins"] == 2 and m["losses"] == 1
    assert m["raw_rows"] == 5 and m["dropped_rows"] == 2
    assert abs(m["won_rate"] - 2 / 3) < 1e-9
    assert m["expected_revenue_coverage"] == 2 / 3
    assert m["weekday_share"] == 2 / 3 and m["weekend_share"] == 1 / 3
    assert m["traffic_tier_distinct"] == 2
    # age_missing_rate is now derived from age_cohort's null share (folded
    # in from the old standalone age_missing flag -- see CHANGELOG).
    assert m["age_missing_rate"] == pytest.approx(1 / 3)
    assert "state" in m["feature_columns"]
    assert "won_flag" not in m["feature_columns"]


def test_run_build_features_raises_when_no_data(monkeypatch):
    """run_build_features propagates StorageError when no data is available."""
    monkeypatch.setattr(
        build.StorageSettings, "from_env", classmethod(lambda cls: object())
    )

    def boom(*a, **k):
        raise storage.StorageError("no data")

    monkeypatch.setattr(storage, "load_leads_raw", boom)
    monkeypatch.setattr(storage, "load_window_raw", boom)
    with pytest.raises(storage.StorageError):
        build.run_build_features(lead_type_id=6, window_days=0)
