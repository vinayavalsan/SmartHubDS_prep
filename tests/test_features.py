"""Tests for the leakage-safe training-table extraction."""

import pandas as pd

from smarthub.features import (
    LEAKAGE_COLUMNS,
    TARGET_COLUMN,
    build_training_table,
)


def _raw():
    return pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "created_at": pd.to_datetime(
                [
                    "2026-06-20 01:00",
                    "2026-06-20 05:00",
                    "2026-06-20 09:00",
                    "2026-06-21 02:00",
                ]
            ),
            "lead_type_id": [6, 6, 6, 1],
            "won": ["true", "false", "", "true"],   # blank -> dropped
            "bid": [12.0, 8.0, 5.0, 20.0],
            "expected_revenue": [15.0, 9.0, 6.0, 25.0],
            "state": ["CA", "NY", "TX", "FL"],
            "age": [40, 55, 33, 60],
            "insured": ["false", "false", "false", "false"],  # zero-variance
            # leakage columns that must be excluded:
            "rev": [12.0, 0.0, 0.0, 20.0],
            "accepted_listings": [1, 0, 0, 2],
            "response_ms": [2000, 2100, 2200, 900],
        }
    )


def test_drops_blank_won_keeps_wins_and_losses():
    table = build_training_table(_raw())  # no lead-type filter
    # row id=3 (blank won) dropped; 1,2,4 kept
    assert set(table["id"]) == {1, 2, 4}
    assert sorted(table[TARGET_COLUMN].unique().tolist()) == [0, 1]


def test_lead_type_filter():
    table = build_training_table(_raw(), lead_type_id=6)
    assert set(table["id"]) == {1, 2}  # id 4 is home (1), id 3 blank
    # lead_type_id is constant after filtering -> dropped as zero-variance
    assert "lead_type_id" not in table.columns


def test_excludes_leakage_columns():
    table = build_training_table(_raw())
    for col in LEAKAGE_COLUMNS:
        assert col not in table.columns
    # but keeps decision + revenue + target
    assert {"bid", "expected_revenue", TARGET_COLUMN}.issubset(table.columns)


def test_drops_zero_variance_feature():
    table = build_training_table(_raw(), lead_type_id=6)
    # insured is constant 'false' across kept rows -> dropped
    assert "insured" not in table.columns
    # a varying feature stays
    assert "state" in table.columns


def test_has_time_features():
    table = build_training_table(_raw())
    assert {"created_hour", "created_dayofweek"}.issubset(table.columns)
