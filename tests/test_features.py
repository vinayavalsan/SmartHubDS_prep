"""Tests for the leakage-safe training-table extraction."""

import pandas as pd

from smarthub.data.features import (
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
            "marital_status": ["Married", "single", "MARRIED", ""],
            "num_vehicles": [2, 1, 3, 1],
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
    # nothing is dropped by default -> constant lead_type_id is kept
    assert "lead_type_id" in table.columns


def test_excludes_leakage_columns():
    table = build_training_table(_raw())
    for col in LEAKAGE_COLUMNS:
        assert col not in table.columns
    # but keeps decision + revenue + target
    assert {"bid", "expected_revenue", TARGET_COLUMN}.issubset(table.columns)


def test_keeps_zero_variance_by_default():
    # "do not drop anything": constant 'insured' is retained by default.
    table = build_training_table(_raw(), lead_type_id=6)
    assert "insured" in table.columns


def test_drops_zero_variance_when_requested():
    table = build_training_table(
        _raw(), lead_type_id=6, drop_zero_variance=True
    )
    # insured is constant 'false' across kept rows -> dropped
    assert "insured" not in table.columns
    # a varying feature stays
    assert "state" in table.columns


def test_derived_features():
    table = build_training_table(_raw())  # rows 1, 2, 4
    assert {"is_married", "multi_vehicle"}.issubset(table.columns)
    by_id = table.set_index("id")
    # marital_status: 'Married'/'MARRIED' -> 1, 'single'/'' -> 0
    assert by_id.loc[1, "is_married"] == 1
    assert by_id.loc[2, "is_married"] == 0
    assert by_id.loc[4, "is_married"] == 0
    # num_vehicles: >1 -> 1
    assert by_id.loc[1, "multi_vehicle"] == 1   # 2 vehicles
    assert by_id.loc[2, "multi_vehicle"] == 0   # 1 vehicle
    assert by_id.loc[4, "multi_vehicle"] == 0   # 1 vehicle


def test_age_cohort_one_hot():
    from smarthub.data.features import AGE_COHORT_COLUMNS

    table = build_training_table(_raw()).set_index("id")
    assert set(AGE_COHORT_COLUMNS).issubset(table.columns)
    # ages: id1=40 -> 35_44, id2=55 -> 55_64, id4=60 -> 55_64
    assert table.loc[1, "age_cohort_35_44"] == 1
    assert table.loc[2, "age_cohort_55_64"] == 1
    assert table.loc[4, "age_cohort_55_64"] == 1
    # exactly one band set per row
    assert table[AGE_COHORT_COLUMNS].sum(axis=1).tolist() == [1, 1, 1]
    # id1 is only in 35_44, not elsewhere
    assert table.loc[1, "age_cohort_55_64"] == 0


def test_age_cohort_missing_all_zero():
    from smarthub.data.features import AGE_COHORT_COLUMNS

    raw = _raw()
    raw.loc[raw["id"] == 1, "age"] = None
    table = build_training_table(raw).set_index("id")
    assert table.loc[1, AGE_COHORT_COLUMNS].sum() == 0


def test_has_time_features():
    table = build_training_table(_raw())
    assert {"created_hour", "created_dayofweek"}.issubset(table.columns)
