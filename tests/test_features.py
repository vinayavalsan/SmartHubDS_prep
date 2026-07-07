"""Tests for the leakage-safe training-table extraction."""

import pandas as pd

from smarthub.feature_engineering.features import (
    LEAKAGE_COLUMNS,
    TARGET_COLUMN,
    build_training_table,
)


def _raw():
    # The warehouse encodes wins as 'true' and losses as NULL/blank (never
    # 'false'); a bid is "placed" when bid > 0. Row 3 has no bid -> excluded.
    return pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "created_at": pd.to_datetime(
                [
                    "2026-06-20 01:00",
                    "2026-06-20 05:00",
                    "2026-06-20 09:00",
                    "2026-06-21 02:00",
                    "2026-06-20 14:00",
                ]
            ),
            "lead_type_id": [6, 6, 6, 1, 6],
            "won": ["true", "", "", "true", ""],   # true=win, blank=loss/no-bid
            "bid": [12.0, 8.0, 0.0, 20.0, 5.0],     # id 3 has no bid placed
            "expected_revenue": [15.0, 9.0, 6.0, 25.0, 7.0],
            "state": ["CA", "NY", "TX", "FL", "WA"],
            "age": [40, 55, 33, 60, 28],
            "insured": ["false", "false", "false", "false", "false"],  # constant
            "marital_status": ["Married", "single", "MARRIED", "", "Married"],
            "num_vehicles": [2, 1, 3, 1, 2],
            # leakage columns that must be excluded:
            "rev": [12.0, 0.0, 0.0, 20.0, 0.0],
            "accepted_listings": [1, 0, 0, 2, 0],
            "response_ms": [2000, 2100, 2200, 900, 1500],
        }
    )


def test_labels_wins_and_placed_bid_losses():
    table = build_training_table(_raw())  # no lead-type filter
    # id 3 has no bid -> excluded; 1,2,4,5 kept
    assert set(table["id"]) == {1, 2, 4, 5}
    flags = table.set_index("id")[TARGET_COLUMN]
    assert flags[1] == 1 and flags[4] == 1   # won == 'true'
    assert flags[2] == 0 and flags[5] == 0   # placed bid, not won -> loss


def test_excludes_no_bid_rows():
    table = build_training_table(_raw())
    assert 3 not in set(table["id"])  # bid == 0, not a bidding decision


def test_lead_type_filter():
    table = build_training_table(_raw(), lead_type_id=6)
    # auto rows with a placed bid: 1 (win), 2 (loss), 5 (loss); 3 no bid, 4 home
    assert set(table["id"]) == {1, 2, 5}
    # nothing is dropped by default -> constant lead_type_id is kept
    assert "lead_type_id" in table.columns


def test_excludes_leakage_columns():
    table = build_training_table(_raw())
    for col in LEAKAGE_COLUMNS:
        assert col not in table.columns
    assert {"bid", "expected_revenue", TARGET_COLUMN}.issubset(table.columns)


def test_keeps_zero_variance_by_default():
    # "do not drop anything": constant 'insured' is retained by default.
    table = build_training_table(_raw(), lead_type_id=6)
    assert "insured" in table.columns


def test_drops_zero_variance_when_requested():
    table = build_training_table(_raw(), lead_type_id=6, drop_zero_variance=True)
    assert "insured" not in table.columns   # constant -> dropped
    assert "state" in table.columns         # varies -> kept


def test_derived_features():
    table = build_training_table(_raw()).set_index("id")  # ids 1,2,4,5
    assert {"is_married", "multi_vehicle"}.issubset(table.columns)
    # marital_status: Married/MARRIED -> 1; single/'' -> 0
    assert table.loc[1, "is_married"] == 1
    assert table.loc[2, "is_married"] == 0
    assert table.loc[4, "is_married"] == 0
    assert table.loc[5, "is_married"] == 1
    # num_vehicles > 1 -> 1
    assert table.loc[1, "multi_vehicle"] == 1
    assert table.loc[2, "multi_vehicle"] == 0
    assert table.loc[5, "multi_vehicle"] == 1


def test_age_cohort_one_hot():
    from smarthub.feature_engineering.features import AGE_COHORT_COLUMNS

    table = build_training_table(_raw()).set_index("id")  # ages 40,55,60,28
    assert set(AGE_COHORT_COLUMNS).issubset(table.columns)
    assert table.loc[1, "age_cohort_35_44"] == 1
    assert table.loc[2, "age_cohort_55_64"] == 1
    assert table.loc[4, "age_cohort_55_64"] == 1
    assert table.loc[5, "age_cohort_25_34"] == 1
    # exactly one band set per row
    assert table[AGE_COHORT_COLUMNS].sum(axis=1).tolist() == [1, 1, 1, 1]


def test_age_cohort_missing_all_zero():
    from smarthub.feature_engineering.features import AGE_COHORT_COLUMNS

    raw = _raw()
    raw.loc[raw["id"] == 1, "age"] = None
    table = build_training_table(raw).set_index("id")
    assert table.loc[1, AGE_COHORT_COLUMNS].sum() == 0


def test_age_missing_sentinel_and_flag():
    from smarthub.feature_engineering.features import AGE_COHORT_COLUMNS

    raw = _raw()
    raw.loc[raw["id"] == 1, "age"] = -7648   # garbage / implausible
    raw.loc[raw["id"] == 2, "age"] = 1828    # garbage / implausible
    table = build_training_table(raw).set_index("id")
    # implausible age -> -1 sentinel (not NaN), age_missing flag set, no cohort
    assert table.loc[1, "age"] == -1
    assert table.loc[2, "age"] == -1
    assert table.loc[1, "age_missing"] == 1
    assert table.loc[2, "age_missing"] == 1
    assert table.loc[1, AGE_COHORT_COLUMNS].sum() == 0
    # a valid age is unchanged and not flagged
    assert table.loc[4, "age"] == 60
    assert table.loc[4, "age_missing"] == 0


def test_is_workday_from_pst_date(monkeypatch, tmp_path):
    from smarthub.core import holidays
    from smarthub.feature_engineering.features import build_training_table

    monkeypatch.setenv("SMARTHUB_HOLIDAYS", str(tmp_path / "none.json"))
    holidays.reload()
    try:
        raw = pd.DataFrame(
            {
                "id": [1, 2, 3],
                "won": ["true", "", "true"],
                "bid": [5.0, 6.0, 7.0],
                "created_at": pd.to_datetime(
                    ["2026-06-22 10:00", "2026-06-20 10:00", "2026-06-22 11:00"]
                ),
                "pst_date": pd.to_datetime(["2026-06-22", "2026-06-20", "2026-06-22"]),
            }
        )
        table = build_training_table(raw).set_index("id")
        assert "is_workday" in table.columns
        assert table.loc[1, "is_workday"] == 1   # Monday
        assert table.loc[2, "is_workday"] == 0   # Saturday
        assert table.loc[3, "is_workday"] == 1   # Monday
    finally:
        holidays.reload()


def test_has_time_features():
    table = build_training_table(_raw())
    assert {"created_hour", "created_dayofweek"}.issubset(table.columns)
