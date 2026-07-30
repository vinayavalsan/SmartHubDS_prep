"""Tests for the leakage-safe training-table extraction."""

import pandas as pd
import pytest

from smarthub.core.lead_types import all_lead_type_ids, lead_type_id
from smarthub.feature_engineering import features as fe
from smarthub.feature_engineering.feature_registry import FEATURES, FeatureSpec
from smarthub.feature_engineering.features import (
    LEAKAGE_COLUMNS,
    TARGET_COLUMN,
    build_training_table,
)


def _raw():
    """Raw lead frame: wins='true', losses blank, bid>0 means a bid placed."""
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
            "won": ["true", "", "", "true", ""],  # true=win, blank=loss/no-bid
            "bid": [12.0, 8.0, 0.0, 20.0, 5.0],  # id 3 has no bid placed
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
    """Wins map to 1 and placed-bid non-wins map to 0."""
    table = build_training_table(_raw())  # no lead-type filter
    # id 3 has no bid -> excluded; 1,2,4,5 kept
    assert set(table["id"]) == {1, 2, 4, 5}
    flags = table.set_index("id")[TARGET_COLUMN]
    assert flags[1] == 1 and flags[4] == 1  # won == 'true'
    assert flags[2] == 0 and flags[5] == 0  # placed bid, not won -> loss


def test_excludes_no_bid_rows():
    """Rows with bid == 0 are excluded (not a bidding decision)."""
    table = build_training_table(_raw())
    assert 3 not in set(table["id"])  # bid == 0, not a bidding decision


def test_lead_type_filter():
    """lead_type_id filters rows to a single lead type."""
    table = build_training_table(_raw(), lead_type_id=6)
    # auto rows with a placed bid: 1 (win), 2 (loss), 5 (loss); 3 no bid, 4 home
    assert set(table["id"]) == {1, 2, 5}
    # nothing is dropped by default -> constant lead_type_id is kept
    assert "lead_type_id" in table.columns


def test_excludes_leakage_columns():
    """Leakage columns are dropped; bid/expected_revenue/target kept."""
    table = build_training_table(_raw())
    for col in LEAKAGE_COLUMNS:
        assert col not in table.columns
    assert {"bid", "expected_revenue", TARGET_COLUMN}.issubset(table.columns)


def test_keeps_zero_variance_by_default():
    """Constant columns are retained by default."""
    table = build_training_table(_raw(), lead_type_id=6)
    assert "insured" in table.columns


def test_drops_zero_variance_when_requested():
    """drop_zero_variance=True removes constant columns, keeps varying ones."""
    table = build_training_table(_raw(), lead_type_id=6, drop_zero_variance=True)
    assert "insured" not in table.columns  # constant -> dropped
    assert "state" in table.columns  # varies -> kept


def test_derived_features():
    """is_married and multi_vehicle are derived from raw columns."""
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


def test_age_cohort_single_categorical():
    """Age is banded into a single categorical age_cohort column."""
    table = build_training_table(_raw()).set_index("id")  # ages 40,55,60,28
    assert "age_cohort" in table.columns
    assert table.loc[1, "age_cohort"] == "35_44"
    assert table.loc[2, "age_cohort"] == "55_64"
    assert table.loc[4, "age_cohort"] == "55_64"
    assert table.loc[5, "age_cohort"] == "25_34"


def test_age_cohort_missing_is_null():
    """Missing age yields a null age_cohort (no band assigned)."""
    raw = _raw()
    raw.loc[raw["id"] == 1, "age"] = None
    table = build_training_table(raw).set_index("id")
    assert pd.isna(table.loc[1, "age_cohort"])


def test_age_sentinel_and_cohort_for_implausible_age():
    """Implausible ages become the -1 sentinel with a null age_cohort."""
    raw = _raw()
    raw.loc[raw["id"] == 1, "age"] = -7648  # garbage / implausible
    raw.loc[raw["id"] == 2, "age"] = 1828  # garbage / implausible
    table = build_training_table(raw).set_index("id")
    # implausible age -> -1 sentinel (not NaN), age_cohort left null
    assert table.loc[1, "age"] == -1
    assert table.loc[2, "age"] == -1
    assert pd.isna(table.loc[1, "age_cohort"])
    assert pd.isna(table.loc[2, "age_cohort"])
    # a valid age is unchanged and gets a real band
    assert table.loc[4, "age"] == 60
    assert table.loc[4, "age_cohort"] == "55_64"


def test_is_workday_from_pst_date(monkeypatch, tmp_path):
    """is_workday is derived from the Pacific date (weekday vs weekend)."""
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
        assert table.loc[1, "is_workday"] == 1  # Monday
        assert table.loc[2, "is_workday"] == 0  # Saturday
        assert table.loc[3, "is_workday"] == 1  # Monday
    finally:
        holidays.reload()


def test_excludes_errored_rows():
    """Errored pings are dropped from the training table."""
    raw = _raw()
    raw["erred"] = ["true", "", "", "false", "1"]  # ids 1 and 5 errored
    table = build_training_table(raw)
    assert set(table["id"]).isdisjoint({1, 5})  # errored pings dropped
    assert set(table["id"]) == {2, 4}  # 3 no-bid excluded; 2,4 kept


def test_expected_revenue_prefers_exp_rev():
    """expected_revenue prefers backend exp_rev, falling back to listings."""
    raw = _raw()
    # exp_rev populated for id 1, zero for id 2 (falls back to listings sum).
    raw["exp_rev"] = [99.0, 0.0, 0.0, 0.0, 0.0]
    table = build_training_table(raw).set_index("id")
    assert table.loc[1, "expected_revenue"] == 99.0  # backend value used
    assert table.loc[2, "expected_revenue"] == 9.0  # fell back to listings


def test_time_features_prefer_pacific():
    """Time features use Pacific pst_hour/pst_date over UTC created_at."""
    raw = _raw()
    # UTC created_at on id 1 is 01:00 Sat; give Pacific pst_hour/pst_date instead.
    raw["pst_hour"] = [17, 9, 9, 2, 14]
    raw["pst_date"] = pd.to_datetime(
        ["2026-06-22", "2026-06-20", "2026-06-20", "2026-06-21", "2026-06-20"]
    )
    table = build_training_table(raw).set_index("id")
    assert table.loc[1, "created_hour"] == 17  # from pst_hour, not UTC 1
    assert table.loc[1, "created_dayofweek"] == 0  # 2026-06-22 = Monday (PT)


def test_training_table_is_lead_type_clean():
    """Lead-type filtering drops the other type's exclusive feature columns."""
    raw = pd.DataFrame(
        {
            "id": [1, 2],
            "won": ["true", "true"],
            "bid": [5.0, 6.0],
            "num_vehicles": [2, 1],
            "num_home_claims": [0, 1],
            "home_property_type": ["SFH", "Condo"],
            "created_at": pd.to_datetime(["2026-06-22 10:00", "2026-06-22 11:00"]),
        }
    )
    auto = build_training_table(raw, lead_type_id=6)
    assert "num_home_claims" not in auto.columns  # home-only dropped
    assert "home_property_type" not in auto.columns
    assert "num_vehicles" in auto.columns  # auto keeps its own

    home = build_training_table(raw, lead_type_id=1)
    assert "num_vehicles" not in home.columns  # auto-only dropped
    assert "num_home_claims" in home.columns
    assert "home_property_type" in home.columns


def test_has_time_features():
    """The training table includes created_hour and created_dayofweek."""
    table = build_training_table(_raw())
    assert {"created_hour", "created_dayofweek"}.issubset(table.columns)


# --- Generic lead-type feature contracts -------------------------------------


def test_model_feature_columns_reject_unknown_lead_type():
    """Unknown IDs fail clearly instead of silently using another schema."""
    with pytest.raises(ValueError, match="Unknown lead_type_id"):
        fe.model_feature_columns(999_999)


def test_every_registered_lead_type_has_model_features():
    """Every lead type in the canonical registry has a usable model schema."""
    for registered_id in all_lead_type_ids():
        numeric, categorical = fe.model_feature_columns(registered_id)
        assert numeric or categorical


def test_model_feature_columns_are_unique_and_disjoint():
    """A feature cannot be duplicated or belong to both preprocessing groups."""
    for registered_id in all_lead_type_ids():
        numeric, categorical = fe.model_feature_columns(registered_id)

        assert len(numeric) == len(set(numeric))
        assert len(categorical) == len(set(categorical))
        assert set(numeric).isdisjoint(categorical)


def test_model_feature_column_order_is_deterministic():
    """Repeated resolution returns the same ordered feature lists."""
    for registered_id in all_lead_type_ids():
        first = fe.model_feature_columns(registered_id)
        second = fe.model_feature_columns(registered_id)
        assert first == second


def test_mandatory_and_optional_features_partition_model_schema():
    """Mandatory and optional sets are complete, non-overlapping partitions."""
    for registered_id in all_lead_type_ids():
        numeric, categorical = fe.model_feature_columns(registered_id)
        selected = set(numeric) | set(categorical)
        mandatory = fe.mandatory_features(registered_id)
        optional = fe.optional_features(registered_id)

        assert mandatory.isdisjoint(optional)
        assert mandatory | optional == selected


def test_disabling_optional_features_keeps_only_mandatory_features():
    """An empty optional selection never removes mandatory model inputs."""
    for registered_id in all_lead_type_ids():
        numeric, categorical = fe.model_feature_columns(
            registered_id, optional_enabled=set()
        )
        assert set(numeric) | set(categorical) == fe.mandatory_features(registered_id)


# --- Mandatory / optional feature selection ---------------------------------


def test_sr22_is_auto_only_model_feature():
    """sr22_required is an auto-only model feature."""
    num_auto, cat_auto = fe.model_feature_columns(lead_type_id("auto"))
    num_home, cat_home = fe.model_feature_columns(lead_type_id("home"))
    auto_features = set(num_auto) | set(cat_auto)
    home_features = set(num_home) | set(cat_home)
    assert "sr22_required" in auto_features
    assert "sr22_required" not in home_features


def test_mandatory_features_kept_when_no_optional():
    """With no optional features, only the auto mandatory core survives."""
    numeric, categorical = fe.model_feature_columns(
        lead_type_id("auto"), optional_enabled=set()
    )
    selected = set(numeric) | set(categorical)
    expected = fe.mandatory_features(lead_type_id("auto"))
    assert selected == expected
    # image criteria all present
    for col in (
        "home_owner",
        "multi_vehicle",
        "num_vehicles",
        "insured",
        "num_auto_accidents",
        "dui",
        "sr22_required",
        "age",
    ):
        assert col in selected
    assert "age_cohort" in selected
    # optional features are gone
    for col in (
        "state",
        "gender",
        "traffic_tier",
        "campaign_id",
        "created_hour",
        "is_workday",
        "num_drivers",
    ):
        assert col not in selected


def test_optional_subset_is_added_to_mandatory():
    """Requested optional features are added on top of the mandatory core."""
    numeric, categorical = fe.model_feature_columns(
        lead_type_id("auto"), optional_enabled={"state", "traffic_tier"}
    )
    selected = set(numeric) | set(categorical)
    assert {"state", "traffic_tier"}.issubset(selected)  # requested optional
    assert fe.mandatory_features(lead_type_id("auto")) <= selected  # core still there
    assert "gender" not in selected and "campaign_id" not in selected


def test_optional_cannot_drop_mandatory():
    """Mandatory features survive even when the optional list omits them."""
    numeric, categorical = fe.model_feature_columns(
        lead_type_id("auto"), optional_enabled={"state"}
    )
    selected = set(numeric) | set(categorical)
    assert "sr22_required" in selected and "home_owner" in selected


def test_config_none_selects_mandatory_only(monkeypatch):
    """Config value 'none' selects the mandatory features only."""
    from smarthub.core import task_config

    monkeypatch.setattr(task_config, "get", lambda *a, **k: "none")
    numeric, categorical = fe.model_feature_columns(lead_type_id("auto"))
    assert set(numeric) | set(categorical) == fe.mandatory_features(
        lead_type_id("auto")
    )


def test_config_comma_list_ignores_unknown(monkeypatch):
    """A comma-list config adds known optionals and ignores unknown names."""
    from smarthub.core import task_config

    monkeypatch.setattr(task_config, "get", lambda *a, **k: "state, not_a_feature")
    numeric, categorical = fe.model_feature_columns(lead_type_id("auto"))
    selected = set(numeric) | set(categorical)
    assert "state" in selected
    assert "not_a_feature" not in selected
    assert fe.mandatory_features(lead_type_id("auto")) <= selected


def test_unknown_lead_type_raises():
    for fn in (fe.model_feature_columns, fe.mandatory_features, fe.optional_features):
        with pytest.raises(ValueError, match="Unknown lead_type_id"):
            fn(999)


def test_new_feature_needs_only_one_registry_entry(monkeypatch):
    """A new feature registry entry flows through normal model selection."""
    monkeypatch.setitem(
        FEATURES,
        "synthetic_score",
        FeatureSpec(
            name="synthetic_score",
            kind="numeric",
            source="raw",
            lead_types=frozenset({"auto"}),
            api_input="synthetic_score",
        ),
    )

    numeric, categorical = fe.model_feature_columns(
        lead_type_id("auto"),
        optional_enabled={"synthetic_score"},
    )

    assert "synthetic_score" in numeric
    assert "synthetic_score" not in categorical


def test_optional_and_mandatory_partition_the_feature_set():
    """Mandatory and optional sets are disjoint and cover the full auto set."""
    num, cat = fe.model_feature_columns(lead_type_id("auto"), optional_enabled=None)
    mand = fe.mandatory_features(lead_type_id("auto"))
    opt = fe.optional_features(lead_type_id("auto"))
    assert mand.isdisjoint(opt)
    assert "sr22_required" in mand and "state" in opt
    # home-only features never leak into the auto optional set
    assert "num_home_claims" not in opt and "home_property_type" not in opt


def test_config_all_keeps_every_feature(monkeypatch):
    """Config value 'all' keeps every optional and mandatory feature."""
    from smarthub.core import task_config

    monkeypatch.setattr(task_config, "get", lambda *a, **k: "all")
    numeric, categorical = fe.model_feature_columns(lead_type_id("auto"))
    selected = set(numeric) | set(categorical)
    # matches the un-filtered universe for auto (all optional + mandatory)
    for col in (
        "state",
        "gender",
        "traffic_tier",
        "campaign_id",
        "created_hour",
        "is_workday",
        "num_drivers",
        "sr22_required",
    ):
        assert col in selected
