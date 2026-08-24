"""Tests for the leakage-safe training-table extraction."""

from dataclasses import replace

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


def test_derived_features():
    """Enabled derived registry features are produced by the training build."""
    table = build_training_table(_raw()).set_index("id")
    enabled_derived = {
        spec.name
        for spec in FEATURES.values()
        if spec.enabled and spec.source == "derived"
    }
    assert enabled_derived.issubset(table.columns)
    if "multi_vehicle" in enabled_derived:
        assert table.loc[1, "multi_vehicle"] == 1
        assert table.loc[2, "multi_vehicle"] == 0
        assert table.loc[5, "multi_vehicle"] == 1


def test_disabled_derived_features_stay_out_of_model_table(monkeypatch):
    """A derived feature forced disabled is not derived or selected."""
    original = FEATURES["is_married"]
    monkeypatch.setitem(FEATURES, "is_married", replace(original, enabled=False))

    table = build_training_table(_raw())

    assert "is_married" not in table.columns
    assert "marital_status" in table.columns


def test_missing_age_uses_numeric_missing_sentinel():
    """Missing raw age uses the registry numeric sentinel."""
    raw = _raw()
    raw.loc[raw["id"] == 1, "age"] = None
    table = build_training_table(raw).set_index("id")
    assert table.loc[1, "age"] == -1


def test_implausible_age_is_not_fixed_when_age_cohort_is_disabled(monkeypatch):
    """Without age-cohort derivation, feature engineering preserves raw ages."""
    original = FEATURES["age_cohort"]
    monkeypatch.setitem(FEATURES, "age_cohort", replace(original, enabled=False))

    raw = _raw()
    raw.loc[raw["id"] == 1, "age"] = -7648
    raw.loc[raw["id"] == 2, "age"] = 1828
    table = build_training_table(raw).set_index("id")

    assert table.loc[1, "age"] == -7648
    assert table.loc[2, "age"] == 1828
    assert "age_cohort" not in table.columns


def test_is_workday_from_created_at_pacific(monkeypatch, tmp_path):
    """is_workday uses the Pacific calendar date derived from UTC created_at."""
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
                    [
                        "2026-06-22 10:00",
                        "2026-06-21 06:00",
                        "2026-06-23 06:30",
                    ]
                ),
            }
        )
        table = build_training_table(raw).set_index("id")
        assert "is_workday" in table.columns
        assert table.loc[1, "is_workday"] == 1  # Mon 03:00 Pacific
        assert table.loc[2, "is_workday"] == 0  # Sat 23:00 Pacific
        assert table.loc[3, "is_workday"] == 1  # Mon 23:30 Pacific
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


def test_time_features_derive_pacific_from_created_at():
    """Time features convert UTC created_at to Pacific before derivation."""
    raw = _raw()

    # Deliberately conflicting warehouse Pacific helper columns must be ignored.
    raw["pst_hour"] = [17, 9, 9, 2, 14]
    raw["pst_date"] = pd.to_datetime(
        ["2026-06-22", "2026-06-20", "2026-06-20", "2026-06-21", "2026-06-20"]
    )

    table = build_training_table(raw).set_index("id")

    # 2026-06-20 01:00 UTC is 2026-06-19 18:00 PDT (Friday).
    assert table.loc[1, "created_hour"] == "18"
    assert table.loc[1, "created_dayofweek"] == "4"


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


def test_every_modeled_lead_type_has_model_features():
    """Every lead type used by the feature registry has a model schema."""
    modeled_lead_types = {
        lead_type for spec in FEATURES.values() for lead_type in spec.lead_types
    }

    for name in sorted(modeled_lead_types):
        numeric, categorical = fe.model_feature_columns(lead_type_id(name))
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


def test_model_schema_uses_enabled_and_lead_types_only():
    """Selected features match enabled registry entries for the lead type."""
    for name in ("auto", "home"):
        registered_id = lead_type_id(name)
        numeric, categorical = fe.model_feature_columns(registered_id)

        expected_numeric = [
            spec.name
            for spec in FEATURES.values()
            if spec.enabled
            and name in spec.lead_types
            and spec.kind in {"numeric", "binary"}
        ]
        expected_categorical = [
            spec.name
            for spec in FEATURES.values()
            if spec.enabled and name in spec.lead_types and spec.kind == "categorical"
        ]

        assert numeric == expected_numeric
        assert categorical == expected_categorical


def test_disabled_features_are_not_model_inputs(monkeypatch):
    """A registry feature forced disabled stays out of every model schema."""
    original = FEATURES["is_married"]
    monkeypatch.setitem(FEATURES, "is_married", replace(original, enabled=False))

    for registered_id in all_lead_type_ids():
        numeric, categorical = fe.model_feature_columns(registered_id)
        selected = set(numeric) | set(categorical)
        assert "is_married" not in selected


def test_sr22_is_auto_only_model_feature():
    """sr22_required is an auto-only model feature."""
    num_auto, cat_auto = fe.model_feature_columns(lead_type_id("auto"))
    num_home, cat_home = fe.model_feature_columns(lead_type_id("home"))
    auto_features = set(num_auto) | set(cat_auto)
    home_features = set(num_home) | set(cat_home)
    assert "sr22_required" in auto_features
    assert "sr22_required" not in home_features


def test_new_enabled_feature_needs_only_one_registry_entry(monkeypatch):
    """One enabled registry entry is enough for model selection."""
    monkeypatch.setitem(
        FEATURES,
        "synthetic_score",
        FeatureSpec(
            name="synthetic_score",
            kind="numeric",
            source="raw",
            lead_types=frozenset({"auto"}),
            enabled=True,
            api_input="synthetic_score",
        ),
    )

    numeric, categorical = fe.model_feature_columns(lead_type_id("auto"))
    assert "synthetic_score" in numeric
    assert "synthetic_score" not in categorical


def test_new_disabled_feature_is_not_selected(monkeypatch):
    """A registry entry with enabled=False is excluded."""
    monkeypatch.setitem(
        FEATURES,
        "disabled_score",
        FeatureSpec(
            name="disabled_score",
            kind="numeric",
            source="raw",
            lead_types=frozenset({"auto"}),
            enabled=False,
            api_input="disabled_score",
        ),
    )

    numeric, categorical = fe.model_feature_columns(lead_type_id("auto"))
    assert "disabled_score" not in set(numeric) | set(categorical)
