"""Tests for the Anton train/predict module — pure logic only.

Covers the feature-parity contract, training-table preparation, and the bid
optimizer math. sklearn/mlflow/fastapi are NOT required here (predict keeps
those imports lazy/guarded), so these run in the base env.
"""

import numpy as np
import pandas as pd
import pytest

from smarthub.feature_engineering import features as fe
from smarthub.train_and_predict import config, predict, preprocessing


# --- Feature schema (single source of truth) --------------------------------


def test_model_feature_columns_auto_vs_home():
    num_auto, cat_auto = fe.model_feature_columns(fe.LEAD_TYPE_AUTO)
    num_home, cat_home = fe.model_feature_columns(fe.LEAD_TYPE_HOME)

    # auto-only features present for auto, absent for home
    assert "multi_vehicle" in num_auto and "num_vehicles" in num_auto
    assert "multi_vehicle" not in num_home and "num_vehicles" not in num_home
    assert "home_owner" in cat_auto and "home_owner" not in cat_home
    # shared features present in both
    for col in fe.AGE_COHORT_COLUMNS + ["is_married", "created_hour"]:
        assert col in num_auto and col in num_home
    assert "state" in cat_auto and "state" in cat_home
    # high-cardinality / redundant identifiers are excluded
    assert "source_type_id" not in cat_auto
    assert "account_id" not in cat_auto


def test_derive_serving_features_parity():
    raw = pd.DataFrame(
        [{"marital_status": "Married", "num_vehicles": 2, "age": 40,
          "created_hour": 9, "created_dayofweek": 2}]
    )
    out = fe.derive_serving_features(raw)
    assert out.loc[0, "is_married"] == 1
    assert out.loc[0, "multi_vehicle"] == 1
    assert out.loc[0, "age_cohort_35_44"] == 1
    # time parts already supplied are preserved
    assert out.loc[0, "created_hour"] == 9


# --- Serving frame -----------------------------------------------------------


def test_serving_frame_selects_and_normalizes():
    record = {
        "campaign_id": 123, "account_id": 7, "source_type_id": 2,
        "state": "TX", "gender": "Female", "marital_status": "",
        "insured": "true", "home_owner": "false", "dui": "false",
        "military_affiliation": "false",
        "num_vehicles": 2, "num_drivers": 1,
        "num_auto_violations": 0, "num_auto_accidents": 0,
        "continuous_coverage_months": 24, "age": 34,
        "created_hour": 14, "created_dayofweek": 2, "bid": 0.25,
    }
    frame = preprocessing.serving_frame([record], fe.LEAD_TYPE_AUTO)
    numeric, categorical = config.feature_columns(fe.LEAD_TYPE_AUTO)

    assert list(frame.columns) == numeric + categorical
    # ids normalised to strings for one-hot; blank -> NAvail
    assert frame.loc[0, "campaign_id"] == "123"
    assert frame.loc[0, "marital_status"] == "NAvail"
    # derived features computed from raw
    assert frame.loc[0, "multi_vehicle"] == 1
    assert frame.loc[0, "age_cohort_25_34"] == 1


# --- Training-table preparation ---------------------------------------------


def _fake_training_table():
    numeric, categorical = fe.model_feature_columns(fe.LEAD_TYPE_AUTO)
    n = 6
    data = {"id": range(n),
            "created_at": pd.to_datetime([f"2026-06-2{i} 01:00" for i in range(n)]),
            "won_flag": [1, 0, 1, 0, 1, 0],
            "expected_revenue": [20.0, 18.0, 25.0, 22.0, 30.0, 15.0]}
    for c in numeric:
        data[c] = np.arange(n, dtype=float) + 1
    for c in categorical:
        data[c] = ["A", "B"] * (n // 2)
    return pd.DataFrame(data)


def test_prepare_training_data(monkeypatch):
    table = _fake_training_table()
    monkeypatch.setattr(
        preprocessing.io, "load_training_table", lambda name, v=None: table
    )
    monkeypatch.setattr(
        preprocessing.io, "training_versions", lambda name: ["2026-07-06T095148Z"]
    )
    monkeypatch.setattr(
        preprocessing.io,
        "load_training_metadata",
        lambda name, v=None: {
            "data_min_created_at": "2026-06-20",
            "data_max_created_at": "2026-07-06",
            "row_count": 999,
        },
    )

    frame, numeric, categorical, summary = preprocessing.prepare_training_data(
        fe.LEAD_TYPE_AUTO, "auto"
    )
    assert summary["training_rows"] == len(table)
    assert summary["win_rate"] == pytest.approx(0.5)
    # lineage captured
    assert summary["training_table_version"] == "2026-07-06T095148Z"
    assert summary["data_max_created_at"] == "2026-07-06"
    assert summary["source_row_count"] == 999
    # frame carries features + target + expected_revenue, sorted by created_at
    assert config.TARGET_COL in frame.columns
    assert config.REVENUE_COL in frame.columns
    assert set(numeric + categorical).issubset(frame.columns)
    assert frame["created_at"].is_monotonic_increasing


def test_model_type_from_ini_and_fallback(monkeypatch):
    from smarthub.core import task_config

    # Missing ini -> model_type falls back to the code default.
    monkeypatch.setenv("SMARTHUB_TASK_CONFIG", "/nonexistent/smarthub.ini")
    task_config.reload()
    assert config.model_type() == config.MODEL_TYPE
    task_config.reload()  # restore real ini for other tests


def test_target_cm_falls_back_when_store_unavailable(monkeypatch):
    # target_cm stays a BUSINESS knob (config store); unreachable -> code default.
    monkeypatch.setenv(
        "SMARTHUB_CONFIG_DB_URL", "postgresql+psycopg2://x:x@127.0.0.1:1/none"
    )
    assert config.target_cm_value() == config.TARGET_CM


def test_drop_zero_variance():
    frame = pd.DataFrame(
        {"a": [1, 2, 3], "b": [5, 5, 5], "c": ["x", "y", "x"], "d": ["k", "k", "k"]}
    )
    num, cat, dropped = preprocessing.drop_zero_variance(frame, ["a", "b"], ["c", "d"])
    assert num == ["a"]
    assert cat == ["c"]
    assert set(dropped) == {"b", "d"}


def test_assert_trainable_rejects_single_class():
    # win_rate=1.0 (all wins) -> not trainable, clear message.
    frame = pd.DataFrame({config.TARGET_COL: [1, 1, 1, 1]})
    with pytest.raises(ValueError, match="only ONE target class"):
        preprocessing.assert_trainable(frame, "auto")


def test_assert_trainable_accepts_two_classes():
    frame = pd.DataFrame({config.TARGET_COL: [1, 0, 1, 0]})
    assert preprocessing.assert_trainable(frame, "auto") == [0, 1]


def test_prepare_training_data_missing_target(monkeypatch):
    table = _fake_training_table().drop(columns=["won_flag"])
    monkeypatch.setattr(
        preprocessing.io, "load_training_table", lambda name, v=None: table
    )
    with pytest.raises(ValueError, match="missing target"):
        preprocessing.prepare_training_data(fe.LEAD_TYPE_AUTO, "auto")


# --- Bid optimizer math ------------------------------------------------------


class _StubModel:
    """Constant win-probability model: predict_proba -> p for every row."""

    def __init__(self, p=0.5):
        self.p = p

    def predict_proba(self, X):
        n = len(X)
        return np.column_stack([np.full(n, 1 - self.p), np.full(n, self.p)])


def test_optimize_bid_picks_max_profit():
    # Constant win rate -> profit = p*(rev - bid) is maximised at the lowest bid.
    row = pd.Series({"bid": 5.0, "state": "TX"})
    result = predict.optimize_bid_for_row(
        row=row, model=_StubModel(0.5), expected_revenue=25.0,
        target_cm=0.25, min_bid=0.25, bid_step=0.25,
    )
    assert result["max_bid"] == pytest.approx(25.0 * 0.75)  # 18.75
    assert result["recommended_bid"] == pytest.approx(0.25)  # lowest bid wins
    assert result["recommended_bid_predicted_win_rate"] == pytest.approx(0.5)
    assert result["n_candidate_bids"] > 0


def test_optimize_bid_skips_when_no_room():
    row = pd.Series({"bid": 1.0})
    # expected_revenue tiny -> max_bid below min_bid -> no candidates
    result = predict.optimize_bid_for_row(
        row=row, model=_StubModel(), expected_revenue=0.1,
        target_cm=0.25, min_bid=0.25, bid_step=0.25,
    )
    assert np.isnan(result["recommended_bid"])
    assert result["n_candidate_bids"] == 0


def test_run_bid_optimizer_evaluation_summary():
    numeric, categorical = fe.model_feature_columns(fe.LEAD_TYPE_AUTO)
    feature_cols = numeric + categorical
    n = 4
    df = pd.DataFrame({c: np.ones(n) for c in numeric})
    for c in categorical:
        df[c] = "A"
    df["bid"] = [5.0, 6.0, 7.0, 8.0]
    df[config.REVENUE_COL] = [20.0, 22.0, 25.0, 30.0]

    out = predict.run_bid_optimizer_evaluation(
        test_eval_df=df, model=_StubModel(0.5), feature_cols=feature_cols,
    )
    assert out is not None
    eval_df, summary = out
    assert summary["optimizer_rows"] == n
    # constant win rate -> optimizer lowers bids -> decreases dominate
    assert summary["bid_decrease_count"] == n
    keys = {"expected_profit_lift_pct", "avg_recommended_bid_cm_if_won"}
    assert keys.issubset(summary)
