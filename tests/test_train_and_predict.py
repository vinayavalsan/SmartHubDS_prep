"""Tests for the Anton train/predict module — pure logic only.

Covers the feature-parity contract, training-table preparation, and the bid
optimizer math. sklearn/mlflow/fastapi are NOT required here (predict keeps
those imports lazy/guarded), so these run in the base env.
"""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from smarthub.core.lead_types import lead_type_id
from smarthub.feature_engineering import features as fe
from smarthub.feature_engineering.feature_registry import FEATURES, MISSING_CATEGORY
from smarthub.server import predict
from smarthub.train_and_predict import (
    config,
    optimizer,
    optimizer_evaluation,
    preprocessing,
    registry,
)

# --- Feature schema (single source of truth) --------------------------------


def test_model_feature_columns_auto_vs_home():
    num_auto, cat_auto = fe.model_feature_columns(lead_type_id("auto"))
    num_home, cat_home = fe.model_feature_columns(lead_type_id("home"))

    auto_features = set(num_auto) | set(cat_auto)
    home_features = set(num_home) | set(cat_home)

    # auto-only features present for auto, absent for home
    for col in ("multi_vehicle", "num_vehicles", "home_owner"):
        assert col in auto_features
        assert col not in home_features

    # home-only features present for home, absent for auto
    for col in ("num_home_claims", "home_property_type"):
        assert col in home_features
        assert col not in auto_features

    # shared features present in both
    for col in (
        "age",
        "marital_status",
        "created_hour",
        "is_workday",
        "state",
        "traffic_tier",
    ):
        assert col in auto_features
        assert col in home_features

    # source_type_id is an enabled shared categorical model feature
    assert "source_type_id" in auto_features
    assert "source_type_id" in home_features

    # account_id is retained metadata, not a model feature
    assert "account_id" not in auto_features
    assert "account_id" not in home_features


def test_derive_serving_features_parity(monkeypatch):
    raw = pd.DataFrame(
        [
            {
                "marital_status": "Married",
                "num_vehicles": 2,
                "age": 40,
                "created_hour": 9,
                "created_dayofweek": 2,
            }
        ]
    )

    original = FEATURES["is_married"]
    monkeypatch.setitem(FEATURES, "is_married", replace(original, enabled=False))

    out = fe.derive_serving_features(raw)

    if FEATURES["multi_vehicle"].enabled:
        assert out.loc[0, "multi_vehicle"] == 1
    assert "is_married" not in out.columns
    assert out.loc[0, "age"] == 40
    assert out.loc[0, "created_hour"] == "9"


# --- Serving frame -----------------------------------------------------------


def test_serving_frame_selects_and_normalizes():
    record = {
        "campaign_id": 123,
        "account_id": 7,
        "source_type_id": 2,
        "state": "TX",
        "gender": "Female",
        "marital_status": "",
        "insured": "true",
        "home_owner": "false",
        "dui": "false",
        "military_affiliation": "false",
        "num_vehicles": 2,
        "num_drivers": 1,
        "num_auto_violations": 0,
        "num_auto_accidents": 0,
        "continuous_coverage_months": 24,
        "age": 34,
        "created_hour": 14,
        "created_dayofweek": 2,
        "bid": 0.25,
    }
    frame = preprocessing.serving_frame([record], lead_type_id("auto"))
    numeric, categorical = fe.model_feature_columns(lead_type_id("auto"))

    assert list(frame.columns) == numeric + categorical
    assert frame.loc[0, "campaign_id"] == "123"
    assert frame.loc[0, "marital_status"] == MISSING_CATEGORY

    if FEATURES["multi_vehicle"].enabled:
        assert frame.loc[0, "multi_vehicle"] == 1

    selected = set(numeric) | set(categorical)
    for name, spec in FEATURES.items():
        if not spec.enabled:
            assert name not in selected


# --- Training-table preparation ---------------------------------------------


def _fake_training_table():
    numeric, categorical = fe.model_feature_columns(lead_type_id("auto"))
    n = 6
    data = {
        "id": range(n),
        "created_at": pd.to_datetime([f"2026-06-2{i} 01:00" for i in range(n)]),
        "won_flag": [1, 0, 1, 0, 1, 0],
        "expected_revenue": [20.0, 18.0, 25.0, 22.0, 30.0, 15.0],
    }
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
        lead_type_id("auto"), "auto"
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


def test_find_zero_variance_features_reports_without_dropping():
    frame = pd.DataFrame(
        {
            "a": [1, 2, 3],
            "b": [5, 5, 5],
            "c": ["x", "y", "x"],
            "d": ["k", "k", "k"],
        }
    )

    zero_variance = preprocessing.find_zero_variance_features(
        frame,
        ["a", "b"],
        ["c", "d"],
    )

    assert set(zero_variance) == {"b", "d"}


def test_assert_trainable_rejects_single_class():
    # win_rate=1.0 (all wins) -> not trainable, clear message.
    frame = pd.DataFrame({config.TARGET_COL: [1, 1, 1, 1]})
    with pytest.raises(ValueError, match="only ONE target class"):
        preprocessing.assert_trainable(frame, "auto")


def test_assert_trainable_accepts_two_classes():
    frame = pd.DataFrame({config.TARGET_COL: [1, 0, 1, 0]})
    assert preprocessing.assert_trainable(frame, "auto") == [0, 1]


def test_assert_partition_has_both_classes_rejects_single_class():
    frame = pd.DataFrame({config.TARGET_COL: [0, 0, 0]})

    with pytest.raises(ValueError, match="Training partition.*only one target class"):
        preprocessing.assert_partition_has_both_classes(
            frame,
            "auto",
            "Training partition",
        )


def test_assert_partition_has_both_classes_accepts_two_classes():
    frame = pd.DataFrame({config.TARGET_COL: [0, 1, 0, 1]})

    preprocessing.assert_partition_has_both_classes(
        frame,
        "auto",
        "Test partition",
    )


def test_prepare_training_data_missing_target(monkeypatch):
    table = _fake_training_table().drop(columns=["won_flag"])
    monkeypatch.setattr(
        preprocessing.io, "load_training_table", lambda name, v=None: table
    )
    monkeypatch.setattr(
        preprocessing.io,
        "load_training_metadata",
        lambda name, v=None: {
            "data_min_created_at": "2026-06-20",
            "data_max_created_at": "2026-07-06",
            "row_count": len(table),
        },
    )

    with pytest.raises(ValueError, match="missing target"):
        preprocessing.prepare_training_data(
            lead_type_id("auto"),
            "auto",
            version="test-version",
        )


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
    result = optimizer.optimize_bid_for_row(
        row=row,
        model=_StubModel(0.5),
        expected_revenue=25.0,
        target_cm=0.25,
        min_bid=0.25,
        bid_step=0.25,
    )
    assert result["max_bid"] == pytest.approx(25.0 * 0.75)  # 18.75
    assert result["recommended_bid"] == pytest.approx(0.25)  # lowest bid wins
    assert result["recommended_bid_predicted_win_rate"] == pytest.approx(0.5)
    assert result["n_candidate_bids"] > 0


def test_optimize_bid_skips_when_no_room():
    row = pd.Series({"bid": 1.0})
    # expected_revenue tiny -> max_bid below min_bid -> no candidates
    result = optimizer.optimize_bid_for_row(
        row=row,
        model=_StubModel(),
        expected_revenue=0.1,
        target_cm=0.25,
        min_bid=0.25,
        bid_step=0.25,
    )
    assert np.isnan(result["recommended_bid"])
    assert result["n_candidate_bids"] == 0


# --- Model resolution (MODEL_URI env > pinned version > currently serving) --


def test_resolve_model_uri_prefers_env_override(tmp_path, monkeypatch):
    monkeypatch.setattr(registry, "MODEL_DIR_ROOT", tmp_path / "models")
    monkeypatch.setenv("MODEL_URI", "s3://somewhere/pinned-model.pkl")
    resolved = predict.resolve_model_uri(lead_type_id("auto"))
    assert resolved == "s3://somewhere/pinned-model.pkl"


def test_resolve_model_uri_uses_pinned_ini_version_over_currently_serving(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(registry, "MODEL_DIR_ROOT", tmp_path / "models")
    monkeypatch.delenv("MODEL_URI", raising=False)
    manifest = registry.save_version(
        {"m": 1},
        "auto",
        feature_cols=["bid"],
        metrics={"roc_auc": 0.7},
        optimizer_summary={},
        lineage={},
        model_params={},
        training_config={},
        promotion_mode="manual",
        eligibility_status="eligible",
        promotion_status="awaiting_manual_promotion",
        promotion_decision_reason="not evaluated",
    )
    registry.promote("auto", manifest["version"])
    monkeypatch.setattr(config, "active_model_version", lambda: manifest["version"])

    resolved = predict.resolve_model_uri(lead_type_id("auto"))
    assert resolved == str(registry.version_path("auto", manifest["version"]))


def test_resolve_model_uri_falls_back_to_currently_serving(tmp_path, monkeypatch):
    monkeypatch.setattr(registry, "MODEL_DIR_ROOT", tmp_path / "models")
    monkeypatch.delenv("MODEL_URI", raising=False)
    monkeypatch.setattr(config, "active_model_version", lambda: None)
    manifest = registry.save_version(
        {"m": 1},
        "auto",
        feature_cols=["bid"],
        metrics={"roc_auc": 0.7},
        optimizer_summary={},
        lineage={},
        model_params={},
        training_config={},
        promotion_mode="manual",
        eligibility_status="eligible",
        promotion_status="awaiting_manual_promotion",
        promotion_decision_reason="not evaluated",
    )
    registry.promote("auto", manifest["version"])

    resolved = predict.resolve_model_uri(lead_type_id("auto"))
    assert resolved == str(registry.currently_serving_model_path("auto"))


def test_resolve_model_uri_nothing_serving_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(registry, "MODEL_DIR_ROOT", tmp_path / "models")
    monkeypatch.delenv("MODEL_URI", raising=False)
    monkeypatch.setattr(config, "active_model_version", lambda: None)
    with pytest.raises(FileNotFoundError):
        predict.resolve_model_uri(lead_type_id("auto"))


def test_run_bid_optimizer_evaluation_summary():
    numeric, categorical = fe.model_feature_columns(lead_type_id("auto"))
    feature_cols = numeric + categorical
    n = 4
    df = pd.DataFrame({c: np.ones(n) for c in numeric})
    for c in categorical:
        df[c] = "A"
    df["bid"] = [5.0, 6.0, 7.0, 8.0]
    df[config.REVENUE_COL] = [20.0, 22.0, 25.0, 30.0]
    df[config.TARGET_COL] = [1, 0, 1, 0]

    out = optimizer_evaluation.run_bid_optimizer_evaluation(
        test_eval_df=df,
        model=_StubModel(0.5),
        feature_cols=feature_cols,
        target_cm=0.25,
        min_bid=0.25,
        bid_step=0.25,
        chunk_size=100,
        monotonicity_enabled=True,
        monotonicity_tolerance=1e-8,
        monotonicity_max_violation_rate=0.0,
    )
    assert out is not None
    eval_df, summary = out
    assert summary.optimizer_rows == n
    # Constant win rate -> optimizer lowers every bid.
    assert summary.bid_decrease_pct == pytest.approx(100.0)
    assert summary.bid_increase_pct == pytest.approx(0.0)
    assert summary.bid_unchanged_pct == pytest.approx(0.0)

    keys = {
        "expected_profit_lift_pct",
        "avg_recommended_bid_cm_if_won",
        "bid_decrease_pct",
    }
    assert keys.issubset(summary.to_dict())
