"""Tests for smarthub.server.predict -- decide_bid / exploration_slot /
bid_curve_around / load_model_and_manifest.

sklearn/mlflow/fastapi are NOT required here -- small fake models with a
hand-computable predict_proba stand in for a real fitted estimator, same
"pure logic only" convention as test_train_and_predict.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from smarthub.feature_engineering import features as fe
from smarthub.server import predict
from smarthub.train_and_predict import config, optimizer, registry


class _ConstantWinRateModel:
    """predict_proba always returns the same win rate regardless of bid --
    makes the profit-maximizing bid deterministically the minimum candidate
    bid (profit = win_rate * (revenue - bid) is then strictly decreasing in
    bid), so exact numbers are easy to assert against."""

    def __init__(self, win_rate: float = 0.6):
        self.win_rate = win_rate

    def predict_proba(self, frame):
        win = np.full(len(frame), self.win_rate)
        return np.column_stack([1 - win, win])


class _LinearWinRateModel:
    """win_rate = intercept - slope * bid (clipped to [0, 1]) -- lets a test
    hand-verify a *specific* bid's win rate, unlike the constant model,
    which can't distinguish "recomputed at the new bid" from "reused the
    old value" since it returns the same number either way."""

    def __init__(self, intercept: float = 1.0, slope: float = 0.02):
        self.intercept = intercept
        self.slope = slope

    def predict_proba(self, frame):
        bids = frame["bid"].to_numpy(dtype=float)
        win = np.clip(self.intercept - self.slope * bids, 0.0, 1.0)
        return np.column_stack([1 - win, win])


def _row():
    return pd.Series({"some_feature": 1.0})


# --- exploration_slot --------------------------------------------------------


def test_exploration_slot_disabled_without_dayofweek_or_hour():
    assert predict.exploration_slot(None, 5) == (False, 0)
    assert predict.exploration_slot(2, None) == (False, 0)


def test_exploration_slot_disabled_at_zero_variance():
    for dow in range(7):
        for hour in range(24):
            assert predict.exploration_slot(dow, hour, variance_pct=0.0) == (
                False,
                0,
            )


def test_exploration_slot_density_and_alternating_direction():
    # variance_pct=0.10 -> N=10 -> every 10th hour-of-week bucket triggers.
    triggered = []
    for bucket in range(168):
        dow, hour = divmod(bucket, 24)
        is_explore, direction = predict.exploration_slot(dow, hour, variance_pct=0.10)
        if is_explore:
            triggered.append((bucket, direction))

    buckets = [b for b, _ in triggered]
    assert buckets == list(range(0, 168, 10))

    directions = [d for _, d in triggered]
    assert directions == [1 if i % 2 == 0 else -1 for i in range(len(directions))]


def test_exploration_slot_is_deterministic():
    a = predict.exploration_slot(3, 10, variance_pct=0.10)
    b = predict.exploration_slot(3, 10, variance_pct=0.10)
    assert a == b


# --- decide_bid: no viable bid / cold start ----------------------------------


def test_decide_bid_no_viable_bid_regardless_of_model():
    result = predict.decide_bid(
        row=_row(),
        model=None,
        manifest=None,
        expected_revenue=0.05,
        target_cm=0.25,
        min_bid=0.25,
        bid_step=0.25,
        created_dayofweek=0,
        created_hour=0,
    )
    assert pd.isna(result["recommended_bid"])
    assert "No viable bid" in result["decision_reason"]


def test_decide_bid_cold_start_fallback_bid_formula(monkeypatch):
    monkeypatch.setattr(config, "cold_start_fallback_bid_pct", lambda: 0.50)
    result = predict.decide_bid(
        row=_row(),
        model=None,
        manifest=None,
        expected_revenue=20.0,
        target_cm=0.25,
        min_bid=0.0,
        bid_step=1.0,
        created_dayofweek=2,
        created_hour=9,
    )
    assert result["decision_path"] == "cold_start_fallback"
    assert result["recommended_bid_predicted_win_rate"] is None
    assert result["recommended_bid_predicted_profit"] is None
    assert result["model_data_age_days"] is None
    # max_bid = 20 * (1 - 0.25) = 15.0; fallback bids 50% of the way from
    # min_bid (0.0) to that ceiling, so squarely in the middle.
    assert result["max_bid"] == pytest.approx(15.0)
    assert 0.0 <= result["recommended_bid"] <= result["max_bid"]
    assert result["recommended_bid"] == pytest.approx(7.5, abs=0.5)


# --- decide_bid: model path ---------------------------------------------------


def test_decide_bid_model_path_matches_raw_optimizer_when_not_exploring():
    model = _ConstantWinRateModel(win_rate=0.6)
    row = _row()
    result = predict.decide_bid(
        row=row,
        model=model,
        manifest=None,
        expected_revenue=20.0,
        target_cm=0.25,
        min_bid=1.0,
        bid_step=1.0,
        created_dayofweek=0,
        created_hour=1,  # bucket 1 -- not a multiple of 10, no explore
    )
    assert result["decision_path"] == "model"
    assert "Standard profit-maximizing bid" in result["decision_reason"]
    # constant win-rate -> profit strictly decreasing in bid -> optimum = floor
    assert result["recommended_bid"] == pytest.approx(1.0)
    assert result["recommended_bid_predicted_win_rate"] == pytest.approx(0.6)
    assert result["recommended_bid_predicted_profit"] == pytest.approx(
        0.6 * (20.0 - 1.0)
    )


def test_decide_bid_recency_flag_when_model_is_stale(monkeypatch):
    monkeypatch.setattr(config, "recency_window_days", lambda: 30)
    stale_manifest = {"created_at": "2020-01-01T00:00:00+00:00"}
    result = predict.decide_bid(
        row=_row(),
        model=_ConstantWinRateModel(),
        manifest=stale_manifest,
        expected_revenue=20.0,
        target_cm=0.25,
        min_bid=1.0,
        bid_step=1.0,
        created_dayofweek=0,
        created_hour=1,
    )
    assert result["model_data_age_days"] > 30
    assert "recency window" in result["decision_reason"]


def test_decide_bid_no_recency_note_when_manifest_missing():
    result = predict.decide_bid(
        row=_row(),
        model=_ConstantWinRateModel(),
        manifest=None,
        expected_revenue=20.0,
        target_cm=0.25,
        min_bid=1.0,
        bid_step=1.0,
        created_dayofweek=0,
        created_hour=1,
    )
    assert result["model_data_age_days"] is None
    assert "recency window" not in result["decision_reason"]


# --- decide_bid: exploration path ---------------------------------------------


def test_decide_bid_exploration_perturbs_and_rescores_at_the_new_bid():
    model = _LinearWinRateModel(intercept=1.0, slope=0.02)
    row = _row()
    expected_revenue, target_cm, min_bid, bid_step = 20.0, 0.25, 5.0, 0.5

    base = optimizer.optimize_bid_for_row(
        row, model, expected_revenue, target_cm, min_bid, bid_step
    )
    explore, direction = predict.exploration_slot(0, 0, variance_pct=0.10)
    assert explore and direction == 1  # bucket 0 always triggers, occurrence 0

    expected_perturbed_raw = base["recommended_bid"] * (1 + direction * 0.10)
    expected_perturbed_bid = predict._snap_to_grid(
        expected_perturbed_raw, min_bid, base["max_bid"], bid_step
    )
    expected_win_rate, expected_profit = predict._score_bid(
        row, model, expected_perturbed_bid, expected_revenue
    )
    # Sanity: the perturbation actually moved the bid for these parameters
    # (otherwise this test wouldn't be exercising the recompute-at-new-bid
    # logic at all).
    assert expected_perturbed_bid != base["recommended_bid"]

    result = predict.decide_bid(
        row=row,
        model=model,
        manifest=None,
        expected_revenue=expected_revenue,
        target_cm=target_cm,
        min_bid=min_bid,
        bid_step=bid_step,
        created_dayofweek=0,
        created_hour=0,
    )
    assert result["decision_path"] == "exploration"
    assert "Scheduled exploration probe" in result["decision_reason"]
    assert result["recommended_bid"] == pytest.approx(expected_perturbed_bid)
    assert result["recommended_bid_predicted_win_rate"] == pytest.approx(
        expected_win_rate
    )
    assert result["recommended_bid_predicted_profit"] == pytest.approx(expected_profit)


# --- bid_curve_around ---------------------------------------------------------


def test_bid_curve_around_empty_on_no_model_or_nan_center():
    assert (
        predict.bid_curve_around(
            row=_row(),
            model=None,
            expected_revenue=20.0,
            min_bid=0.0,
            max_bid=15.0,
            bid_step=1.0,
            center_bid=5.0,
        )
        == []
    )
    assert (
        predict.bid_curve_around(
            row=_row(),
            model=_ConstantWinRateModel(),
            expected_revenue=20.0,
            min_bid=0.0,
            max_bid=15.0,
            bid_step=1.0,
            center_bid=float("nan"),
        )
        == []
    )


def test_bid_curve_around_symmetric_spanning():
    model = _ConstantWinRateModel(win_rate=0.6)
    curve = predict.bid_curve_around(
        row=_row(),
        model=model,
        expected_revenue=20.0,
        min_bid=0.0,
        max_bid=15.0,
        bid_step=1.0,
        center_bid=1.0,
        n_points=3,
    )
    assert [p["bid"] for p in curve] == [0.0, 1.0, 2.0]
    for p in curve:
        assert p["predicted_win_rate"] == pytest.approx(0.6)
        assert p["expected_profit"] == pytest.approx(0.6 * (20.0 - p["bid"]))


def test_bid_curve_around_edge_clip_without_duplicate_bids():
    model = _ConstantWinRateModel()
    curve = predict.bid_curve_around(
        row=_row(),
        model=model,
        expected_revenue=20.0,
        min_bid=0.0,
        max_bid=15.0,
        bid_step=1.0,
        center_bid=0.0,
        n_points=3,
    )
    # offsets -1/0/+1 around 0.0 clip to {0, 0, 1} -> deduped to [0.0, 1.0]
    assert [p["bid"] for p in curve] == [0.0, 1.0]


# --- load_model_and_manifest ---------------------------------------------------


def test_load_model_and_manifest_cold_start(tmp_path, monkeypatch):
    monkeypatch.setattr(registry, "MODEL_DIR_ROOT", tmp_path / "models")
    monkeypatch.delenv("MODEL_URI", raising=False)
    monkeypatch.setattr(config, "active_model_version", lambda: None)
    predict.clear_model_cache()

    model, manifest = predict.load_model_and_manifest(fe.LEAD_TYPE_AUTO)
    assert model is None
    assert manifest is None


def test_load_model_and_manifest_currently_serving(tmp_path, monkeypatch):
    monkeypatch.setattr(registry, "MODEL_DIR_ROOT", tmp_path / "models")
    monkeypatch.delenv("MODEL_URI", raising=False)
    monkeypatch.setattr(config, "active_model_version", lambda: None)
    predict.clear_model_cache()

    fake_model = _ConstantWinRateModel(win_rate=0.42)
    saved_manifest = registry.save_version(
        fake_model,
        "auto",
        feature_cols=["bid"],
        metrics={"roc_auc": 0.7},
        optimizer_summary={},
        lineage={},
        model_params={},
        promotion_mode="manual",
        promotion_eligible=None,
        promotion_decision_reason="not evaluated",
    )
    registry.promote("auto", saved_manifest["version"])

    model, manifest = predict.load_model_and_manifest(fe.LEAD_TYPE_AUTO)
    assert isinstance(model, _ConstantWinRateModel)
    assert model.win_rate == pytest.approx(0.42)
    assert manifest["version"] == saved_manifest["version"]


def test_load_model_and_manifest_pinned_version(tmp_path, monkeypatch):
    monkeypatch.setattr(registry, "MODEL_DIR_ROOT", tmp_path / "models")
    monkeypatch.delenv("MODEL_URI", raising=False)
    predict.clear_model_cache()

    fake_model = _ConstantWinRateModel(win_rate=0.11)
    saved_manifest = registry.save_version(
        fake_model,
        "auto",
        feature_cols=["bid"],
        metrics={"roc_auc": 0.7},
        optimizer_summary={},
        lineage={},
        model_params={},
        promotion_mode="manual",
        promotion_eligible=None,
        promotion_decision_reason="not evaluated",
    )
    # Deliberately do NOT promote it -- pinning should still find it.
    monkeypatch.setattr(
        config, "active_model_version", lambda: saved_manifest["version"]
    )

    model, manifest = predict.load_model_and_manifest(fe.LEAD_TYPE_AUTO)
    assert model.win_rate == pytest.approx(0.11)
    assert manifest["version"] == saved_manifest["version"]


def test_load_model_and_manifest_env_override_has_no_manifest(tmp_path, monkeypatch):
    import joblib

    model_path = tmp_path / "pinned.pkl"
    joblib.dump(_ConstantWinRateModel(win_rate=0.7), model_path)
    monkeypatch.setenv("MODEL_URI", str(model_path))
    predict.clear_model_cache()

    model, manifest = predict.load_model_and_manifest(fe.LEAD_TYPE_AUTO)
    assert model.win_rate == pytest.approx(0.7)
    assert manifest is None
