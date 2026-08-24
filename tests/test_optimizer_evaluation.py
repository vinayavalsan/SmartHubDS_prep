"""Tests for offline bid-optimizer evaluation and diagnostics."""

from __future__ import annotations

from contextlib import contextmanager

import numpy as np
import pandas as pd
import pytest

from smarthub.train_and_predict import optimizer_evaluation as oe


class _BidModel:
    """Simple model whose win rate increases linearly with bid."""

    def predict_proba(self, frame):
        bids = frame["bid"].to_numpy(dtype=float)
        win = np.clip(0.10 + 0.05 * bids, 0.0, 1.0)
        return np.column_stack([1.0 - win, win])


def _eval_frame():
    return pd.DataFrame(
        {
            "bid": [1.0, 2.0, 3.0],
            "expected_revenue": [10.0, 20.0, 30.0],
            "feature": [5.0, 6.0, 7.0],
            "won_flag": [1, 0, 1],
        }
    )


def test_prepare_frame_requires_revenue_column():
    frame = pd.DataFrame({"bid": [1.0, 2.0]})

    assert oe._prepare_frame(frame) is None


def test_prepare_frame_drops_missing_and_nonpositive_revenue():
    frame = pd.DataFrame(
        {
            "bid": [1.0, None, 2.0, 3.0],
            "expected_revenue": [10.0, 20.0, 0.0, -5.0],
            "row_id": [1, 2, 3, 4],
        }
    )

    result = oe._prepare_frame(frame)

    assert result["row_id"].tolist() == [1]


def test_score_current_bids_uses_historical_bid_and_expected_profit():
    frame = _eval_frame()

    result = oe._score_current_bids(frame, _BidModel(), ["bid", "feature"])

    expected_win = np.array([0.15, 0.20, 0.25])
    expected_profit = expected_win * (
        frame["expected_revenue"].to_numpy() - frame["bid"].to_numpy()
    )
    assert result["current_bid_predicted_win_rate"].to_numpy() == pytest.approx(
        expected_win
    )
    assert result["current_bid_expected_profit"].to_numpy() == pytest.approx(
        expected_profit
    )


def test_add_diagnostics_computes_lift_change_and_cm():
    frame = pd.DataFrame(
        {
            "expected_revenue": [10.0, 20.0],
            "bid": [2.0, 5.0],
            "current_bid_expected_profit": [1.0, 2.0],
            "recommended_bid": [3.0, 4.0],
            "recommended_bid_expected_profit": [1.5, 3.0],
            "won_flag": [1, 0],
        }
    )

    result = oe._add_diagnostics(frame)

    assert result["expected_profit_lift"].tolist() == pytest.approx([0.5, 1.0])
    assert result["bid_change"].tolist() == pytest.approx([1.0, -1.0])
    assert result["recommended_bid_cm_if_won"].tolist() == pytest.approx([0.7, 0.8])


def test_summarize_results_aggregates_directional_bid_diagnostics():
    frame = pd.DataFrame(
        {
            "current_bid_expected_profit": [10.0, 20.0, 30.0],
            "recommended_bid_expected_profit": [12.0, 18.0, 35.0],
            "current_bid_predicted_win_rate": [0.1, 0.2, 0.3],
            "recommended_bid_predicted_win_rate": [0.2, 0.25, 0.35],
            "bid_change": [1.0, -2.0, 0.0],
            "recommended_bid_cm_if_won": [0.5, 0.6, 0.7],
            "won_flag": [1, 0, 1],
            "observed_policy_expected_revenue": [10.0, 0.0, 30.0],
            "observed_policy_bid_cost": [2.0, 0.0, 5.0],
            "observed_policy_expected_profit": [8.0, 0.0, 25.0],
        }
    )

    summary = oe.summarize_results(frame, target_cm=0.25)

    assert summary.optimizer_rows == 3
    assert summary.current_bid_total_expected_profit == pytest.approx(60.0)
    assert summary.recommended_bid_total_expected_profit == pytest.approx(65.0)
    assert summary.expected_profit_lift_total == pytest.approx(5.0)
    assert summary.expected_profit_lift_pct == pytest.approx(5.0 / 60.0)
    assert summary.avg_current_bid_predicted_win_rate == pytest.approx(0.2)
    assert summary.avg_recommended_bid_predicted_win_rate == pytest.approx(0.8 / 3.0)
    assert summary.avg_bid_change == pytest.approx(-1.0 / 3.0)
    assert summary.median_bid_change == pytest.approx(0.0)
    assert summary.bid_increase_pct == pytest.approx(100.0 / 3.0)
    assert summary.bid_decrease_pct == pytest.approx(100.0 / 3.0)
    assert summary.bid_unchanged_pct == pytest.approx(100.0 / 3.0)
    assert summary.avg_recommended_bid_cm_if_won == pytest.approx(0.6)


def test_summarize_results_zero_current_profit_returns_nan_lift_ratio():
    frame = pd.DataFrame(
        {
            "current_bid_expected_profit": [0.0, 0.0],
            "recommended_bid_expected_profit": [1.0, 2.0],
            "current_bid_predicted_win_rate": [0.0, 0.0],
            "recommended_bid_predicted_win_rate": [0.1, 0.2],
            "bid_change": [1.0, 1.0],
            "recommended_bid_cm_if_won": [0.5, 0.5],
            "won_flag": [0, 0],
            "observed_policy_expected_revenue": [0.0, 0.0],
            "observed_policy_bid_cost": [0.0, 0.0],
            "observed_policy_expected_profit": [0.0, 0.0],
        }
    )

    summary = oe.summarize_results(frame, target_cm=0.25)

    assert np.isnan(summary.expected_profit_lift_pct)


def test_optimizer_summary_mlflow_metrics_are_prefixed():
    summary = oe.OptimizerSummary(
        optimizer_rows=2,
        target_cm=0.25,
        observed_policy_wins=1,
        observed_policy_win_rate=0.5,
        observed_policy_total_expected_revenue=10.0,
        observed_policy_total_bid_cost=2.0,
        observed_policy_total_expected_profit=8.0,
        observed_policy_expected_cm=0.8,
        current_bid_total_expected_profit=10.0,
        recommended_bid_total_expected_profit=12.0,
        expected_profit_lift_total=2.0,
        expected_profit_lift_pct=0.2,
        avg_current_bid_predicted_win_rate=0.2,
        avg_recommended_bid_predicted_win_rate=0.3,
        avg_bid_change=1.0,
        median_bid_change=1.0,
        bid_increase_pct=100.0,
        bid_decrease_pct=0.0,
        bid_unchanged_pct=0.0,
        avg_recommended_bid_cm_if_won=0.5,
    )

    metrics = summary.mlflow_metrics()

    assert metrics["optimizer_optimizer_rows"] == 2
    assert metrics["optimizer_expected_profit_lift_total"] == pytest.approx(2.0)
    assert set(metrics) == {f"optimizer_{name}" for name in summary.to_dict()}


def test_summarize_monotonicity_disabled_returns_skipped_summary():
    summary = oe._summarize_monotonicity(
        {
            "checked_rows": 99,
            "checked_steps": 99,
            "violation_count": 99,
        },
        enabled=False,
        max_violation_rate=0.01,
    )

    assert summary.enabled is False
    assert summary.checked_rows == 0
    assert summary.checked_steps == 0
    assert summary.violation_count == 0
    assert summary.passed is None
    assert summary.max_allowed_violation_rate == pytest.approx(0.01)


def test_summarize_monotonicity_computes_rates_and_magnitudes():
    summary = oe._summarize_monotonicity(
        {
            "checked_rows": 4,
            "checked_steps": 20,
            "violation_count": 2,
            "rows_with_violation": 1,
            "violation_magnitude_sum": 0.06,
            "max_violation_magnitude": 0.04,
        },
        enabled=True,
        max_violation_rate=0.10,
    )

    assert summary.checked_rows == 4
    assert summary.checked_steps == 20
    assert summary.violation_count == 2
    assert summary.violation_rate == pytest.approx(0.10)
    assert summary.rows_with_violation_pct == pytest.approx(0.25)
    assert summary.mean_violation_magnitude == pytest.approx(0.03)
    assert summary.max_violation_magnitude == pytest.approx(0.04)
    assert summary.passed is True


def test_summarize_monotonicity_fails_above_allowed_rate():
    summary = oe._summarize_monotonicity(
        {
            "checked_rows": 2,
            "checked_steps": 10,
            "violation_count": 2,
            "rows_with_violation": 2,
        },
        enabled=True,
        max_violation_rate=0.10,
    )

    assert summary.violation_rate == pytest.approx(0.20)
    assert summary.passed is False


def test_summarize_monotonicity_handles_zero_checked_steps():
    summary = oe._summarize_monotonicity(
        {},
        enabled=True,
        max_violation_rate=0.0,
    )

    assert summary.violation_rate == 0.0
    assert summary.rows_with_violation_pct == 0.0
    assert summary.mean_violation_magnitude == 0.0
    assert summary.passed is True


def test_run_bid_optimizer_evaluation_returns_none_without_valid_rows():
    frame = pd.DataFrame(
        {
            "bid": [1.0, 2.0],
            "expected_revenue": [0.0, -1.0],
        }
    )

    result = oe.run_bid_optimizer_evaluation(
        frame,
        _BidModel(),
        ["bid"],
        target_cm=0.25,
        min_bid=0.25,
        bid_step=0.25,
        chunk_size=100,
        monotonicity_enabled=True,
        monotonicity_tolerance=1e-8,
        monotonicity_max_violation_rate=0.0,
    )

    assert result is None


def test_run_bid_optimizer_evaluation_propagates_monotonicity_diagnostics(
    monkeypatch,
):
    frame = _eval_frame()

    @contextmanager
    def _quiet():
        yield

    monkeypatch.setattr(oe.optimizer, "quiet_feature_name_warning", _quiet)

    def _fake_score_recommended_bids(
        eval_df,
        model,
        feature_cols,
        target_cm,
        min_bid,
        bid_step,
        chunk_size,
        monotonicity_tolerance=None,
    ):
        result = eval_df.copy()
        result["recommended_bid"] = [2.0, 1.0, 3.0]
        result["recommended_bid_predicted_win_rate"] = [0.2, 0.15, 0.25]
        result["recommended_bid_expected_profit"] = [1.6, 2.85, 6.75]
        result.attrs["monotonicity_diagnostics"] = {
            "checked_rows": 3,
            "checked_steps": 12,
            "violation_count": 1,
            "rows_with_violation": 1,
            "violation_magnitude_sum": 0.02,
            "max_violation_magnitude": 0.02,
        }
        return result

    monkeypatch.setattr(
        oe.optimizer,
        "score_recommended_bids",
        _fake_score_recommended_bids,
    )

    evaluated, summary = oe.run_bid_optimizer_evaluation(
        frame,
        _BidModel(),
        ["bid", "feature"],
        target_cm=0.25,
        min_bid=0.25,
        bid_step=0.25,
        chunk_size=2,
        monotonicity_enabled=True,
        monotonicity_tolerance=0.001,
        monotonicity_max_violation_rate=0.10,
        log_summary_result=False,
    )

    monotonicity = evaluated.attrs["monotonicity_summary"]
    assert monotonicity["checked_rows"] == 3
    assert monotonicity["checked_steps"] == 12
    assert monotonicity["violation_count"] == 1
    assert monotonicity["violation_rate"] == pytest.approx(1.0 / 12.0)
    assert monotonicity["passed"] is True
    assert summary.optimizer_rows == 3


def test_run_bid_optimizer_evaluation_disables_monotonicity_collection(
    monkeypatch,
):
    frame = _eval_frame()
    seen = {}

    def _fake_score_recommended_bids(
        eval_df,
        model,
        feature_cols,
        target_cm,
        min_bid,
        bid_step,
        chunk_size,
        monotonicity_tolerance=None,
    ):
        seen["tolerance"] = monotonicity_tolerance
        result = eval_df.copy()
        result["recommended_bid"] = result["bid"]
        result["recommended_bid_predicted_win_rate"] = result[
            "current_bid_predicted_win_rate"
        ]
        result["recommended_bid_expected_profit"] = result[
            "current_bid_expected_profit"
        ]
        return result

    monkeypatch.setattr(
        oe.optimizer,
        "score_recommended_bids",
        _fake_score_recommended_bids,
    )

    evaluated, _ = oe.run_bid_optimizer_evaluation(
        frame,
        _BidModel(),
        ["bid", "feature"],
        target_cm=0.25,
        min_bid=0.25,
        bid_step=0.25,
        chunk_size=2,
        monotonicity_enabled=False,
        monotonicity_tolerance=0.001,
        monotonicity_max_violation_rate=0.0,
        log_summary_result=False,
    )

    assert seen["tolerance"] is None
    monotonicity = evaluated.attrs["monotonicity_summary"]
    assert monotonicity["enabled"] is False
    assert monotonicity["passed"] is None


def test_run_bid_optimizer_evaluation_returns_none_if_optimizer_fails(
    monkeypatch,
):
    monkeypatch.setattr(
        oe.optimizer,
        "score_recommended_bids",
        lambda *args, **kwargs: None,
    )

    result = oe.run_bid_optimizer_evaluation(
        _eval_frame(),
        _BidModel(),
        ["bid", "feature"],
        target_cm=0.25,
        min_bid=0.25,
        bid_step=0.25,
        chunk_size=100,
        monotonicity_enabled=True,
        monotonicity_tolerance=1e-8,
        monotonicity_max_violation_rate=0.0,
    )

    assert result is None
