"""Tests for SmartHub hyperparameter-search decision logic.

These tests focus on the parts of HPO that protect data separation and model
selection: final-test reservation, finalist holdout construction, probability
ranking, optimizer shortlisting, and bid-response guardrails. They intentionally
avoid running a full Optuna search or fitting production model families.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import optuna
import pandas as pd
import pytest
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit

from smarthub.train_and_predict import hyperparameter_search as hpo


def _frame(n: int = 20) -> pd.DataFrame:
    """Return a small chronologically ordered binary-classification frame."""
    return pd.DataFrame(
        {
            "row_id": range(n),
            "created_at": pd.date_range("2026-08-01", periods=n, freq="h"),
            hpo.config.TARGET_COL: [i % 2 for i in range(n)],
            "feature": np.arange(n, dtype=float),
        }
    )


def _probability_result(
    trial_number: int,
    *,
    log_loss: float,
    brier_score: float,
    profit: float | None = None,
    evaluated_rows: int = 10,
    optimizer_selected: bool = False,
    passes_log_loss_guardrail: bool = True,
    violation_rate: float = 0.0,
) -> dict:
    """Build one synthetic HPO finalist result."""
    optimizer_metrics = None
    if profit is not None:
        optimizer_metrics = {
            "evaluated_rows": evaluated_rows,
            "total_expected_profit": profit,
        }

    return {
        "trial_number": trial_number,
        "calibration_method": "none",
        "probability_metrics": {
            "log_loss": log_loss,
            "brier_score": brier_score,
        },
        "optimizer_selected": optimizer_selected,
        "optimizer_metrics": optimizer_metrics,
        "passes_log_loss_guardrail": passes_log_loss_guardrail,
        "monotonicity": {
            "violation_rate": violation_rate,
            "violation_count": int(violation_rate > 0),
            "checked_steps": 10,
        },
        "_model": object(),
    }


def _optimizer_settings(
    *,
    monotonicity_enabled: bool = True,
    max_violation_rate: float = 0.0,
    max_log_loss_regression: float = 0.02,
    optimizer_top_n: int = 2,
) -> dict:
    """Return the minimal settings needed by shortlist/finalist helpers."""
    return {
        "max_log_loss_regression": max_log_loss_regression,
        "optimizer_top_n": optimizer_top_n,
        "monotonicity": {
            "enabled": monotonicity_enabled,
            "max_violation_rate": max_violation_rate,
        },
    }


# --- probability scoring contract -------------------------------------------


@pytest.mark.parametrize("scoring", ["neg_log_loss", "neg_brier_score"])
def test_validate_probability_scoring_accepts_supported_metrics(scoring):
    hpo._validate_probability_scoring(scoring)


@pytest.mark.parametrize("scoring", ["accuracy", "roc_auc", "f1"])
def test_validate_probability_scoring_rejects_non_probability_metrics(scoring):
    with pytest.raises(ValueError, match="probability-quality"):
        hpo._validate_probability_scoring(scoring)


# --- final HPO test reservation ---------------------------------------------


def test_reserve_final_test_time_uses_newest_rows():
    frame = _frame(10).sample(frac=1.0, random_state=7).reset_index(drop=True)

    pool, final_test = hpo._reserve_final_test(
        frame,
        split_settings={"strategy": "time", "test_size": 0.2},
        random_seed=123,
    )

    assert pool["row_id"].tolist() == list(range(8))
    assert final_test["row_id"].tolist() == [8, 9]
    assert pool["created_at"].max() < final_test["created_at"].min()


def test_reserve_final_test_time_requires_created_at():
    frame = _frame(10).drop(columns="created_at")

    with pytest.raises(ValueError, match="created_at"):
        hpo._reserve_final_test(
            frame,
            split_settings={"strategy": "time", "test_size": 0.2},
            random_seed=123,
        )


@pytest.mark.parametrize("test_size", [0.0, 1.0, -0.1, 1.1])
def test_reserve_final_test_rejects_invalid_test_size(test_size):
    with pytest.raises(ValueError, match="test_size"):
        hpo._reserve_final_test(
            _frame(10),
            split_settings={"strategy": "time", "test_size": test_size},
            random_seed=123,
        )


def test_reserve_final_test_random_is_reproducible():
    frame = _frame(40)
    settings = {"strategy": "random", "test_size": 0.25, "stratify": True}

    pool_a, test_a = hpo._reserve_final_test(frame, settings, random_seed=17)
    pool_b, test_b = hpo._reserve_final_test(frame, settings, random_seed=17)

    assert pool_a["row_id"].tolist() == pool_b["row_id"].tolist()
    assert test_a["row_id"].tolist() == test_b["row_id"].tolist()
    assert test_a[hpo.config.TARGET_COL].mean() == pytest.approx(
        frame[hpo.config.TARGET_COL].mean()
    )


def test_reserve_final_test_rejects_unknown_strategy():
    with pytest.raises(ValueError, match="Unsupported HPO split strategy"):
        hpo._reserve_final_test(
            _frame(10),
            split_settings={"strategy": "future", "test_size": 0.2},
            random_seed=123,
        )


# --- development / finalist holdout -----------------------------------------


def test_split_development_and_holdout_reserves_newest_rows():
    frame = _frame(10).sample(frac=1.0, random_state=9).reset_index(drop=True)

    development, holdout = hpo._split_development_and_holdout(
        frame,
        holdout_fraction=0.3,
    )

    assert development["row_id"].tolist() == list(range(7))
    assert holdout["row_id"].tolist() == [7, 8, 9]
    assert development["created_at"].max() < holdout["created_at"].min()


def test_split_development_and_holdout_uses_ceiling_for_fraction():
    development, holdout = hpo._split_development_and_holdout(
        _frame(7),
        holdout_fraction=0.2,
    )

    assert len(holdout) == 2
    assert len(development) == 5


def test_split_development_and_holdout_requires_created_at():
    with pytest.raises(ValueError, match="created_at"):
        hpo._split_development_and_holdout(
            _frame(10).drop(columns="created_at"),
            holdout_fraction=0.2,
        )


def test_split_development_and_holdout_rejects_all_rows_in_holdout():
    with pytest.raises(ValueError, match="Not enough rows"):
        hpo._split_development_and_holdout(
            _frame(3),
            holdout_fraction=1.0,
        )


# --- CV construction ---------------------------------------------------------


def test_build_cv_time_returns_time_series_split():
    cv = hpo._build_cv("time", n_splits=4, random_seed=12)
    assert isinstance(cv, TimeSeriesSplit)
    assert cv.n_splits == 4


def test_build_cv_random_returns_seeded_stratified_split():
    cv = hpo._build_cv("random", n_splits=3, random_seed=12)
    assert isinstance(cv, StratifiedKFold)
    assert cv.n_splits == 3
    assert cv.shuffle is True
    assert cv.random_state == 12


# --- CV stability summary ----------------------------------------------------


def test_trial_stability_reports_expected_values():
    result = hpo._trial_stability([0.1, 0.3, 0.2])

    assert result["cv_mean"] == pytest.approx(0.2)
    assert result["cv_std"] == pytest.approx(np.std([0.1, 0.3, 0.2]))
    assert result["cv_min"] == pytest.approx(0.1)
    assert result["cv_max"] == pytest.approx(0.3)
    assert result["fold_scores"] == [0.1, 0.3, 0.2]


# --- probability-only finalist selection ------------------------------------


def test_select_probability_finalist_uses_log_loss_primary():
    better_log_loss = _probability_result(
        1,
        log_loss=0.30,
        brier_score=0.25,
    )
    better_brier = _probability_result(
        2,
        log_loss=0.31,
        brier_score=0.10,
    )

    selected = hpo._select_probability_finalist(
        [better_brier, better_log_loss],
        scoring="neg_log_loss",
    )

    assert selected["trial_number"] == 1
    assert selected["eligible"] is True


def test_select_probability_finalist_uses_brier_primary_and_log_loss_tiebreak():
    candidate_a = _probability_result(
        1,
        log_loss=0.29,
        brier_score=0.12,
    )
    candidate_b = _probability_result(
        2,
        log_loss=0.31,
        brier_score=0.12,
    )

    selected = hpo._select_probability_finalist(
        [candidate_b, candidate_a],
        scoring="neg_brier_score",
    )

    assert selected["trial_number"] == 1


def test_select_probability_finalist_rejects_empty_results():
    with pytest.raises(RuntimeError, match="No probability finalist"):
        hpo._select_probability_finalist([], scoring="neg_log_loss")


# --- optimizer shortlist -----------------------------------------------------


def test_optimizer_shortlist_applies_log_loss_guardrail_and_top_n():
    results = [
        _probability_result(1, log_loss=0.30, brier_score=0.20),
        _probability_result(2, log_loss=0.31, brier_score=0.15),
        _probability_result(3, log_loss=0.40, brier_score=0.10),
    ]
    settings = _optimizer_settings(
        max_log_loss_regression=0.02,
        optimizer_top_n=2,
    )

    shortlist = hpo._optimizer_shortlist(results, settings)

    assert [result["trial_number"] for result in shortlist] == [1, 2]
    assert results[0]["passes_log_loss_guardrail"] is True
    assert results[1]["passes_log_loss_guardrail"] is True
    assert results[2]["passes_log_loss_guardrail"] is False
    assert results[0]["optimizer_selected"] is True
    assert results[1]["optimizer_selected"] is True
    assert results[2]["optimizer_selected"] is False
    assert "_model" not in results[2]


def test_optimizer_shortlist_uses_brier_as_tiebreaker():
    results = [
        _probability_result(1, log_loss=0.30, brier_score=0.20),
        _probability_result(2, log_loss=0.30, brier_score=0.10),
    ]

    shortlist = hpo._optimizer_shortlist(
        results,
        _optimizer_settings(optimizer_top_n=1),
    )

    assert shortlist[0]["trial_number"] == 2


# --- optimizer finalist guardrails ------------------------------------------


def test_select_finalist_chooses_highest_profit_eligible_candidate():
    results = [
        _probability_result(
            1,
            log_loss=0.30,
            brier_score=0.20,
            profit=100.0,
            optimizer_selected=True,
        ),
        _probability_result(
            2,
            log_loss=0.31,
            brier_score=0.19,
            profit=125.0,
            optimizer_selected=True,
        ),
    ]

    selected = hpo._select_finalist(results, _optimizer_settings())

    assert selected["trial_number"] == 2
    assert all(result["eligible"] for result in results)


def test_select_finalist_rejects_highest_profit_when_log_loss_guardrail_fails():
    high_profit_bad_probability = _probability_result(
        1,
        log_loss=0.40,
        brier_score=0.20,
        profit=500.0,
        optimizer_selected=True,
        passes_log_loss_guardrail=False,
    )
    lower_profit_good_probability = _probability_result(
        2,
        log_loss=0.30,
        brier_score=0.18,
        profit=100.0,
        optimizer_selected=True,
    )

    selected = hpo._select_finalist(
        [high_profit_bad_probability, lower_profit_good_probability],
        _optimizer_settings(),
    )

    assert selected["trial_number"] == 2
    assert high_profit_bad_probability["eligible"] is False


def test_select_finalist_rejects_monotonicity_violation():
    violating = _probability_result(
        1,
        log_loss=0.30,
        brier_score=0.18,
        profit=500.0,
        optimizer_selected=True,
        violation_rate=0.02,
    )
    monotonic = _probability_result(
        2,
        log_loss=0.31,
        brier_score=0.19,
        profit=100.0,
        optimizer_selected=True,
        violation_rate=0.0,
    )

    selected = hpo._select_finalist(
        [violating, monotonic],
        _optimizer_settings(
            monotonicity_enabled=True,
            max_violation_rate=0.0,
        ),
    )

    assert selected["trial_number"] == 2
    assert violating["passes_monotonicity_guardrail"] is False
    assert violating["eligible"] is False


def test_select_finalist_ignores_monotonicity_when_disabled():
    violating = _probability_result(
        1,
        log_loss=0.30,
        brier_score=0.18,
        profit=500.0,
        optimizer_selected=True,
        violation_rate=0.50,
    )

    selected = hpo._select_finalist(
        [violating],
        _optimizer_settings(monotonicity_enabled=False),
    )

    assert selected["trial_number"] == 1
    assert violating["passes_monotonicity_guardrail"] is True


@pytest.mark.parametrize(
    "evaluated_rows,profit",
    [
        (0, 100.0),
        (10, float("inf")),
        (10, float("-inf")),
        (10, float("nan")),
    ],
)
def test_select_finalist_rejects_invalid_optimizer_evidence(
    evaluated_rows,
    profit,
):
    result = _probability_result(
        1,
        log_loss=0.30,
        brier_score=0.18,
        profit=profit,
        evaluated_rows=evaluated_rows,
        optimizer_selected=True,
    )

    with pytest.raises(RuntimeError, match="No optimizer finalist passed"):
        hpo._select_finalist([result], _optimizer_settings())

    assert not bool(result["eligible"])


def test_select_finalist_rejects_when_nothing_was_optimizer_evaluated():
    result = _probability_result(
        1,
        log_loss=0.30,
        brier_score=0.18,
        optimizer_selected=False,
    )

    with pytest.raises(RuntimeError, match="No optimizer finalist passed"):
        hpo._select_finalist([result], _optimizer_settings())


# --- optimizer evaluation adapter -------------------------------------------


def test_evaluate_optimizer_and_monotonicity_maps_missing_result(monkeypatch):
    monkeypatch.setattr(
        hpo.optimizer_evaluation,
        "run_bid_optimizer_evaluation",
        lambda **kwargs: None,
    )
    settings = {
        "optimizer": {
            "target_cm": 0.25,
            "minimum_bid": 0.25,
            "bid_step": 0.25,
            "chunk_size": 100,
        },
        "monotonicity": {
            "enabled": True,
            "tolerance": 1e-8,
            "max_violation_rate": 0.0,
        },
    }

    optimizer_metrics, monotonicity = hpo._evaluate_optimizer_and_monotonicity(
        model=object(),
        holdout=pd.DataFrame(),
        feature_cols=["bid"],
        settings=settings,
    )

    assert optimizer_metrics["evaluated_rows"] == 0
    assert optimizer_metrics["total_expected_profit"] == float("-inf")
    assert np.isnan(optimizer_metrics["mean_expected_profit"])
    assert monotonicity["enabled"] is True
    assert monotonicity["checked_rows"] == 0
    assert monotonicity["passed"] is True


def test_evaluate_optimizer_and_monotonicity_maps_success(monkeypatch):
    scored = pd.DataFrame(
        {
            "recommended_bid": [1.0, 2.0],
            "recommended_bid_expected_profit": [3.0, 5.0],
        }
    )
    scored.attrs["monotonicity_summary"] = {
        "enabled": True,
        "violation_rate": 0.01,
        "passed": False,
    }
    summary = SimpleNamespace(
        optimizer_rows=2,
        recommended_bid_total_expected_profit=8.0,
        avg_recommended_bid_predicted_win_rate=0.4,
    )
    monkeypatch.setattr(
        hpo.optimizer_evaluation,
        "run_bid_optimizer_evaluation",
        lambda **kwargs: (scored, summary),
    )
    settings = {
        "optimizer": {
            "target_cm": 0.25,
            "minimum_bid": 0.25,
            "bid_step": 0.25,
            "chunk_size": 100,
        },
        "monotonicity": {
            "enabled": True,
            "tolerance": 1e-8,
            "max_violation_rate": 0.0,
        },
    }

    optimizer_metrics, monotonicity = hpo._evaluate_optimizer_and_monotonicity(
        model=object(),
        holdout=pd.DataFrame(),
        feature_cols=["bid"],
        settings=settings,
    )

    assert optimizer_metrics == {
        "evaluated_rows": 2,
        "total_expected_profit": 8.0,
        "mean_expected_profit": 4.0,
        "mean_recommended_bid": 1.5,
        "mean_predicted_win_rate": 0.4,
    }
    assert monotonicity["violation_rate"] == 0.01
    assert monotonicity["passed"] is False


# --- serialization -----------------------------------------------------------


def test_serializable_finalist_results_removes_only_fitted_model():
    result = _probability_result(
        1,
        log_loss=0.30,
        brier_score=0.18,
        profit=100.0,
        optimizer_selected=True,
    )
    result["parameters"] = {"max_depth": 4}

    serialized = hpo._serializable_finalist_results([result])

    assert "_model" not in serialized[0]
    assert serialized[0]["trial_number"] == 1
    assert serialized[0]["parameters"] == {"max_depth": 4}
    assert serialized[0]["optimizer_metrics"]["total_expected_profit"] == 100.0


# --- parameter suggestion ----------------------------------------------------


def test_suggest_parameters_uses_configured_types():
    study = optuna.create_study(direction="maximize")
    trial = study.ask()
    search_space = {
        "kind": {"type": "categorical", "choices": ["a", "b"]},
        "depth": {"type": "int", "low": 2, "high": 4},
        "rate": {"type": "float", "low": 0.1, "high": 0.3},
    }

    params = hpo._suggest_parameters(trial, search_space)

    assert params["kind"] in {"a", "b"}
    assert 2 <= params["depth"] <= 4
    assert 0.1 <= params["rate"] <= 0.3


def test_suggest_parameter_rejects_unknown_type():
    study = optuna.create_study(direction="maximize")
    trial = study.ask()

    with pytest.raises(ValueError, match="Unsupported parameter type"):
        hpo._suggest_parameter(
            trial,
            "x",
            {"type": "unsupported"},
        )
