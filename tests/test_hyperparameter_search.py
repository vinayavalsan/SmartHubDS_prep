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

from smarthub.core import notifications
from smarthub.train_and_predict import hyperparameter_search as hpo


@pytest.fixture(autouse=True)
def disable_slack_delivery(monkeypatch):
    payloads = []
    monkeypatch.setattr(
        notifications,
        "_post",
        lambda payload, category=None: payloads.append(payload) or True,
    )
    return payloads


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


def _parallel_test_job(value, delay=0, fail=False):
    import time

    time.sleep(delay)
    if fail:
        raise ValueError("worker failed")
    return value


def test_parallel_map_preserves_input_order_and_propagates_errors():
    jobs = [dict(value=1, delay=0.1), dict(value=2)]
    assert hpo._parallel_map(_parallel_test_job, jobs, 2) == [1, 2]
    assert hpo._parallel_map(_parallel_test_job, [], 2) == []
    with pytest.raises(ValueError, match="worker failed"):
        hpo._parallel_map(_parallel_test_job, [dict(value=1, fail=True)] * 2, 2)


def _small_hpo_frame():
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "bid": rng.uniform(0.25, 4, 120),
            "state": ["CA", "TX", "NY"] * 40,
            hpo.config.TARGET_COL: [0, 1] * 60,
        }
    )


@pytest.mark.parametrize("early_stopping", [False, True])
def test_parallel_cv_matches_sequential_lightgbm(early_stopping):
    pytest.importorskip("lightgbm")
    frame = _small_hpo_frame()
    estimator = hpo._build_estimator(
        "lightgbm",
        ["bid"],
        ["state"],
        dict(
            n_estimators=12,
            num_leaves=4,
            min_child_samples=5,
            random_state=42,
            n_jobs=1,
            verbosity=-1,
        ),
        "none",
        2,
    )
    kwargs = dict(
        estimator=estimator,
        X=frame[["bid", "state"]],
        y=frame[hpo.config.TARGET_COL],
        scoring="neg_log_loss",
        cross_validation=TimeSeriesSplit(n_splits=3),
        trial_number=1,
        total_folds=3,
        early_stopping_settings={
            "enabled": early_stopping,
            "stopping_rounds": 3,
            "metric": "binary_logloss",
        },
    )
    sequential = hpo._score_trial_folds(**kwargs, n_jobs=1)
    parallel = hpo._score_trial_folds(**kwargs, n_jobs=2)
    assert parallel[0] == pytest.approx(sequential[0], abs=1e-12)
    assert parallel[1] == sequential[1]


def test_parallel_cv_reports_single_class_fold():
    from sklearn.dummy import DummyClassifier

    kwargs = dict(
        estimator=DummyClassifier(),
        X=pd.DataFrame({"x": range(12)}),
        y=pd.Series([0] * 12),
        scoring="neg_log_loss",
        cross_validation=TimeSeriesSplit(n_splits=2),
        trial_number=1,
        total_folds=2,
        early_stopping_settings={"enabled": False},
        n_jobs=2,
    )
    with pytest.raises(ValueError, match="only one target class"):
        hpo._score_trial_folds(**kwargs)


def test_parallel_probability_finalists_match_sequential():
    pytest.importorskip("lightgbm")
    frame = _small_hpo_frame()
    trial = optuna.trial.create_trial(
        params={},
        distributions={},
        value=-0.7,
        user_attrs={"best_iteration_median": 8},
    )
    study = SimpleNamespace(trials=[trial])
    settings = {
        "early_stopping": {"enabled": True},
        "calibration_cv": 2,
        "calibration_methods": ["none", "sigmoid", "isotonic"],
        "probability_shortlist_top_n": 1,
    }
    kwargs = dict(
        study=study,
        fixed_parameters=dict(
            n_estimators=12,
            num_leaves=4,
            min_child_samples=5,
            random_state=42,
            n_jobs=1,
            verbosity=-1,
        ),
        model_type="lightgbm",
        numeric=["bid"],
        categorical=["state"],
        development=frame.iloc[:90],
        holdout=frame.iloc[90:],
    )
    seq = hpo._evaluate_probability_candidates(**kwargs, settings=settings)
    par = hpo._evaluate_probability_candidates(
        **kwargs, settings={**settings, "probability_jobs": 2}
    )
    assert [r["calibration_method"] for r in par] == settings["calibration_methods"]
    for left, right in zip(seq, par):
        assert right["probability_metrics"] == pytest.approx(
            left["probability_metrics"]
        )
        assert right["_model"].predict_proba(frame[["bid", "state"]]) == pytest.approx(
            left["_model"].predict_proba(frame[["bid", "state"]])
        )
    assert hpo._select_probability_finalist(seq, "neg_log_loss")[
        "calibration_method"
    ] == (hpo._select_probability_finalist(par, "neg_log_loss")["calibration_method"])


def test_probability_worker_retains_skip_policy(monkeypatch):
    def fail(**kwargs):
        raise ValueError("invalid candidate")

    monkeypatch.setattr(hpo, "_evaluate_probability_candidate", fail)
    assert (
        hpo._evaluate_probability_candidate_safely(
            trial=SimpleNamespace(number=3), calibration_method="none"
        )
        is None
    )


def test_optimizer_parallel_results_update_original_candidates(monkeypatch):
    shortlist = [
        {
            "_model": model,
            "trial_number": trial_number,
            "calibration_method": "none",
            "probability_metrics": {"log_loss": 0.5},
        }
        for trial_number, model in enumerate(("a", "b"))
    ]

    def evaluate(function, jobs, n_jobs):
        assert n_jobs == 2
        assert [job["model"] for job in jobs] == ["a", "b"]
        return [
            ({"total_expected_profit": 10}, {"passed": True}),
            ({"total_expected_profit": 20}, {"passed": False}),
        ]

    monkeypatch.setattr(hpo, "_parallel_map", evaluate)
    hpo._evaluate_optimizer_shortlist(
        shortlist, pd.DataFrame(), [], {"optimizer_jobs": 2}
    )
    assert shortlist[0]["optimizer_metrics"]["total_expected_profit"] == 10
    assert shortlist[1]["monotonicity"]["passed"] is False


def test_parallel_hpo_run_writes_timings_and_evaluates_optimizer(
    tmp_path, monkeypatch, disable_slack_delivery
):
    """Exercise the full search with real fits and optimizer workers."""
    pytest.importorskip("lightgbm")
    pytest.importorskip("mlflow")
    import json
    from pathlib import Path

    import yaml

    payload = yaml.safe_load(
        hpo.config.paths.resolve("config/hyperparameter_search.yaml").read_text()
    )
    defaults = payload["hyperparameter_search"]["defaults"]
    defaults["search"].update(n_trials=2, cv_folds=2)
    defaults["early_stopping"].update(max_estimators=8, stopping_rounds=2)
    defaults["finalists"].update(probability_shortlist_top_n=2, optimizer_top_n=2)
    defaults["parallelism"]["optimizer_jobs"] = 2
    defaults["calibration"]["methods"] = ["none", "sigmoid"]
    defaults["output"]["root"] = str(tmp_path / "outputs")
    defaults["mlflow"]["tracking_db_path"] = str(tmp_path / "mlflow.db")
    defaults["mlflow"]["artifact_root"] = str(tmp_path / "mlruns")
    path = tmp_path / "hpo.yaml"
    path.write_text(yaml.safe_dump(payload))
    frame = _small_hpo_frame()
    frame["created_at"] = pd.date_range("2026-09-01", periods=len(frame), freq="h")
    frame[hpo.config.REVENUE_COL] = 5.0
    summary = {
        "training_table_version": "snapshot-test",
        "training_rows": len(frame),
        "data_min_created_at": str(frame["created_at"].min()),
        "data_max_created_at": str(frame["created_at"].max()),
        "source_row_count": len(frame),
    }
    monkeypatch.setattr(
        hpo.preprocessing,
        "prepare_training_data",
        lambda *args: (frame, ["bid"], ["state"], summary),
    )
    monkeypatch.setattr(hpo, "_write_optuna_plots", lambda **kwargs: {})
    result = hpo.run_hyperparameter_search(6, "snapshot-test", path)
    assert "HPO started" in disable_slack_delivery[0]["text"]
    assert "HPO completed; parameters saved" in disable_slack_delivery[-1]["text"]
    assert result["hpo_run_id"] in disable_slack_delivery[-1]["text"]
    saved = json.loads(Path(result["summary_path"]).read_text())
    assert saved["training_table_version"] == "snapshot-test"
    assert saved["parallelism"] == {
        "cv_jobs": 2,
        "probability_jobs": 2,
        "optimizer_jobs": 2,
    }
    assert saved["timings"] == result["timings"]
    assert all(value >= 0 for value in saved["timings"].values())
    assert result["optimizer_metrics"]["evaluated_rows"] > 0
    assert result["monotonicity"]["passed"] is True

    # The real HPO result is consumed directly by training, including calibration
    # and the reserved rows, without copying model values into training.yaml.
    from smarthub.train_and_predict import model_parameters, train

    artifact = model_parameters.load_artifact(result["parameters_path"], 6)
    assert artifact["parameter_version"] == result["parameter_version"]
    assert artifact["hpo_run_id"] == result["hpo_run_id"]
    assert artifact["hpo_mlflow_run_id"] == result["hpo_mlflow_run_id"]
    assert artifact["hpo_mlflow_run_id"]
    training_payload = yaml.safe_load(
        hpo.config.paths.resolve("config/training.yaml").read_text()
    )
    training_payload["training"]["lead_types"][6]["parameters"]["current_file"] = str(
        tmp_path / "current.yaml"
    )
    training_payload["training"]["defaults"]["early_stopping"].update(
        max_estimators=8, stopping_rounds=2
    )
    training_path = tmp_path / "training.yaml"
    training_path.write_text(yaml.safe_dump(training_payload))
    original_load = hpo.config.load_training_config
    monkeypatch.setattr(
        hpo.config,
        "load_training_config",
        lambda lead_type_id, config_path=None, model_settings=None: original_load(
            lead_type_id,
            config_path=config_path or training_path,
            model_settings=model_settings,
        ),
    )
    summary.update(dropped_rows=0, win_rate=0.5, missing_feature_columns=[])
    ctx = train.TrainingContext(
        lead_type_id=6, parameter_file=result["parameters_path"]
    )
    train.stage_prepare_data(ctx)
    train.stage_split_and_diagnostics(ctx)
    train.stage_fit_model(ctx)
    assert ctx.version == "snapshot-test"
    assert (
        model_parameters.settings_from_config(ctx.training_config)
        == artifact["model_settings"]
    )
    assert ctx.test_df.index.tolist() == artifact["data"]["test_positions"]
    assert ctx.train_df.index.tolist() == artifact["data"]["training_positions"]
    assert not (tmp_path / "current.yaml").exists()

    model_parameters.write_current_parameters(
        ctx.training_config,
        6,
        {
            "model_settings": artifact["model_settings"],
            "parameter_version": artifact["parameter_version"],
            "training_run_id": "approved_candidate",
            "hpo_run_id": artifact["hpo_run_id"],
            "hpo_mlflow_run_id": artifact["hpo_mlflow_run_id"],
        },
    )
    current_path = tmp_path / "current.yaml"
    current_yaml = yaml.safe_load(current_path.read_text())
    hpo_yaml = yaml.safe_load(Path(result["parameters_path"]).read_text())
    assert current_yaml.keys() == hpo_yaml.keys()
    assert current_yaml["model_settings"] == hpo_yaml["model_settings"]
    assert "models" not in hpo_yaml and "calibration" not in hpo_yaml
    # The approved current artifact is also accepted explicitly, using fresh
    # data rather than treating inherited HPO provenance as a pending candidate.
    versions = []

    def prepare_latest(*args):
        versions.append(args[-1])
        return frame, ["bid"], ["state"], summary

    monkeypatch.setattr(hpo.preprocessing, "prepare_training_data", prepare_latest)
    daily = train.TrainingContext(lead_type_id=6, parameter_file=str(current_path))
    train.stage_prepare_data(daily)
    train.stage_split_and_diagnostics(daily)
    assert versions == [None]
    assert daily.parameter_info["approved_training_run_id"] == "approved_candidate"


def test_search_reports_started_and_completed_without_claiming_promotion(
    disable_slack_delivery,
):
    result = {
        "hpo_run_id": "hpo_selected",
        "parameter_version": "params_selected",
        "hpo_mlflow_run_id": "mlflow_selected",
        "model_type": "lightgbm",
        "selected_trial": 0,
        "selected_calibration_method": "none",
        "holdout_probability_metrics": {"log_loss": 0.4},
        "parameters_path": "/data/hpo/best_parameters.yaml",
    }

    @hpo._report_search
    def search(lead_type_id, version, config_path):
        assert lead_type_id == 6 and version == "dataset_v1" and config_path is None
        return result

    assert search(6, version="dataset_v1") is result
    assert len(disable_slack_delivery) == 2
    assert "started" in disable_slack_delivery[0]["text"]
    assert "auto (6)" in disable_slack_delivery[0]["text"]
    text = disable_slack_delivery[1]["text"]
    assert "HPO completed; parameters saved" in text
    assert "hpo_selected" in text and "mlflow_selected" in text
    assert "params_selected" in text and "0.4" in text
    assert "promoted" not in text.lower()


def test_search_failure_alert_preserves_original_exception(disable_slack_delivery):
    error = ValueError("insufficient data")

    @hpo._report_search
    def search(*args):
        raise error

    with pytest.raises(ValueError) as caught:
        search(1)
    assert caught.value is error
    assert len(disable_slack_delivery) == 2
    text = disable_slack_delivery[-1]["text"]
    assert "FAILED" in text and "HPO failed" in text
    assert "home (1)" in text and "ValueError: insufficient data" in text


def test_slack_delivery_error_never_changes_search_result(monkeypatch):
    def broken(*args, **kwargs):
        raise OSError("Slack unavailable")

    monkeypatch.setattr(notifications, "_post", broken)
    result = {"hpo_run_id": "completed"}

    @hpo._report_search
    def search(*args):
        return result

    assert search(6) is result


def test_slack_delivery_error_never_masks_search_exception(monkeypatch):
    def broken(*args, **kwargs):
        raise OSError("Slack unavailable")

    monkeypatch.setattr(notifications, "_post", broken)

    @hpo._report_search
    def search(*args):
        raise ValueError("search failed")

    with pytest.raises(ValueError, match="search failed"):
        search(6)


@pytest.mark.parametrize(
    "status,headline,category,display_status",
    [
        ("started", "HPO started", "updates", "started"),
        ("success", "HPO completed; parameters saved", "updates", "completed"),
        ("success", "HPO candidate promoted", "updates", "promoted"),
        (
            "success",
            "HPO candidate not promoted",
            "updates",
            "completed (not promoted)",
        ),
        ("failure", "HPO failed; retry scheduled", "failures", "FAILED"),
        ("failure", "HPO candidate training failed", "failures", "FAILED"),
    ],
)
def test_hpo_notification_uses_shared_format_and_category(
    monkeypatch, status, headline, category, display_status
):
    delivered = []

    def capture(payload, channel=None):
        delivered.append((payload, channel))
        return True

    monkeypatch.setattr(notifications, "_post", capture)
    hpo._send_notification(
        status,
        {"Lead type": "auto (6)", "Status": headline, "HPO run": "hpo_test"},
        error="test error" if status == "failure" else None,
    )
    payload, channel = delivered[0]
    assert channel == category
    title = payload["blocks"][0]["text"]["text"].splitlines()[0]
    assert f"· hpo · {display_status} · auto (6)" in title
    assert headline in payload["text"]
    assert "```" in payload["blocks"][0]["text"]["text"]
    assert payload["blocks"][1]["type"] == "context"
    assert ("<!here>" in title) == (status == "failure")
    if status == "failure":
        assert "Error: test error" in payload["text"]
