"""SmartHub-aware hyperparameter search for win-probability models.

This module tunes raw classifier hyperparameters with time-aware cross-validation,
then evaluates the strongest trials on a recent untouched holdout. Calibration
search and downstream bid optimization are independently optional.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import optuna
import pandas as pd
import yaml
from sklearn.base import clone
from sklearn.metrics import brier_score_loss, get_scorer, log_loss
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit, train_test_split

from smarthub.core.logging_utils import get_logger

from . import config, models, optimizer_evaluation, preprocessing

logger = get_logger(__name__)

_PROBABILITY_SCORERS = {"neg_log_loss", "neg_brier_score"}


def _suggest_parameter(
    trial: optuna.Trial,
    parameter_name: str,
    specification: dict[str, Any],
) -> Any:
    """Suggest one parameter value from its configured search space."""
    parameter_type = specification["type"]

    if parameter_type == "categorical":
        return trial.suggest_categorical(
            parameter_name,
            specification["choices"],
        )

    if parameter_type == "int":
        return trial.suggest_int(
            parameter_name,
            int(specification["low"]),
            int(specification["high"]),
            step=int(specification.get("step", 1)),
            log=bool(specification.get("log", False)),
        )

    if parameter_type == "float":
        step = specification.get("step")
        return trial.suggest_float(
            parameter_name,
            float(specification["low"]),
            float(specification["high"]),
            step=float(step) if step is not None else None,
            log=bool(specification.get("log", False)),
        )

    raise ValueError(
        f"Unsupported parameter type {parameter_type!r} " f"for {parameter_name!r}."
    )


def _suggest_parameters(
    trial: optuna.Trial,
    search_space: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Suggest all configured parameters for one Optuna trial."""
    return {
        parameter_name: _suggest_parameter(
            trial,
            parameter_name,
            specification,
        )
        for parameter_name, specification in search_space.items()
    }


def _scoring_plot_config(scoring: str) -> tuple[None, str]:
    """Build a human-readable plot label for an sklearn scorer."""
    metric_names = {
        "neg_brier_score": "Negative Brier Score",
        "neg_log_loss": "Negative Log Loss",
    }
    metric_name = metric_names.get(
        scoring,
        scoring.replace("neg_", "negative_").replace("_", " ").title(),
    )
    target_name = f"Mean Cross-Validation {metric_name} (higher is better)"
    return None, target_name


def _write_optuna_plots(
    run_output_dir: Path,
    study: optuna.Study,
    parameter_names: list[str],
    scoring: str,
) -> dict[str, Path]:
    """Write interactive Optuna visualizations as HTML artifacts."""
    plots_dir = run_output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    target, target_name = _scoring_plot_config(scoring)
    plot_builders = {
        "optimization_history": lambda: (
            optuna.visualization.plot_optimization_history(
                study,
                target=target,
                target_name=target_name,
            )
        ),
        "parameter_importance": lambda: (
            optuna.visualization.plot_param_importances(
                study,
                target=target,
                target_name=target_name,
            )
        ),
        "contour_matrix": lambda: optuna.visualization.plot_contour(
            study,
            params=parameter_names,
            target=target,
            target_name=target_name,
        ),
    }

    plot_paths: dict[str, Path] = {}
    for plot_name, build_plot in plot_builders.items():
        plot_path = plots_dir / f"{plot_name}.html"
        try:
            figure = build_plot()
            figure.write_html(
                str(plot_path),
                include_plotlyjs=True,
                full_html=True,
            )
            plot_paths[plot_name] = plot_path
        except (ImportError, RuntimeError, ValueError, ZeroDivisionError) as exc:
            logger.warning("Unable to create %s plot: %s", plot_name, exc)

    return plot_paths


def _validate_probability_scoring(scoring: str) -> None:
    """Require a probability-quality objective for SmartHub HPO."""
    if scoring not in _PROBABILITY_SCORERS:
        allowed = ", ".join(sorted(_PROBABILITY_SCORERS))
        raise ValueError(
            "SmartHub hyperparameter search must optimize a probability-quality "
            f"metric. search.scoring={scoring!r}; supported: {allowed}."
        )


def _hpo_settings(
    search_config: config.HyperparameterSearchConfig,
) -> dict[str, Any]:
    """Return normalized SmartHub-specific HPO settings."""
    return {
        "validation_strategy": search_config.validation_strategy,
        "holdout_fraction": search_config.holdout_fraction,
        "probability_shortlist_top_n": (search_config.probability_shortlist_top_n),
        "optimizer_top_n": search_config.optimizer_top_n,
        "max_log_loss_regression": search_config.max_log_loss_regression,
        "calibration_enabled": search_config.calibration_enabled,
        "calibration_methods": list(search_config.calibration_methods),
        "calibration_cv": search_config.calibration_cv,
        "optimizer_enabled": search_config.optimizer_enabled,
        "optimizer": (
            search_config.optimizer.as_dict() if search_config.optimizer else None
        ),
        "monotonicity": search_config.monotonicity.as_dict(),
    }


def _reserve_final_training_test(
    frame: pd.DataFrame,
    split_settings: dict[str, Any],
    random_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reserve the final test partition that training will evaluate later.

    Hyperparameter search must never inspect this partition. The split mirrors
    ``train.split_training_data`` so that the rows held out here are the same
    rows later used for the final training evaluation and promotion decision.

    Inputs
    ------
    frame : pandas.DataFrame
        Prepared model-ready training data.
    split_settings : dict[str, Any]
        Resolved training split settings from ``training.yaml``.
    random_seed : int
        Training random seed used for reproducible random splitting.

    Returns
    -------
    tuple[pandas.DataFrame, pandas.DataFrame]
        HPO-eligible rows followed by the untouched final training test rows.
    """
    strategy = str(split_settings["strategy"]).strip().lower()
    test_size = float(split_settings["test_size"])

    if not 0.0 < test_size < 1.0:
        raise ValueError("Training split test_size must be between 0 and 1.")

    if strategy == "time":
        if "created_at" not in frame.columns:
            raise ValueError(
                "Time-based final test reservation requires a 'created_at' column."
            )
        ordered = frame.sort_values("created_at").reset_index(drop=True)
        n_test = max(1, int(round(len(ordered) * test_size)))
        split_index = len(ordered) - n_test
        hpo_pool = ordered.iloc[:split_index].copy()
        final_test = ordered.iloc[split_index:].copy()
    elif strategy == "random":
        stratify = None
        if bool(split_settings.get("stratify", False)):
            stratify = frame[config.TARGET_COL]
        hpo_pool, final_test = train_test_split(
            frame,
            test_size=test_size,
            random_state=random_seed,
            shuffle=True,
            stratify=stratify,
        )
        hpo_pool = hpo_pool.copy()
        final_test = final_test.copy()
    else:
        raise ValueError(f"Unsupported training split strategy: {strategy!r}.")

    if hpo_pool.empty or final_test.empty:
        raise ValueError(
            "Training split produced an empty HPO pool or final test partition."
        )

    return hpo_pool.reset_index(drop=True), final_test.reset_index(drop=True)


def _split_development_and_holdout(
    frame: pd.DataFrame,
    holdout_fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reserve the newest rows as an untouched finalist holdout."""
    if "created_at" not in frame.columns:
        raise ValueError(
            "Time-aware SmartHub HPO requires a 'created_at' column in the "
            "prepared training table."
        )

    ordered = frame.sort_values("created_at").reset_index(drop=True)
    holdout_rows = max(1, int(np.ceil(len(ordered) * holdout_fraction)))
    split_index = len(ordered) - holdout_rows
    if split_index <= 0:
        raise ValueError("Not enough rows remain after finalist holdout split.")

    development = ordered.iloc[:split_index].reset_index(drop=True)
    holdout = ordered.iloc[split_index:].reset_index(drop=True)
    return development, holdout


def _build_cv(
    strategy: str,
    n_splits: int,
    random_seed: int,
):
    """Build the configured cross-validation splitter."""
    if strategy == "time":
        return TimeSeriesSplit(n_splits=n_splits)
    return StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_seed,
    )


def _iter_splits(cross_validation, X: pd.DataFrame, y: pd.Series) -> Iterable:
    """Yield CV splits for either time-based or stratified validation."""
    if isinstance(cross_validation, StratifiedKFold):
        return cross_validation.split(X, y)
    return cross_validation.split(X)


def _build_hpo_coverage_diagnostics(
    development: pd.DataFrame,
    holdout: pd.DataFrame,
    numeric: list[str],
    categorical: list[str],
    cross_validation,
) -> list[dict[str, Any]]:
    """Build feature-coverage diagnostics once for all HPO evaluation boundaries."""
    feature_cols = numeric + categorical
    X = development[feature_cols]
    y = development[config.TARGET_COL]
    features = preprocessing.coverage_features(development, numeric, categorical)
    diagnostics: list[dict[str, Any]] = []

    for fold_number, (train_idx, valid_idx) in enumerate(
        _iter_splits(cross_validation, X, y),
        start=1,
    ):
        diagnostics.extend(
            preprocessing.feature_coverage_rows(
                development.iloc[train_idx],
                development.iloc[valid_idx],
                features,
                partition=f"cv_fold_{fold_number}",
            )
        )

    diagnostics.extend(
        preprocessing.feature_coverage_rows(
            development,
            holdout,
            features,
            partition="finalist_holdout",
        )
    )
    return diagnostics


def _partition_time_range(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    partition: str,
) -> dict[str, Any]:
    """Return created-at ranges for one training/evaluation boundary."""
    train_created = pd.to_datetime(train_df["created_at"], errors="coerce")
    eval_created = pd.to_datetime(eval_df["created_at"], errors="coerce")

    train_min = train_created.min()
    train_max = train_created.max()
    eval_min = eval_created.min()
    eval_max = eval_created.max()

    return {
        "partition": partition,
        "train_min_created_at": train_min.isoformat(),
        "train_max_created_at": train_max.isoformat(),
        "train_span_days": round(
            float((train_max - train_min).total_seconds() / 86400.0),
            2,
        ),
        "eval_min_created_at": eval_min.isoformat(),
        "eval_max_created_at": eval_max.isoformat(),
        "eval_span_days": round(
            float((eval_max - eval_min).total_seconds() / 86400.0),
            2,
        ),
    }


def _build_hpo_time_range_diagnostics(
    development: pd.DataFrame,
    holdout: pd.DataFrame,
    feature_cols: list[str],
    cross_validation,
) -> list[dict[str, Any]]:
    """Build time ranges once for all HPO evaluation boundaries."""
    X = development[feature_cols]
    y = development[config.TARGET_COL]
    diagnostics: list[dict[str, Any]] = []

    for fold_number, (train_idx, valid_idx) in enumerate(
        _iter_splits(cross_validation, X, y),
        start=1,
    ):
        diagnostics.append(
            _partition_time_range(
                development.iloc[train_idx],
                development.iloc[valid_idx],
                partition=f"cv_fold_{fold_number}",
            )
        )

    diagnostics.append(
        _partition_time_range(
            development,
            holdout,
            partition="finalist_holdout",
        )
    )
    return diagnostics


def _log_hpo_time_range_diagnostics(
    diagnostics: list[dict[str, Any]],
) -> None:
    """Log train/evaluation date ranges for each HPO boundary."""
    logger.info("Cross-Validation Time Ranges")
    if not diagnostics:
        logger.info("  No time-range diagnostics available.")
        return

    table = pd.DataFrame(diagnostics)
    logger.info("\n%s", table.to_string(index=False))


def _log_hpo_coverage_diagnostics(diagnostics: list[dict[str, Any]]) -> None:
    """Log compact feature-coverage diagnostics without affecting HPO."""
    if not diagnostics:
        logger.info("Feature Coverage Diagnostics: no categorical/binary features.")
        return

    table = pd.DataFrame(diagnostics)
    report = table[
        (table["train_unique"] != table["eval_unique"])
        | (table["unseen_eval_unique"] > 0)
    ].copy()

    logger.info("Feature Coverage Diagnostics")
    if report.empty:
        logger.info("  No train/evaluation coverage differences detected.")
        return

    report["eval_pct_unseen"] = report["eval_pct_unseen"].round(2)
    logger.info(
        "\n%s",
        report[
            [
                "partition",
                "feature",
                "train_unique",
                "eval_unique",
                "unseen_eval_unique",
                "eval_rows_unseen",
                "eval_pct_unseen",
                "min_train_support_for_eval_values",
            ]
        ].to_string(index=False),
    )


def _score_trial_folds(
    estimator,
    X: pd.DataFrame,
    y: pd.Series,
    scoring: str,
    cross_validation,
    trial_number: int,
    total_folds: int,
) -> list[float]:
    """Fit and score one estimator independently on every CV fold."""
    scorer = get_scorer(scoring)
    scores: list[float] = []

    for fold_number, (train_idx, valid_idx) in enumerate(
        _iter_splits(cross_validation, X, y),
        start=1,
    ):
        y_train = y.iloc[train_idx]
        y_valid = y.iloc[valid_idx]
        if y_train.nunique() < 2 or y_valid.nunique() < 2:
            raise ValueError(
                "Cross-validation fold contains only one target class. "
                f"fold={fold_number}. Increase the dataset/window size or "
                "reduce search.cv_folds."
            )

        logger.info(
            "Trial %s | fold %s/%s | fitting | train=%s valid=%s",
            trial_number,
            fold_number,
            total_folds,
            f"{len(train_idx):,}",
            f"{len(valid_idx):,}",
        )
        fold_started = time.perf_counter()
        fitted = clone(estimator)
        fitted.fit(X.iloc[train_idx], y_train)
        score = float(scorer(fitted, X.iloc[valid_idx], y_valid))
        fold_elapsed = time.perf_counter() - fold_started
        scores.append(score)

        logger.info(
            "Trial %s | fold %s/%s | complete | score=%.6f | " "elapsed=%.1fs",
            trial_number,
            fold_number,
            total_folds,
            score,
            fold_elapsed,
        )

    return scores


def _trial_stability(scores: list[float]) -> dict[str, Any]:
    """Summarize fold-level stability for one Optuna trial."""
    array = np.asarray(scores, dtype=float)
    return {
        "cv_mean": float(np.mean(array)),
        "cv_std": float(np.std(array, ddof=0)),
        "cv_min": float(np.min(array)),
        "cv_max": float(np.max(array)),
        "fold_scores": [float(value) for value in array],
    }


def _build_estimator(
    model_type: str,
    numeric: list[str],
    categorical: list[str],
    model_parameters: dict[str, Any],
    calibration_method: str,
    calibration_cv: int,
):
    """Build an estimator, optionally with configured probability calibration."""
    calibration_enabled = calibration_method != "none"
    return models.build_model(
        model_type=model_type,
        numeric_features=numeric,
        categorical_features=categorical,
        model_params=model_parameters,
        calibration_enabled=calibration_enabled,
        calibration_method=(
            None if calibration_method == "none" else calibration_method
        ),
        calibration_cv=(None if calibration_method == "none" else calibration_cv),
    )


def _evaluate_optimizer_and_monotonicity(
    model,
    holdout: pd.DataFrame,
    feature_cols: list[str],
    settings: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Evaluate optimizer performance and monotonicity in one shared pass.

    This uses the same optimizer-evaluation path as normal training.
    Monotonicity diagnostics are calculated from the candidate-bid predictions
    already produced during optimizer scoring, so no second prediction sweep
    is performed.

    Inputs
    ------
    model : Any
        Fitted finalist model.
    holdout : pandas.DataFrame
        Untouched HPO finalist holdout.
    feature_cols : list[str]
        Ordered model feature columns.
    settings : dict[str, Any]
        Resolved HPO optimizer and monotonicity settings.

    Returns
    -------
    tuple[dict[str, Any], dict[str, Any]]
        Optimizer metrics followed by monotonicity metrics.
    """
    optimizer_settings = settings["optimizer"]
    monotonicity_settings = settings["monotonicity"]

    result = optimizer_evaluation.run_bid_optimizer_evaluation(
        test_eval_df=holdout,
        model=model,
        feature_cols=feature_cols,
        target_cm=optimizer_settings["target_cm"],
        min_bid=optimizer_settings["minimum_bid"],
        bid_step=optimizer_settings["bid_step"],
        chunk_size=optimizer_settings["chunk_size"],
        monotonicity_enabled=monotonicity_settings["enabled"],
        monotonicity_tolerance=monotonicity_settings["tolerance"],
        monotonicity_max_violation_rate=(monotonicity_settings["max_violation_rate"]),
        log_summary_result=False,
    )

    if result is None:
        optimizer_metrics = {
            "evaluated_rows": 0,
            "total_expected_profit": float("-inf"),
            "mean_expected_profit": float("nan"),
            "mean_recommended_bid": float("nan"),
            "mean_predicted_win_rate": float("nan"),
        }
        monotonicity = {
            "enabled": monotonicity_settings["enabled"],
            "checked_rows": 0,
            "checked_steps": 0,
            "violation_count": 0,
            "violation_rate": 0.0,
            "rows_with_violation_pct": 0.0,
            "mean_violation_magnitude": 0.0,
            "max_violation_magnitude": 0.0,
            "max_allowed_violation_rate": monotonicity_settings["max_violation_rate"],
            "passed": None if not monotonicity_settings["enabled"] else True,
        }
        return optimizer_metrics, monotonicity

    scored, summary = result
    optimizer_metrics = {
        "evaluated_rows": int(summary.optimizer_rows),
        "total_expected_profit": float(summary.recommended_bid_total_expected_profit),
        "mean_expected_profit": float(scored["recommended_bid_expected_profit"].mean()),
        "mean_recommended_bid": float(scored["recommended_bid"].mean()),
        "mean_predicted_win_rate": float(
            summary.avg_recommended_bid_predicted_win_rate
        ),
    }
    monotonicity = dict(scored.attrs.get("monotonicity_summary", {}))
    return optimizer_metrics, monotonicity


def _evaluate_probability_candidate(
    trial: optuna.trial.FrozenTrial,
    calibration_method: str,
    fixed_parameters: dict[str, Any],
    model_type: str,
    numeric: list[str],
    categorical: list[str],
    development: pd.DataFrame,
    holdout: pd.DataFrame,
    settings: dict[str, Any],
) -> dict[str, Any]:
    """Fit one trial/calibration pair and score probability quality only."""
    feature_cols = numeric + categorical
    model_parameters = {**fixed_parameters, **trial.params}
    estimator = _build_estimator(
        model_type=model_type,
        numeric=numeric,
        categorical=categorical,
        model_parameters=model_parameters,
        calibration_method=calibration_method,
        calibration_cv=settings["calibration_cv"],
    )
    estimator.fit(development[feature_cols], development[config.TARGET_COL])

    probabilities = estimator.predict_proba(holdout[feature_cols])[:, 1]
    holdout_y = holdout[config.TARGET_COL]
    probability_metrics = {
        "log_loss": float(log_loss(holdout_y, probabilities, labels=[0, 1])),
        "brier_score": float(brier_score_loss(holdout_y, probabilities)),
    }

    return {
        "trial_number": int(trial.number),
        "calibration_method": calibration_method,
        "calibration_cv": settings["calibration_cv"],
        "parameters": model_parameters,
        "cv_score": float(trial.value),
        "cv_std": float(trial.user_attrs.get("cv_std", float("nan"))),
        "cv_min": float(trial.user_attrs.get("cv_min", float("nan"))),
        "cv_max": float(trial.user_attrs.get("cv_max", float("nan"))),
        "fold_scores": trial.user_attrs.get("fold_scores", []),
        "probability_metrics": probability_metrics,
        "optimizer_selected": False,
        "optimizer_metrics": None,
        "monotonicity": None,
        "_model": estimator,
    }


def _evaluate_probability_candidates(
    study: optuna.Study,
    fixed_parameters: dict[str, Any],
    model_type: str,
    numeric: list[str],
    categorical: list[str],
    development: pd.DataFrame,
    holdout: pd.DataFrame,
    settings: dict[str, Any],
) -> list[dict[str, Any]]:
    """Evaluate calibration variants for a wider probability shortlist."""
    completed = [
        trial
        for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE and trial.value is not None
    ]
    completed.sort(key=lambda trial: float(trial.value), reverse=True)
    top_trials = completed[: settings["probability_shortlist_top_n"]]
    results: list[dict[str, Any]] = []

    for trial in top_trials:
        for calibration_method in settings["calibration_methods"]:
            logger.info(
                "Probability finalist: trial=%s calibration=%s",
                trial.number,
                calibration_method,
            )
            try:
                result = _evaluate_probability_candidate(
                    trial=trial,
                    calibration_method=calibration_method,
                    fixed_parameters=fixed_parameters,
                    model_type=model_type,
                    numeric=numeric,
                    categorical=categorical,
                    development=development,
                    holdout=holdout,
                    settings=settings,
                )
                results.append(result)
                logger.info(
                    "  Holdout log loss=%.6f | Brier=%.6f",
                    result["probability_metrics"]["log_loss"],
                    result["probability_metrics"]["brier_score"],
                )
            except (RuntimeError, ValueError) as exc:
                logger.warning(
                    "Skipping probability finalist trial=%s calibration=%s: %s",
                    trial.number,
                    calibration_method,
                    exc,
                )

    if not results:
        raise RuntimeError("No probability finalist completed evaluation successfully.")
    return results


def _optimizer_shortlist(
    probability_results: list[dict[str, Any]],
    settings: dict[str, Any],
) -> list[dict[str, Any]]:
    """Choose the small set that receives expensive optimizer evaluation."""
    best_log_loss = min(
        result["probability_metrics"]["log_loss"] for result in probability_results
    )
    log_loss_ceiling = best_log_loss + settings["max_log_loss_regression"]

    for result in probability_results:
        result["passes_log_loss_guardrail"] = (
            result["probability_metrics"]["log_loss"] <= log_loss_ceiling
        )

    acceptable = [
        result for result in probability_results if result["passes_log_loss_guardrail"]
    ]
    acceptable.sort(
        key=lambda result: (
            result["probability_metrics"]["log_loss"],
            result["probability_metrics"]["brier_score"],
        )
    )
    shortlist = acceptable[: settings["optimizer_top_n"]]
    if not shortlist:
        raise RuntimeError("No probability candidate passed the log-loss guardrail.")

    shortlisted_ids = {id(result) for result in shortlist}
    for result in probability_results:
        if id(result) in shortlisted_ids:
            result["optimizer_selected"] = True
        else:
            result.pop("_model", None)
    return shortlist


def _evaluate_optimizer_shortlist(
    shortlist: list[dict[str, Any]],
    holdout: pd.DataFrame,
    feature_cols: list[str],
    settings: dict[str, Any],
) -> None:
    """Run optimizer and monotonicity only for shortlisted candidates."""
    for rank, result in enumerate(shortlist, start=1):
        logger.info(
            "Optimizer finalist %s/%s: trial=%s calibration=%s | log_loss=%.6f",
            rank,
            len(shortlist),
            result["trial_number"],
            result["calibration_method"],
            result["probability_metrics"]["log_loss"],
        )
        optimizer_metrics, monotonicity = _evaluate_optimizer_and_monotonicity(
            model=result["_model"],
            holdout=holdout,
            feature_cols=feature_cols,
            settings=settings,
        )
        result["optimizer_metrics"] = optimizer_metrics
        result["monotonicity"] = monotonicity


def _select_probability_finalist(
    finalist_results: list[dict[str, Any]],
    scoring: str,
) -> dict[str, Any]:
    """Select the strongest finalist using held-out probability quality."""
    if not finalist_results:
        raise RuntimeError("No probability finalist is available for selection.")

    primary_metric = "brier_score" if scoring == "neg_brier_score" else "log_loss"
    secondary_metric = "log_loss" if primary_metric == "brier_score" else "brier_score"
    selected = min(
        finalist_results,
        key=lambda result: (
            result["probability_metrics"][primary_metric],
            result["probability_metrics"][secondary_metric],
        ),
    )
    selected["eligible"] = True
    return selected


def _select_finalist(
    finalist_results: list[dict[str, Any]],
    settings: dict[str, Any],
) -> dict[str, Any]:
    """Select highest-profit optimizer finalist subject to guardrails."""
    max_violation_rate = settings["monotonicity"]["max_violation_rate"]
    evaluated = [
        result
        for result in finalist_results
        if result.get("optimizer_selected") and result.get("optimizer_metrics")
    ]

    for result in evaluated:
        monotonicity = result.get("monotonicity") or {}
        result["passes_monotonicity_guardrail"] = (
            not settings["monotonicity"]["enabled"]
            or float(monotonicity.get("violation_rate", 0.0)) <= max_violation_rate
        )
        optimizer_metrics = result["optimizer_metrics"]
        result["eligible"] = (
            result["passes_log_loss_guardrail"]
            and result["passes_monotonicity_guardrail"]
            and optimizer_metrics["evaluated_rows"] > 0
            and np.isfinite(optimizer_metrics["total_expected_profit"])
        )

        monotonicity_status = (
            "PASS" if result["passes_monotonicity_guardrail"] else "FAIL"
        )
        logger.info(
            "Optimizer finalist trial=%s calibration=%s | monotonicity=%s",
            result["trial_number"],
            result["calibration_method"],
            monotonicity_status,
        )
        if not result["passes_monotonicity_guardrail"]:
            logger.info(
                "  Monotonicity violation rate: %.6f%% (%s/%s bid transitions)",
                float(monotonicity.get("violation_rate", 0.0)) * 100.0,
                monotonicity.get("violation_count", 0),
                monotonicity.get("checked_steps", 0),
            )

    eligible = [result for result in evaluated if result.get("eligible")]
    if not eligible:
        raise RuntimeError(
            "No optimizer finalist passed the probability-quality and "
            "bid-response guardrails. Review finalist_results.json."
        )

    return max(
        eligible,
        key=lambda result: result["optimizer_metrics"]["total_expected_profit"],
    )


def _serializable_finalist_results(
    finalist_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Remove fitted model objects before writing HPO artifacts."""
    return [
        {key: value for key, value in result.items() if key != "_model"}
        for result in finalist_results
    ]


def _write_outputs(
    output_dir: Path,
    lead_type_id: int,
    lead_type_name: str,
    model_type: str,
    search_config: config.HyperparameterSearchConfig,
    prep_summary: dict[str, Any],
    study: optuna.Study,
    selected: dict[str, Any],
    finalist_results: list[dict[str, Any]],
    parameter_names: list[str],
    hpo_pool_rows: int,
    development_rows: int,
    holdout_rows: int,
    final_training_test_rows: int,
    zero_variance_features: list[str],
    feature_coverage_diagnostics: list[dict[str, Any]],
    time_range_diagnostics: list[dict[str, Any]],
    settings: dict[str, Any],
) -> tuple[Path, Path, Path, dict[str, Path]]:
    """Write tuning summary, finalist details, YAML, and Optuna plots."""
    run_timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_output_dir = output_dir / run_timestamp
    run_output_dir.mkdir(parents=True, exist_ok=False)

    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "lead_type_id": lead_type_id,
        "lead_type_name": lead_type_name,
        "model_type": model_type,
        "training_table_version": prep_summary.get("training_table_version"),
        "training_rows": prep_summary.get("training_rows"),
        "hpo_pool_rows": hpo_pool_rows,
        "development_rows": development_rows,
        "finalist_holdout_rows": holdout_rows,
        "final_training_test_rows": final_training_test_rows,
        "zero_variance_features": list(zero_variance_features),
        "feature_coverage_diagnostics": feature_coverage_diagnostics,
        "time_range_diagnostics": time_range_diagnostics,
        "scoring": search_config.scoring,
        "n_trials": len(study.trials),
        "cv_folds": search_config.cv_folds,
        "probability_shortlist_top_n": settings["probability_shortlist_top_n"],
        "calibration_enabled": settings["calibration_enabled"],
        "optimizer_enabled": settings["optimizer_enabled"],
        "optimizer_top_n": (
            settings["optimizer_top_n"] if settings["optimizer_enabled"] else None
        ),
        "optuna_best_trial": int(study.best_trial.number),
        "optuna_best_score": float(study.best_value),
        "selected_trial": selected["trial_number"],
        "selected_calibration_method": selected["calibration_method"],
        "selected_cv_score": selected["cv_score"],
        "selected_cv_std": selected["cv_std"],
        "selected_holdout_probability_metrics": selected["probability_metrics"],
        "selected_optimizer_metrics": selected["optimizer_metrics"],
        "selected_monotonicity": selected["monotonicity"],
        "selected_parameters": selected["parameters"],
    }

    summary_path = run_output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, default=str),
        encoding="utf-8",
    )

    finalist_path = run_output_dir / "finalist_results.json"
    finalist_path.write_text(
        json.dumps(
            _serializable_finalist_results(finalist_results),
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    calibration_method = selected["calibration_method"]
    yaml_payload = {
        "calibration": {
            "enabled": calibration_method != "none",
        },
        "models": {
            model_type: selected["parameters"],
        },
    }
    if calibration_method != "none":
        yaml_payload["calibration"]["method"] = calibration_method
        yaml_payload["calibration"]["cv"] = selected["calibration_cv"]

    parameters_path = run_output_dir / "best_parameters.yaml"
    parameters_path.write_text(
        yaml.safe_dump(yaml_payload, sort_keys=False),
        encoding="utf-8",
    )

    source_config_path = Path(search_config.raw["resolved"]["config_path"])
    config_copy_path = run_output_dir / "hyperparameter_search.yaml"
    config_copy_path.write_bytes(source_config_path.read_bytes())

    plot_paths = _write_optuna_plots(
        run_output_dir=run_output_dir,
        study=study,
        parameter_names=parameter_names,
        scoring=search_config.scoring,
    )
    return summary_path, parameters_path, finalist_path, plot_paths


def run_hyperparameter_search(
    lead_type_id: int,
    version: str | None = None,
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run SmartHub-aware hyperparameter search for one model family."""
    search_config = config.load_hyperparameter_search_config(config_path)
    _validate_probability_scoring(search_config.scoring)
    settings = _hpo_settings(search_config)

    normalized_model_type = search_config.model_type.strip().lower()
    model_config = search_config.model_config(normalized_model_type)
    lead_type_name = config.lead_type_name(lead_type_id)
    np.random.seed(search_config.random_seed)

    logger.info(
        "Loading training table for lead_type=%s (%s)",
        lead_type_name,
        lead_type_id,
    )
    frame, numeric, categorical, prep_summary = preprocessing.prepare_training_data(
        lead_type_id,
        lead_type_name,
        version,
    )
    preprocessing.assert_trainable(frame, lead_type_name)

    if settings["optimizer_enabled"] and config.REVENUE_COL not in frame.columns:
        raise ValueError(
            "Enabled SmartHub optimizer evaluation requires expected revenue "
            f"column {config.REVENUE_COL!r}."
        )

    # Protect the same final test partition that ``smarthub-train`` will use.
    # HPO must not use these rows for CV, finalist selection, calibration
    # selection, optimizer scoring, monotonicity checks, or feature cleanup.
    training_config = config.load_training_config()
    hpo_pool, final_training_test = _reserve_final_training_test(
        frame,
        split_settings=training_config.split,
        random_seed=training_config.random_seed,
    )
    preprocessing.assert_trainable(hpo_pool, lead_type_name)

    development, holdout = _split_development_and_holdout(
        hpo_pool,
        settings["holdout_fraction"],
    )
    preprocessing.assert_trainable(development, lead_type_name)

    # Detect zero-variance features using only model-fitting rows. Keep them
    # in the feature schema; the finalist holdout must not influence this
    # diagnostic.
    zero_variance_features = preprocessing.find_zero_variance_features(
        development,
        numeric,
        categorical,
    )
    logger.info("Feature Diagnostics")
    logger.info(
        "  Zero-variance features                : %s",
        f"{len(zero_variance_features):,}",
    )
    logger.info(
        "  Zero-variance feature names           : %s",
        ", ".join(zero_variance_features) or "none",
    )
    logger.info("  Zero-variance features retained       : yes")

    if len(development) <= search_config.cv_folds:
        raise ValueError(
            "Development rows must exceed search.cv_folds after reserving the "
            "finalist holdout."
        )

    feature_cols = numeric + categorical
    X = development[feature_cols]
    y = development[config.TARGET_COL]
    cross_validation = _build_cv(
        strategy=settings["validation_strategy"],
        n_splits=search_config.cv_folds,
        random_seed=search_config.random_seed,
    )
    feature_coverage_diagnostics = _build_hpo_coverage_diagnostics(
        development=development,
        holdout=holdout,
        numeric=numeric,
        categorical=categorical,
        cross_validation=cross_validation,
    )
    _log_hpo_coverage_diagnostics(feature_coverage_diagnostics)
    time_range_diagnostics = _build_hpo_time_range_diagnostics(
        development=development,
        holdout=holdout,
        feature_cols=feature_cols,
        cross_validation=cross_validation,
    )
    _log_hpo_time_range_diagnostics(time_range_diagnostics)

    fixed_parameters = {
        **model_config["fixed_parameters"],
        "random_state": search_config.random_seed,
    }
    search_space = model_config["search_space"]

    def objective(trial: optuna.Trial) -> float:
        trial_started = time.perf_counter()
        trial_parameters = _suggest_parameters(trial, search_space)
        model_parameters = {**fixed_parameters, **trial_parameters}

        parameter_text = ", ".join(
            f"{name}={value}" for name, value in trial_parameters.items()
        )
        logger.info(
            "Trial %s/%s started | %s",
            trial.number + 1,
            search_config.n_trials,
            parameter_text,
        )

        estimator = _build_estimator(
            model_type=normalized_model_type,
            numeric=numeric,
            categorical=categorical,
            model_parameters=model_parameters,
            calibration_method="none",
            calibration_cv=settings["calibration_cv"],
        )
        scores = _score_trial_folds(
            estimator=estimator,
            X=X,
            y=y,
            scoring=search_config.scoring,
            cross_validation=cross_validation,
            trial_number=trial.number + 1,
            total_folds=search_config.cv_folds,
        )
        stability = _trial_stability(scores)
        for name, value in stability.items():
            trial.set_user_attr(name, value)

        trial_elapsed = time.perf_counter() - trial_started
        logger.info(
            "Trial %s/%s complete | mean=%.6f | std=%.6f | "
            "min=%.6f | max=%.6f | elapsed=%.1fs",
            trial.number + 1,
            search_config.n_trials,
            stability["cv_mean"],
            stability["cv_std"],
            stability["cv_min"],
            stability["cv_max"],
            trial_elapsed,
        )
        return stability["cv_mean"]

    logger.info("Hyperparameter Search")
    logger.info("  Model type                            : %s", normalized_model_type)
    logger.info("  Total training rows                   : %s", f"{len(frame):,}")
    logger.info("  HPO-eligible rows                     : %s", f"{len(hpo_pool):,}")
    logger.info("  Development rows                      : %s", f"{len(development):,}")
    logger.info("  Finalist holdout rows                 : %s", f"{len(holdout):,}")
    logger.info(
        "  Final training test rows (untouched)  : %s",
        f"{len(final_training_test):,}",
    )
    logger.info(
        "  Validation strategy                   : %s",
        settings["validation_strategy"],
    )
    logger.info("  Trials                                : %s", search_config.n_trials)
    logger.info("  Cross-validation folds                : %s", search_config.cv_folds)
    logger.info("  Parallel Optuna trials                : %s", search_config.n_jobs)
    logger.info("  Scoring                               : %s", search_config.scoring)
    logger.info(
        "  Probability shortlist trials          : %s",
        settings["probability_shortlist_top_n"],
    )
    logger.info(
        "  Calibration search enabled            : %s",
        settings["calibration_enabled"],
    )
    logger.info(
        "  Optimizer evaluation enabled          : %s",
        settings["optimizer_enabled"],
    )
    if settings["optimizer_enabled"]:
        logger.info(
            "  Optimizer finalist candidates         : %s",
            settings["optimizer_top_n"],
        )

    study = optuna.create_study(
        study_name=f"{lead_type_name}_{normalized_model_type}",
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=search_config.random_seed),
    )
    study.optimize(
        objective,
        n_trials=search_config.n_trials,
        timeout=search_config.timeout_seconds,
        n_jobs=search_config.n_jobs,
    )

    finalist_results = _evaluate_probability_candidates(
        study=study,
        fixed_parameters=fixed_parameters,
        model_type=normalized_model_type,
        numeric=numeric,
        categorical=categorical,
        development=development,
        holdout=holdout,
        settings=settings,
    )
    if settings["optimizer_enabled"]:
        optimizer_shortlist = _optimizer_shortlist(finalist_results, settings)
        _evaluate_optimizer_shortlist(
            shortlist=optimizer_shortlist,
            holdout=holdout,
            feature_cols=feature_cols,
            settings=settings,
        )
        selected = _select_finalist(finalist_results, settings)
    else:
        selected = _select_probability_finalist(
            finalist_results,
            search_config.scoring,
        )

    output_dir = Path(
        search_config.output_dir(
            lead_type_name,
            normalized_model_type,
        )
    )
    summary_path, parameters_path, finalist_path, plot_paths = _write_outputs(
        output_dir=output_dir,
        lead_type_id=lead_type_id,
        lead_type_name=lead_type_name,
        model_type=normalized_model_type,
        search_config=search_config,
        prep_summary=prep_summary,
        study=study,
        selected=selected,
        finalist_results=finalist_results,
        parameter_names=list(search_space),
        hpo_pool_rows=len(hpo_pool),
        development_rows=len(development),
        holdout_rows=len(holdout),
        final_training_test_rows=len(final_training_test),
        zero_variance_features=zero_variance_features,
        feature_coverage_diagnostics=feature_coverage_diagnostics,
        time_range_diagnostics=time_range_diagnostics,
        settings=settings,
    )

    yaml_text = parameters_path.read_text(encoding="utf-8").rstrip()
    logger.info("Optuna best score: %.6f", study.best_value)
    logger.info("Selected finalist trial: %s", selected["trial_number"])
    logger.info("Selected calibration: %s", selected["calibration_method"])
    logger.info(
        "Selected holdout log loss: %.6f",
        selected["probability_metrics"]["log_loss"],
    )
    if settings["optimizer_enabled"]:
        logger.info(
            "Selected optimizer expected profit: %.6f",
            selected["optimizer_metrics"]["total_expected_profit"],
        )
    else:
        logger.info("Selected finalist by held-out probability quality.")
    logger.info("Copy the following into config/training.yaml:\n%s", yaml_text)
    logger.info("Saved summary: %s", summary_path)
    logger.info("Saved finalist results: %s", finalist_path)
    logger.info("Saved parameters: %s", parameters_path)
    for plot_name, plot_path in plot_paths.items():
        logger.info("Saved %s plot: %s", plot_name, plot_path)

    return {
        "lead_type_id": lead_type_id,
        "lead_type_name": lead_type_name,
        "model_type": normalized_model_type,
        "optuna_best_score": float(study.best_value),
        "selected_trial": int(selected["trial_number"]),
        "selected_calibration_method": selected["calibration_method"],
        "best_parameters": selected["parameters"],
        "holdout_probability_metrics": selected["probability_metrics"],
        "optimizer_metrics": selected["optimizer_metrics"],
        "monotonicity": selected["monotonicity"],
        "hpo_pool_rows": int(len(hpo_pool)),
        "development_rows": int(len(development)),
        "finalist_holdout_rows": int(len(holdout)),
        "final_training_test_rows": int(len(final_training_test)),
        "zero_variance_features": list(zero_variance_features),
        "feature_coverage_diagnostics": feature_coverage_diagnostics,
        "time_range_diagnostics": time_range_diagnostics,
        "summary_path": str(summary_path),
        "parameters_path": str(parameters_path),
        "finalist_results_path": str(finalist_path),
        "plot_paths": {
            plot_name: str(plot_path) for plot_name, plot_path in plot_paths.items()
        },
    }


def main(argv: list[str] | None = None) -> int:
    """Run SmartHub-aware hyperparameter search from command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run SmartHub-aware Optuna hyperparameter search."
    )
    parser.add_argument(
        "--lead-type-id",
        type=int,
        required=True,
        help="Lead type ID: 6=auto, 1=home, 5=commercial",
    )
    parser.add_argument(
        "--version",
        default=None,
        help="Training-table version (default: latest).",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional hyperparameter-search YAML path.",
    )
    args = parser.parse_args(argv)
    optuna.logging.set_verbosity(optuna.logging.INFO)

    run_hyperparameter_search(
        lead_type_id=args.lead_type_id,
        version=args.version,
        config_path=args.config,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
