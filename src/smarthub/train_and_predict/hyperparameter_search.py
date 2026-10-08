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
from functools import wraps
from pathlib import Path
from typing import Any, Iterable
from uuid import uuid4

import numpy as np
import optuna
import pandas as pd
import yaml
from joblib import Parallel, delayed, parallel_config
from sklearn.base import clone
from sklearn.metrics import brier_score_loss, get_scorer, log_loss
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit

from smarthub.core import notifications
from smarthub.core.lead_types import lead_type_name as resolve_lead_type_name
from smarthub.core.logging_utils import get_logger

from . import (
    config,
    feature_diagnostics,
    model_parameters,
    models,
    optimizer_evaluation,
    preprocessing,
)

logger = get_logger(__name__)

_PROBABILITY_SCORERS = {"neg_log_loss", "neg_brier_score"}


def _notification_fields(lead_type_id):
    try:
        label = f"{resolve_lead_type_name(lead_type_id)} ({lead_type_id})"
    except (ValueError, KeyError, TypeError):
        label = str(lead_type_id)
    return {"Lead type": label}


def _send_notification(status, fields, error=None):
    try:
        fields = dict(fields)
        subject = fields.pop("Lead type", None)
        headline = fields.pop("Status", None)
        severity = "success" if status == "started" else status
        display_status = {
            "started": "started",
            "success": "completed",
            "warning": "WARNING",
            "failure": "FAILED",
        }.get(status, status)
        if headline == "HPO candidate promoted":
            display_status = "promoted"
        elif headline == "HPO candidate not promoted":
            display_status = "completed (not promoted)"
        if error:
            fields["Error"] = str(error).strip()[:1500]
        delivered = notifications.notify_grouped(
            severity,
            "hpo",
            subject=subject,
            status=display_status,
            headline=headline,
            groups=[("HPO", fields)],
        )
        if not delivered:
            logger.info("HPO Slack alert was not delivered or Slack is disabled.")
    except Exception:
        logger.warning("HPO Slack notification failed; run continues.", exc_info=True)


def _report_search(function):
    """Report the public search entrypoint, including CLI and Prefect calls."""

    @wraps(function)
    def wrapped(lead_type_id, version=None, config_path=None):
        started = time.perf_counter()
        fields = {
            **_notification_fields(lead_type_id),
            "Status": "HPO started",
            "Requested dataset": version or "latest",
        }
        _send_notification("started", fields)
        try:
            result = function(lead_type_id, version, config_path)
        except Exception as exc:
            _send_notification(
                "failure",
                {
                    **fields,
                    "Status": "HPO failed",
                    "Duration (seconds)": round(time.perf_counter() - started, 1),
                },
                error=f"{type(exc).__name__}: {exc}",
            )
            raise
        _send_notification(
            "success",
            {
                **fields,
                "Status": "HPO completed; parameters saved",
                "HPO run": result.get("hpo_run_id"),
                "Parameter version": result.get("parameter_version"),
                "HPO MLflow run": result.get("hpo_mlflow_run_id"),
                "Model": result.get("model_type"),
                "Selected trial": result.get("selected_trial"),
                "Calibration": result.get("selected_calibration_method"),
                "Holdout log loss": (
                    result.get("holdout_probability_metrics") or {}
                ).get("log_loss"),
                "Parameter file": result.get("parameters_path"),
                "Duration (seconds)": round(time.perf_counter() - started, 1),
            },
        )
        return result

    return wrapped


def notify_candidate_result(lead_type_id, hpo_result, training_result, state=None):
    """Keep successful search separate from acceptance of its trained candidate."""
    promoted = training_result.get("promoted") is True
    state = state or {}
    _send_notification(
        "success",
        {
            **_notification_fields(lead_type_id),
            "Status": (
                "HPO candidate promoted" if promoted else "HPO candidate not promoted"
            ),
            "HPO run": hpo_result.get("hpo_run_id"),
            "Parameter version": hpo_result.get("parameter_version"),
            "Training run": training_result.get("training_run_id"),
            "Promotion status": training_result.get("promotion_status"),
            "Reason": training_result.get("promotion_reason"),
            "Current parameters": "Updated" if promoted else "Unchanged",
            "Next HPO retry": state.get("retry_date") if not promoted else None,
            "Next scheduled HPO": state.get("next_scheduled_date"),
            "Timezone": (state.get("schedule") or {}).get("timezone"),
        },
    )


def notify_candidate_error(lead_type_id, hpo_result, state, error):
    """Report execution errors and the persistent next-day retry decision."""
    phase = state["status"]
    _send_notification(
        "failure",
        {
            **_notification_fields(lead_type_id),
            "Status": (
                "HPO candidate training failed"
                if phase == "training_failed"
                else "HPO failed; retry scheduled"
            ),
            "HPO run": (hpo_result or {}).get("hpo_run_id"),
            "Promotion": (
                "Outcome not confirmed"
                if phase == "training_failed"
                else "Candidate training not reached"
            ),
            "Next HPO retry": state.get("retry_date"),
            "Next scheduled HPO": state.get("next_scheduled_date"),
            "Timezone": (state.get("schedule") or {}).get("timezone"),
        },
        error=f"{type(error).__name__}: {error}",
    )


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
        "cv_jobs": search_config.cv_jobs,
        "probability_jobs": search_config.probability_jobs,
        "optimizer_jobs": search_config.optimizer_jobs,
        "validation_strategy": search_config.validation_strategy,
        "split": dict(search_config.split),
        "early_stopping": search_config.early_stopping.as_dict(),
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
        "mlflow": search_config.raw.get("mlflow") or {"enabled": False},
    }


def _reserve_final_test(
    frame: pd.DataFrame,
    split_settings: dict[str, Any],
    random_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reserve HPO's untouched final test partition.

    The split is configured entirely by ``hyperparameter_search.yaml``. HPO
    must not inspect these rows during CV, finalist selection, calibration
    selection, optimizer scoring, or monotonicity evaluation.

    Inputs
    ------
    frame : pandas.DataFrame
        Prepared model-ready training data.
    split_settings : dict[str, Any]
        HPO-owned split strategy and options from ``hyperparameter_search.yaml``.
    random_seed : int
        HPO random seed used for reproducible random splitting.

    Returns
    -------
    tuple[pandas.DataFrame, pandas.DataFrame]
        HPO-eligible rows followed by the untouched final HPO test rows.
    """
    hpo_pool, final_test = model_parameters.split_training_data(
        frame, config.TARGET_COL, split_settings, random_seed
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
    features = feature_diagnostics.coverage_features(development, numeric, categorical)
    diagnostics: list[dict[str, Any]] = []

    for fold_number, (train_idx, valid_idx) in enumerate(
        _iter_splits(cross_validation, X, y),
        start=1,
    ):
        diagnostics.extend(
            feature_diagnostics.feature_coverage_rows(
                development.iloc[train_idx],
                development.iloc[valid_idx],
                features,
                partition=f"cv_fold_{fold_number}",
            )
        )

    diagnostics.extend(
        feature_diagnostics.feature_coverage_rows(
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
    """Log feature-coverage diagnostics for HPO evaluation boundaries."""
    feature_diagnostics.log_feature_coverage_diagnostics(
        diagnostics,
        evaluation_label="eval",
        include_partition=True,
    )


def _parallel_map(function, jobs: list[dict[str, Any]], n_jobs: int):
    """Evaluate independent jobs in input order with bounded native threads."""
    workers = min(n_jobs, len(jobs))
    if workers <= 1:
        return [function(**job) for job in jobs]
    with parallel_config(backend="loky", inner_max_num_threads=1):
        return Parallel(n_jobs=workers, pre_dispatch=workers)(
            delayed(function)(**job) for job in jobs
        )


def _score_trial_fold(
    estimator,
    X,
    y,
    scoring,
    train_idx,
    valid_idx,
    fold_number,
    trial_number,
    total_folds,
    early_stopping_settings,
):
    """Fit an isolated estimator on one fold and return its diagnostics."""
    scorer = get_scorer(scoring)
    best_iteration = None
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
    if early_stopping_settings["enabled"]:
        early_stopping_result = models.fit_lightgbm_with_early_stopping(
            fitted,
            X.iloc[train_idx],
            y_train,
            X.iloc[valid_idx],
            y_valid,
            stopping_rounds=early_stopping_settings["stopping_rounds"],
            eval_metric=early_stopping_settings["metric"],
        )
        best_iteration = early_stopping_result["best_iteration"]
    else:
        fitted.fit(X.iloc[train_idx], y_train)

    score = float(scorer(fitted, X.iloc[valid_idx], y_valid))
    fold_elapsed = time.perf_counter() - fold_started

    if early_stopping_settings["enabled"]:
        best_score = early_stopping_result["best_score"]
        logger.info(
            "Trial %s | fold %s/%s | complete | score=%.6f | "
            "best_iteration=%s | stopped_at=%s | best_%s=%s | "
            "early_stop=%s | elapsed=%.1fs",
            trial_number,
            fold_number,
            total_folds,
            score,
            best_iteration,
            early_stopping_result["stopped_iteration"],
            early_stopping_settings["metric"],
            f"{best_score:.6f}" if best_score is not None else "n/a",
            "yes" if early_stopping_result["stopped_early"] else "no",
            fold_elapsed,
        )
    else:
        logger.info(
            "Trial %s | fold %s/%s | complete | score=%.6f | " "elapsed=%.1fs",
            trial_number,
            fold_number,
            total_folds,
            score,
            fold_elapsed,
        )
    return score, best_iteration


def _score_trial_folds(
    estimator,
    X: pd.DataFrame,
    y: pd.Series,
    scoring: str,
    cross_validation,
    trial_number: int,
    total_folds: int,
    early_stopping_settings: dict[str, Any],
    n_jobs: int = 1,
) -> tuple[list[float], list[int]]:
    """Fit independent CV folds and collect results in fold order."""
    jobs = [
        dict(
            estimator=estimator,
            X=X,
            y=y,
            scoring=scoring,
            train_idx=train_idx,
            valid_idx=valid_idx,
            fold_number=fold_number,
            trial_number=trial_number,
            total_folds=total_folds,
            early_stopping_settings=early_stopping_settings,
        )
        for fold_number, (train_idx, valid_idx) in enumerate(
            _iter_splits(cross_validation, X, y), start=1
        )
    ]
    results = _parallel_map(_score_trial_fold, jobs, n_jobs)
    return (
        [score for score, _ in results],
        [iteration for _, iteration in results if iteration is not None],
    )


def _trial_stability(
    scores: list[float],
    best_iterations: list[int] | None = None,
) -> dict[str, Any]:
    """Summarize fold-level score and boosting-iteration stability."""
    array = np.asarray(scores, dtype=float)
    result = {
        "cv_mean": float(np.mean(array)),
        "cv_std": float(np.std(array, ddof=0)),
        "cv_min": float(np.min(array)),
        "cv_max": float(np.max(array)),
        "fold_scores": [float(value) for value in array],
    }
    if best_iterations:
        iterations = np.asarray(best_iterations, dtype=int)
        result.update(
            {
                "fold_best_iterations": [int(value) for value in iterations],
                "best_iteration_median": int(round(float(np.median(iterations)))),
                "best_iteration_min": int(np.min(iterations)),
                "best_iteration_max": int(np.max(iterations)),
            }
        )
    return result


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
    best_iteration = trial.user_attrs.get("best_iteration_median")
    if settings["early_stopping"]["enabled"]:
        if best_iteration is None:
            raise ValueError(
                "Early-stopped HPO trial is missing best_iteration_median."
            )
        model_parameters["n_estimators"] = int(best_iteration)
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
        "fold_best_iterations": trial.user_attrs.get("fold_best_iterations", []),
        "best_iteration": (int(best_iteration) if best_iteration is not None else None),
        "probability_metrics": probability_metrics,
        "optimizer_selected": False,
        "optimizer_metrics": None,
        "monotonicity": None,
        "_model": estimator,
    }


def _evaluate_probability_candidate_safely(**kwargs):
    """Keep the existing per-candidate failure policy inside each worker."""
    trial = kwargs["trial"]
    method = kwargs["calibration_method"]
    logger.info("Probability finalist: trial=%s calibration=%s", trial.number, method)
    try:
        result = _evaluate_probability_candidate(**kwargs)
    except (RuntimeError, ValueError) as exc:
        logger.warning(
            "Skipping probability finalist trial=%s calibration=%s: %s",
            trial.number,
            method,
            exc,
        )
        return None
    logger.info(
        "Trial=%s calibration=%s | log loss=%.6f | Brier=%.6f",
        trial.number,
        method,
        result["probability_metrics"]["log_loss"],
        result["probability_metrics"]["brier_score"],
    )
    return result


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

    jobs = [
        dict(
            trial=trial,
            calibration_method=method,
            fixed_parameters=fixed_parameters,
            model_type=model_type,
            numeric=numeric,
            categorical=categorical,
            development=development,
            holdout=holdout,
            settings=settings,
        )
        for trial in top_trials
        for method in settings["calibration_methods"]
    ]
    evaluated = _parallel_map(
        _evaluate_probability_candidate_safely,
        jobs,
        settings.get("probability_jobs", 1),
    )
    results = [result for result in evaluated if result is not None]

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
    jobs = [
        dict(
            model=result["_model"],
            holdout=holdout,
            feature_cols=feature_cols,
            settings=settings,
        )
        for result in shortlist
    ]
    evaluated = _parallel_map(
        _evaluate_optimizer_and_monotonicity,
        jobs,
        settings.get("optimizer_jobs", 1),
    )
    for result, (optimizer_metrics, monotonicity) in zip(shortlist, evaluated):
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
    feature_target_diagnostics: list[dict[str, Any]],
    feature_coverage_diagnostics: list[dict[str, Any]],
    time_range_diagnostics: list[dict[str, Any]],
    settings: dict[str, Any],
) -> tuple[Path, Path, Path, dict[str, Path]]:
    """Write tuning summary, finalist details, YAML, and Optuna plots."""
    run_timestamp = (
        datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + "_" + uuid4().hex[:8]
    )
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
        "feature_target_association": feature_target_diagnostics,
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
        "selected_best_iteration": selected.get("best_iteration"),
        "early_stopping": settings["early_stopping"],
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
    selected_model_parameters = {
        key: value
        for key, value in selected["parameters"].items()
        if key != "random_state"
        and (key != "n_estimators" or not settings["early_stopping"]["enabled"])
    }
    calibration = {"enabled": calibration_method != "none"}
    if calibration_method != "none":
        calibration.update(method=calibration_method, cv=selected["calibration_cv"])
    model_settings = model_parameters.normalize_settings(
        {
            "model_type": model_type,
            "model_parameters": selected_model_parameters,
            "calibration": calibration,
        }
    )
    hpo_run_id = f"hpo_{lead_type_name}_{run_timestamp}"
    yaml_payload = model_parameters.make_artifact(
        model_settings,
        lead_type_id,
        hpo_run_id=hpo_run_id,
        created_at=summary["created_at"],
        code_version=model_parameters.code_version(),
        data=settings.get("candidate_data"),
    )
    summary["selected_cv_median_best_iteration"] = selected.get("best_iteration")

    parameters_path = run_output_dir / "best_parameters.yaml"
    parameters_path.write_text(
        yaml.safe_dump(yaml_payload, sort_keys=False),
        encoding="utf-8",
    )

    source_config_path = Path(search_config.raw["resolved"]["config_path"])
    config_copy_path = run_output_dir / "hyperparameter_search.yaml"
    config_copy_path.write_bytes(source_config_path.read_bytes())
    summary.update(
        {
            "hpo_run_id": hpo_run_id,
            "parameter_version": yaml_payload["parameter_version"],
            "code_version": yaml_payload["code_version"],
            "data": yaml_payload["data"],
        }
    )
    summary_path.write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )

    plot_paths = _write_optuna_plots(
        run_output_dir=run_output_dir,
        study=study,
        parameter_names=parameter_names,
        scoring=search_config.scoring,
    )
    return summary_path, parameters_path, finalist_path, plot_paths


@_report_search
def run_hyperparameter_search(
    lead_type_id: int,
    version: str | None = None,
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run SmartHub-aware hyperparameter search for one model family."""
    run_started = time.perf_counter()
    search_config = config.load_hyperparameter_search_config(
        lead_type_id,
        config_path,
    )
    _validate_probability_scoring(search_config.scoring)
    settings = _hpo_settings(search_config)
    logger.info(
        "HPO workers: CV=%s probability=%s optimizer=%s",
        settings["cv_jobs"],
        settings["probability_jobs"],
        settings["optimizer_jobs"],
    )

    normalized_model_type = search_config.model_type.strip().lower()
    model_config = search_config.model_config(normalized_model_type)
    lead_type_name = resolve_lead_type_name(lead_type_id)
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

    feature_target_diagnostics = feature_diagnostics.build_feature_target_association(
        frame=frame,
        numeric_features=numeric,
        categorical_features=categorical,
        target_column=config.TARGET_COL,
        random_seed=search_config.random_seed,
    )
    feature_diagnostics.log_feature_target_association(
        feature_target_diagnostics,
    )

    if settings["optimizer_enabled"] and config.REVENUE_COL not in frame.columns:
        raise ValueError(
            "Enabled SmartHub optimizer evaluation requires expected revenue "
            f"column {config.REVENUE_COL!r}."
        )

    # Reserve an HPO-only final test partition using HPO configuration only.
    # No settings are read from training.yaml.
    hpo_pool, final_training_test = _reserve_final_test(
        frame,
        split_settings=search_config.split,
        random_seed=search_config.random_seed,
    )
    settings["candidate_data"] = {
        "training_table_version": prep_summary["training_table_version"],
        "frame_fingerprint": model_parameters.dataset_fingerprint(frame),
        "split_version": 1,
        "split_settings": dict(search_config.split),
        "random_seed": search_config.random_seed,
        "data_min_created_at": prep_summary.get("data_min_created_at"),
        "data_max_created_at": prep_summary.get("data_max_created_at"),
    }
    preprocessing.assert_trainable(hpo_pool, lead_type_name)

    development, holdout = _split_development_and_holdout(
        hpo_pool,
        settings["holdout_fraction"],
    )
    preprocessing.assert_trainable(development, lead_type_name)

    # Detect zero-variance features using only model-fitting rows. Keep them
    # in the feature schema; the finalist holdout must not influence this
    # diagnostic.
    zero_variance_features = feature_diagnostics.find_zero_variance_features(
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
    if settings["early_stopping"]["enabled"]:
        fixed_parameters["n_estimators"] = settings["early_stopping"]["max_estimators"]
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
        scores, best_iterations = _score_trial_folds(
            estimator=estimator,
            X=X,
            y=y,
            scoring=search_config.scoring,
            cross_validation=cross_validation,
            trial_number=trial.number + 1,
            total_folds=search_config.cv_folds,
            early_stopping_settings=settings["early_stopping"],
            n_jobs=settings["cv_jobs"],
        )
        stability = _trial_stability(scores, best_iterations)
        for name, value in stability.items():
            trial.set_user_attr(name, value)

        trial_elapsed = time.perf_counter() - trial_started
        logger.info(
            "Trial %s/%s complete | mean=%.6f | std=%.6f | "
            "min=%.6f | max=%.6f | median_best_iteration=%s | elapsed=%.1fs",
            trial.number + 1,
            search_config.n_trials,
            stability["cv_mean"],
            stability["cv_std"],
            stability["cv_min"],
            stability["cv_max"],
            stability.get("best_iteration_median", "n/a"),
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
        "  Early stopping enabled                : %s",
        settings["early_stopping"]["enabled"],
    )
    if settings["early_stopping"]["enabled"]:
        logger.info(
            "  Early stopping maximum estimators     : %s",
            settings["early_stopping"]["max_estimators"],
        )
        logger.info(
            "  Early stopping rounds                 : %s",
            settings["early_stopping"]["stopping_rounds"],
        )
        logger.info(
            "  Early stopping metric                 : %s",
            settings["early_stopping"]["metric"],
        )
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
    stage_started = time.perf_counter()
    study.optimize(
        objective,
        n_trials=search_config.n_trials,
        timeout=search_config.timeout_seconds,
        n_jobs=search_config.n_jobs,
    )

    timings = {"search_seconds": time.perf_counter() - stage_started}
    stage_started = time.perf_counter()
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
    timings["probability_seconds"] = time.perf_counter() - stage_started
    stage_started = time.perf_counter()
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

    timings["optimizer_and_selection_seconds"] = time.perf_counter() - stage_started
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
        feature_target_diagnostics=feature_target_diagnostics,
        feature_coverage_diagnostics=feature_coverage_diagnostics,
        time_range_diagnostics=time_range_diagnostics,
        settings=settings,
    )

    timings["total_seconds"] = time.perf_counter() - run_started
    summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
    summary_payload["timings"] = timings
    summary_payload["parallelism"] = {
        key: settings[key] for key in ("cv_jobs", "probability_jobs", "optimizer_jobs")
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    mlflow_settings = settings["mlflow"]
    if mlflow_settings.get("enabled"):
        try:
            from . import mlflow_utils

            mlflow_metadata = mlflow_utils.log_hpo_run(
                run_output_dir=summary_path.parent,
                settings=mlflow_settings,
                lead_type_name=lead_type_name,
                resolved_search_config=search_config.as_dict(),
            )
        except Exception:
            logger.exception("HPO MLflow logging failed; parameter artifact retained.")
            if mlflow_settings.get("required", True):
                raise
        else:
            summary_payload.update(mlflow_metadata)
            summary_path.write_text(
                json.dumps(summary_payload, indent=2), encoding="utf-8"
            )
    logger.info("HPO stage timings (seconds): %s", timings)
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
            "Selected optimizer probability-weighted expected profit: %.6f",
            selected["optimizer_metrics"]["total_expected_profit"],
        )
    else:
        logger.info("Selected finalist by held-out probability quality.")
    logger.info("Selected parameters YAML:\n%s", yaml_text)
    logger.info("Saved summary: %s", summary_path)
    logger.info("Saved finalist results: %s", finalist_path)
    logger.info("Saved parameters: %s", parameters_path)
    logger.info(
        "Candidate training command: smarthub-train --lead-type-id %s "
        "--parameter-file %s",
        lead_type_id,
        parameters_path,
    )
    for plot_name, plot_path in plot_paths.items():
        logger.info("Saved %s plot: %s", plot_name, plot_path)

    return {
        "hpo_run_id": summary_payload["hpo_run_id"],
        "parameter_version": summary_payload["parameter_version"],
        "hpo_mlflow_run_id": summary_payload.get("hpo_mlflow_run_id"),
        "lead_type_id": lead_type_id,
        "lead_type_name": lead_type_name,
        "model_type": normalized_model_type,
        "optuna_best_score": float(study.best_value),
        "selected_trial": int(selected["trial_number"]),
        "selected_calibration_method": selected["calibration_method"],
        "selected_best_iteration": selected.get("best_iteration"),
        "best_parameters": selected["parameters"],
        "holdout_probability_metrics": selected["probability_metrics"],
        "optimizer_metrics": selected["optimizer_metrics"],
        "monotonicity": selected["monotonicity"],
        "hpo_pool_rows": int(len(hpo_pool)),
        "development_rows": int(len(development)),
        "finalist_holdout_rows": int(len(holdout)),
        "final_training_test_rows": int(len(final_training_test)),
        "zero_variance_features": list(zero_variance_features),
        "feature_target_association": feature_target_diagnostics,
        "feature_coverage_diagnostics": feature_coverage_diagnostics,
        "time_range_diagnostics": time_range_diagnostics,
        "summary_path": str(summary_path),
        "parameters_path": str(parameters_path),
        "finalist_results_path": str(finalist_path),
        "timings": timings,
        "parallelism": summary_payload["parallelism"],
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
