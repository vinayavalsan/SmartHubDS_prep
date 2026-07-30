"""Manual hyperparameter search for SmartHub win-probability models.

This module runs an Optuna study and writes the best parameters as a
copy-paste-ready YAML block for the official training configuration.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import yaml
from sklearn.model_selection import StratifiedKFold, cross_val_score

from . import config, models, preprocessing

logger = logging.getLogger("smarthub.train_and_predict.hyperparameter_search")


def _suggest_parameter(
    trial: optuna.Trial,
    parameter_name: str,
    specification: dict[str, Any],
) -> Any:
    """Suggest one parameter value from its configured search space.

    Inputs
    ------
    trial : optuna.Trial
        Active Optuna trial.
    parameter_name : str
        Hyperparameter name.
    specification : dict[str, Any]
        Validated parameter search-space definition.

    Returns
    -------
    Any
        Suggested parameter value.

    Raises
    ------
    ValueError
        If the configured parameter type is unsupported.
    """
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
            step=int(specification["step"]),
            log=specification["log"],
        )

    if parameter_type == "float":
        step = specification["step"]
        return trial.suggest_float(
            parameter_name,
            float(specification["low"]),
            float(specification["high"]),
            step=float(step) if step is not None else None,
            log=specification["log"],
        )

    raise ValueError(
        f"Unsupported parameter type {parameter_type!r} " f"for {parameter_name!r}."
    )


def _suggest_parameters(
    trial: optuna.Trial,
    search_space: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Suggest all configured parameters for one trial.

    Inputs
    ------
    trial : optuna.Trial
        Active Optuna trial.
    search_space : dict[str, dict[str, Any]]
        Model-specific parameter search space.

    Returns
    -------
    dict[str, Any]
        Suggested model parameters.
    """
    return {
        parameter_name: _suggest_parameter(
            trial,
            parameter_name,
            specification,
        )
        for parameter_name, specification in search_space.items()
    }


def _scoring_plot_config(scoring: str) -> tuple[None, str]:
    """Build a plot label from the configured sklearn scoring name.

    All plots use the raw cross-validation score returned by scikit-learn,
    so larger values are always better. Negative loss scorers therefore
    remain negative in the plots (for example, ``neg_log_loss`` is shown as
    ``Negative Log Loss``).

    Inputs
    ------
    scoring : str
        Scikit-learn scoring name used by cross-validation.

    Returns
    -------
    tuple[None, str]
        No target transformation and a human-readable plot label.
    """
    metric_names = {
        "accuracy": "Accuracy",
        "average_precision": "Average Precision",
        "balanced_accuracy": "Balanced Accuracy",
        "f1": "F1 Score",
        "neg_brier_score": "Negative Brier Score",
        "neg_log_loss": "Negative Log Loss",
        "neg_mean_absolute_error": "Negative Mean Absolute Error",
        "neg_mean_absolute_percentage_error": (
            "Negative Mean Absolute Percentage Error"
        ),
        "neg_mean_gamma_deviance": "Negative Mean Gamma Deviance",
        "neg_mean_poisson_deviance": "Negative Mean Poisson Deviance",
        "neg_mean_squared_error": "Negative Mean Squared Error",
        "neg_mean_squared_log_error": "Negative Mean Squared Log Error",
        "neg_median_absolute_error": "Negative Median Absolute Error",
        "neg_root_mean_squared_error": "Negative Root Mean Squared Error",
        "precision": "Precision",
        "r2": "R²",
        "recall": "Recall",
        "roc_auc": "ROC AUC",
    }

    metric_name = metric_names.get(
        scoring,
        scoring.replace("neg_", "negative_").replace("_", " ").title(),
    )
    target_name = f"Mean Cross-Validation {metric_name} " "(higher is better)"
    return None, target_name


def _write_optuna_plots(
    run_output_dir: Path,
    study: optuna.Study,
    parameter_names: list[str],
    scoring: str,
    log: logging.Logger,
) -> dict[str, Path]:
    """Write interactive Optuna visualizations as HTML artifacts.

    Inputs
    ------
    run_output_dir : pathlib.Path
        Directory for the current hyperparameter-search run.
    study : optuna.Study
        Completed Optuna study.
    parameter_names : list[str]
        Hyperparameters to include in the multi-parameter contour matrix.
    scoring : str
        Scikit-learn scoring name used by the Optuna objective.
    log : logging.Logger
        Logger used to report any visualization failures.

    Returns
    -------
    dict[str, pathlib.Path]
        Successfully written visualization names and paths.
    """
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
            log.warning("Unable to create %s plot: %s", plot_name, exc)

    return plot_paths


def _write_outputs(
    output_dir: Path,
    lead_type_id: int,
    lead_type_name: str,
    model_type: str,
    search_config: config.HyperparameterSearchConfig,
    prep_summary: dict[str, Any],
    study: optuna.Study,
    best_parameters: dict[str, Any],
    parameter_names: list[str],
    log: logging.Logger,
) -> tuple[Path, Path, dict[str, Path]]:
    """Write the minimal tuning summary and copy-paste YAML block.

    Inputs
    ------
    output_dir : pathlib.Path
        Directory for tuning results.
    lead_type_id : int
        SmartHub lead type identifier.
    lead_type_name : str
        Human-readable lead type name.
    model_type : str
        Model family optimized by the study.
    search_config : HyperparameterSearchConfig
        Resolved search configuration.
    prep_summary : dict[str, Any]
        Training-data preparation summary.
    study : optuna.Study
        Completed Optuna study.
    best_parameters : dict[str, Any]
        Winning parameters including fixed values.
    parameter_names : list[str]
        Hyperparameters to include in the contour matrix.
    log : logging.Logger
        Logger used to report visualization output and failures.

    Returns
    -------
    tuple[pathlib.Path, pathlib.Path, dict[str, pathlib.Path]]
        Summary JSON path, best-parameters YAML path, and visualization paths.
    """
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
        "scoring": search_config.scoring,
        "n_trials": len(study.trials),
        "cv_folds": search_config.cv_folds,
        "best_trial": study.best_trial.number,
        "best_score": float(study.best_value),
        "best_parameters": best_parameters,
    }

    summary_path = run_output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, default=str),
        encoding="utf-8",
    )

    yaml_payload = {
        "models": {
            model_type: best_parameters,
        }
    }
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
        log=log,
    )

    return summary_path, parameters_path, plot_paths


def run_hyperparameter_search(
    lead_type_id: int,
    version: str | None = None,
    config_path: str | Path | None = None,
    log: logging.Logger | None = None,
) -> dict[str, Any]:
    """Run manual Bayesian hyperparameter search for one model.

    Inputs
    ------
    lead_type_id : int
        SmartHub lead type identifier.
    version : str | None
        Optional training-table version. The latest is used when omitted.
    config_path : str | pathlib.Path | None
        Optional hyperparameter-search YAML path.
    log : logging.Logger | None
        Optional logger for structured output.

    Returns
    -------
    dict[str, Any]
        Search result and generated output paths.
    """
    log = log or logger
    search_config = config.load_hyperparameter_search_config(config_path)
    normalized_model_type = search_config.model_type.strip().lower()
    model_config = search_config.model_config(normalized_model_type)
    lead_type_name = config.lead_type_name(lead_type_id)

    np.random.seed(search_config.random_seed)

    log.info(
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

    if prep_summary["training_rows"] < search_config.cv_folds:
        raise ValueError("Training rows must be at least as large as search.cv_folds.")

    numeric, categorical, dropped = preprocessing.drop_zero_variance(
        frame,
        numeric,
        categorical,
    )
    if dropped:
        log.info("Dropped zero-variance features: %s", ", ".join(dropped))

    feature_cols = numeric + categorical
    X = frame[feature_cols]
    y = frame[config.TARGET_COL]

    cross_validation = StratifiedKFold(
        n_splits=search_config.cv_folds,
        shuffle=True,
        random_state=search_config.random_seed,
    )
    fixed_parameters = {
        **model_config["fixed_parameters"],
        "random_state": search_config.random_seed,
    }
    search_space = model_config["search_space"]

    def objective(trial: optuna.Trial) -> float:
        trial_parameters = _suggest_parameters(trial, search_space)
        model_parameters = {
            **fixed_parameters,
            **trial_parameters,
        }
        estimator = models.build_model(
            model_type=normalized_model_type,
            numeric_features=numeric,
            categorical_features=categorical,
            model_params=model_parameters,
            calibration_enabled=False,
            calibration_method=None,
            calibration_cv=None,
        )
        scores = cross_val_score(
            estimator,
            X,
            y,
            scoring=search_config.scoring,
            cv=cross_validation,
            n_jobs=search_config.n_jobs,
            error_score="raise",
        )
        return float(np.mean(scores))

    log.info("Hyperparameter Search")
    log.info("  Model type                            : %s", normalized_model_type)
    log.info("  Training rows                         : %s", f"{len(frame):,}")
    log.info("  Trials                                : %s", search_config.n_trials)
    log.info("  Cross-validation folds                : %s", search_config.cv_folds)
    log.info("  Scoring                               : %s", search_config.scoring)

    study = optuna.create_study(
        study_name=f"{lead_type_name}_{normalized_model_type}",
        direction="maximize",
        sampler=optuna.samplers.TPESampler(
            seed=search_config.random_seed,
        ),
    )
    study.optimize(
        objective,
        n_trials=search_config.n_trials,
        timeout=search_config.timeout_seconds,
    )

    best_parameters = {
        **fixed_parameters,
        **study.best_params,
    }
    output_dir = Path(
        search_config.output_dir(
            lead_type_name,
            normalized_model_type,
        )
    )
    summary_path, parameters_path, plot_paths = _write_outputs(
        output_dir=output_dir,
        lead_type_id=lead_type_id,
        lead_type_name=lead_type_name,
        model_type=normalized_model_type,
        search_config=search_config,
        prep_summary=prep_summary,
        study=study,
        best_parameters=best_parameters,
        parameter_names=list(search_space),
        log=log,
    )

    yaml_text = parameters_path.read_text(encoding="utf-8").rstrip()
    log.info("Best score: %.6f", study.best_value)
    log.info("Best trial: %s", study.best_trial.number)
    log.info("Copy the following into config/training.yaml:\n%s", yaml_text)
    log.info("Saved summary: %s", summary_path)
    log.info("Saved parameters: %s", parameters_path)
    for plot_name, plot_path in plot_paths.items():
        log.info("Saved %s plot: %s", plot_name, plot_path)

    return {
        "lead_type_id": lead_type_id,
        "lead_type_name": lead_type_name,
        "model_type": normalized_model_type,
        "best_score": float(study.best_value),
        "best_parameters": best_parameters,
        "summary_path": str(summary_path),
        "parameters_path": str(parameters_path),
        "plot_paths": {
            plot_name: str(plot_path) for plot_name, plot_path in plot_paths.items()
        },
    }


def main(argv: list[str] | None = None) -> int:
    """Run manual hyperparameter search from command-line arguments.

    Inputs
    ------
    argv : list[str] | None
        Optional command-line argument sequence.

    Returns
    -------
    int
        Process exit status.
    """
    parser = argparse.ArgumentParser(
        description="Run manual Optuna hyperparameter search."
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

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    optuna.logging.set_verbosity(optuna.logging.INFO)

    run_hyperparameter_search(
        lead_type_id=args.lead_type_id,
        version=args.version,
        config_path=args.config,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
