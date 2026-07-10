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


def _write_outputs(
    output_dir: Path,
    lead_type_id: int,
    lead_type_name: str,
    model_type: str,
    search_config: config.HyperparameterSearchConfig,
    prep_summary: dict[str, Any],
    study: optuna.Study,
    best_parameters: dict[str, Any],
) -> tuple[Path, Path]:
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

    Returns
    -------
    tuple[pathlib.Path, pathlib.Path]
        Summary JSON path followed by best-parameters YAML path.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

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

    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, default=str),
        encoding="utf-8",
    )

    yaml_payload = {
        "models": {
            model_type: best_parameters,
        }
    }
    parameters_path = output_dir / "best_parameters.yaml"
    parameters_path.write_text(
        yaml.safe_dump(yaml_payload, sort_keys=False),
        encoding="utf-8",
    )
    return summary_path, parameters_path


def run_hyperparameter_search(
    lead_type_id: int,
    model_type: str,
    version: str | None = None,
    config_path: str | Path | None = None,
    log: logging.Logger | None = None,
) -> dict[str, Any]:
    """Run manual Bayesian hyperparameter search for one model.

    Inputs
    ------
    lead_type_id : int
        SmartHub lead type identifier.
    model_type : str
        Model family to optimize.
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
    normalized_model_type = model_type.strip().lower()
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
            calibrate=False,
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
    summary_path, parameters_path = _write_outputs(
        output_dir=output_dir,
        lead_type_id=lead_type_id,
        lead_type_name=lead_type_name,
        model_type=normalized_model_type,
        search_config=search_config,
        prep_summary=prep_summary,
        study=study,
        best_parameters=best_parameters,
    )

    yaml_text = parameters_path.read_text(encoding="utf-8").rstrip()
    log.info("Best score: %.6f", study.best_value)
    log.info("Best trial: %s", study.best_trial.number)
    log.info("Copy the following into config/training.yaml:\n%s", yaml_text)
    log.info("Saved summary: %s", summary_path)
    log.info("Saved parameters: %s", parameters_path)

    return {
        "lead_type_id": lead_type_id,
        "lead_type_name": lead_type_name,
        "model_type": normalized_model_type,
        "best_score": float(study.best_value),
        "best_parameters": best_parameters,
        "summary_path": str(summary_path),
        "parameters_path": str(parameters_path),
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
        "--model-type",
        required=True,
        choices=["logistic_regression", "xgboost", "lightgbm"],
        help="Model family to optimize.",
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
        model_type=args.model_type,
        version=args.version,
        config_path=args.config,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
