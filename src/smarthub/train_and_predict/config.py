"""Configuration for Anton model training and bid optimization.

This module loads and validates YAML settings for training and serving.
"""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from smarthub.core import paths, task_config
from smarthub.feature_engineering import features as fe

TARGET_COL = fe.TARGET_COLUMN
REVENUE_COL = fe.REVENUE_COLUMN
_SUPPORTED_MODELS = {"logistic_regression", "xgboost", "lightgbm"}
_SUPPORTED_SPLIT_STRATEGIES = {"time", "random"}
_SUPPORTED_PROMOTION_MODES = {"manual", "automatic", "disabled"}


@dataclass(frozen=True)
class OptimizerConfig:
    """Store resolved bid-optimizer configuration values."""

    target_cm: float
    minimum_bid: float
    bid_step: float

    def as_dict(self) -> dict[str, float]:
        """Return the configuration values as a dictionary.

        Returns
        -------
        dict
            Dictionary representation of the configuration.
        """
        return {
            "target_cm": self.target_cm,
            "minimum_bid": self.minimum_bid,
            "bid_step": self.bid_step,
        }


@dataclass(frozen=True)
class TrainingConfig:
    """Store resolved settings for one model-training run."""

    raw: dict[str, Any]
    model_type: str
    random_seed: int
    drop_zero_variance: bool
    calibrate: bool
    split: dict[str, Any]
    model_parameters: dict[str, Any]
    optimizer: OptimizerConfig
    promotion_mode: str
    promotion_min_roc_auc_regression: float
    promotion_min_profit_ratio: float
    report_root: str
    model_root: str
    mlflow_tracking_db_path: str
    mlflow_artifact_root: str
    mlflow_experiment_name: str
    mlflow_registered_model_name: str

    def as_dict(self) -> dict[str, Any]:
        """Return the configuration values as a dictionary.

        Returns
        -------
        dict
            Dictionary representation of the configuration.
        """
        return copy.deepcopy(self.raw)

    def report_dir(self, lead_type_name_value: str) -> str:
        """Return the report directory for a lead type.

        Inputs
        ------
        lead_type_name_value : str
            Human-readable lead type name.

        Returns
        -------
        str
            Absolute report directory.
        """
        path = f"{self.report_root}/{lead_type_name_value}"
        return str(paths.resolve(path))

    def model_path(self, lead_type_name_value: str) -> str:
        """Return the legacy model path for a lead type.

        Inputs
        ------
        lead_type_name_value : str
            Human-readable lead type name.

        Returns
        -------
        str
            Absolute legacy model path.
        """
        filename = f"anton_model_{lead_type_name_value}.pkl"
        return str(paths.resolve(f"{self.model_root}/{filename}"))


def lead_type_name(lead_type_id: int) -> str:
    """Return the human-readable name for a lead type.

    Inputs
    ------
    lead_type_id : int
        SmartHub lead type identifier.

    Returns
    -------
    str
        Human-readable lead type name.
    """
    return fe.lead_type_name(lead_type_id)


def feature_columns(lead_type_id: int) -> tuple[list[str], list[str]]:
    """Return numeric and categorical model features for a lead type.

    Inputs
    ------
    lead_type_id : int
        SmartHub lead type identifier.

    Returns
    -------
    tuple[list[str], list[str]]
        Numeric features followed by categorical features.
    """
    return fe.model_feature_columns(lead_type_id)


def _training_config_path() -> Path:
    """Return the resolved training configuration path."""
    configured_path = os.getenv("SMARTHUB_TRAINING_CONFIG")
    return paths.resolve(configured_path or "config/training.yaml")


def _mapping(value: Any, section: str) -> dict[str, Any]:
    """Validate and return a configuration mapping.

    Inputs
    ------
    value : Any
        Value to process.
    section : str
        Configuration section name.

    Returns
    -------
    dict[str, Any]
        Validated configuration mapping.

    Raises
    ------
    ValueError
        If the value is not a mapping.
    """
    if not isinstance(value, dict):
        raise ValueError(f"Configuration section {section!r} must be a mapping.")
    return value


def _fraction(value: Any, field_name: str) -> float:
    """Validate and return a fractional configuration value.

    Inputs
    ------
    value : Any
        Value to process.
    field_name : str
        Configuration field name.

    Returns
    -------
    float
        Validated fractional value.

    Raises
    ------
    ValueError
        If the value is outside the open interval from zero to one.
    """
    result = float(value)
    if not 0.0 < result < 1.0:
        raise ValueError(f"{field_name} must be between 0 and 1.")
    return result


def _positive_float(value: Any, field_name: str) -> float:
    """Validate and return a positive floating-point value.

    Inputs
    ------
    value : Any
        Value to process.
    field_name : str
        Configuration field name.

    Returns
    -------
    float
        Validated positive value.

    Raises
    ------
    ValueError
        If the value is not greater than zero.
    """
    result = float(value)
    if result <= 0.0:
        raise ValueError(f"{field_name} must be greater than 0.")
    return result


def load_training_config(
    config_path: str | Path | None = None,
) -> TrainingConfig:
    """Load and validate the complete training configuration.

    Inputs
    ------
    config_path : str | pathlib.Path | None
        Optional path to the training YAML file.

    Returns
    -------
    TrainingConfig
        Validated immutable training configuration.

    Raises
    ------
    FileNotFoundError
        If the YAML configuration file does not exist.
    ValueError
        If a required configuration section or value is invalid.
    """
    if config_path is None:
        resolved_path = _training_config_path()
    else:
        resolved_path = paths.resolve(config_path)

    if not resolved_path.exists():
        raise FileNotFoundError(f"Training configuration not found: {resolved_path}")

    with resolved_path.open("r", encoding="utf-8") as config_file:
        root = _mapping(yaml.safe_load(config_file) or {}, "root")

    training = _mapping(root.get("training"), "training")
    split_root = _mapping(root.get("split"), "split")
    model_configs = _mapping(root.get("models"), "models")
    optimizer_root = _mapping(root.get("optimizer"), "optimizer")
    promotion = _mapping(root.get("promotion"), "promotion")
    output = _mapping(root.get("output"), "output")
    mlflow = _mapping(root.get("mlflow"), "mlflow")

    model_type = str(training.get("model_type", "")).strip().lower()
    if model_type not in _SUPPORTED_MODELS or model_type not in model_configs:
        choices = ", ".join(sorted(_SUPPORTED_MODELS))
        raise ValueError(f"Unsupported training.model_type {model_type!r}: {choices}")

    strategy = str(split_root.get("strategy", "")).strip().lower()
    if strategy not in _SUPPORTED_SPLIT_STRATEGIES:
        choices = ", ".join(sorted(_SUPPORTED_SPLIT_STRATEGIES))
        raise ValueError(f"Unsupported split.strategy {strategy!r}: {choices}")

    selected_split = copy.deepcopy(
        _mapping(split_root.get(strategy), f"split.{strategy}")
    )
    selected_split["strategy"] = strategy
    selected_split["test_size"] = _fraction(
        selected_split.get("test_size"),
        f"split.{strategy}.test_size",
    )

    random_seed = int(training["random_seed"])
    model_parameters = copy.deepcopy(model_configs[model_type])
    model_parameters["random_state"] = random_seed

    optimizer = OptimizerConfig(
        target_cm=_fraction(
            optimizer_root.get("target_cm"),
            "optimizer.target_cm",
        ),
        minimum_bid=_positive_float(
            optimizer_root.get("minimum_bid"),
            "optimizer.minimum_bid",
        ),
        bid_step=_positive_float(
            optimizer_root.get("bid_step"),
            "optimizer.bid_step",
        ),
    )

    promotion_mode = str(promotion.get("mode", "")).strip().lower()
    if promotion_mode not in _SUPPORTED_PROMOTION_MODES:
        choices = ", ".join(sorted(_SUPPORTED_PROMOTION_MODES))
        raise ValueError(f"Unsupported promotion.mode {promotion_mode!r}: {choices}")

    resolved_raw = copy.deepcopy(root)
    resolved_raw["resolved"] = {
        "config_path": str(resolved_path),
        "selected_model": model_type,
        "selected_split": copy.deepcopy(selected_split),
        "selected_model_parameters": copy.deepcopy(model_parameters),
        "optimizer": optimizer.as_dict(),
        "promotion_mode": promotion_mode,
    }

    return TrainingConfig(
        raw=resolved_raw,
        model_type=model_type,
        random_seed=random_seed,
        drop_zero_variance=bool(training["drop_zero_variance"]),
        calibrate=bool(training["calibrate"]),
        split=selected_split,
        model_parameters=model_parameters,
        optimizer=optimizer,
        promotion_mode=promotion_mode,
        promotion_min_roc_auc_regression=float(promotion["min_roc_auc_regression"]),
        promotion_min_profit_ratio=float(promotion["min_profit_ratio"]),
        report_root=str(output["report_root"]),
        model_root=str(output["model_root"]),
        mlflow_tracking_db_path=str(mlflow["tracking_db_path"]),
        mlflow_artifact_root=str(mlflow["artifact_root"]),
        mlflow_experiment_name=str(mlflow["experiment_name"]),
        mlflow_registered_model_name=str(mlflow["registered_model_name"]),
    )


def active_model_version() -> str | None:
    """Return the explicitly pinned serving model version.

    Returns
    -------
    str | None
        Pinned model version, or ``None``.
    """
    raw = task_config.get("prediction", "active_model_version", "none")
    raw = (raw or "none").strip()
    return None if raw.lower() == "none" else raw


def exploration_variance_pct() -> float:
    """Return the scheduled-exploration bid perturbation fraction.

    Also sets the exploration schedule's density: ``N = round(1 / this)``
    hour-of-week buckets are scheduled explore slots (see
    ``server.predict.exploration_slot``).

    Returns
    -------
    float
        Configured ``[prediction] exploration_variance_pct`` (default 0.10).
    """
    return task_config.get_float("prediction", "exploration_variance_pct", 0.10)


def recency_window_days() -> int:
    """Return how many days old a serving model's training data can be
    before ``/recommend_bid``/``/explain_bid`` flag it as stale.

    Informational only -- doesn't change the bid itself, just the
    ``model_data_age_days`` retraining-cadence signal.

    Returns
    -------
    int
        Configured ``[prediction] recency_window_days`` (default 30).
    """
    return task_config.get_int("prediction", "recency_window_days", 30)


def cold_start_fallback_bid_pct() -> float:
    """Return the true-cold-start fallback bid fraction.

    Fraction of the way from ``min_bid`` to the CM-respecting ceiling to bid
    when a lead type has no model ever trained/promoted yet.

    Returns
    -------
    float
        Configured ``[prediction] cold_start_fallback_bid_pct`` (default 0.50).
    """
    return task_config.get_float("prediction", "cold_start_fallback_bid_pct", 0.50)


# Compatibility aliases allow existing prediction and optimizer modules to use
# the new YAML-backed values without a second configuration source.
_DEFAULT_CONFIG = load_training_config()
RANDOM_SEED = _DEFAULT_CONFIG.random_seed
TARGET_CM = _DEFAULT_CONFIG.optimizer.target_cm
MIN_BID = _DEFAULT_CONFIG.optimizer.minimum_bid
BID_STEP = _DEFAULT_CONFIG.optimizer.bid_step


def target_cm_value() -> float:
    """Return the configured target contribution margin.

    Returns
    -------
    float
        Configured target contribution margin.
    """
    return _DEFAULT_CONFIG.optimizer.target_cm


def min_bid_value() -> float:
    """Return the configured minimum candidate bid.

    Returns
    -------
    float
        Configured minimum candidate bid.
    """
    return _DEFAULT_CONFIG.optimizer.minimum_bid


@dataclass(frozen=True)
class HyperparameterSearchConfig:
    """Store resolved settings for manual hyperparameter search."""

    raw: dict[str, Any]
    scoring: str
    n_trials: int
    cv_folds: int
    timeout_seconds: int | None
    random_seed: int
    n_jobs: int
    output_root: str
    model_configs: dict[str, dict[str, Any]]

    def as_dict(self) -> dict[str, Any]:
        """Return the configuration values as a dictionary.

        Returns
        -------
        dict
            Dictionary representation of the configuration.
        """
        return copy.deepcopy(self.raw)

    def model_config(self, model_type: str) -> dict[str, Any]:
        """Return fixed parameters and search space for a model.

        Inputs
        ------
        model_type : str
            Configured model family name.

        Returns
        -------
        dict[str, Any]
            Model-specific hyperparameter-search configuration.

        Raises
        ------
        ValueError
            If the model type is unsupported or not configured.
        """
        normalized = model_type.strip().lower()
        if normalized not in _SUPPORTED_MODELS:
            choices = ", ".join(sorted(_SUPPORTED_MODELS))
            raise ValueError(f"Unsupported model type {model_type!r}: {choices}")
        try:
            return copy.deepcopy(self.model_configs[normalized])
        except KeyError as exc:
            raise ValueError(
                f"No hyperparameter-search configuration for {normalized!r}."
            ) from exc

    def output_dir(self, lead_type_name_value: str, model_type: str) -> str:
        """Return the output directory for one tuning result.

        Inputs
        ------
        lead_type_name_value : str
            Human-readable lead type name.
        model_type : str
            Configured model family name.

        Returns
        -------
        str
            Absolute output directory.
        """
        path = (
            f"{self.output_root}/{lead_type_name_value.strip().lower()}/"
            f"{model_type.strip().lower()}"
        )
        return str(paths.resolve(path))


def _hyperparameter_search_config_path() -> Path:
    """Return the resolved hyperparameter-search configuration path."""
    configured_path = os.getenv("SMARTHUB_HYPERPARAMETER_SEARCH_CONFIG")
    return paths.resolve(configured_path or "config/hyperparameter_search.yaml")


def _positive_int(value: Any, field_name: str) -> int:
    """Validate and return a positive integer configuration value.

    Inputs
    ------
    value : Any
        Value to process.
    field_name : str
        Configuration field name.

    Returns
    -------
    int
        Validated positive integer.

    Raises
    ------
    ValueError
        If the value is not greater than zero.
    """
    result = int(value)
    if result <= 0:
        raise ValueError(f"{field_name} must be greater than 0.")
    return result


def _validate_search_parameter(
    parameter_name: str,
    specification: Any,
    section: str,
) -> dict[str, Any]:
    """Validate one hyperparameter search-space specification.

    Inputs
    ------
    parameter_name : str
        Hyperparameter name.
    specification : Any
        Hyperparameter search-space definition.
    section : str
        Configuration section containing the definition.

    Returns
    -------
    dict[str, Any]
        Validated search-space specification.

    Raises
    ------
    ValueError
        If the specification is invalid.
    """
    spec = copy.deepcopy(_mapping(specification, f"{section}.{parameter_name}"))
    parameter_type = str(spec.get("type", "")).strip().lower()
    if parameter_type not in {"int", "float", "categorical"}:
        raise ValueError(
            f"{section}.{parameter_name}.type must be int, float, or categorical."
        )

    if parameter_type == "categorical":
        choices = spec.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ValueError(
                f"{section}.{parameter_name}.choices must be a non-empty list."
            )
        return spec

    if "low" not in spec or "high" not in spec:
        raise ValueError(f"{section}.{parameter_name} must define low and high values.")
    if float(spec["low"]) >= float(spec["high"]):
        raise ValueError(f"{section}.{parameter_name}.low must be less than high.")
    if bool(spec.get("log", False)) and "step" in spec:
        raise ValueError(
            f"{section}.{parameter_name} cannot define both log=true and step."
        )
    if "step" in spec and float(spec["step"]) <= 0:
        raise ValueError(f"{section}.{parameter_name}.step must be greater than 0.")
    return spec


def load_hyperparameter_search_config(
    config_path: str | Path | None = None,
) -> HyperparameterSearchConfig:
    """Load and validate manual hyperparameter-search configuration.

    Inputs
    ------
    config_path : str | pathlib.Path | None
        Optional path to the hyperparameter-search YAML file.

    Returns
    -------
    HyperparameterSearchConfig
        Validated immutable hyperparameter-search configuration.

    Raises
    ------
    FileNotFoundError
        If the YAML configuration file does not exist.
    ValueError
        If a required configuration section or value is invalid.
    """
    if config_path is None:
        resolved_path = _hyperparameter_search_config_path()
    else:
        resolved_path = paths.resolve(config_path)

    if not resolved_path.exists():
        raise FileNotFoundError(
            "Hyperparameter-search configuration not found: " f"{resolved_path}"
        )

    with resolved_path.open("r", encoding="utf-8") as config_file:
        root = _mapping(yaml.safe_load(config_file) or {}, "root")

    search = _mapping(root.get("search"), "search")
    output = _mapping(root.get("output"), "output")
    models_root = _mapping(root.get("models"), "models")

    scoring = str(search.get("scoring", "")).strip()
    if not scoring:
        raise ValueError("search.scoring must not be empty.")

    timeout_value = search.get("timeout_seconds")
    timeout_seconds = (
        None
        if timeout_value in (None, "", "none", "None")
        else _positive_int(timeout_value, "search.timeout_seconds")
    )

    model_configs = {}
    for model_type, raw_model_config in models_root.items():
        normalized = str(model_type).strip().lower()
        if normalized not in _SUPPORTED_MODELS:
            choices = ", ".join(sorted(_SUPPORTED_MODELS))
            raise ValueError(f"Unsupported models entry {model_type!r}: {choices}")

        model_config = _mapping(raw_model_config, f"models.{normalized}")
        fixed_parameters = copy.deepcopy(
            _mapping(
                model_config.get("fixed_parameters", {}),
                f"models.{normalized}.fixed_parameters",
            )
        )
        raw_search_space = _mapping(
            model_config.get("search_space"),
            f"models.{normalized}.search_space",
        )
        if not raw_search_space:
            raise ValueError(f"models.{normalized}.search_space must not be empty.")
        search_space = {
            parameter_name: _validate_search_parameter(
                parameter_name,
                specification,
                f"models.{normalized}.search_space",
            )
            for parameter_name, specification in raw_search_space.items()
        }
        model_configs[normalized] = {
            "fixed_parameters": fixed_parameters,
            "search_space": search_space,
        }

    output_root = str(output.get("root", "")).strip()
    if not output_root:
        raise ValueError("output.root must not be empty.")

    resolved_raw = copy.deepcopy(root)
    resolved_raw["resolved"] = {
        "config_path": str(resolved_path),
        "models": sorted(model_configs),
    }

    return HyperparameterSearchConfig(
        raw=resolved_raw,
        scoring=scoring,
        n_trials=_positive_int(search.get("n_trials"), "search.n_trials"),
        cv_folds=_positive_int(search.get("cv_folds"), "search.cv_folds"),
        timeout_seconds=timeout_seconds,
        random_seed=int(search["random_seed"]),
        n_jobs=int(search["n_jobs"]),
        output_root=output_root,
        model_configs=model_configs,
    )
