"""Configuration for Anton model training and bid optimization.

This module loads and validates YAML settings for training and serving.
"""

from __future__ import annotations

import copy
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from smarthub.core import paths, task_config
from smarthub.core.lead_types import lead_type_name as canonical_lead_type_name
from smarthub.feature_engineering import features as fe

logger = logging.getLogger("smarthub.train_and_predict.config")

TARGET_COL = fe.TARGET_COLUMN
REVENUE_COL = fe.REVENUE_COLUMN
_SUPPORTED_MODELS = {"logistic_regression", "xgboost", "lightgbm"}
_SUPPORTED_SPLIT_STRATEGIES = {"time", "random"}
_SUPPORTED_PROMOTION_MODES = {"manual", "automatic", "disabled"}
_DEFAULT_TRAINING_CONFIG_REL_PATH = "config/training.yaml"
_DEFAULT_HYPERPARAMETER_SEARCH_CONFIG_REL_PATH = "config/hyperparameter_search.yaml"
_DEFAULT_PROMOTION_MAX_LOG_LOSS = 0.55
_DEFAULT_PROMOTION_MIN_EXPECTED_PROFIT = 0.0
_DEFAULT_PROMOTION_MAX_ABSOLUTE_PROFIT_LOSS_TOLERANCE = 5000.0


@dataclass(frozen=True)
class OptimizerConfig:
    """Store resolved bid-optimizer configuration values."""

    target_cm: float
    minimum_bid: float
    bid_step: float
    chunk_size: int

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
            "chunk_size": self.chunk_size,
        }


@dataclass(frozen=True)
class TrainingConfig:
    """Store resolved settings for one model-training run."""

    raw: dict[str, Any]
    model_type: str
    random_seed: int
    drop_zero_variance: bool
    calibration_enabled: bool
    calibration_method: str
    calibration_cv: int
    split: dict[str, Any]
    model_parameters: dict[str, Any]
    optimizer: OptimizerConfig
    promotion_mode: str
    promotion_max_log_loss_regression: float
    promotion_min_profit_ratio: float
    promotion_max_log_loss: float
    promotion_min_expected_profit: float
    promotion_max_absolute_profit_loss_tolerance: float
    report_root: str
    model_root: str
    mlflow_tracking_db_path: str
    mlflow_artifact_root: str
    mlflow_experiment_name: str
    mlflow_registered_model_name: str
    production_storage: dict[str, Any] | None = None
    mlflow_production_tracking_uri: str = ""
    mlflow_production_experiment_name: str = "SmartHub Production"
    mlflow_production_registered_model_name: str = ""

    def local_model_store(self):
        """Filesystem store for local (all-runs) model artifacts."""
        from smarthub.train_and_predict.model_storage import FilesystemModelStore

        return FilesystemModelStore(str(paths.resolve(self.model_root)))

    def production_model_store(self):
        """Production model store (promoted models), or None when disabled."""
        spec = self.production_storage
        if not spec:
            return None
        from smarthub.train_and_predict import model_storage

        backend = spec["backend"]
        if backend == "s3":
            return model_storage.S3ModelStore(
                spec["bucket"],
                prefix=spec.get("prefix", ""),
                endpoint_url=spec.get("endpoint_url"),
                region=spec.get("region"),
            )
        if backend == "filesystem":
            root = spec.get("root") or "data/production_models"
            return model_storage.FilesystemModelStore(str(paths.resolve(root)))
        raise ValueError(f"Unsupported production storage backend: {backend!r}")

    @property
    def production_mlflow_enabled(self) -> bool:
        """Whether a production MLflow tracking server is configured."""
        return bool(self.mlflow_production_tracking_uri)

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
    return canonical_lead_type_name(lead_type_id)


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


def _env_or_default_path(variable_name: str, default_rel_path: str) -> Path:
    """Return a configuration path from the environment or the packaged default.

    Inputs
    ------
    variable_name : str
        Optional environment-variable override name.
    default_rel_path : str
        Repository-relative fallback path.

    Returns
    -------
    pathlib.Path
        Resolved path to the requested YAML file.
    """
    configured_path = os.getenv(variable_name)
    if configured_path is not None and configured_path.strip():
        return paths.resolve(configured_path.strip())
    return paths.resolve(default_rel_path)


def _training_config_path() -> Path:
    """Return the training configuration path, preferring any env override."""
    return _env_or_default_path(
        "SMARTHUB_TRAINING_CONFIG",
        _DEFAULT_TRAINING_CONFIG_REL_PATH,
    )


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


def _non_negative_float(value: Any, field_name: str) -> float:
    """Validate and return a non-negative floating-point value.

    Inputs
    ------
    value : Any
        Value to process.
    field_name : str
        Configuration field name.

    Returns
    -------
    float
        Validated non-negative value.

    Raises
    ------
    ValueError
        If the value is less than zero.
    """
    result = float(value)
    if result < 0.0:
        raise ValueError(f"{field_name} must be greater than or equal to 0.")
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


def _resolve_production_storage(cfg: Any) -> dict[str, Any] | None:
    """Resolve the production model-storage spec, or None when disabled.

    An empty ``backend`` (or an absent ``storage.production`` section) means
    local-only, preserving today's behaviour. ``SMARTHUB_S3_ENDPOINT_URL``
    overrides the endpoint so the same config targets AWS or a MinIO/
    S3-compatible service per environment.
    """
    cfg = cfg or {}
    backend = (
        (
            os.getenv("SMARTHUB_PRODUCTION_STORAGE_BACKEND")
            or str(cfg.get("backend", "") or "")
        )
        .strip()
        .lower()
    )
    if not backend:
        return None
    # Endpoint selects the S3 target: a value (default MinIO in Docker) points
    # at an S3-compatible service; EMPTY means real AWS S3 (boto3 talks to AWS
    # directly). `SMARTHUB_S3_ENDPOINT_URL` overrides -- set it empty for AWS.
    endpoint = (
        os.getenv("SMARTHUB_S3_ENDPOINT_URL") or str(cfg.get("endpoint_url", "") or "")
    ) or None
    bucket = os.getenv("SMARTHUB_S3_BUCKET") or str(cfg.get("bucket", "") or "")
    prefix = os.getenv("SMARTHUB_S3_PREFIX") or str(cfg.get("prefix", "") or "")
    # Region: explicit override, else boto3's standard AWS_DEFAULT_REGION, else
    # config. None lets boto3 resolve it (fine for MinIO and AWS with a region
    # in the env/instance profile).
    region = (
        os.getenv("SMARTHUB_S3_REGION")
        or os.getenv("AWS_DEFAULT_REGION")
        or str(cfg.get("region", "") or "")
    ) or None
    return {
        "backend": backend,
        "bucket": bucket,
        "prefix": prefix,
        "endpoint_url": endpoint,
        "region": region,
        "root": str(cfg.get("root", "") or ""),
    }


def _resolve_mlflow_production(cfg: Any) -> dict[str, str]:
    """Resolve production MLflow settings; empty ``tracking_uri`` disables it.

    ``SMARTHUB_MLFLOW_PROD_TRACKING_URI`` overrides the tracking URI.
    """
    cfg = cfg or {}
    tracking_uri = os.getenv("SMARTHUB_MLFLOW_PROD_TRACKING_URI") or str(
        cfg.get("tracking_uri", "") or ""
    )
    return {
        "tracking_uri": tracking_uri,
        "experiment_name": str(
            cfg.get("experiment_name", "SmartHub Production") or "SmartHub Production"
        ),
        "registered_model_name": str(cfg.get("registered_model_name", "") or ""),
    }


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
    calibration_root = _mapping(root.get("calibration"), "calibration")
    split_root = _mapping(root.get("split"), "split")
    models_root = _mapping(root.get("models"), "models")
    optimizer_root = _mapping(root.get("optimizer"), "optimizer")
    promotion = _mapping(root.get("promotion"), "promotion")
    promotion_criteria = _mapping(
        promotion.get("criteria"),
        "promotion.criteria",
    )
    output = _mapping(root.get("output"), "output")
    mlflow = _mapping(root.get("mlflow"), "mlflow")
    storage_root = root.get("storage") or {}
    production_storage = _resolve_production_storage(storage_root.get("production"))
    mlflow_production = _resolve_mlflow_production(mlflow.get("production"))

    model_type = str(training.get("model_type", "")).strip().lower()
    if model_type not in _SUPPORTED_MODELS:
        choices = ", ".join(sorted(_SUPPORTED_MODELS))
        raise ValueError(f"Unsupported training.model_type {model_type!r}: {choices}")

    if model_type not in models_root:
        raise ValueError(f"No training configuration found for model {model_type!r}.")

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

    if strategy == "random":
        if "stratify" not in selected_split:
            raise ValueError("Missing required config: split.random.stratify")
        if not isinstance(selected_split["stratify"], bool):
            raise TypeError("split.random.stratify must be a YAML boolean.")

    random_seed = int(training["random_seed"])
    if "enabled" not in calibration_root:
        raise ValueError("Missing required config: calibration.enabled")
    calibration_enabled = calibration_root["enabled"]
    if not isinstance(calibration_enabled, bool):
        raise TypeError("calibration.enabled must be a YAML boolean.")

    calibration_method = str(calibration_root["method"]).strip().lower()
    if calibration_method not in {"sigmoid", "isotonic"}:
        raise ValueError("calibration.method must be one of 'sigmoid' or 'isotonic'.")
    calibration_cv = _positive_int(
        calibration_root["cv"],
        "calibration.cv",
    )
    model_parameters = copy.deepcopy(
        _mapping(
            models_root[model_type],
            f"models.{model_type}",
        )
    )
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
        chunk_size=_positive_int(
            optimizer_root.get("chunk_size"),
            "optimizer.chunk_size",
        ),
    )

    promotion_mode = str(promotion.get("mode", "")).strip().lower()
    if promotion_mode not in _SUPPORTED_PROMOTION_MODES:
        choices = ", ".join(sorted(_SUPPORTED_PROMOTION_MODES))
        raise ValueError(f"Unsupported promotion.mode {promotion_mode!r}: {choices}")

    resolved_raw = copy.deepcopy(root)
    resolved_raw["models"] = {model_type: copy.deepcopy(models_root[model_type])}
    promotion_max_log_loss_regression = _non_negative_float(
        promotion_criteria.get("max_log_loss_regression"),
        "promotion.criteria.max_log_loss_regression",
    )
    promotion_min_profit_ratio = _positive_float(
        promotion_criteria.get("min_profit_ratio"),
        "promotion.criteria.min_profit_ratio",
    )
    promotion_max_log_loss = _positive_float(
        promotion_criteria.get(
            "max_log_loss",
            _DEFAULT_PROMOTION_MAX_LOG_LOSS,
        ),
        "promotion.criteria.max_log_loss",
    )
    promotion_min_expected_profit = _non_negative_float(
        promotion_criteria.get(
            "min_expected_profit",
            _DEFAULT_PROMOTION_MIN_EXPECTED_PROFIT,
        ),
        "promotion.criteria.min_expected_profit",
    )
    promotion_max_absolute_profit_loss_tolerance = _non_negative_float(
        promotion_criteria.get(
            "max_absolute_profit_loss_tolerance",
            _DEFAULT_PROMOTION_MAX_ABSOLUTE_PROFIT_LOSS_TOLERANCE,
        ),
        "promotion.criteria.max_absolute_profit_loss_tolerance",
    )

    resolved_raw["resolved"] = {
        "config_path": str(resolved_path),
        "selected_model": model_type,
        "selected_split": copy.deepcopy(selected_split),
        "calibration": {
            "enabled": calibration_enabled,
            "method": calibration_method,
            "cv": calibration_cv,
        },
        "selected_model_parameters": copy.deepcopy(model_parameters),
        "optimizer": optimizer.as_dict(),
        "promotion": {
            "mode": promotion_mode,
            "criteria": {
                "max_log_loss_regression": (promotion_max_log_loss_regression),
                "min_profit_ratio": promotion_min_profit_ratio,
                "max_log_loss": promotion_max_log_loss,
                "min_expected_profit": promotion_min_expected_profit,
                "max_absolute_profit_loss_tolerance": (
                    promotion_max_absolute_profit_loss_tolerance
                ),
            },
        },
    }

    return TrainingConfig(
        raw=resolved_raw,
        model_type=model_type,
        random_seed=random_seed,
        drop_zero_variance=bool(training["drop_zero_variance"]),
        calibration_enabled=calibration_enabled,
        calibration_method=calibration_method,
        calibration_cv=calibration_cv,
        split=selected_split,
        model_parameters=model_parameters,
        optimizer=optimizer,
        promotion_mode=promotion_mode,
        promotion_max_log_loss_regression=(promotion_max_log_loss_regression),
        promotion_min_profit_ratio=promotion_min_profit_ratio,
        promotion_max_log_loss=promotion_max_log_loss,
        promotion_min_expected_profit=promotion_min_expected_profit,
        promotion_max_absolute_profit_loss_tolerance=(
            promotion_max_absolute_profit_loss_tolerance
        ),
        report_root=str(output["report_root"]),
        model_root=str(output["model_root"]),
        mlflow_tracking_db_path=str(mlflow["tracking_db_path"]),
        mlflow_artifact_root=str(mlflow["artifact_root"]),
        mlflow_experiment_name=str(mlflow["experiment_name"]),
        mlflow_registered_model_name=str(mlflow["registered_model_name"]),
        production_storage=production_storage,
        mlflow_production_tracking_uri=mlflow_production["tracking_uri"],
        mlflow_production_experiment_name=mlflow_production["experiment_name"],
        mlflow_production_registered_model_name=(
            mlflow_production["registered_model_name"]
        ),
    )


def active_model_version() -> str | None:
    """Return the explicitly pinned serving model version.

    Returns
    -------
    str | None
        Pinned model version, or ``None``.
    """
    raw = task_config.get("prediction", "active_model_version", None)
    if raw is None or not str(raw).strip():
        raise ValueError(
            "Missing required configuration: prediction.active_model_version"
        )
    raw = str(raw).strip()
    return None if raw.lower() == "none" else raw


def exploration_variance_pct() -> float:
    """Return the scheduled-exploration bid perturbation fraction.

    Also sets the exploration schedule's density: ``N = round(1 / this)``
    hour-of-week buckets are scheduled explore slots (see
    ``server.predict.exploration_slot``).

    Returns
    -------
    float
        Required ``[prediction] exploration_variance_pct`` value.
    """
    return task_config.get_float("prediction", "exploration_variance_pct", None)


def recency_window_days() -> int:
    """Return how many days old a serving model's training data can be
    before ``/recommend_bid``/``/explain_bid`` flag it as stale.

    Informational only -- doesn't change the bid itself, just the
    ``model_data_age_days`` retraining-cadence signal.

    Returns
    -------
    int
        Required ``[prediction] recency_window_days`` value.
    """
    return task_config.get_int("prediction", "recency_window_days", None)


def cold_start_fallback_bid_pct() -> float:
    """Return the true-cold-start fallback bid fraction.

    Fraction of the way from ``min_bid`` to the CM-respecting ceiling to bid
    when a lead type has no model ever trained/promoted yet.

    Returns
    -------
    float
        Required ``[prediction] cold_start_fallback_bid_pct`` value.
    """
    return task_config.get_float("prediction", "cold_start_fallback_bid_pct", None)


# Valid SHAP enrichment modes for /recommend_bid's post-response explanation.
SHAP_MODE_INPROCESS = "inprocess"
SHAP_MODE_OFFLOAD = "offload"
SHAP_MODE_OFF = "off"
_VALID_SHAP_MODES = {SHAP_MODE_INPROCESS, SHAP_MODE_OFFLOAD, SHAP_MODE_OFF}


def shap_enrichment_mode() -> str:
    """How ``/recommend_bid`` attaches SHAP factors to a prediction's log row.

    Three modes, so the current behaviour and the offloaded behaviour can run
    side by side and be compared before either is removed:

    - ``"inprocess"`` (default -- unchanged legacy behaviour): the serving
      process computes SHAP in a Starlette ``BackgroundTask`` right after the
      response is sent. Correct, but the ~1.5 s of CPU-bound work runs on the
      same worker and contends for the GIL with later requests.
    - ``"offload"``: the serving process does NOT compute SHAP. The prediction
      is logged with ``shap_explanation = NULL``; a separate ``shap-worker``
      process (see ``server.shap_worker``) drains those rows and backfills the
      explanation. Keeps serving CPU free for the 1 s bid TAT.
    - ``"off"``: no SHAP enrichment at all (useful as a control when
      benchmarking the pure bid path).

    Resolution order: the ``SMARTHUB_SHAP_MODE`` env var wins (so a load test
    can flip modes without editing YAML), then ``[prediction]
    shap_enrichment_mode`` in ``config/smarthub.yaml``, then the default.

    Returns
    -------
    str
        One of ``"inprocess"``, ``"offload"``, ``"off"``.
    """
    raw = os.getenv("SMARTHUB_SHAP_MODE") or task_config.get(
        "prediction", "shap_enrichment_mode", SHAP_MODE_INPROCESS
    )
    mode = str(raw).strip().lower()
    if mode not in _VALID_SHAP_MODES:
        logger.warning(
            "Unknown shap_enrichment_mode %r; falling back to %r. Valid: %s",
            raw,
            SHAP_MODE_INPROCESS,
            sorted(_VALID_SHAP_MODES),
        )
        return SHAP_MODE_INPROCESS
    return mode


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
    model_type: str
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
    """Return the hyperparameter-search configuration path,
    preferring any env override.
    """
    return _env_or_default_path(
        "SMARTHUB_HYPERPARAMETER_SEARCH_CONFIG",
        _DEFAULT_HYPERPARAMETER_SEARCH_CONFIG_REL_PATH,
    )


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

    if "log" not in spec:
        raise ValueError(f"{section}.{parameter_name}.log must be configured.")
    if not isinstance(spec["log"], bool):
        raise ValueError(f"{section}.{parameter_name}.log must be true or false.")

    if "step" not in spec:
        raise ValueError(f"{section}.{parameter_name}.step must be configured.")

    step = spec["step"]
    if parameter_type == "int":
        if step is None:
            raise ValueError(
                f"{section}.{parameter_name}.step must be a positive integer."
            )
        if isinstance(step, bool) or int(step) != step or int(step) <= 0:
            raise ValueError(
                f"{section}.{parameter_name}.step must be a positive integer."
            )
        spec["step"] = int(step)
    else:
        if step is not None and float(step) <= 0:
            raise ValueError(
                f"{section}.{parameter_name}.step must be null or greater than 0."
            )
        if step is not None:
            spec["step"] = float(step)

    if spec["log"] and step is not None:
        raise ValueError(
            f"{section}.{parameter_name} must set step: null when log: true."
        )
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
            f"Hyperparameter-search configuration not found: {resolved_path}"
        )

    with resolved_path.open("r", encoding="utf-8") as config_file:
        root = _mapping(yaml.safe_load(config_file) or {}, "root")

    search = _mapping(root.get("search"), "search")
    output = _mapping(root.get("output"), "output")
    models_root = _mapping(root.get("models"), "models")

    selected_model_type = str(search.get("model_type", "")).strip().lower()
    if selected_model_type not in _SUPPORTED_MODELS:
        choices = ", ".join(sorted(_SUPPORTED_MODELS))
        raise ValueError(
            f"Unsupported search.model_type " f"{selected_model_type!r}: {choices}"
        )

    if selected_model_type not in models_root:
        raise ValueError(
            "No hyperparameter-search configuration found for "
            f"{selected_model_type!r}."
        )

    scoring = str(search.get("scoring", "")).strip()
    if not scoring:
        raise ValueError("search.scoring must not be empty.")

    timeout_value = search.get("timeout_seconds")
    timeout_seconds = (
        None
        if timeout_value in (None, "", "none", "None")
        else _positive_int(timeout_value, "search.timeout_seconds")
    )

    raw_model_config = _mapping(
        models_root[selected_model_type],
        f"models.{selected_model_type}",
    )
    fixed_parameters = copy.deepcopy(
        _mapping(
            raw_model_config.get("fixed_parameters"),
            f"models.{selected_model_type}.fixed_parameters",
        )
    )
    raw_search_space = _mapping(
        raw_model_config.get("search_space"),
        f"models.{selected_model_type}.search_space",
    )
    if not raw_search_space:
        raise ValueError(
            f"models.{selected_model_type}.search_space must not be empty."
        )

    search_space = {
        parameter_name: _validate_search_parameter(
            parameter_name,
            specification,
            f"models.{selected_model_type}.search_space",
        )
        for parameter_name, specification in raw_search_space.items()
    }

    model_configs = {
        selected_model_type: {
            "fixed_parameters": fixed_parameters,
            "search_space": search_space,
        }
    }

    output_root = str(output.get("root", "")).strip()
    if not output_root:
        raise ValueError("output.root must not be empty.")

    resolved_raw = copy.deepcopy(root)
    resolved_raw["models"] = {
        selected_model_type: copy.deepcopy(root["models"][selected_model_type])
    }
    resolved_raw["resolved"] = {
        "config_path": str(resolved_path),
        "selected_model": selected_model_type,
        "models": [selected_model_type],
    }

    return HyperparameterSearchConfig(
        raw=resolved_raw,
        model_type=selected_model_type,
        scoring=scoring,
        n_trials=_positive_int(search.get("n_trials"), "search.n_trials"),
        cv_folds=_positive_int(search.get("cv_folds"), "search.cv_folds"),
        timeout_seconds=timeout_seconds,
        random_seed=int(search["random_seed"]),
        n_jobs=int(search["n_jobs"]),
        output_root=output_root,
        model_configs=model_configs,
    )
