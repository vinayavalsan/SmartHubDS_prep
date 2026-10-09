"""Validated parameter artifacts shared by HPO, training, and serving metadata."""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import os
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

from smarthub.core import paths

logger = logging.getLogger(__name__)
SCHEMA_VERSION = 1

_ARTIFACT_METADATA_DEFAULTS = {
    "created_at": None,
    "code_version": None,
    "hpo_run_id": None,
    "hpo_mlflow_run_id": None,
    "hpo_mlflow_tracking_uri": None,
    "data": None,
    "approved_training_run_id": None,
    "production_model_version": None,
    "initialized_from_bootstrap": False,
}


def make_artifact(settings, lead_type_id, **metadata) -> dict:
    """Use one persisted format for HPO, current, and manual parameters."""
    unknown = set(metadata) - set(_ARTIFACT_METADATA_DEFAULTS)
    if unknown:
        raise ValueError(f"Unsupported parameter artifact metadata: {sorted(unknown)}")
    settings = normalize_settings(settings)
    return {
        "schema_version": SCHEMA_VERSION,
        "lead_type_id": lead_type_id,
        "parameter_version": parameter_version(settings),
        "model_settings": settings,
        **copy.deepcopy(_ARTIFACT_METADATA_DEFAULTS),
        **copy.deepcopy(metadata),
    }


def normalize_settings(settings: dict) -> dict:
    """Validate the common model-settings format, irrespective of its origin."""
    if not isinstance(settings, dict):
        raise ValueError("model_settings must be a mapping.")
    model_type = settings.get("model_type")
    if model_type not in {"lightgbm", "xgboost", "logistic_regression"}:
        raise ValueError(f"Unsupported model_settings.model_type: {model_type!r}")
    parameters = settings.get("model_parameters")
    if not isinstance(parameters, dict) or not all(
        isinstance(key, str) for key in parameters
    ):
        raise ValueError("model_settings.model_parameters must be a mapping.")
    calibration = settings.get("calibration")
    if not isinstance(calibration, dict) or not isinstance(
        calibration.get("enabled"), bool
    ):
        raise ValueError("model_settings.calibration.enabled must be a boolean.")
    calibration = copy.deepcopy(calibration)
    if calibration["enabled"]:
        if calibration.get("method") not in {"sigmoid", "isotonic"}:
            raise ValueError("Calibration method must be sigmoid or isotonic.")
        cv = calibration.get("cv")
        if isinstance(cv, bool) or not isinstance(cv, int) or cv < 2:
            raise ValueError("Calibration cv must be an integer >= 2.")
    else:
        calibration = {"enabled": False}
    normalized = {
        "model_type": model_type,
        "model_parameters": copy.deepcopy(parameters),
        "calibration": calibration,
    }
    # The training policy owns the seed; it is not a tuned model parameter.
    normalized["model_parameters"].pop("random_state", None)
    json.dumps(normalized, allow_nan=False)
    return normalized


def parameter_version(settings: dict) -> str:
    """Content-based identity for the model settings, including calibration."""
    payload = json.dumps(
        normalize_settings(settings),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return "params_" + hashlib.sha256(payload).hexdigest()[:24]


def dataset_fingerprint(frame: pd.DataFrame) -> str:
    """Detect changes to a pinned prepared dataset before candidate fitting."""
    digest = hashlib.sha256(json.dumps(list(frame.columns)).encode())
    digest.update(pd.util.hash_pandas_object(frame, index=False).values.tobytes())
    return digest.hexdigest()


def code_version() -> str | None:
    """Return the checkout commit when available; never invent a code version."""
    try:
        return subprocess.check_output(
            ["git", "-C", str(paths.project_root()), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        logger.info("model_parameters.code_version: checkout commit unavailable.")
        return None


def load_artifact(file: str | Path, lead_type_id: int) -> dict:
    """Load an explicit manual/HPO artifact and reject cross-lead-type reuse."""
    path = paths.resolve(file)
    with path.open(encoding="utf-8") as stream:
        artifact = yaml.safe_load(stream)
    if (
        not isinstance(artifact, dict)
        or artifact.get("schema_version") != SCHEMA_VERSION
    ):
        raise ValueError("Unsupported parameter artifact schema_version.")
    if artifact.get("lead_type_id") != lead_type_id:
        raise ValueError("Parameter artifact lead_type_id does not match training.")
    settings = normalize_settings(artifact.get("model_settings"))
    version = parameter_version(settings)
    if artifact.get("parameter_version", version) != version:
        raise ValueError("Parameter artifact checksum does not match its settings.")
    # Normalize earlier generated artifacts too; legacy duplicate settings are
    # ignored in favor of the validated model_settings section.
    artifact = make_artifact(
        settings,
        lead_type_id,
        **{
            key: artifact[key] for key in _ARTIFACT_METADATA_DEFAULTS if key in artifact
        },
    )
    artifact["parameter_file"] = str(path)
    return artifact


def settings_from_config(cfg) -> dict:
    """Snapshot parameters intended for reuse; fitted iteration count is separate."""
    calibration = {"enabled": cfg.calibration_enabled}
    if cfg.calibration_enabled:
        calibration.update(method=cfg.calibration_method, cv=cfg.calibration_cv)
    return normalize_settings(
        {
            "model_type": cfg.model_type,
            "model_parameters": cfg.model_parameters,
            "calibration": calibration,
        }
    )


def settings_from_manifest(manifest: dict) -> dict:
    """Read new snapshots or migrate a legacy model's saved training settings."""
    if manifest.get("model_settings") is not None:
        settings = normalize_settings(manifest["model_settings"])
        recorded = manifest.get("parameter_version")
        if recorded and recorded != parameter_version(settings):
            raise ValueError("Serving model parameter checksum is invalid.")
        return settings
    raw = manifest.get("training_config")
    if not isinstance(raw, dict):
        raise ValueError("Serving model has no reusable training configuration.")
    model_type = raw.get("model_type")
    model_parameters = copy.deepcopy((raw.get("models") or {}).get(model_type))
    if not isinstance(model_parameters, dict):
        raise ValueError("Serving model has no saved model parameters.")
    if (raw.get("early_stopping") or {}).get("enabled"):
        model_parameters.pop("n_estimators", None)
    return normalize_settings(
        {
            "model_type": model_type,
            "model_parameters": model_parameters,
            "calibration": raw.get("calibration"),
        }
    )


def _with_parameter_details(result, cfg, explicit_file=None):
    """Describe the file consumed without changing selection or provenance."""
    policy = cfg.raw.get("parameters") or {}
    training_file = (cfg.raw.get("resolved") or {}).get("config_path")
    if explicit_file is not None:
        file = explicit_file
        if result.get("approved_training_run_id"):
            label = "Current (explicit file)"
        elif result.get("hpo_run_id"):
            label = "HPO candidate"
        else:
            label = "Manual"
    elif result.get("parameter_source") == "current_file":
        file = policy.get("current_file")
        label = (
            "Bootstrap (current copy)"
            if result.get("initialized_from_bootstrap")
            else "Current"
        )
    elif policy.get("bootstrap_file"):
        file = policy["bootstrap_file"]
        label = "Bootstrap"
    else:
        file = training_file
        label = "Training YAML"
    result["parameter_source_label"] = label
    result["parameter_file"] = str(paths.resolve(file)) if file else None
    result["bootstrap_parameter_file"] = None
    if explicit_file is None and result.get("initialized_from_bootstrap"):
        origin = policy.get("bootstrap_file") or training_file
        result["bootstrap_parameter_file"] = (
            str(paths.resolve(origin)) if origin else None
        )
    return result


def resolve_training_parameters(cfg, lead_type_id: int, parameter_file=None) -> dict:
    """Choose an explicit candidate or the lead type's current parameter file."""
    if parameter_file is not None:
        result = load_artifact(parameter_file, lead_type_id)
        result["parameter_source"] = "parameter_file"
        return _with_parameter_details(result, cfg, parameter_file)
    policy = cfg.raw.get("parameters") or {}
    current_file = policy.get("current_file")
    if current_file:
        from smarthub.core.lead_types import lead_type_name

        from . import registry

        path = paths.resolve(current_file)
        current = load_artifact(path, lead_type_id) if path.exists() else None
        name = lead_type_name(lead_type_id)
        serving = registry.currently_serving_version(name)
        # Repair a crash between promotion and the parameter-file write. Also
        # follow an explicit model rollback. Manual settings without an approved
        # run ID remain a deliberate operator choice.
        if serving and (
            current is None
            or current.get("initialized_from_bootstrap")
            or (
                current.get("approved_training_run_id")
                and current["approved_training_run_id"] != serving
            )
        ):
            manifest = registry.load_manifest(name, serving)
            current = write_current_parameters(cfg, lead_type_id, manifest)
        if current is None:
            settings = settings_from_config(cfg)
            current = make_artifact(
                settings,
                lead_type_id,
                created_at=datetime.now(timezone.utc).isoformat(),
                initialized_from_bootstrap=True,
            )
            atomic_write_yaml(path, current)
            logger.info("Initialized current parameters from bootstrap: %s", path)
        current["parameter_source"] = "current_file"
        current["parameter_parent_training_run_id"] = current.get(
            "approved_training_run_id"
        )
        return _with_parameter_details(current, cfg)
    logger.info("model_parameters.resolve_training_parameters: using YAML settings.")
    settings = settings_from_config(cfg)
    return _with_parameter_details(
        {
            "model_settings": settings,
            "parameter_version": parameter_version(settings),
            "parameter_source": "yaml",
            "hpo_run_id": None,
            "hpo_mlflow_run_id": None,
        },
        cfg,
    )


def split_training_data(frame, target_column, split_settings, random_seed):
    """Reproduce a train/test partition from compact settings and ordered data."""
    strategy = str(split_settings["strategy"]).strip().lower()
    test_size = float(split_settings["test_size"])
    if not 0.0 < test_size < 1.0:
        raise ValueError("Split test_size must be between 0 and 1.")
    if strategy == "time":
        if "created_at" not in frame.columns:
            raise ValueError("Time-based splitting requires a 'created_at' column.")
        ordered = frame.sort_values("created_at", kind="stable")
        n_test = max(1, int(round(len(ordered) * test_size)))
        train_df = ordered.iloc[:-n_test].copy()
        test_df = ordered.iloc[-n_test:].copy()
    elif strategy == "random":
        stratify = None
        if split_settings.get("stratify", False):
            if target_column not in frame.columns:
                raise ValueError(
                    "Cannot stratify because target column "
                    f"{target_column!r} is missing."
                )
            stratify = frame[target_column]
        train_df, test_df = train_test_split(
            frame,
            test_size=test_size,
            random_state=random_seed,
            shuffle=True,
            stratify=stratify,
        )
        train_df, test_df = train_df.copy(), test_df.copy()
    else:
        raise ValueError(f"Unsupported split strategy: {strategy!r}.")
    if train_df.empty or test_df.empty:
        raise ValueError(
            "Configured train/test split produced an empty dataset. "
            "Adjust split.test_size."
        )
    return train_df, test_df


def candidate_partitions(frame: pd.DataFrame, data: dict, target_column=None):
    """Validate and recover the exact HPO fitting and final-test partitions."""
    if dataset_fingerprint(frame) != data.get("frame_fingerprint"):
        raise ValueError("Pinned HPO dataset changed; refusing candidate training.")
    if "split_settings" in data:
        if data.get("split_version") != 1:
            raise ValueError("Unsupported HPO split_version.")
        settings = data["split_settings"]
        seed = data.get("random_seed")
        if not isinstance(settings, dict) or type(seed) is not int:
            raise ValueError("HPO result is missing its reproducible split settings.")
        return split_training_data(frame, target_column, settings, seed)
    # Read previously generated HPO artifacts without rewriting their partitions.
    fit = data.get("training_positions")
    test = data.get("test_positions")
    if not isinstance(fit, list) or not isinstance(test, list) or not fit or not test:
        raise ValueError("HPO result is missing its reserved dataset partitions.")
    positions = fit + test
    if any(type(i) is not int or i < 0 or i >= len(frame) for i in positions):
        raise ValueError("HPO dataset partition positions are invalid.")
    if len(positions) != len(frame) or len(set(positions)) != len(frame):
        raise ValueError("HPO dataset partitions overlap or omit rows.")
    return frame.iloc[fit].copy(), frame.iloc[test].copy()


def atomic_write_yaml(file, payload: dict) -> None:
    """Replace a local/shared-filesystem YAML without exposing partial content."""
    path = paths.resolve(file)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            yaml.safe_dump(payload, stream, sort_keys=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def write_current_parameters(cfg, lead_type_id: int, manifest: dict) -> dict:
    """Persist a serving model's settings only after registry promotion succeeds."""
    settings = settings_from_manifest(manifest)
    artifact = make_artifact(
        settings,
        lead_type_id,
        created_at=manifest.get("created_at"),
        code_version=manifest.get("code_version"),
        approved_training_run_id=manifest["training_run_id"],
        production_model_version=manifest.get("production_model_version"),
        hpo_run_id=manifest.get("hpo_run_id"),
        hpo_mlflow_run_id=manifest.get("hpo_mlflow_run_id"),
        hpo_mlflow_tracking_uri=manifest.get("hpo_mlflow_tracking_uri"),
    )
    current_file = (cfg.raw.get("parameters") or {}).get("current_file")
    if current_file:
        atomic_write_yaml(current_file, artifact)
    return artifact
