"""MLflow integration for SmartHub model training.

This module configures local MLflow storage and logs training runs.
"""

import math
from pathlib import Path

import mlflow
import mlflow.sklearn
from mlflow import MlflowClient

from smarthub.core import paths


def _is_loggable_number(value):
    """Return whether a value is a finite MLflow metric."""
    return isinstance(value, (int, float)) and not math.isnan(float(value))


def _configure_tracking(tracking_db_path, artifact_root, experiment_name):
    """Configure local MLflow metadata and artifact storage.

    Inputs
    ------
    tracking_db_path : str | pathlib.Path
        SQLite database path from the training configuration.
    artifact_root : str | pathlib.Path
        Root directory for MLflow run artifacts.
    experiment_name : str
        MLflow experiment name.

    Returns
    -------
    str
        Resolved MLflow experiment ID.

    Raises
    ------
    ValueError
        If an existing experiment points to a different artifact location.
    """
    db_path = Path(paths.resolve(tracking_db_path))
    artifact_path = Path(paths.resolve(artifact_root))
    db_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.mkdir(parents=True, exist_ok=True)

    tracking_uri = f"sqlite:///{db_path}"
    artifact_uri = artifact_path.as_uri()
    mlflow.set_tracking_uri(tracking_uri)

    client = MlflowClient(tracking_uri=tracking_uri)
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        return client.create_experiment(
            experiment_name,
            artifact_location=artifact_uri,
        )

    existing_location = str(experiment.artifact_location).rstrip("/")
    configured_location = artifact_uri.rstrip("/")
    if existing_location != configured_location:
        raise ValueError(
            f"MLflow experiment {experiment_name!r} uses artifact location "
            f"{existing_location!r}, but configuration specifies "
            f"{configured_location!r}. Use a new experiment name or migrate "
            "the existing experiment."
        )
    return experiment.experiment_id


def log_training_run(
    model,
    model_params,
    feature_cols,
    metrics,
    report_dir,
    report_artifact_path,
    tracking_db_path,
    artifact_root,
    experiment_name,
    run_name,
    registered_model_name=None,
    extra_params=None,
    optimizer_metrics=None,
):
    """Log a complete SmartHub training run to MLflow.

    Inputs
    ------
    model : Any
        Fitted model or model pipeline.
    model_params : dict
        Parameters passed to the classifier.
    feature_cols : list[str]
        Ordered model feature names.
    metrics : dict
        Model evaluation metrics.
    report_dir : str | pathlib.Path
        Local directory containing report artifacts.
    report_artifact_path : str
        MLflow artifact path taken from ``output.report_root``.
    tracking_db_path : str | pathlib.Path
        SQLite database path used by MLflow.
    artifact_root : str | pathlib.Path
        Root directory used for MLflow artifacts.
    experiment_name : str
        MLflow experiment name.
    run_name : str
        MLflow run name.
    registered_model_name : str | None
        Optional MLflow registered-model name.
    extra_params : dict | None
        Optional lineage parameters and tags.
    optimizer_metrics : dict | None
        Optional flat optimizer metrics.
    """
    experiment_id = _configure_tracking(
        tracking_db_path,
        artifact_root,
        experiment_name,
    )

    with mlflow.start_run(experiment_id=experiment_id, run_name=run_name):
        mlflow.log_param("feature_count", len(feature_cols))
        mlflow.log_param("features", ",".join(feature_cols))
        for name, value in (extra_params or {}).items():
            mlflow.log_param(name, value)
            mlflow.set_tag(name, value)

        for metric_name, metric_value in metrics.items():
            if _is_loggable_number(metric_value):
                mlflow.log_metric(metric_name, metric_value)

        for metric_name, metric_value in (optimizer_metrics or {}).items():
            if _is_loggable_number(metric_value):
                mlflow.log_metric(metric_name, metric_value)

        mlflow.log_artifacts(
            report_dir,
            artifact_path=report_artifact_path,
        )
        mlflow.log_params(
            {
                param_name: param_value
                for param_name, param_value in model_params.items()
            }
        )

        model_log_kwargs = {
            "sk_model": model,
            "name": "model",
            "serialization_format": "pickle",
        }
        if registered_model_name:
            model_log_kwargs["registered_model_name"] = registered_model_name
        mlflow.sklearn.log_model(**model_log_kwargs)
