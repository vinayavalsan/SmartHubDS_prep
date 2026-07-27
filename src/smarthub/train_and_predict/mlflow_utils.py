"""MLflow integration for SmartHub model training and promotion."""

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
    """Configure local MLflow metadata and artifact storage."""
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
        experiment_id = client.create_experiment(
            experiment_name,
            artifact_location=artifact_uri,
        )
        return tracking_uri, experiment_id

    existing_location = str(experiment.artifact_location).rstrip("/")
    configured_location = artifact_uri.rstrip("/")
    if existing_location != configured_location:
        raise ValueError(
            f"MLflow experiment {experiment_name!r} uses artifact location "
            f"{existing_location!r}, but configuration specifies "
            f"{configured_location!r}. Use a new experiment name or migrate "
            "the existing experiment."
        )
    return tracking_uri, experiment.experiment_id


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
    extra_tags=None,
    optimizer_metrics=None,
):
    """Log a complete SmartHub training run to MLflow.

    Mutable lifecycle fields such as ``promotion_status`` and ``promoted`` are
    stored only as tags. MLflow parameters are immutable and therefore must not
    be used for values that change during a later manual promotion.

    ``registered_model_name`` is retained for backward compatibility but model
    registration is intentionally handled by :func:`promote_training_run`, so
    automatic and manual promotion use the same registration path.
    """
    del registered_model_name
    tracking_uri, experiment_id = _configure_tracking(
        tracking_db_path,
        artifact_root,
        experiment_name,
    )

    with mlflow.start_run(experiment_id=experiment_id, run_name=run_name) as run:
        mlflow.log_param("feature_count", len(feature_cols))
        mlflow.log_param("features", ",".join(feature_cols))

        for name, value in (extra_params or {}).items():
            if value is not None:
                mlflow.log_param(name, value)

        for name, value in (extra_tags or {}).items():
            if value is not None:
                mlflow.set_tag(name, value)

        for metric_name, metric_value in metrics.items():
            if _is_loggable_number(metric_value):
                mlflow.log_metric(metric_name, metric_value)

        for metric_name, metric_value in (optimizer_metrics or {}).items():
            if _is_loggable_number(metric_value):
                mlflow.log_metric(metric_name, metric_value)

        mlflow.log_artifacts(report_dir, artifact_path=report_artifact_path)
        mlflow.log_params(dict(model_params))
        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            serialization_format="pickle",
        )
        return {
            "mlflow_run_id": run.info.run_id,
            "mlflow_experiment_id": experiment_id,
            "mlflow_tracking_uri": tracking_uri,
            "mlflow_model_uri": f"runs:/{run.info.run_id}/model",
        }


def _find_training_run(client, experiment_id, training_run_id):
    """Find the MLflow run associated with one SmartHub training run."""
    for filter_string in (
        f"tags.training_run_id = '{training_run_id}'",
        f"params.training_run_id = '{training_run_id}'",
    ):
        runs = client.search_runs(
            experiment_ids=[experiment_id],
            filter_string=filter_string,
            max_results=2,
            order_by=["attributes.start_time DESC"],
        )
        if runs:
            return runs[0]
    raise LookupError(
        f"No MLflow run found for training_run_id={training_run_id!r} "
        f"in experiment_id={experiment_id!r}."
    )


def _existing_registered_version(client, registered_model_name, run_id):
    """Return an existing model version for this run, if already registered."""
    try:
        versions = client.search_model_versions(f"name = '{registered_model_name}'")
    except Exception:
        return None
    for version in versions:
        if getattr(version, "run_id", None) == run_id:
            return version
    return None


def promote_training_run(
    *,
    training_run_id,
    lead_type_name,
    production_model_version,
    reason,
    tracking_db_path,
    artifact_root,
    experiment_name,
    registered_model_name,
    mlflow_run_id=None,
):
    """Register a promoted run and update its mutable MLflow lifecycle tags.

    The operation is idempotent: rerunning it for the same training run reuses
    the existing MLflow registered-model version instead of creating a duplicate.
    """
    tracking_uri, experiment_id = _configure_tracking(
        tracking_db_path,
        artifact_root,
        experiment_name,
    )
    client = MlflowClient(tracking_uri=tracking_uri)

    if mlflow_run_id:
        run = client.get_run(mlflow_run_id)
    else:
        run = _find_training_run(client, experiment_id, training_run_id)
        mlflow_run_id = run.info.run_id

    existing = _existing_registered_version(
        client,
        registered_model_name,
        mlflow_run_id,
    )
    if existing is None:
        registered = mlflow.register_model(
            model_uri=f"runs:/{mlflow_run_id}/model",
            name=registered_model_name,
        )
        mlflow_model_version = str(registered.version)
    else:
        mlflow_model_version = str(existing.version)

    mutable_tags = {
        "promotion_status": "promoted",
        "promoted": "True",
        "production_model_version": production_model_version,
        "model_version": production_model_version,
        "promotion_reason": reason,
    }
    for name, value in mutable_tags.items():
        client.set_tag(mlflow_run_id, name, value)

    model_version_tags = {
        "training_run_id": training_run_id,
        "lead_type_name": lead_type_name,
        "production_model_version": production_model_version,
        "promotion_reason": reason,
    }
    for name, value in model_version_tags.items():
        client.set_model_version_tag(
            registered_model_name,
            mlflow_model_version,
            name,
            str(value),
        )

    if hasattr(client, "set_registered_model_alias"):
        try:
            client.set_registered_model_alias(
                registered_model_name,
                production_model_version,
                mlflow_model_version,
            )
        except Exception:
            pass

    return {
        "mlflow_run_id": mlflow_run_id,
        "mlflow_experiment_id": experiment_id,
        "mlflow_tracking_uri": tracking_uri,
        "mlflow_model_uri": f"models:/{registered_model_name}/{mlflow_model_version}",
        "mlflow_registered_model_name": registered_model_name,
        "mlflow_registered_model_version": mlflow_model_version,
    }
