"""MLflow integration for SmartHub model training and promotion."""

import logging
import math
import os
from pathlib import Path

import mlflow
import mlflow.sklearn
from mlflow import MlflowClient

from smarthub.core import paths
from smarthub.core.logging_utils import get_logger

logger = get_logger(__name__)

logger = logging.getLogger("smarthub.train_and_predict.mlflow_utils")


def _ensure_database_exists(tracking_uri: str) -> None:
    """For a Postgres tracking URI, create the target database if it's missing.

    MLflow creates its own tables but NOT the database itself, so pointing it at
    ``.../mlflow`` on the shared Postgres normally needs a one-time manual
    ``CREATE DATABASE mlflow``. This does it automatically: connect to the
    server's ``postgres`` maintenance DB and create the target database if it
    doesn't exist yet. No-op for SQLite / non-Postgres URIs, and best-effort --
    a failure here just falls through to MLflow's own (clearer) error.
    """
    if not tracking_uri.startswith("postgresql"):
        return
    try:
        from sqlalchemy import create_engine, text
        from sqlalchemy.engine import make_url

        url = make_url(tracking_uri)
        db_name = url.database
        if not db_name or db_name == "postgres":
            return
        admin_engine = create_engine(
            url.set(database="postgres"), isolation_level="AUTOCOMMIT"
        )
        with admin_engine.connect() as conn:
            exists = conn.execute(
                text("SELECT 1 FROM pg_database WHERE datname = :n"),
                {"n": db_name},
            ).scalar()
            if not exists:
                conn.execute(text(f'CREATE DATABASE "{db_name}"'))
                logger.info(
                    "Created MLflow database %r on the shared Postgres.", db_name
                )
        admin_engine.dispose()
    except Exception:  # noqa: BLE001 -- best-effort; MLflow will surface a clear error
        logger.warning(
            "Could not ensure the MLflow database exists (%s); MLflow will "
            "report the underlying error if the DB is truly missing.",
            tracking_uri,
            exc_info=True,
        )


def _is_loggable_number(value):
    """Return whether a value is a finite MLflow metric."""
    return isinstance(value, (int, float)) and not math.isnan(float(value))


def _resolve_tracking_uri(tracking_db_path):
    """Resolve MLflow's backend store (metadata) URI.

    ``SMARTHUB_MLFLOW_TRACKING_URI`` wins when set -- e.g. the shared Prefect
    Postgres (``postgresql+psycopg2://prefect:prefect@postgres:5432/mlflow``) --
    so Docker deployments consolidate metadata onto one database instead of a
    SQLite file (which also sidesteps the absolute-artifact-path bug that a
    file store bakes in). Otherwise fall back to the local SQLite file at
    ``tracking_db_path`` for non-Docker/dev use. Note: this is only the
    *metadata* store; artifacts still live under ``artifact_root`` (the
    ``mlruns`` tree / later S3).
    """
    override = os.getenv("SMARTHUB_MLFLOW_TRACKING_URI")
    if override:
        return override
    db_path = Path(paths.resolve(tracking_db_path))
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{db_path}"


def _configure_tracking(tracking_db_path, artifact_root, experiment_name):
    """Configure MLflow metadata (backend store) and artifact storage."""
    artifact_path = Path(paths.resolve(artifact_root))
    artifact_path.mkdir(parents=True, exist_ok=True)

    tracking_uri = _resolve_tracking_uri(tracking_db_path)
    artifact_uri = artifact_path.as_uri()
    # Auto-create the target Postgres database if it's missing (no-op for
    # SQLite), so MLflow-on-Postgres needs no manual `CREATE DATABASE mlflow`.
    _ensure_database_exists(tracking_uri)
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
    tracking_db_path,
    artifact_root,
    experiment_name,
    run_name,
    training_config_path,
    comparison_artifact_dir=None,
    registered_model_name=None,
    extra_params=None,
    extra_tags=None,
    optimizer_metrics=None,
):
    """Log a complete SmartHub training run to MLflow.

    Mutable lifecycle fields such as ``promotion_status`` and ``promoted`` are
    stored only as tags. MLflow parameters are immutable and therefore must not
    be used for values that change during a later manual promotion.

    The exact training YAML used for the run is logged under the MLflow
    ``config`` artifact directory.

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

        mlflow.log_artifacts(report_dir, artifact_path="reports")
        if comparison_artifact_dir:
            comparison_artifact_path = "comparison"
            mlflow.log_artifacts(
                comparison_artifact_dir,
                artifact_path=comparison_artifact_path,
            )
            logger.info("Saved Model Comparison Artifacts")
            logger.info(
                "  MLflow run ID                         : %s",
                run.info.run_id,
            )
            logger.info(
                "  Artifact directory                    : %s/",
                comparison_artifact_path,
            )
            logger.info(
                "  Evaluation dataset                    : %s",
                f"{comparison_artifact_path}/evaluation_dataset.parquet",
            )
            logger.info(
                "  Optimizer results                     : %s",
                f"{comparison_artifact_path}/optimizer_results.parquet",
            )
            logger.info(
                "  Metadata                              : %s",
                f"{comparison_artifact_path}/evaluation_metadata.json",
            )
        mlflow.log_artifact(
            str(training_config_path),
            artifact_path="config",
        )
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


def log_production_model(
    *,
    tracking_uri,
    experiment_name,
    registered_model_name,
    run_name,
    model,
    model_params=None,
    feature_cols=None,
    metrics=None,
    tags=None,
):
    """Log a promoted model to the PRODUCTION MLflow server and register it.

    Production MLflow only ever sees promoted models. The training run lived in
    the *local* MLflow (via :func:`log_training_run`), so here we create a fresh
    run on the production tracking server, log its params/metrics/model, and
    register it under ``registered_model_name`` -- giving production a clean
    registry + lineage of only the models that were actually promoted.

    Returns production MLflow metadata (prefixed ``production_``) to persist on
    the training-run manifest.
    """
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)
    experiment = client.get_experiment_by_name(experiment_name)
    experiment_id = (
        experiment.experiment_id
        if experiment is not None
        else client.create_experiment(experiment_name)
    )

    with mlflow.start_run(experiment_id=experiment_id, run_name=run_name) as run:
        for name, value in (model_params or {}).items():
            if value is not None:
                mlflow.log_param(name, value)
        if feature_cols:
            mlflow.log_param("feature_count", len(feature_cols))
            mlflow.log_param("features", ",".join(feature_cols))
        for name, value in (metrics or {}).items():
            if _is_loggable_number(value):
                mlflow.log_metric(name, value)
        for name, value in (tags or {}).items():
            if value is not None:
                mlflow.set_tag(name, str(value))
        mlflow.sklearn.log_model(
            sk_model=model, name="model", serialization_format="pickle"
        )
        model_uri = f"runs:/{run.info.run_id}/model"
        registered = mlflow.register_model(
            model_uri=model_uri, name=registered_model_name
        )
        return {
            "production_mlflow_run_id": run.info.run_id,
            "production_mlflow_experiment_id": experiment_id,
            "production_mlflow_tracking_uri": tracking_uri,
            "production_registered_model_name": registered_model_name,
            "production_registered_model_version": str(registered.version),
        }
