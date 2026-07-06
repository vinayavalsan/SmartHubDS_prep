"""
MLflow logging utilities for Anton training.
"""

import mlflow
import mlflow.sklearn


def log_training_run(
    model,
    model_params,
    feature_cols,
    metrics,
    report_dir,
    experiment_name,
    run_name,
    registered_model_name=None,
    extra_params=None,
):
    """Log model, metrics, parameters, and artifacts to MLflow.

    ``extra_params`` (e.g. lineage: training_table_version, data date range,
    model_type) is logged as params AND tags so a model traces back to the exact
    data it was trained on.
    """
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("feature_count", len(feature_cols))
        mlflow.log_param("features", ",".join(feature_cols))
        for name, value in (extra_params or {}).items():
            mlflow.log_param(name, value)
            mlflow.set_tag(name, value)

        for metric_name, metric_value in metrics.items():
            if isinstance(metric_value, (int, float)):
                mlflow.log_metric(metric_name, metric_value)

        mlflow.log_artifacts(report_dir, artifact_path="training_report")
        mlflow.log_params(
            {
                param_name: param_value
                for param_name, param_value in model_params.items()
            }
        )

        if registered_model_name:
            mlflow.sklearn.log_model(
                sk_model=model,
                name="model",
                serialization_format="pickle",
                registered_model_name=registered_model_name,
            )
        else:
            mlflow.sklearn.log_model(
                sk_model=model,
                name="model",
                serialization_format="pickle",
            )
