"""Training promotion checks and full-run MLflow lifecycle integration.

The lifecycle test uses real LightGBM fits and MLflow. It is skipped when those
optional ML dependencies are absent, while the remaining helper tests still run.
"""

from types import SimpleNamespace

import pandas as pd
import pytest

pytest.importorskip("sklearn")

from smarthub.train_and_predict import registry, train  # noqa: E402


def _monotonicity_config():
    """Return the minimal monotonicity config required by the helper."""
    return SimpleNamespace(
        tolerance=1e-8,
        max_violation_rate=0.0,
    )


@pytest.mark.parametrize("threaded", [False, True])
def test_training_lifecycle_logs_one_run_and_preserves_model_outputs(
    tmp_path, monkeypatch, threaded
):
    """Exercise real fits and MLflow, including stages on separate task threads."""
    import json
    from concurrent.futures import ThreadPoolExecutor

    import yaml

    mlflow = pytest.importorskip("mlflow")
    pytest.importorskip("lightgbm")
    from mlflow import MlflowClient

    payload = yaml.safe_load(
        train.config.paths.resolve("config/training.yaml").read_text()
    )
    defaults = payload["training"]["defaults"]
    defaults["early_stopping"].update(max_estimators=8, stopping_rounds=2)
    defaults["promotion"]["mode"] = "disabled"
    defaults["output"].update(
        report_root=str(tmp_path / "reports"),
        model_root=str(tmp_path / "models"),
        comparison_artifacts=False,
    )
    defaults["mlflow"].update(
        tracking_db_path=str(tmp_path / "mlflow.db"),
        artifact_root=str(tmp_path / "mlruns"),
        required=True,
    )
    defaults["mlflow"]["system_metrics"] = {"enabled": True, "interval_seconds": 0.1}
    payload["training"]["lead_types"][6]["parameters"]["current_file"] = str(
        tmp_path / "current.yaml"
    )
    path = tmp_path / "training.yaml"
    path.write_text(yaml.safe_dump(payload))
    monkeypatch.setenv("SMARTHUB_TRAINING_CONFIG", str(path))
    monkeypatch.delenv("SMARTHUB_MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.setattr(registry, "MODEL_DIR_ROOT", tmp_path / "models")
    monkeypatch.setattr(registry, "_production_store", lambda _: None)
    frame = pd.DataFrame(
        {
            "bid": [1.0, 2.0] * 60,
            "state": ["CA", "NY"] * 60,
            "won_flag": [0, 1] * 60,
            "expected_revenue": [5.0] * 120,
            "created_at": pd.date_range("2026-10-01", periods=120, freq="h"),
        }
    )
    summary = {
        "training_table_version": "snapshot",
        "training_rows": 120,
        "dropped_rows": 0,
        "win_rate": 0.5,
        "missing_feature_columns": [],
        "data_min_created_at": str(frame.created_at.min()),
        "data_max_created_at": str(frame.created_at.max()),
        "source_row_count": 120,
    }
    monkeypatch.setattr(
        train.preprocessing,
        "prepare_training_data",
        lambda *args: (frame.copy(), ["bid"], ["state"], summary),
    )
    monkeypatch.setattr(train.notifications, "_post", lambda *args, **kwargs: True)
    # Report rendering is covered elsewhere; retain the real fitting/evaluation.
    monkeypatch.setattr(
        train.training_artifacts, "save_performance_plots", lambda **kwargs: None
    )
    if threaded:
        with ThreadPoolExecutor(max_workers=1) as pool:
            functions = [
                lambda ctx, stage=stage: pool.submit(stage, ctx).result()
                for stage in train.TRAINING_STAGES
            ]
            result = train.execute_training(
                train.TrainingContext(lead_type_id=6),
                stage_functions=functions,
                mlflow_function=lambda ctx: pool.submit(
                    train.stage_mlflow, ctx
                ).result(),
            )
    else:
        result = train.run_training(6)
    client = MlflowClient(tracking_uri=f"sqlite:///{tmp_path / 'mlflow.db'}")
    experiment = client.get_experiment_by_name("anton_win_probability_auto")
    runs = client.search_runs([experiment.experiment_id])
    assert len(runs) == 1
    run = runs[0]
    assert run.info.status == "FINISHED"
    assert run.data.params["training_run_id"] == result["training_run_id"]
    assert run.data.params["parameter_source"] == "current_file"
    assert "timing.fit_model_seconds" in run.data.metrics
    assert run.data.metrics["log_loss"] == result["metrics"]["log_loss"]
    assert "system/process_tree_rss_mb" in run.data.metrics
    assert run.data.tags["tracking.degraded"] == "False"
    assert {item.dataset.name for item in run.inputs.dataset_inputs} == {
        "training",
        "reserved_test",
    }
    manifest_path = registry.manifest_path("auto", result["training_run_id"])
    manifest = json.loads(manifest_path.read_text())
    assert manifest["mlflow_run_id"] == run.info.run_id
    assert client.list_artifacts(run.info.run_id, "results")
    assert mlflow.active_run() is None


def test_evaluate_currently_serving_model_none_when_nothing_serving(monkeypatch):
    """No promoted version yet -> (None, None), logged as the bootstrap case."""
    monkeypatch.setattr(
        registry, "currently_serving_version", lambda lead_type_name: None
    )
    logged_messages = []
    monkeypatch.setattr(
        train.logger,
        "info",
        lambda message, *args: logged_messages.append(
            message % args if args else message
        ),
    )
    result = train._evaluate_currently_serving_model(
        "auto",
        pd.DataFrame(),
        pd.Series(dtype=int),
        0.25,
        0.25,
        0.25,
        100,
        _monotonicity_config(),
    )
    assert result == (None, None)
    assert any("first model" in message for message in logged_messages)


def test_evaluate_currently_serving_model_propagates_registry_load_error(monkeypatch):
    """Registry loading errors currently propagate to the training workflow."""

    def _boom(lead_type_name):
        raise ValueError("Expecting value: line 1 column 1 (char 0)")

    monkeypatch.setattr(
        registry, "currently_serving_version", lambda lead_type_name: "v1_x"
    )
    monkeypatch.setattr(registry, "load_currently_serving_model", _boom)
    with pytest.raises(ValueError, match="Expecting value"):
        train._evaluate_currently_serving_model(
            "auto",
            pd.DataFrame(),
            pd.Series(dtype=int),
            0.25,
            0.25,
            0.25,
            100,
            _monotonicity_config(),
        )


def test_evaluate_currently_serving_model_skips_incompatible_schema(monkeypatch):
    """An incompatible serving feature schema is treated as not comparable
    (logged) rather than raising -- so a feature-pipeline migration (e.g. the
    one-hot age_cohort_* / age_missing columns replaced by a single age_cohort)
    doesn't fail the whole training run. The challenger is then judged on the
    absolute promotion gates alone (decide_promotion handles None serving
    metrics)."""
    monkeypatch.setattr(
        registry, "currently_serving_version", lambda lead_type_name: "v1_x"
    )
    monkeypatch.setattr(
        registry,
        "load_currently_serving_model",
        lambda lead_type_name: (
            object(),
            {"version": "v1_x", "feature_cols": ["not_a_real_column"]},
        ),
    )
    test_df = pd.DataFrame({"bid": [1.0, 2.0]})
    logged = []
    monkeypatch.setattr(
        train.logger,
        "warning",
        lambda message, *args: logged.append(message % args if args else message),
    )
    result = train._evaluate_currently_serving_model(
        "auto",
        test_df,
        pd.Series([0, 1]),
        0.25,
        0.25,
        0.25,
        100,
        _monotonicity_config(),
    )
    assert result == (None, None)
    assert any(
        "not present in the current training data" in message for message in logged
    )


def test_evaluate_currently_serving_model_skips_on_scoring_failure(monkeypatch):
    """If the champion pipeline cannot score, treat it as not comparable."""

    class _Boom:
        def predict_proba(self, X):
            raise RuntimeError("incompatible encoders")

        def predict(self, X):
            raise RuntimeError("incompatible encoders")

    monkeypatch.setattr(
        registry, "currently_serving_version", lambda lead_type_name: "v2_x"
    )
    monkeypatch.setattr(
        registry,
        "load_currently_serving_model",
        lambda lead_type_name: (
            _Boom(),
            {"version": "v2_x", "feature_cols": ["bid"]},
        ),
    )

    logged = []
    monkeypatch.setattr(
        train.logger,
        "warning",
        lambda message, *args: logged.append(message % args if args else message),
    )

    test_df = pd.DataFrame({"bid": [1.0, 2.0]})
    result = train._evaluate_currently_serving_model(
        "auto",
        test_df,
        pd.Series([0, 1]),
        0.25,
        0.25,
        0.25,
        100,
        _monotonicity_config(),
    )
    assert result == (None, None)
    assert any("not comparable" in message for message in logged)
