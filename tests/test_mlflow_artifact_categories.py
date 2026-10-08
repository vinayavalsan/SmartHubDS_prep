"""Training reports use the shared MLflow artifact categories."""

import json
from importlib import import_module
from pathlib import Path

import pytest

pytest.importorskip("mlflow")
mlflow_utils = import_module("smarthub.train_and_predict.mlflow_utils")


def test_training_reports_are_categorized_without_changing_local_files(
    tmp_path, monkeypatch
):
    contents = {
        "roc_curve.png": b"image",
        "feature_summary.csv": b"feature,count\nbid,10\n",
        "bid_optimizer_test_rows.csv": b"bid\n1\n",
        "model_evaluation_summary.json": b'{"log_loss": 0.3}',
        "nested/calibration.html": b"<html>interactive</html>",
    }
    for name, data in contents.items():
        file = tmp_path / name
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_bytes(data)
    files, dictionaries = [], []
    monkeypatch.setattr(
        mlflow_utils.mlflow,
        "log_artifact",
        lambda file, artifact_path: files.append((Path(file).name, artifact_path)),
    )
    monkeypatch.setattr(
        mlflow_utils.mlflow,
        "log_dict",
        lambda payload, path: dictionaries.append((payload, path)),
    )
    mlflow_utils._log_training_artifacts(tmp_path)
    assert set(files) == {
        ("roc_curve.png", "plots"),
        ("feature_summary.csv", "data"),
        ("bid_optimizer_test_rows.csv", "data"),
        ("calibration.html", "plots/nested"),
    }
    assert dictionaries == [({"log_loss": 0.3}, "results/summary.json")]
    for name, data in contents.items():
        assert (tmp_path / name).read_bytes() == data


def test_hpo_logging_records_resolved_config_and_dataset_metadata(
    tmp_path, monkeypatch
):
    from contextlib import nullcontext
    from types import SimpleNamespace

    import yaml

    artifact = {
        "hpo_run_id": "hpo_test",
        "model_settings": {
            "model_parameters": {"num_leaves": 7},
            "calibration": {"enabled": False},
        },
        "data": {
            "training_table_version": "snapshot",
            "split_settings": {"strategy": "time", "test_size": 0.2},
        },
    }
    (tmp_path / "best_parameters.yaml").write_text(yaml.safe_dump(artifact))
    (tmp_path / "summary.json").write_text(json.dumps({"hpo_run_id": "hpo_test"}))
    monkeypatch.setattr(
        mlflow_utils, "_configure_tracking", lambda *args: ("sqlite:///local", "1")
    )
    monkeypatch.setattr(
        mlflow_utils.mlflow,
        "start_run",
        lambda **kwargs: nullcontext(
            SimpleNamespace(info=SimpleNamespace(run_id="mlflow_test"))
        ),
    )
    for name in ["log_param", "log_params", "log_metric", "set_tag"]:
        monkeypatch.setattr(mlflow_utils.mlflow, name, lambda *args, **kwargs: None)
    monkeypatch.setattr(mlflow_utils, "_log_hpo_artifacts", lambda folder: None)
    logged = {}
    monkeypatch.setattr(
        mlflow_utils.mlflow,
        "log_dict",
        lambda payload, path: logged.update({path: payload}),
    )
    resolved = {"search": {"n_trials": 50}}
    result = mlflow_utils.log_hpo_run(
        tmp_path,
        {
            "tracking_db_path": "local",
            "artifact_root": "artifacts",
            "experiment_name": "hpo",
        },
        "auto",
        resolved_search_config=resolved,
    )
    assert logged["config/resolved_config.json"] == resolved
    assert logged["data/dataset.json"] == artifact["data"]
    assert result["hpo_mlflow_run_id"] == "mlflow_test"
    assert (
        yaml.safe_load((tmp_path / "best_parameters.yaml").read_text())[
            "hpo_mlflow_run_id"
        ]
        == "mlflow_test"
    )
