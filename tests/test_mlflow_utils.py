"""MLflow artifact uploads for training and HPO."""

from __future__ import annotations

import json
from importlib import import_module
from pathlib import Path

import pytest

pytest.importorskip("mlflow")
mlflow_utils = import_module("smarthub.train_and_predict.mlflow_utils")


def test_hpo_artifact_layout_keeps_html_interactive_and_metadata_separate(
    tmp_path, monkeypatch
):
    metadata = [
        "best_parameters.yaml",
        "summary.json",
        "finalist_results.json",
        "hyperparameter_search.yaml",
    ]
    for name in metadata:
        (tmp_path / name).write_text("{}", encoding="utf-8")
    plots_dir = tmp_path / "plots"
    plots_dir.mkdir()
    plots = [
        "optimization_history.html",
        "parameter_importance.html",
        "contour_matrix.html",
    ]
    html = "<html><script>window.interactive = true;</script></html>"
    for name in plots:
        (plots_dir / name).write_text(html, encoding="utf-8")
    (plots_dir / "summary.json").write_text("{}", encoding="utf-8")
    uploads = []

    def capture(file, artifact_path):
        uploads.append((Path(file).name, artifact_path))

    monkeypatch.setattr(mlflow_utils.mlflow, "log_artifact", capture)
    mlflow_utils._log_hpo_artifacts(tmp_path)
    assert set(uploads) == {
        *(
            (name, "results")
            for name in metadata
            if name != "hyperparameter_search.yaml"
        ),
        ("hyperparameter_search.yaml", "config"),
        *((name, "plots") for name in plots),
    }
    for name in plots:
        assert (plots_dir / name).read_text(encoding="utf-8") == html


@pytest.mark.parametrize("empty_plots_folder", [False, True])
def test_missing_hpo_plots_warns_without_discarding_metadata(
    tmp_path, monkeypatch, caplog, empty_plots_folder
):
    (tmp_path / "summary.json").write_text("{}", encoding="utf-8")
    if empty_plots_folder:
        (tmp_path / "plots").mkdir()
    uploads = []
    monkeypatch.setattr(
        mlflow_utils.mlflow,
        "log_artifact",
        lambda file, artifact_path: uploads.append((Path(file).name, artifact_path)),
    )
    mlflow_utils._log_hpo_artifacts(tmp_path)
    assert uploads == [("summary.json", "results")]
    assert "No HPO HTML plots found" in caplog.text


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
    files, dictionaries = ([], [])
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
