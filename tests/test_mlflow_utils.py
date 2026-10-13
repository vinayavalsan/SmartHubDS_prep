"""MLflow artifact uploads for training and HPO."""

from __future__ import annotations

import json
from importlib import import_module
from pathlib import Path

import pytest

pytest.importorskip("mlflow")
mlflow_utils = import_module("smarthub.train_and_predict.mlflow_utils")


class TrackingClient:
    """Record explicit-ID writes and enforce immutable parameters in tests."""

    def __init__(self):
        self.runs = {}
        self.artifacts = {}
        self.inputs = []

    def create_run(self, experiment_id, tags):
        from types import SimpleNamespace

        run_id = f"tracking_{len(self.runs)}"
        self.runs[run_id] = {
            "tags": dict(tags),
            "params": {},
            "metrics": {},
            "status": "RUNNING",
        }
        return SimpleNamespace(info=SimpleNamespace(run_id=run_id))

    def log_param(self, run_id, key, value):
        params = self.runs[run_id]["params"]
        if key in params and params[key] != value:
            raise ValueError("Cannot change an immutable parameter")
        params[key] = value

    def set_tag(self, run_id, key, value):
        self.runs[run_id]["tags"][key] = value

    def log_batch(self, run_id, metrics=(), params=(), synchronous=True):
        for item in params:
            self.log_param(run_id, item.key, item.value)
        self.runs[run_id]["metrics"].update({item.key: item.value for item in metrics})

    def log_dict(self, run_id, payload, file):
        self.artifacts[(run_id, file)] = payload

    def log_inputs(self, run_id, inputs):
        self.inputs.extend((run_id, item) for item in inputs)

    def set_terminated(self, run_id, status):
        self.runs[run_id]["status"] = status


@pytest.fixture
def tracker_setup(monkeypatch):
    import mlflow

    from smarthub.train_and_predict import mlflow_tracking

    client = TrackingClient()
    monkeypatch.setattr(mlflow, "MlflowClient", lambda **kwargs: client)
    monkeypatch.setattr(
        mlflow_utils, "_configure_tracking", lambda *args: ("local", "1")
    )
    monkeypatch.setattr(
        mlflow_tracking, "environment_snapshot", lambda: {"python": "test"}
    )
    settings = {
        "enabled": True,
        "required": True,
        "tracking_db_path": "local",
        "artifact_root": "artifacts",
        "experiment_name": "test",
        "system_metrics": {"enabled": False},
    }
    return mlflow_tracking, client, settings


def test_full_run_is_running_before_work_and_finishes_with_stage_timings(tracker_setup):
    module, client, settings = tracker_setup
    with module.RunTracker(settings, "run_test", "training", 6) as tracker:
        assert client.runs[tracker.run_id]["status"] == "RUNNING"
        with tracker.stage("fit"):
            assert client.runs[tracker.run_id]["tags"]["tracking.stage"] == "fit"
            tracker.metrics({"log_loss": 0.3, "invalid": float("inf")})
    record = client.runs[tracker.run_id]
    assert record["status"] == "FINISHED"
    assert record["params"]["training_run_id"] == "run_test"
    assert record["metrics"]["timing.fit_seconds"] >= 0
    assert "timing.total_seconds" in record["metrics"]
    assert "invalid" not in record["metrics"]
    assert (tracker.run_id, "config/environment.json") in client.artifacts


def test_failed_stage_preserves_error_and_failure_artifact(tracker_setup):
    module, client, settings = tracker_setup
    error = ValueError("invalid dataset")
    with pytest.raises(ValueError) as caught:
        with module.RunTracker(settings, "hpo_test", "hpo", 6) as tracker:
            with tracker.stage("prepare_data"):
                raise error
    assert caught.value is error
    assert client.runs[tracker.run_id]["status"] == "FAILED"
    failure = client.artifacts[(tracker.run_id, "results/failure.json")]
    assert failure["stage"] == "prepare_data"
    assert failure["type"] == "ValueError"
    assert "invalid dataset" in failure["traceback"]


def test_optional_write_failures_do_not_change_execution(tracker_setup, monkeypatch):
    module, client, settings = tracker_setup
    settings["required"] = False
    monkeypatch.setattr(
        client,
        "log_param",
        lambda *args: (_ for _ in ()).throw(OSError("tracking unavailable")),
    )
    executed = []
    with module.RunTracker(settings, "run_test", "training", 6) as tracker:
        with tracker.stage("fit"):
            executed.append("fit")
    assert executed == ["fit"]
    assert client.runs[tracker.run_id]["status"] == "FINISHED"
    assert client.runs[tracker.run_id]["tags"]["tracking.degraded"] == "True"


def test_required_tracking_startup_failure_terminates_created_run(
    tracker_setup, monkeypatch
):
    module, client, settings = tracker_setup
    error = OSError("write failed")
    monkeypatch.setattr(client, "log_param", lambda *args: (_ for _ in ()).throw(error))
    with pytest.raises(OSError) as caught:
        with module.RunTracker(settings, "hpo_test", "hpo", 6):
            pytest.fail("Required tracking failure must stop before model work")
    assert caught.value is error
    assert client.runs["tracking_0"]["status"] == "FAILED"


def test_tracking_finalize_failure_never_masks_model_failure(
    tracker_setup, monkeypatch
):
    module, client, settings = tracker_setup
    error = ValueError("model failed")
    with pytest.raises(ValueError) as caught:
        with module.RunTracker(settings, "hpo_test", "hpo", 6) as tracker:
            monkeypatch.setattr(
                client,
                "log_dict",
                lambda *args: (_ for _ in ()).throw(OSError("tracking failed")),
            )
            raise error
    assert caught.value is error
    assert client.runs[tracker.run_id]["status"] == "FAILED"


def test_child_runs_are_idempotent_and_isolated_across_threads(tracker_setup):
    from concurrent.futures import ThreadPoolExecutor

    module, client, settings = tracker_setup
    with module.RunTracker(settings, "hpo_test", "hpo", 6) as tracker:

        def record(number):
            tracker.child(
                f"trial_{number}",
                "hpo_trial",
                {"trial": number},
                {"score": number / 10},
            )

        with ThreadPoolExecutor(max_workers=3) as pool:
            list(pool.map(record, [0, 1, 2, 0, 1, 2]))
    assert len(client.runs) == 4
    for number in range(3):
        child = client.runs[tracker.children[f"trial_{number}"]]
        assert child["params"] == {"trial": str(number)}
        assert child["metrics"]["score"] == number / 10
        assert child["tags"]["mlflow.parentRunId"] == tracker.run_id
        assert child["status"] == "FINISHED"


def test_dataset_inputs_distinguish_training_and_reserved_test(tracker_setup):
    import pandas as pd

    module, client, settings = tracker_setup
    frame = pd.DataFrame({"bid": [1, 2], "won_flag": [0, 1]})
    with module.RunTracker(settings, "run_test", "training", 6) as tracker:
        tracker.dataset(frame, "train", "training", "snapshot.parquet", "won_flag")
        tracker.dataset(
            frame.iloc[1:], "test", "evaluation", "snapshot.parquet", "won_flag"
        )
    assert len(client.inputs) == 2
    assert {item.dataset.name for _, item in client.inputs} == {"train", "test"}
    assert client.artifacts[(tracker.run_id, "data/test.json")]["rows"] == 1


def test_disabled_tracking_creates_no_runs(tracker_setup):
    module, client, settings = tracker_setup
    with module.RunTracker(
        settings, "run_test", "training", 6, enabled=False
    ) as tracker:
        with tracker.stage("fit"):
            tracker.metrics({"log_loss": 0.3})
    assert client.runs == {}


def test_optional_dataset_failure_is_visible_as_degraded_tracking(
    tracker_setup, monkeypatch
):
    import mlflow.data
    import pandas as pd

    module, client, settings = tracker_setup
    settings["required"] = False

    def fail(*args, **kwargs):
        raise ValueError("dataset schema unavailable")

    monkeypatch.setattr(mlflow.data, "from_pandas", fail)
    with module.RunTracker(settings, "run_test", "training", 6) as tracker:
        tracker.dataset(pd.DataFrame({"bid": [1.0]}), "train", "training", "source")
    assert client.runs[tracker.run_id]["status"] == "FINISHED"
    assert client.runs[tracker.run_id]["tags"]["tracking.degraded"] == "True"


def test_failed_parent_closes_unfinished_children(tracker_setup):
    module, client, settings = tracker_setup
    with pytest.raises(ValueError):
        with module.RunTracker(settings, "hpo_test", "hpo", 6) as tracker:
            tracker.child("finalist_1_none", "hpo_finalist", {}, {}, status="RUNNING")
            raise ValueError("optimizer worker failed")
    child = client.runs[tracker.children["finalist_1_none"]]
    assert child["status"] == "FAILED"
    assert child["tags"]["error"] == "Parent ended before result"


@pytest.mark.parametrize("io_available", [True, False])
def test_resource_samples_include_driver_and_local_workers(
    tracker_setup, monkeypatch, io_available
):
    from types import SimpleNamespace

    import psutil

    module, client, settings = tracker_setup

    class Process:
        def __init__(self, pid, rss):
            self.pid, self.rss = pid, rss

        def children(self, recursive):
            return [Process(20, 20 * 2**20)]

        def memory_info(self):
            return SimpleNamespace(rss=self.rss)

        def cpu_percent(self):
            return 50

        def io_counters(self):
            return SimpleNamespace(read_bytes=2**20, write_bytes=2 * 2**20)

    if not io_available:
        delattr(Process, "io_counters")
    monkeypatch.setattr(psutil, "Process", lambda pid: Process(10, 80 * 2**20))
    monkeypatch.setattr(psutil, "virtual_memory", lambda: SimpleNamespace(percent=55))
    settings["system_metrics"] = {"enabled": True, "interval_seconds": 3600}
    with module.RunTracker(settings, "hpo_test", "hpo", 6) as tracker:
        assert tracker._thread.is_alive()
    assert not tracker._thread.is_alive()
    metrics = client.runs[tracker.run_id]["metrics"]
    assert metrics["system/process_tree_rss_mb"] == 100
    assert metrics["system/process_tree_cpu_percent"] == 100
    if io_available:
        assert metrics["system/process_tree_read_mb"] == 2
    else:
        assert "system/process_tree_read_mb" not in metrics
        assert client.runs[tracker.run_id]["tags"]["resource_io_available"] == "False"
    assert metrics["system/host_memory_percent"] == 55


def test_unavailable_telemetry_does_not_fail_required_tracking(
    tracker_setup, monkeypatch
):
    from types import SimpleNamespace

    import psutil

    module, client, settings = tracker_setup

    def unavailable(**kwargs):
        raise RuntimeError("process sampling unavailable")

    process = SimpleNamespace(pid=1, children=unavailable)
    monkeypatch.setattr(psutil, "Process", lambda pid: process)
    settings["system_metrics"] = {"enabled": True}
    with module.RunTracker(settings, "hpo_test", "hpo", 6) as tracker:
        pass
    assert client.runs[tracker.run_id]["status"] == "FINISHED"
    assert client.runs[tracker.run_id]["tags"]["resource_monitor"] == "unavailable"


def test_training_failure_is_tracked_before_any_model_is_saved(
    tracker_setup, monkeypatch
):
    from smarthub.train_and_predict import train

    module, client, settings = tracker_setup
    monkeypatch.setattr(module, "tracking_settings", lambda *args: settings)
    error = ValueError("training data missing")

    def fail(ctx):
        assert client.runs[ctx.tracker.run_id]["status"] == "RUNNING"
        raise error

    stages = [fail, *train.TRAINING_STAGES[1:]]
    with pytest.raises(ValueError) as caught:
        train.execute_training(train.TrainingContext(lead_type_id=6), stages)
    assert caught.value is error
    assert len(client.runs) == 1
    assert client.runs["tracking_0"]["status"] == "FAILED"
    assert (
        client.artifacts[("tracking_0", "results/failure.json")]["stage"]
        == "prepare_data"
    )


def test_hpo_failure_is_tracked_before_search_work(tracker_setup, monkeypatch):
    from smarthub.train_and_predict import hyperparameter_search as hpo

    module, client, settings = tracker_setup
    monkeypatch.setattr(module, "tracking_settings", lambda *args: settings)
    error = ValueError("HPO dataset missing")

    def fail(*args):
        assert client.runs["tracking_0"]["status"] == "RUNNING"
        raise error

    monkeypatch.setattr(hpo.preprocessing, "prepare_training_data", fail)
    with pytest.raises(ValueError) as caught:
        hpo.run_hyperparameter_search(6)
    assert caught.value is error
    assert len(client.runs) == 1
    assert client.runs["tracking_0"]["status"] == "FAILED"
    assert client.artifacts[("tracking_0", "results/failure.json")]["stage"] == (
        "prepare_data"
    )


def test_required_finalization_error_propagates_after_successful_work(
    tracker_setup, monkeypatch
):
    module, client, settings = tracker_setup
    error = OSError("status update failed")

    def fail(*args, **kwargs):
        raise error

    with pytest.raises(OSError) as caught:
        with module.RunTracker(settings, "hpo_test", "hpo", 6):
            monkeypatch.setattr(client, "set_terminated", fail)
    assert caught.value is error


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
    parameter_bytes = (tmp_path / "best_parameters.yaml").read_bytes()
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
    assert (tmp_path / "best_parameters.yaml").read_bytes() == parameter_bytes
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["hpo_mlflow_run_id"] == "mlflow_test"
