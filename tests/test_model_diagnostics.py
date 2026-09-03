"""Tests for the model-diagnostics MLflow run discovery + artifact loading.

The MLflow client and artifact download are mocked, so no MLflow install or
network is needed (mlflow is imported lazily inside mlflow_runs).
"""

from __future__ import annotations

import pytest

from smarthub.model_diagnostics import mlflow_runs


class _RunData:
    def __init__(self, tags, metrics):
        self.tags = tags
        self.metrics = metrics


class _RunInfo:
    def __init__(self, run_id, exp_id, start_time, run_name):
        self.run_id = run_id
        self.experiment_id = exp_id
        self.start_time = start_time
        self.run_name = run_name


class _Run:
    def __init__(
        self,
        run_id,
        exp_id,
        start_time=1_700_000_000_000,
        run_name="run",
        tags=None,
        metrics=None,
    ):
        self.info = _RunInfo(run_id, exp_id, start_time, run_name)
        self.data = _RunData(tags or {}, metrics or {})


class _Exp:
    def __init__(self, exp_id, name):
        self.experiment_id = exp_id
        self.name = name


class _Client:
    def __init__(self, experiments, runs):
        self._experiments = experiments
        self._runs = runs

    def search_experiments(self):
        return self._experiments

    def search_runs(self, experiment_ids, order_by=None, max_results=None):
        wanted = set(experiment_ids)
        return [r for r in self._runs if r.info.experiment_id in wanted]


def test_summarise_builds_label_and_id():
    run = _Run(
        "abcdef1234567890",
        "1",
        run_name="auto-train",
        tags={"lead_type_name": "auto"},
        metrics={"roc_auc": 0.8123},
    )
    info = mlflow_runs._summarise(run, "SmartHub Production_auto")
    assert info.run_id == "abcdef1234567890"
    assert info.lead_type == "auto"
    assert "abcdef12" in info.label  # short id in the label
    assert "ROC 0.812" in info.label
    assert "SmartHub Production_auto" in info.label


def test_list_runs_filters_by_prefix(monkeypatch):
    experiments = [
        _Exp("1", "SmartHub Production_auto"),
        _Exp("2", "Something Else"),
    ]
    runs = [_Run("r1", "1"), _Run("r2", "2")]
    monkeypatch.setattr(mlflow_runs, "_client", lambda: _Client(experiments, runs))

    got = mlflow_runs.list_runs(experiment_prefix="SmartHub Production")
    assert [r.run_id for r in got] == ["r1"]


def test_list_runs_all_experiments(monkeypatch):
    experiments = [_Exp("1", "A"), _Exp("2", "B")]
    runs = [_Run("r1", "1"), _Run("r2", "2")]
    monkeypatch.setattr(mlflow_runs, "_client", lambda: _Client(experiments, runs))

    got = mlflow_runs.list_runs()
    assert {r.run_id for r in got} == {"r1", "r2"}


def test_list_runs_no_experiments_returns_empty(monkeypatch):
    monkeypatch.setattr(mlflow_runs, "_client", lambda: _Client([], []))
    assert mlflow_runs.list_runs() == []


def test_optimizer_csv_path_finds_top_level(monkeypatch, tmp_path):
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / mlflow_runs.OPTIMIZER_CSV).write_text("a,b\n1,2\n")
    monkeypatch.setattr(
        mlflow_runs, "download_reports", lambda run_id, dst_path=None: str(reports)
    )
    path = mlflow_runs.optimizer_csv_path("run1")
    assert path.endswith(mlflow_runs.OPTIMIZER_CSV)


def test_optimizer_csv_path_recursive(monkeypatch, tmp_path):
    nested = tmp_path / "reports" / "sub"
    nested.mkdir(parents=True)
    (nested / mlflow_runs.OPTIMIZER_CSV).write_text("x\n")
    monkeypatch.setattr(
        mlflow_runs,
        "download_reports",
        lambda run_id, dst_path=None: str(tmp_path / "reports"),
    )
    assert mlflow_runs.optimizer_csv_path("run1").endswith(mlflow_runs.OPTIMIZER_CSV)


def test_optimizer_csv_path_missing_raises(monkeypatch, tmp_path):
    (tmp_path / "reports").mkdir()
    monkeypatch.setattr(
        mlflow_runs,
        "download_reports",
        lambda run_id, dst_path=None: str(tmp_path / "reports"),
    )
    with pytest.raises(FileNotFoundError):
        mlflow_runs.optimizer_csv_path("run1")


def test_diagnostics_url_tag_helper(monkeypatch):
    # mlflow_utils imports mlflow at module top, so gate on the ml extra.
    pytest.importorskip("mlflow")
    from smarthub.train_and_predict import mlflow_utils

    monkeypatch.delenv("SMARTHUB_MODEL_DIAGNOSTICS_URL", raising=False)
    assert mlflow_utils._model_diagnostics_url("r1") is None

    monkeypatch.setenv("SMARTHUB_MODEL_DIAGNOSTICS_URL", "http://host:8511")
    assert mlflow_utils._model_diagnostics_url("r1") == "http://host:8511?run_id=r1"

    monkeypatch.setenv("SMARTHUB_MODEL_DIAGNOSTICS_URL", "http://host:8511/d?x=1")
    assert (
        mlflow_utils._model_diagnostics_url("r1") == "http://host:8511/d?x=1&run_id=r1"
    )
