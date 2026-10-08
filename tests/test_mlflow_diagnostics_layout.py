"""Diagnostics can read both current and historical artifact layouts."""

import sys
from types import SimpleNamespace

import pytest

from smarthub.model_diagnostics import mlflow_runs


@pytest.mark.parametrize(
    "folders,expected",
    [(["data", "plots", "results"], "data"), (["reports", "comparison"], "reports")],
)
def test_diagnostics_downloads_new_or_legacy_evaluation_tables(
    monkeypatch, folders, expected
):
    client = SimpleNamespace(
        list_artifacts=lambda run_id: [
            SimpleNamespace(path=folder, is_dir=True) for folder in folders
        ]
    )
    monkeypatch.setattr(mlflow_runs, "_client", lambda: client)
    monkeypatch.setattr(mlflow_runs, "tracking_uri", lambda: None)
    calls = []

    def download(**kwargs):
        calls.append(kwargs)
        return "/tmp/evaluation"

    monkeypatch.setitem(
        sys.modules,
        "mlflow",
        SimpleNamespace(artifacts=SimpleNamespace(download_artifacts=download)),
    )
    assert mlflow_runs.download_reports("run_test", "/tmp") == "/tmp/evaluation"
    assert calls == [
        {"run_id": "run_test", "artifact_path": expected, "dst_path": "/tmp"}
    ]
