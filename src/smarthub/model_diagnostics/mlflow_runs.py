"""Discover training runs and load their evaluation artifacts from MLflow.

The diagnostics app addresses evaluation artifacts by MLflow ``run_id`` rather
than by walking ``data/model_evaluations`` directly. That keeps a single hosted
app working across every training run and makes the underlying artifact store
(local ``mlruns`` file store today, S3 later) transparent — ``MlflowClient``
resolves the location, so no code here changes when the store moves.

Only the tracking URI is configuration: ``SMARTHUB_MLFLOW_TRACKING_URI`` (the
shared Postgres backend in Docker), the same value the training pipeline uses.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone

# Training logs each run's evaluation files under this artifact sub-path
# (mlflow.log_artifacts(report_dir, artifact_path="reports") in mlflow_utils).
REPORTS_ARTIFACT_PATH = "reports"
OPTIMIZER_CSV = "bid_optimizer_test_rows.csv"


def tracking_uri() -> str | None:
    """MLflow tracking (metadata) URI from the env, or None for MLflow's default."""
    return os.getenv("SMARTHUB_MLFLOW_TRACKING_URI") or None


def _client():
    """Build an MlflowClient bound to the configured tracking URI."""
    import mlflow
    from mlflow.tracking import MlflowClient

    uri = tracking_uri()
    if uri:
        mlflow.set_tracking_uri(uri)
    return MlflowClient(tracking_uri=uri)


@dataclass(frozen=True)
class RunInfo:
    """A training run, summarised for the dropdown."""

    run_id: str
    label: str
    experiment_name: str
    lead_type: str
    started_at: str


def _fmt_time(ms: int | None) -> str:
    """Format an MLflow epoch-millis start time as a readable UTC string."""
    if not ms:
        return "?"
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")


def _summarise(run, experiment_name: str) -> RunInfo:
    """Build a RunInfo (id + human label) from an MLflow run object."""
    tags = dict(run.data.tags or {})
    metrics = dict(run.data.metrics or {})
    lead = tags.get("lead_type_name") or tags.get("lead_type_id") or ""
    name = run.info.run_name or tags.get("mlflow.runName") or ""
    started = _fmt_time(run.info.start_time)
    short = run.info.run_id[:8]
    label = f"{started}  ·  {experiment_name}"
    if lead:
        label += f"  ·  {lead}"
    if name:
        label += f"  ·  {name}"
    roc = metrics.get("roc_auc")
    if isinstance(roc, (int, float)):
        label += f"  ·  ROC {roc:.3f}"
    label += f"  ·  {short}"
    return RunInfo(
        run_id=run.info.run_id,
        label=label,
        experiment_name=experiment_name,
        lead_type=str(lead),
        started_at=started,
    )


def list_runs(
    experiment_prefix: str | None = None, max_results: int = 300
) -> list[RunInfo]:
    """Return training runs (newest first) for the diagnostics dropdown.

    Inputs
    ------
    experiment_prefix : str | None
        Restrict to experiments whose name starts with this (e.g.
        ``"SmartHub Production"``); ``None`` searches every experiment.
    max_results : int
        Cap on the number of runs returned.

    Returns
    -------
    list[RunInfo]
        One entry per finished/failed run, newest first.
    """
    client = _client()
    experiments = client.search_experiments()
    if experiment_prefix:
        experiments = [
            e for e in experiments if str(e.name).startswith(experiment_prefix)
        ]
    if not experiments:
        return []

    name_by_id = {e.experiment_id: e.name for e in experiments}
    runs = client.search_runs(
        experiment_ids=list(name_by_id),
        order_by=["attributes.start_time DESC"],
        max_results=max_results,
    )
    return [_summarise(r, name_by_id.get(r.info.experiment_id, "?")) for r in runs]


def download_reports(run_id: str, dst_path: str | None = None) -> str:
    """Download a run's ``reports/`` artifacts locally; return the local dir path.

    MLflow resolves the artifact location from the run (file store or S3), so
    this works regardless of where artifacts live.
    """
    import mlflow

    uri = tracking_uri()
    if uri:
        mlflow.set_tracking_uri(uri)
    return mlflow.artifacts.download_artifacts(
        run_id=run_id,
        artifact_path=REPORTS_ARTIFACT_PATH,
        dst_path=dst_path,
    )


def optimizer_csv_path(run_id: str, dst_path: str | None = None) -> str:
    """Return the local path to a run's ``bid_optimizer_test_rows.csv``.

    Downloads the run's ``reports/`` artifacts and locates the optimizer CSV
    within them.

    Raises
    ------
    FileNotFoundError
        If the run has no optimizer evaluation CSV under ``reports/``.
    """
    import os.path as osp

    reports_dir = download_reports(run_id, dst_path=dst_path)
    candidate = osp.join(reports_dir, OPTIMIZER_CSV)
    if osp.isfile(candidate):
        return candidate
    # Fall back to a recursive search in case the layout nests it.
    for root, _dirs, files in os.walk(reports_dir):
        if OPTIMIZER_CSV in files:
            return osp.join(root, OPTIMIZER_CSV)
    raise FileNotFoundError(
        f"No {OPTIMIZER_CSV} found in the 'reports' artifacts of run {run_id}."
    )
