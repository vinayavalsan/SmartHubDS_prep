"""Tests for the optional --include-prediction-logs monitoring pull."""

from __future__ import annotations

import dataclasses
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from smarthub.core import storage
from smarthub.core.config import StorageSettings
from smarthub.data_pull import prediction_logs as pl


@pytest.fixture
def store(tmp_path, monkeypatch):
    """A prediction-log store backed by a temp SQLite DB."""
    monkeypatch.setenv(
        "SMARTHUB_PREDICTION_LOG_DB_URL", f"sqlite:///{tmp_path/'pred.db'}"
    )
    from smarthub.train_and_predict.prediction_log_schema import PredictionLogStore

    return PredictionLogStore()


@pytest.fixture
def settings(tmp_path):
    """DuckDB-only storage in a temp dir."""
    return dataclasses.replace(
        StorageSettings.from_env(),
        backend="duckdb",
        duckdb_path=str(tmp_path / "leads.duckdb"),
    )


def _log(store, prediction_id, lead_ping_id, lead_type_id=6):
    return store.log_prediction(
        endpoint="recommend_bid",
        lead_type_id=lead_type_id,
        lead_type_name="auto" if lead_type_id == 6 else "home",
        campaign_id=40088,
        input_features={},
        expected_revenue=100.0,
        target_cm=0.25,
        min_bid=0.25,
        bid_step=0.25,
        lead_ping_id=lead_ping_id,
        recommended_bid=10.0,
        recommended_bid_predicted_win_rate=0.9,
        recommended_bid_predicted_profit=80.0,
        model_name="auto_v1.0.0",
        model_version="run_x",
        prediction_id=prediction_id,
    )


def _window(hours=1):
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    return now - timedelta(hours=hours), now + timedelta(minutes=1)


def test_fetch_filters_window_and_lead_type(store):
    _log(store, "p_auto", 111, lead_type_id=6)
    _log(store, "p_home", 222, lead_type_id=1)
    since, until = _window()

    auto = pl.fetch_prediction_logs(since, until, lead_type_id=6, store=store)
    assert list(auto["prediction_id"]) == ["p_auto"]

    both = pl.fetch_prediction_logs(since, until, lead_type_id=None, store=store)
    assert set(both["prediction_id"]) == {"p_auto", "p_home"}

    empty = pl.fetch_prediction_logs(until, until, lead_type_id=6, store=store)
    assert empty.empty


def test_build_dataset_joins_and_drops_null_ping(store):
    _log(store, "matched", 111)
    _log(store, "unmatched", 222)
    _log(store, "no_ping", None)
    since, until = _window()
    preds = pl.fetch_prediction_logs(since, until, 6, store=store)

    leads = pd.DataFrame([{"id": 111, "won": "true", "rev": 50.0, "exp_rev": 60.0}])
    out = pl.build_monitoring_dataset(preds, leads)

    ids = set(out["prediction_id"])
    assert "no_ping" not in ids  # dropped: no join key
    assert ids == {"matched", "unmatched"}
    row = out.set_index("prediction_id")
    assert row.loc["matched", "lead_won"] == "true"
    assert pd.isna(row.loc["unmatched", "lead_won"])  # outcome not available yet


def test_persist_upserts_on_late_resolving_outcome(store, settings):
    _log(store, "p1", 111)
    _log(store, "p2", 222)
    since, until = _window()
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    # Only lead 111 is in the raw store so far.
    storage.append_duckdb(
        pd.DataFrame([{"id": 111, "won": "true", "rev": 50.0, "created_at": now}]),
        path=settings.duckdb_path,
    )

    r1 = pl.pull_and_persist_prediction_logs(
        since, until, 6, settings=settings, store=store
    )
    assert r1["prediction_rows"] == 2 and r1["monitoring_rows"] == 2

    # Lead 222's outcome resolves later; a re-pull must UPDATE p2 in place.
    storage.append_duckdb(
        pd.DataFrame([{"id": 222, "won": "false", "rev": 0.0, "created_at": now}]),
        path=settings.duckdb_path,
    )
    pl.pull_and_persist_prediction_logs(since, until, 6, settings=settings, store=store)

    m = storage.load_monitoring(settings)
    assert len(m) == 2  # upsert, not duplicate
    assert m.set_index("prediction_id").loc["p2", "lead_won"] == "false"


def test_end_to_end_on_parquet_backend(store, tmp_path):
    """The full pull works when STORAGE_BACKEND=parquet (the prod config)."""
    parquet_settings = dataclasses.replace(
        StorageSettings.from_env(),
        backend="parquet",
        parquet_dir=str(tmp_path / "leads"),
        partition_date_col="created_at",
        duckdb_path=str(tmp_path / "unused.duckdb"),
    )
    _log(store, "p1", 111)
    _log(store, "p2", 222)
    since, until = _window()
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    # Seed the raw leads as PARQUET (not duckdb).
    storage.append_parquet(
        pd.DataFrame([{"id": 111, "won": "true", "rev": 50.0, "created_at": now}]),
        parquet_settings.parquet_dir,
        "created_at",
    )

    r = pl.pull_and_persist_prediction_logs(
        since, until, 6, settings=parquet_settings, store=store
    )
    assert r["prediction_rows"] == 2 and r["monitoring_rows"] == 2

    m = storage.load_monitoring(parquet_settings)
    assert len(m) == 2
    by_id = m.set_index("prediction_id")
    assert by_id.loc["p1", "lead_won"] == "true"  # joined from parquet leads
    assert pd.isna(by_id.loc["p2", "lead_won"])  # lead 222 not in store yet


def test_no_predictions_is_noop(store, settings):
    since, until = _window()
    r = pl.pull_and_persist_prediction_logs(
        since, until, 6, settings=settings, store=store
    )
    assert r == {"prediction_rows": 0, "monitoring_rows": 0}
    assert storage.load_monitoring(settings).empty


def test_cli_flag_optional_and_defaults_false():
    from smarthub.data_pull.cli import build_pull_parser

    base = [
        "--min-created-at",
        "2026-08-01 00:00:00",
        "--max-created-at",
        "2026-08-02 00:00:00",
        "--lead-type-id",
        "6",
    ]
    assert build_pull_parser().parse_args(base).include_prediction_logs is False
    assert (
        build_pull_parser()
        .parse_args(base + ["--include-prediction-logs"])
        .include_prediction_logs
        is True
    )
