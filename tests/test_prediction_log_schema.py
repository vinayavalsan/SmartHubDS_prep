"""Tests for smarthub.train_and_predict.prediction_log_schema.

Pure logic + a real (SQLite) engine -- no ml/fastapi extras required, same
convention as test_registry.py's use of a temp SQLite/temp-dir backend.
"""

from __future__ import annotations

import tempfile

import pytest

from smarthub.train_and_predict.prediction_log_schema import PredictionLogStore


@pytest.fixture
def store():
    """A PredictionLogStore backed by a throwaway SQLite file."""
    return PredictionLogStore(f"sqlite:///{tempfile.mktemp()}.db")


def _minimal_kwargs(**overrides):
    kwargs = dict(
        endpoint="recommend_bid",
        lead_type_id=6,
        lead_type_name="auto",
        campaign_id=4021,
        input_features={"state": "CA", "age": 34},
        expected_revenue=50.0,
        target_cm=0.25,
        min_bid=0.25,
        bid_step=0.25,
    )
    kwargs.update(overrides)
    return kwargs


# --- log_prediction / get: success path -------------------------------------


def test_log_and_get_round_trips_json_columns(store):
    pid = store.log_prediction(
        **_minimal_kwargs(
            model_input_features={"age": 34.0, "state": "CA", "bid": 12.5},
            feature_cols=["age", "state", "bid"],
            candidate_bid_generation={
                "method": "equally_spaced",
                "min_bid": 0.25,
                "max_bid": 37.5,
                "bid_step": 0.25,
                "n_candidates": 150,
            },
            shap_explanation={
                "top_factors": [
                    {
                        "feature": "age",
                        "value": 34.0,
                        "shap": -0.31,
                        "direction": "decreased",
                    }
                ],
                "base_win_rate": 0.114,
                "bid_curve": [],
                "explanation": "test",
            },
            serving_config={"exploration_variance_pct": 0.1},
            model_version="v4_2026-07-09T140501Z",
            model_type="lightgbm",
            model_calibrated=True,
            recommended_bid=12.5,
            recommended_bid_predicted_win_rate=0.34,
            recommended_bid_expected_profit=12.75,
            decision_path="model",
            decision_reason="Standard profit-maximizing bid.",
            lead_ping_id=82931,
        )
    )

    row = store.get(pid)
    assert row is not None
    assert row["status"] == "success"
    assert row["schema_version"] == 2
    assert row["lead_ping_id"] == 82931
    assert row["model_input_features"] == {"age": 34.0, "state": "CA", "bid": 12.5}
    assert row["feature_cols"] == ["age", "state", "bid"]
    assert row["candidate_bid_generation"]["n_candidates"] == 150
    assert row["shap_explanation"]["base_win_rate"] == 0.114
    assert row["shap_explanation"]["top_factors"][0]["feature"] == "age"
    assert row["serving_config"] == {"exploration_variance_pct": 0.1}
    assert float(row["recommended_bid"]) == pytest.approx(12.5)
    assert row["decision_path"] == "model"


def test_prediction_id_is_generated_when_omitted(store):
    pid = store.log_prediction(**_minimal_kwargs())
    assert pid
    assert store.get(pid) is not None


def test_explicit_prediction_id_is_used_verbatim(store):
    pid = store.log_prediction(**_minimal_kwargs(prediction_id="fixed-id-123"))
    assert pid == "fixed-id-123"
    assert store.get("fixed-id-123") is not None


# --- log_prediction: failure path --------------------------------------------


def test_error_status_row_has_no_bid_but_keeps_request_context(store):
    pid = store.log_prediction(
        **_minimal_kwargs(
            status="error",
            error_message="model.predict_proba raised ValueError",
            input_features={"state": "TX"},
        )
    )
    row = store.get(pid)
    assert row["status"] == "error"
    assert row["error_message"] == "model.predict_proba raised ValueError"
    assert row["recommended_bid"] is None
    assert row["model_input_features"] is None
    assert row["input_features"] == {"state": "TX"}


def test_null_json_columns_decode_to_none_not_a_json_string(store):
    pid = store.log_prediction(**_minimal_kwargs())
    row = store.get(pid)
    assert row["shap_explanation"] is None
    assert row["candidate_bid_generation"] is None
    assert row["serving_config"] is None


# --- validation ---------------------------------------------------------------


def test_invalid_endpoint_raises():
    store = PredictionLogStore(f"sqlite:///{tempfile.mktemp()}.db")
    with pytest.raises(ValueError):
        store.log_prediction(**_minimal_kwargs(endpoint="not_a_real_endpoint"))


def test_invalid_status_raises():
    store = PredictionLogStore(f"sqlite:///{tempfile.mktemp()}.db")
    with pytest.raises(ValueError):
        store.log_prediction(**_minimal_kwargs(status="not_a_real_status"))


# --- get / recent ---------------------------------------------------------------


def test_get_returns_none_for_unknown_id(store):
    assert store.get("does-not-exist") is None


def test_recent_filters_by_lead_type_and_status(store):
    store.log_prediction(**_minimal_kwargs(lead_type_id=6, status="success"))
    store.log_prediction(**_minimal_kwargs(lead_type_id=6, status="error"))
    store.log_prediction(**_minimal_kwargs(lead_type_id=1, status="success"))

    assert len(store.recent(limit=10)) == 3
    assert len(store.recent(lead_type_id=6, limit=10)) == 2
    assert len(store.recent(lead_type_id=1, limit=10)) == 1
    assert len(store.recent(status="error", limit=10)) == 1


def test_recent_orders_newest_first(store):
    import datetime as dt

    older = dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc)
    newer = dt.datetime(2026, 7, 1, tzinfo=dt.timezone.utc)
    store.log_prediction(**_minimal_kwargs(prediction_id="old", served_at=older))
    store.log_prediction(**_minimal_kwargs(prediction_id="new", served_at=newer))

    rows = store.recent(limit=10)
    assert [r["prediction_id"] for r in rows] == ["new", "old"]


# --- JSON encoding robustness -------------------------------------------------


def test_non_native_json_values_serialize_via_default_str(store):
    """A value json.dumps can't natively encode (e.g. a numpy-like scalar)
    must not crash the write -- mirrors the real numpy scalars that show up
    in model_input_features at the actual call site (predict.py)."""

    class _FakeNumpyFloat:
        def __str__(self):
            return "34.0"

    pid = store.log_prediction(
        **_minimal_kwargs(model_input_features={"age": _FakeNumpyFloat()})
    )
    row = store.get(pid)
    assert row["model_input_features"]["age"] == "34.0"
