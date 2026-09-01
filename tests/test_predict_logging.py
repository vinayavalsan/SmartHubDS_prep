"""Integration tests: /recommend_bid and /explain_bid actually write a
prediction-log row (success and failure), per the 2026-07-22 DS meeting
decision -- see docs/PREDICTION_LOG_SCHEMA.md.

Uses `_ConstantWinRateModel` (no sklearn/lightgbm required) with
`registry.save_version`/`promote`, same convention as test_predict.py's
`load_model_and_manifest` tests -- only `fastapi` is needed here (for
`TestClient`), gated via `pytest.importorskip`.
"""

from __future__ import annotations

import tempfile

import numpy as np
import pandas as pd
import pytest

fastapi = pytest.importorskip("fastapi")  # noqa: F841 -- import-gate only

from fastapi.testclient import TestClient  # noqa: E402

from smarthub.core.lead_types import lead_type_id  # noqa: E402
from smarthub.feature_engineering import features as fe  # noqa: E402
from smarthub.server import predict  # noqa: E402
from smarthub.train_and_predict import config, registry  # noqa: E402
from smarthub.train_and_predict.prediction_log_schema import (  # noqa: E402
    PredictionLogStore,
)


class _ConstantWinRateModel:
    """Same fake used in test_predict.py -- deterministic, no sklearn needed."""

    def __init__(self, win_rate: float = 0.6):
        self.win_rate = win_rate

    def predict_proba(self, frame):
        win = np.full(len(frame), self.win_rate)
        return np.column_stack([1 - win, win])


PAYLOAD = {
    "expected_revenue": 25.0,
    "target_cm": 0.25,
    "min_bid": 0.25,
    "bid_step": 0.25,
    "campaign_id": 12345,
    "account_id": 118,
    "lead_ping_id": 82931,
    "source_type_id": 10,
    "traffic_tier": "1",
    "lead_type_id": lead_type_id("auto"),
    "created_at": "2026-08-20T21:00:00Z",
    "state": "TX",
    "age": 34,
}


@pytest.fixture
def log_store(tmp_path, monkeypatch):
    """Point the API's prediction logger at a fresh temp SQLite file and
    reset its lazily-constructed store so each test starts clean."""
    url = f"sqlite:///{tempfile.mktemp(dir=tmp_path)}.db"
    monkeypatch.setenv("SMARTHUB_PREDICTION_LOG_DB_URL", url)
    predict._prediction_log_store_holder.clear()
    return PredictionLogStore(url)


@pytest.fixture
def client(tmp_path, monkeypatch, log_store):
    monkeypatch.setattr(registry, "MODEL_DIR_ROOT", tmp_path / "models")
    monkeypatch.delenv("MODEL_URI", raising=False)
    monkeypatch.setattr(config, "active_model_version", lambda: None)
    monkeypatch.setattr(config, "exploration_variance_pct", lambda: 0.0)
    # Prediction logging is decoupled (async writer thread) in production; force
    # synchronous inline writes in tests so the row is visible immediately after
    # the request (no polling), keeping these assertions deterministic.
    monkeypatch.setenv("SMARTHUB_PREDICTION_LOG_SYNC", "1")
    predict.clear_model_cache()
    with TestClient(predict.app, raise_server_exceptions=False) as c:
        yield c


def _promote_constant_model(win_rate=0.42):
    manifest = registry.save_version(
        _ConstantWinRateModel(win_rate=win_rate),
        "auto",
        feature_cols=["bid"],
        metrics={"roc_auc": 0.7},
        optimizer_summary={},
        lineage={"model_type": "constant_test_model", "calibrated": False},
        model_params={},
        training_config={},
        promotion_mode="manual",
        eligibility_status="eligible",
        promotion_status="awaiting_manual_promotion",
        promotion_decision_reason="test",
    )
    registry.promote("auto", manifest["version"])
    predict.clear_model_cache()
    return manifest


# --- /recommend_bid: success paths -------------------------------------------


def test_recommend_bid_logs_success_row_with_model(client, log_store):
    manifest = _promote_constant_model()

    resp = client.post("/recommend_bid", json=PAYLOAD)
    assert resp.status_code == 200

    rows = log_store.recent(limit=10)
    assert len(rows) == 1
    row = rows[0]
    assert row["endpoint"] == "recommend_bid"
    assert row["status"] == "success"
    assert row["error_message"] is None
    assert row["decision_path"] == "model"
    assert row["model_version"] == manifest["version"]
    assert row["model_type"] == "constant_test_model"
    assert row["lead_type_id"] == lead_type_id("auto")
    assert row["campaign_id"] == PAYLOAD["campaign_id"]
    assert row["input_features"]["state"] == "TX"
    assert row["model_input_features"] is not None
    assert row["candidate_bid_generation"]["method"] == "equally_spaced"
    assert row["serving_config"]["exploration_variance_pct"] is not None
    assert float(row["recommended_bid"]) == resp.json()["recommended_bid"]
    # prediction_id is returned as a receipt AND matches the actual log row --
    # see docs/PREDICTION_LOG_SCHEMA.md §8.
    assert resp.json()["prediction_id"] == row["prediction_id"]


def test_recommend_bid_threads_lead_ping_id_through_to_log_row(client, log_store):
    _promote_constant_model()

    resp = client.post("/recommend_bid", json={**PAYLOAD, "lead_ping_id": 98765})
    assert resp.status_code == 200
    # Echoed back in the response, not just written to the log row.
    assert resp.json()["lead_ping_id"] == 98765

    row = log_store.recent(limit=10)[0]
    assert row["lead_ping_id"] == 98765
    assert resp.json()["prediction_id"] == row["prediction_id"]


def test_recommend_bid_requires_lead_ping_id(client, log_store):
    # lead_ping_id is required: it's the join key linking a prediction to its
    # raw lead/outcome for post-bid monitoring, so a request without it is
    # rejected (422) and nothing is logged.
    _promote_constant_model()
    payload = {key: value for key, value in PAYLOAD.items() if key != "lead_ping_id"}

    resp = client.post("/recommend_bid", json=payload)

    assert resp.status_code == 422
    assert log_store.recent(limit=10) == []


def test_recommend_bid_cold_start_logs_null_model_fields(client, log_store):
    # No model ever saved/promoted -- true cold start.
    resp = client.post("/recommend_bid", json=PAYLOAD)
    assert resp.status_code == 200
    assert resp.json()["decision_path"] == "cold_start_fallback"

    row = log_store.recent(limit=10)[0]
    assert row["status"] == "success"
    assert row["decision_path"] == "cold_start_fallback"
    assert row["model_version"] is None
    assert row["model_uri"] is None
    # Cold start has no predicted profit to derive a CM from either.
    assert row["recommended_bid_predicted_profit"] is None
    assert row["recommended_bid_predicted_cm"] is None


def test_recommend_bid_logs_predicted_profit_and_cm(client, log_store):
    # 2026-07-23: recommended_bid_expected_profit renamed to
    # recommended_bid_predicted_profit, plus a new derived
    # recommended_bid_predicted_cm = predicted_profit / expected_revenue.
    _promote_constant_model()

    resp = client.post("/recommend_bid", json=PAYLOAD)
    assert resp.status_code == 200
    body = resp.json()
    assert body["recommended_bid_predicted_profit"] is not None
    assert "recommended_bid_expected_profit" not in body  # old name is gone

    row = log_store.recent(limit=10)[0]
    assert float(row["recommended_bid_predicted_profit"]) == pytest.approx(
        body["recommended_bid_predicted_profit"]
    )
    assert float(row["recommended_bid_predicted_cm"]) == pytest.approx(
        body["recommended_bid_predicted_profit"] / PAYLOAD["expected_revenue"]
    )


def test_recommend_bid_prediction_id_generated_before_background_write(
    client, log_store
):
    """prediction_id is generated by the route itself (uuid4), not read back
    from the store -- confirm it's a well-formed uuid4 string and that the
    background-logged row uses that exact id, even though the actual DB
    insert happens strictly after the response is built."""
    import uuid

    _promote_constant_model()

    resp = client.post("/recommend_bid", json=PAYLOAD)
    assert resp.status_code == 200
    prediction_id = resp.json()["prediction_id"]

    # Well-formed uuid4 (predict.recommend_bid generates it via uuid.uuid4()).
    parsed = uuid.UUID(prediction_id)
    assert parsed.version == 4

    row = log_store.get(prediction_id)
    assert row is not None
    assert row["prediction_id"] == prediction_id


def test_recommend_bid_default_response_is_lean(client, log_store):
    """Default response returns the decision essentials only -- the heavy
    logging-only payload (sweep, snapshots, model identity) stays off the wire."""
    _promote_constant_model()

    body = client.post("/recommend_bid", json=PAYLOAD).json()

    assert "recommended_bid" in body and "prediction_id" in body
    for k in (
        "candidate_evaluations",
        "input_features",
        "model_input_features",
        "model_version",
        "serving_config",
        "package_version",
    ):
        assert k not in body


def test_recommend_bid_verbose_returns_full_payload_capped(client, log_store):
    """`verbose=True` returns the full logging-schema payload with the candidate
    sweep capped (selected bid + first 19) and SHAP omitted; the log keeps the
    complete sweep."""
    manifest = _promote_constant_model()

    body = client.post("/recommend_bid", json={**PAYLOAD, "verbose": True}).json()

    # Full logging-schema fields now on the response:
    assert body["model_version"] == manifest["version"]
    assert body["input_features"]["state"] == "TX"
    assert body["model_input_features"] is not None
    assert body["serving_config"]["exploration_variance_pct"] is not None
    assert body["package_version"]
    assert body["served_at"]

    # Candidate sweep is capped (<=20) and always contains the selected bid:
    cands = body["candidate_evaluations"]
    assert isinstance(cands, list) and 0 < len(cands) <= 20
    assert sum(1 for c in cands if c["selected"]) == 1

    # SHAP is never on the response path (attached asynchronously by id):
    assert "shap_explanation" not in body

    # The log row still holds the FULL sweep (not the capped 20):
    row = log_store.recent(limit=1)[0]
    assert len(row["candidate_evaluations"]) >= len(cands)


# --- /recommend_bid: failure path --------------------------------------------


def test_recommend_bid_error_logs_error_row_and_still_returns_500(
    client, log_store, monkeypatch
):
    _promote_constant_model()
    monkeypatch.setenv("MODEL_URI", "/nonexistent/path/model.pkl")
    predict.clear_model_cache()

    resp = client.post("/recommend_bid", json=PAYLOAD)
    assert resp.status_code == 500  # logging must not swallow the real error

    row = log_store.recent(limit=10)[0]
    assert row["status"] == "error"
    assert row["error_message"]
    assert row["recommended_bid"] is None
    assert row["decision_path"] is None
    # Request context is still captured even though scoring never ran.
    assert row["input_features"]["state"] == "TX"


# --- /explain_bid: consumes an already-logged prediction (production mode) ---
# The route now takes {prediction_id} and explains a *persisted* prediction --
# it never re-decides the bid. So each test first makes a /recommend_bid call to
# create the prediction, then explains it by id.


def test_explain_bid_cold_start_consumes_prediction(client, log_store):
    """Cold start: the logged prediction has no model, so /explain_bid returns
    the policy explanation (no SHAP) and persists it onto the same row."""
    pred = client.post("/recommend_bid", json={**PAYLOAD, "lead_ping_id": 555})
    assert pred.status_code == 200
    assert pred.json()["decision_path"] == "cold_start_fallback"
    prediction_id = pred.json()["prediction_id"]

    resp = client.post("/explain_bid", json={"prediction_id": prediction_id})
    assert resp.status_code == 200
    body = resp.json()
    assert body["prediction_id"] == prediction_id
    assert body["lead_ping_id"] == 555
    assert body["top_factors"] == []
    assert body["explanation"]  # the cold-start policy reason

    # Persisted back onto the same (recommend_bid) row -- no new row created.
    row = log_store.get(prediction_id)
    assert row["shap_explanation"]["explanation"] == body["explanation"]


def test_explain_bid_unknown_prediction_id_returns_404(client, log_store):
    """Explaining an id that was never logged is a 404, not a crash."""
    resp = client.post("/explain_bid", json={"prediction_id": "does-not-exist"})
    assert resp.status_code == 404


def test_explain_bid_non_lgbm_model_degrades_gracefully(client, log_store):
    """A non-LightGBM model can't be SHAP-explained -> graceful 200 with a
    fallback message (never a 500)."""
    _promote_constant_model()
    pred = client.post("/recommend_bid", json=PAYLOAD)
    assert pred.json()["decision_path"] == "model"
    prediction_id = pred.json()["prediction_id"]

    resp = client.post("/explain_bid", json={"prediction_id": prediction_id})
    assert resp.status_code == 200
    body = resp.json()
    assert body["top_factors"] == []
    assert "unavailable" in body["explanation"].lower()


def test_explain_bid_uses_the_predictions_model_not_the_current(
    client, log_store, monkeypatch
):
    """The correctness win: even after a newer model is promoted, /explain_bid
    loads the model that *served* the logged prediction (by its logged
    model_uri), not whatever is currently serving."""
    model_a = _promote_constant_model(win_rate=0.3)  # A serves the prediction
    pred = client.post("/recommend_bid", json=PAYLOAD)
    prediction_id = pred.json()["prediction_id"]
    model_a_uri = log_store.get(prediction_id)["model_uri"]
    assert model_a_uri

    model_b = _promote_constant_model(win_rate=0.9)  # B is now current-serving
    assert model_b["version"] != model_a["version"]

    loaded = {}
    real_load = predict.load_model

    def _spy(model_uri=None, lead_type_id=lead_type_id("auto")):
        loaded["uri"] = model_uri
        return real_load(model_uri=model_uri, lead_type_id=lead_type_id)

    monkeypatch.setattr(predict, "load_model", _spy)

    client.post("/explain_bid", json={"prediction_id": prediction_id})
    assert loaded["uri"] == model_a_uri  # A, not B


# --- /recommend_bid: background SHAP enrichment ------------------------------
#
# /recommend_bid never computes SHAP on its own response path (that stays
# /explain_bid's job) -- but per the 2026-07-23 follow-up, a real model's
# prediction is enriched with top_factors/base_win_rate in a BackgroundTask
# that runs AFTER the response is sent, updating the same prediction_id's log
# row via PredictionLogStore.update_shap_explanation. Starlette's
# BackgroundTasks run synchronously as part of finishing the response cycle,
# and FastAPI's TestClient waits for that to complete before returning from
# client.post(...) -- so these tests can check the DB row immediately, no
# polling/sleep needed.

NUMERIC = ["bid", "age"]
CATEGORICAL = ["state", "created_hour", "created_dayofweek"]


@pytest.fixture
def small_feature_columns(monkeypatch):
    """Use a small feature set for the LightGBM SHAP integration test."""
    monkeypatch.setattr(
        fe, "model_feature_columns", lambda lead_type_id: (NUMERIC, CATEGORICAL)
    )


def _promote_tiny_lightgbm_model():
    """Fit + promote a real (tiny) LightGBM pipeline. Needed here (unlike the
    rest of this file's _ConstantWinRateModel fake) because the background
    SHAP task calls explain.explain_row, which only supports
    model_type='lightgbm' (see explain._fitted_lgbm_estimators)."""
    pytest.importorskip("lightgbm")
    pytest.importorskip("shap")
    from smarthub.train_and_predict import models

    n = 40
    frame = pd.DataFrame(
        {
            "bid": [float(i % 10) for i in range(n)],
            "age": [20 + (i % 40) for i in range(n)],
            "state": ["TX", "CA"] * (n // 2),
            "created_hour": [str(i % 24) for i in range(n)],
            "created_dayofweek": [str(i % 7) for i in range(n)],
        }
    )
    y = [1 if (i % 10) >= 5 else 0 for i in range(n)]

    model = models.build_model(
        "lightgbm",
        NUMERIC,
        CATEGORICAL,
        model_params={"n_estimators": 5, "min_child_samples": 1, "num_leaves": 7},
        calibration_enabled=False,
        calibration_method="isotonic",
        calibration_cv=3,
    )
    model.fit(frame, y)

    manifest = registry.save_version(
        model,
        "auto",
        feature_cols=NUMERIC + CATEGORICAL,
        metrics={"roc_auc": 0.7},
        optimizer_summary={},
        lineage={"model_type": "lightgbm", "calibrated": False},
        model_params={},
        training_config={},
        promotion_mode="manual",
        eligibility_status="eligible",
        promotion_status="awaiting_manual_promotion",
        promotion_decision_reason="test",
    )
    registry.promote("auto", manifest["version"])
    predict.clear_model_cache()
    return manifest


def test_recommend_bid_background_task_attaches_shap_explanation(
    client, log_store, small_feature_columns, monkeypatch
):
    """A real LightGBM model triggers a background task that attaches
    top_factors/base_win_rate to the already-logged row -- without the LLM
    'explanation'/'bid_curve' fields /explain_bid's shap_explanation carries,
    since those stay off the /recommend_bid path entirely (see
    _log_shap_background's docstring)."""
    # This test exercises the in-process SHAP path specifically; pin the mode
    # so the offload default (config/smarthub.yaml) doesn't turn it off here.
    monkeypatch.setenv("SMARTHUB_SHAP_MODE", "inprocess")
    _promote_tiny_lightgbm_model()

    resp = client.post("/recommend_bid", json=PAYLOAD)
    assert resp.status_code == 200
    prediction_id = resp.json()["prediction_id"]
    assert prediction_id is not None

    row = log_store.get(prediction_id)
    shap = row["shap_explanation"]
    assert shap is not None
    assert "top_factors" in shap and shap["top_factors"]
    assert 0.0 <= shap["base_win_rate"] <= 1.0
    assert "explanation" not in shap  # no LLM narrative on this path
    assert "bid_curve" not in shap


def test_recommend_bid_cold_start_never_schedules_shap_task(client, log_store):
    """No model at all (true cold start) -> no background SHAP task --
    nothing to explain, and load_model_and_manifest returned model=None."""
    resp = client.post("/recommend_bid", json=PAYLOAD)
    assert resp.status_code == 200
    assert resp.json()["decision_path"] == "cold_start_fallback"

    row = log_store.get(resp.json()["prediction_id"])
    assert row["shap_explanation"] is None


def test_recommend_bid_non_lgbm_model_leaves_shap_explanation_null(
    client, log_store, monkeypatch
):
    """A non-LightGBM model (_ConstantWinRateModel) makes the background SHAP
    task fail internally (explain_row only supports lightgbm) -- caught and
    swallowed by _log_shap_background, leaving shap_explanation null. Must
    never surface as an error to the caller (response is still 200) or break
    anything else already logged on the row."""
    # In-process path is what this test asserts; pin it against the offload
    # default so the failure-is-swallowed behaviour is what's exercised.
    monkeypatch.setenv("SMARTHUB_SHAP_MODE", "inprocess")
    _promote_constant_model()

    resp = client.post("/recommend_bid", json=PAYLOAD)
    assert resp.status_code == 200

    row = log_store.get(resp.json()["prediction_id"])
    assert row["shap_explanation"] is None
