"""Tests for server.explain.explain_from_prediction (production consume path).

SHAP/LLM are mocked so these stay fast; the point is that the explanation is
built from an already-computed prediction (its prepared row + chosen bid) and
never re-runs the bid decision.
"""

from smarthub.server import explain as se
from smarthub.train_and_predict import llm_explain, shap_explain


def _prediction(**over):
    base = dict(
        lead_type_id=6,
        model_input_features={"bid": 0.25, "age": 34.0, "state": "CA"},
        recommended_bid=12.5,
        recommended_bid_predicted_win_rate=0.34,
        recommended_bid_predicted_profit=12.75,
        decision_path="model",
        decision_reason="Standard profit-maximizing bid.",
    )
    base.update(over)
    return se.PredictionOutput(**base)


def test_explain_consumes_prediction_without_recompute(monkeypatch):
    """SHAP runs on the prediction's prepared row at its chosen bid; LLM added."""
    calls = {}

    def _fake_shap(model, mif, lead_type_id, recommended_bid=None, top_n=None):
        calls["model_input_features"] = mif
        calls["recommended_bid"] = recommended_bid
        return {
            "top_factors": [
                {"feature": "age", "value": 34.0, "shap": 0.5, "direction": "increased"}
            ],
            "base_win_rate": 0.11,
        }

    monkeypatch.setattr(shap_explain, "explain_prepared_row", _fake_shap)
    monkeypatch.setattr(llm_explain, "format_llm_prompt", lambda facts: "PROMPT")
    monkeypatch.setattr(llm_explain, "call_ollama", lambda prompt: "because age.")

    out = se.explain_from_prediction(object(), _prediction(), with_llm=True)

    assert out["base_win_rate"] == 0.11
    assert out["top_factors"][0]["feature"] == "age"
    assert out["explanation"] == "because age."
    # SHAP was taken at the chosen bid, from the prediction's own prepared row.
    assert calls["recommended_bid"] == 12.5
    assert calls["model_input_features"] == {"bid": 0.25, "age": 34.0, "state": "CA"}


def test_payload_carries_all_feature_contributions(monkeypatch):
    """The result exposes the spec fields: base_prediction, prediction, and
    the full feature_contributions list (not just the top-N view)."""

    def _fake_shap(model, mif, lead_type_id, recommended_bid=None, top_n=None):
        return {
            "base_prediction": 0.11,
            "prediction": 0.78,
            "feature_contributions": [
                {"feature": "bid", "value": 8.25, "contribution": 0.13},
                {"feature": "age", "value": 34.0, "contribution": -0.04},
            ],
            "top_factors": [
                {
                    "feature": "bid",
                    "value": 8.25,
                    "shap": 0.13,
                    "direction": "increased",
                }
            ],
            "base_win_rate": 0.11,
        }

    monkeypatch.setattr(shap_explain, "explain_prepared_row", _fake_shap)

    out = se.explain_from_prediction(object(), _prediction(), with_llm=False)

    assert out["base_prediction"] == 0.11
    # prediction is reconciled to the served win rate (0.34), not the fake's 0.78
    assert out["prediction"] == 0.34
    assert [c["feature"] for c in out["feature_contributions"]] == ["bid", "age"]
    assert out["feature_contributions"][0]["contribution"] == 0.13
    # compat view still present
    assert out["top_factors"][0]["feature"] == "bid"


def test_prediction_reconciled_to_calibrated_served_win_rate(monkeypatch):
    """`prediction` echoes the served (calibrated) win rate, not the SHAP
    reconstruction, so it matches the logged column."""

    def _fake_shap(model, mif, lead_type_id, recommended_bid=None, top_n=None):
        return {
            "base_prediction": 0.11,
            "prediction": 0.90,  # SHAP-reconstructed (uncalibrated) -- must NOT win
            "feature_contributions": [],
            "top_factors": [],
            "base_win_rate": 0.11,
        }

    monkeypatch.setattr(shap_explain, "explain_prepared_row", _fake_shap)

    # served/calibrated win rate is 0.34 on the prediction
    out = se.explain_from_prediction(object(), _prediction(), with_llm=False)
    assert out["prediction"] == 0.34


def test_prediction_falls_back_to_shap_when_no_served_value(monkeypatch):
    """With no served win rate (raw-lead dev path), fall back to the SHAP
    reconstruction."""

    def _fake_shap(model, mif, lead_type_id, recommended_bid=None, top_n=None):
        return {
            "base_prediction": 0.11,
            "prediction": 0.72,
            "feature_contributions": [],
            "top_factors": [],
            "base_win_rate": 0.11,
        }

    monkeypatch.setattr(shap_explain, "explain_prepared_row", _fake_shap)

    pred = _prediction(recommended_bid_predicted_win_rate=None)
    out = se.explain_from_prediction(object(), pred, with_llm=False)
    assert out["prediction"] == 0.72


def test_cold_start_payload_has_spec_keys(monkeypatch):
    """Even with no model, the payload keeps the spec shape (empty)."""
    monkeypatch.setattr(shap_explain, "explain_prepared_row", lambda *a, **k: {})
    out = se.explain_from_prediction(None, _prediction(), with_llm=False)
    assert out["feature_contributions"] == []
    assert out["base_prediction"] is None
    assert out["prediction"] is None


def test_no_llm_when_disabled(monkeypatch):
    """with_llm=False returns SHAP only and never calls the LLM."""
    monkeypatch.setattr(
        shap_explain,
        "explain_prepared_row",
        lambda *a, **k: {"top_factors": [], "base_win_rate": 0.1},
    )

    def _boom(prompt):
        raise AssertionError("LLM must not be called when with_llm=False")

    monkeypatch.setattr(llm_explain, "call_ollama", _boom)

    out = se.explain_from_prediction(object(), _prediction(), with_llm=False)
    assert "explanation" not in out
    assert out["base_win_rate"] == 0.1


def test_cold_start_skips_shap(monkeypatch):
    """model=None (cold start) returns the policy reason, never runs SHAP."""

    def _boom(*a, **k):
        raise AssertionError("SHAP must not run for a cold-start prediction")

    monkeypatch.setattr(shap_explain, "explain_prepared_row", _boom)

    out = se.explain_from_prediction(
        None, _prediction(decision_reason="cold start reason"), with_llm=True
    )
    assert out["top_factors"] == []
    assert out["base_win_rate"] is None
    assert out["explanation"] == "cold start reason"


def test_no_viable_bid_skips_shap(monkeypatch):
    """A None recommended_bid (no viable bid) skips SHAP and returns the reason."""

    def _boom(*a, **k):
        raise AssertionError("SHAP must not run when there's no viable bid")

    monkeypatch.setattr(shap_explain, "explain_prepared_row", _boom)

    out = se.explain_from_prediction(
        object(),
        _prediction(recommended_bid=None, decision_reason="no viable bid"),
        with_llm=True,
    )
    assert out["top_factors"] == []
    assert out["explanation"] == "no viable bid"
