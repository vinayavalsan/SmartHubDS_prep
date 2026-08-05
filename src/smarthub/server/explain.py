"""Explanation surface (server-side).

Produces SHAP factor breakdowns and optional LLM narratives. The production
entry point, :func:`explain_from_prediction`, **consumes** an already-computed
prediction — its prepared feature vector and its chosen bid — and never re-runs
the bid decision. So an explanation always corresponds to the bid that was
logged and to the model version that served it, and the optimizer sweep isn't
duplicated on the explain path.

The reusable SHAP / LLM logic lives in ``train_and_predict`` (``shap_explain`` /
``llm_explain``); this module only orchestrates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from smarthub.train_and_predict import llm_explain, shap_explain


@dataclass
class PredictionOutput:
    """What an explanation needs from an already-computed prediction.

    Built in-memory by ``/recommend_bid`` (from the ``decide_bid`` result) or
    reconstructed from a persisted prediction-log row by ``/explain_bid`` — the
    field names mirror the prediction-log columns. No raw lead / re-derivation
    and no bid recomputation are involved.
    """

    lead_type_id: int
    model_input_features: dict  # the prepared row the model scored
    recommended_bid: float | None = None
    recommended_bid_predicted_win_rate: float | None = None
    recommended_bid_predicted_profit: float | None = None
    decision_path: str | None = None
    decision_reason: str | None = None
    expected_revenue: float | None = None
    min_bid: float | None = None
    max_bid: float | None = None
    bid_step: float | None = None
    prediction_id: str | None = None
    lead_ping_id: int | None = None


def _no_viable_bid(prediction: PredictionOutput) -> bool:
    return prediction.recommended_bid is None or pd.isna(prediction.recommended_bid)


def explain_from_prediction(
    model,
    prediction: PredictionOutput,
    *,
    with_llm: bool = True,
    with_bid_curve: bool = False,
) -> dict:
    """Explain an already-computed prediction. Never re-runs the bid decision.

    Inputs
    ------
    model : fitted model | None
        The model that served the prediction (loaded by its logged version).
        ``None`` (true cold start) or a no-viable-bid prediction skips SHAP.
    prediction : PredictionOutput
        The already-computed prediction to explain.
    with_llm : bool
        Also produce the plain-English LLM narrative (heavier; off for the
        per-bid background enrichment, on for ``/explain_bid``).
    with_bid_curve : bool
        Also compute predicted win rate/profit at a few nearby bids (needs the
        model; illustrative only).

    Returns
    -------
    dict
        ``top_factors`` / ``base_win_rate`` (+ ``bid_curve`` / ``explanation``
        when requested). For cold start / no-viable-bid, factors are empty and
        ``explanation`` carries the policy's own reason.
    """
    if model is None or _no_viable_bid(prediction):
        reason = prediction.decision_reason or (
            "No viable bid: nothing for the model to explain."
        )
        # Keep the full spec shape even when empty, so every logged prediction
        # carries a `feature_contributions` key (empty here) plus a reason.
        return {
            "base_prediction": None,
            "prediction": None,
            "feature_contributions": [],
            "top_factors": [],
            "base_win_rate": None,
            "explanation": reason,
        }

    factors = shap_explain.explain_prepared_row(
        model,
        prediction.model_input_features,
        prediction.lead_type_id,
        recommended_bid=prediction.recommended_bid,
    )
    # `prediction` is reconciled to the *calibrated* win rate actually served
    # and logged (`recommended_bid_predicted_win_rate`), so the payload's
    # headline number matches the log's own column -- no two-figure mismatch.
    # The SHAP-reconstructed (uncalibrated) win rate is used only as a fallback
    # when no served value is available (e.g. the raw-lead dev path). Note the
    # `feature_contributions` + `base_prediction` sum to the *uncalibrated*
    # model output, so they won't exactly reconstruct a calibrated `prediction`.
    served_win_rate = prediction.recommended_bid_predicted_win_rate
    result: dict[str, Any] = {
        # Spec fields: base rate, final predicted win rate, and EVERY feature's
        # contribution -- the canonical stored payload.
        "base_prediction": factors.get("base_prediction", factors.get("base_win_rate")),
        "prediction": (
            served_win_rate
            if served_win_rate is not None
            else factors.get("prediction")
        ),
        "feature_contributions": factors.get("feature_contributions", []),
        # Compatibility view: top-N ranked factors + base rate, kept for
        # /explain_bid consumers and existing tooling.
        "top_factors": factors["top_factors"],
        "base_win_rate": factors["base_win_rate"],
    }

    bid_curve = None
    if with_bid_curve:
        bid_curve = _bid_curve(model, prediction)
        result["bid_curve"] = bid_curve

    if with_llm:
        facts = {
            "recommended_bid": prediction.recommended_bid,
            "predicted_win_rate": prediction.recommended_bid_predicted_win_rate,
            "expected_profit": prediction.recommended_bid_predicted_profit,
            "base_win_rate": factors["base_win_rate"],
            "top_factors": factors["top_factors"],
            "bid_curve": bid_curve or [],
        }
        if prediction.decision_path and prediction.decision_path != "model":
            facts["decision_note"] = prediction.decision_reason
        result["explanation"] = llm_explain.call_ollama(
            llm_explain.format_llm_prompt(facts)
        )
    return result


def _bid_curve(model, prediction: PredictionOutput) -> list:
    """Nearby-bid win-rate/profit curve, built from the prepared row.

    Lazy imports keep this module free of a load-time dependency on
    ``server.predict`` (which imports this module) and on ``config``.
    """
    if prediction.expected_revenue is None or prediction.max_bid is None:
        return []
    from smarthub.server import predict
    from smarthub.train_and_predict import config

    numeric, categorical = config.feature_columns(prediction.lead_type_id)
    feature_cols = list(numeric) + list(categorical)
    row = pd.Series(
        {col: prediction.model_input_features.get(col) for col in feature_cols}
    )
    return predict.bid_curve_around(
        row=row,
        model=model,
        expected_revenue=prediction.expected_revenue,
        min_bid=prediction.min_bid if prediction.min_bid is not None else 0.25,
        max_bid=prediction.max_bid,
        bid_step=prediction.bid_step if prediction.bid_step is not None else 0.25,
        center_bid=prediction.recommended_bid,
    )
