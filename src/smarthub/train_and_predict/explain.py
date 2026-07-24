"""Offline/on-demand explanations for Anton's bid decisions.

NOT part of the live bidding path (`predict.optimize_bid_for_row`,
`predict.run_bid_optimizer_evaluation`) — this is a separate, slower path a
human calls on demand for one lead at a time, e.g. via the `/explain_bid`
endpoint:

    lead features -> SHAP (which factors moved the win-probability
    prediction, and by how much, for THIS lead) -> a small local LLM (via
    Ollama) turns that numeric breakdown into a couple of plain-English
    sentences.

This module is the thin orchestrator only (``explain_bid``) — the two
pipeline stages themselves live in their own files, split out 2026-07-24 for
readability (no logic changed, only which file each piece lives in):

    - ``shap_explain.py`` — the SHAP factor breakdown (``explain_row`` and
      its helpers).
    - ``llm_explain.py`` — the LLM prompt template, Ollama calls, and the
      model-pull/dedup infrastructure (``format_llm_prompt``, ``call_ollama``,
      ``ensure_model_pulled_async``, ...).

Everything from both is re-imported here (rather than referenced as
``shap_explain.foo`` / ``llm_explain.foo``) so that ``explain_bid`` keeps
resolving them as plain module-level names — this preserves both this
module's public surface (``explain.explain_row``, ``explain.call_ollama``,
``explain.TOP_N_FACTORS``, etc. all still work) and existing
``monkeypatch.setattr(explain, "name", ...)``-based tests, which patch names
on *this* module's namespace and rely on ``explain_bid`` looking them up here
rather than on the sub-modules directly.
"""

from __future__ import annotations

import pandas as pd

from smarthub.server import predict

from . import preprocessing
from .llm_explain import (  # noqa: F401 -- re-exported, see module docstring
    _OLLAMA_PULL_LOCK_PATH,
    DEFAULT_OLLAMA_HOST,
    LLM_MODEL,
    LLM_TIMEOUT_SECONDS,
    OLLAMA_HOST,
    _ensure_model_pulled_locked,
    _ensure_model_pulled_sync,
    call_ollama,
    ensure_model_pulled_async,
    format_llm_prompt,
    is_model_pulled,
    logger,
    pull_model,
)
from .shap_explain import (  # noqa: F401 -- re-exported, see module docstring
    TOP_N_FACTORS,
    _fitted_lgbm_estimators,
    _to_native,
    explain_row,
)


def explain_bid(
    model,
    record,
    lead_type_id,
    expected_revenue,
    manifest=None,
    target_cm=0.25,
    min_bid=0.25,
    bid_step=0.25,
    created_dayofweek=None,
    created_hour=None,
):
    """Full "why did Anton bid $X for this lead" explanation for one lead.

    Runs the lead through ``predict.decide_bid`` — the same cold-start
    fallback / scheduled exploration / normal-optimizer policy
    ``/recommend_bid`` uses — so the bid and ``decision_path`` shown here
    always match what live serving would do for the same inputs. When the
    bid came from the model, adds a SHAP-based factor breakdown for its
    prediction *at that specific bid* and asks a local LLM to write the
    whole thing up in plain English (including *why* a bid was a fallback or
    an exploration probe, not just the winning price).

    ``model``/``manifest`` should come from
    ``predict.load_model_and_manifest`` — ``model=None`` means true cold
    start (nothing ever promoted for this lead type); SHAP needs a fitted
    estimator, so that path skips straight to a canned, fully deterministic
    explanation (the policy itself already says everything there is to say).

    Returns
    -------
    dict
        Everything from ``decide_bid`` (``recommended_bid``,
        ``decision_path``, ``decision_reason``, ``model_data_age_days``, …)
        plus ``base_win_rate``, ``top_factors``, ``bid_curve`` (predicted
        win rate/profit at a few nearby bids — see
        ``predict.bid_curve_around``), and ``explanation`` (the LLM text, or
        a fallback message if the LLM is unreachable).
    """
    frame = preprocessing.serving_frame([record], lead_type_id)
    row = frame.iloc[0]

    decision = predict.decide_bid(
        row=row,
        model=model,
        manifest=manifest,
        expected_revenue=expected_revenue,
        target_cm=target_cm,
        min_bid=min_bid,
        bid_step=bid_step,
        created_dayofweek=(
            created_dayofweek
            if created_dayofweek is not None
            else record.get("created_dayofweek")
        ),
        created_hour=(
            created_hour if created_hour is not None else record.get("created_hour")
        ),
    )

    if pd.isna(decision["recommended_bid"]):
        return {
            **decision,
            "top_factors": [],
            "base_win_rate": None,
            "explanation": (
                "No viable bid: expected revenue is too low to bid anything "
                "at or above the minimum bid while still meeting the target "
                "margin."
            ),
        }

    if decision["decision_path"] == "cold_start_fallback":
        # No fitted model to run SHAP against, and there's nothing more to
        # say than the policy itself — skip SHAP + the LLM.
        return {
            **decision,
            "top_factors": [],
            "base_win_rate": None,
            "explanation": decision["decision_reason"],
        }

    # Explain the model's prediction AT the recommended bid specifically —
    # `record` may carry a placeholder bid (the optimizer sweeps it), so swap
    # in the actual chosen bid (post-exploration-perturbation, if any) before
    # computing SHAP values.
    explained_record = dict(record)
    explained_record["bid"] = decision["recommended_bid"]
    factors = explain_row(model, explained_record, lead_type_id)

    # "The shape of the market" around this bid (Kiran, docs/CONTEXT.md §7) —
    # a few nearby win-rate/profit points, not just the one chosen number.
    # Offline/explanatory only; never computed for /recommend_bid.
    bid_curve = predict.bid_curve_around(
        row=row,
        model=model,
        expected_revenue=expected_revenue,
        min_bid=min_bid,
        max_bid=decision["max_bid"],
        bid_step=bid_step,
        center_bid=decision["recommended_bid"],
    )

    facts = {
        "recommended_bid": decision["recommended_bid"],
        "predicted_win_rate": decision["recommended_bid_predicted_win_rate"],
        "expected_profit": decision["recommended_bid_predicted_profit"],
        "base_win_rate": factors["base_win_rate"],
        "top_factors": factors["top_factors"],
        "bid_curve": bid_curve,
    }
    if decision["decision_path"] != "model":
        facts["decision_note"] = decision["decision_reason"]
    explanation = call_ollama(format_llm_prompt(facts))

    return {
        **decision,
        "top_factors": factors["top_factors"],
        "base_win_rate": factors["base_win_rate"],
        "bid_curve": bid_curve,
        "explanation": explanation,
    }
