"""Offline/on-demand explanations for Anton's bid decisions.

NOT part of the live bidding path (`predict.optimize_bid_for_row`,
`predict.run_bid_optimizer_evaluation`) — this is a separate, slower path a
human calls on demand for one lead at a time, e.g. via the `/explain_bid`
endpoint:

    lead features -> SHAP (which factors moved the win-probability
    prediction, and by how much, for THIS lead) -> a small local LLM (via
    Ollama) turns that numeric breakdown into a couple of plain-English
    sentences.

The LLM never sees model internals and never computes anything — it only
formats facts it's handed, with an explicit "don't invent numbers"
instruction, since this is a formatting task, not a reasoning task, and small
models hallucinate numbers readily if given room to reason freely.

Heavy/optional deps (shap, lightgbm, requests) are imported lazily so the rest
of `train_and_predict` keeps working without the `explain` extra installed —
same pattern as `predict.py`'s lazy joblib/mlflow/fastapi imports.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from smarthub.core import task_config
from smarthub.server import predict

from . import config, preprocessing

DEFAULT_OLLAMA_HOST = "http://localhost:11434"

# Task config: smarthub.yaml `explain` section — used only by this module, not live
# bidding path, so it's kept local rather than in train_and_predict/config.py.
LLM_MODEL = task_config.get("explain", "llm_model", "qwen2.5:1.5b-instruct")
OLLAMA_HOST = task_config.get("explain", "ollama_host", DEFAULT_OLLAMA_HOST)
TOP_N_FACTORS = task_config.get_int("explain", "top_n_factors", 5)
LLM_TIMEOUT_SECONDS = task_config.get_float("explain", "timeout_seconds", 30.0)


def _fitted_lgbm_estimators(model):
    """Return the fitted (preprocessor, LGBMClassifier) pair(s) inside ``model``.

    Handles both a plain sklearn ``Pipeline`` (preprocessor + classifier, see
    ``models.build_lightgbm_model``) and a ``CalibratedClassifierCV`` wrapping
    one — isotonic calibration is a monotonic rescaling of the final
    probability, so it doesn't change *which* features mattered or their
    ranking, only the final number; that's why explaining the underlying
    LightGBM model(s) is still valid even though the served model is
    calibrated. When calibrated, there's one fitted pipeline per CV fold
    (``models.py``'s ``cv=3``) — all are returned so callers can average.

    Raises
    ------
    ValueError
        If ``model`` isn't (or doesn't wrap) a LightGBM pipeline — SHAP
        explanations here only support ``model_type=lightgbm`` for now.
    """
    from lightgbm import LGBMClassifier

    calibrated_classifiers = getattr(model, "calibrated_classifiers_", None)
    if calibrated_classifiers:
        pipelines = [
            getattr(cc, "estimator", None) or getattr(cc, "base_estimator", None)
            for cc in calibrated_classifiers
        ]
        pipelines = [p for p in pipelines if p is not None]
    else:
        pipelines = [model]

    pairs = []
    for pipe in pipelines:
        preprocessor = pipe.named_steps["preprocessor"]
        classifier = pipe.named_steps["classifier"]
        if not isinstance(classifier, LGBMClassifier):
            raise ValueError(
                "SHAP explanations currently support model_type='lightgbm' "
                f"only (got {type(classifier).__name__}). Train an LGBM "
                "model to use /explain_bid for this lead type."
            )
        pairs.append((preprocessor, classifier))

    if not pairs:
        raise ValueError("Could not find a fitted LightGBM estimator in this model.")
    return pairs


def _shap_for_row(model, row_frame, feature_cols):
    """SHAP values for one row, averaged across calibration folds if any.

    ``shap.TreeExplainer`` on an ``LGBMClassifier`` works in **margin
    (log-odds) space**, not probability space: raw feature contributions and
    ``explainer.expected_value`` are unbounded reals that sum to the model's
    raw score, not a 0-1 probability (e.g. a base value of ``-2.0`` is a
    normal log-odds figure — ``sigmoid(-2.0) ≈ 0.12``). Per-feature
    contributions are left in log-odds units (their *sign* and *relative
    magnitude* — which factor mattered most — are unaffected by the
    sigmoid's monotonicity, so ranking/direction stay valid), but the base
    value is converted to an actual probability here since callers use it as
    "the model's average predicted win rate".

    Returns
    -------
    tuple[dict[str, float], float]
        ``(feature -> log-odds shap contribution, base_win_rate)`` where
        ``base_win_rate`` IS a 0-1 probability — the model's average
        predicted win rate before this lead's specific factors are applied.
    """
    import numpy as np
    import shap

    pairs = _fitted_lgbm_estimators(model)
    per_fold_shap = []
    base_values = []
    for preprocessor, classifier in pairs:
        transformed = preprocessor.transform(row_frame)
        explainer = shap.TreeExplainer(classifier)
        raw = explainer.shap_values(transformed)

        # SHAP's return shape for binary classifiers has changed across
        # versions: a [class0, class1] list (older), or a single array with a
        # trailing class axis (newer). Normalize to "one row of feature
        # contributions to the positive (win) class".
        if isinstance(raw, list):
            values = raw[1][0]
        elif np.asarray(raw).ndim == 3:
            values = np.asarray(raw)[0, :, 1]
        else:
            values = np.asarray(raw)[0]
        per_fold_shap.append(np.asarray(values, dtype=float))

        base = explainer.expected_value
        base = base[1] if isinstance(base, (list, np.ndarray)) else base
        base_values.append(float(base))

    avg_shap = np.mean(per_fold_shap, axis=0)
    avg_base_margin = float(np.mean(base_values))
    avg_base_win_rate = float(1.0 / (1.0 + np.exp(-avg_base_margin)))
    return dict(zip(feature_cols, avg_shap.tolist())), avg_base_win_rate


def _to_native(value):
    """Convert a numpy scalar (int64/float64/bool_) to a plain Python type.

    ``frame.iloc[0][name]`` returns numpy scalars for numeric columns, and
    some FastAPI/pydantic version combinations can't JSON-encode those (seen
    in the wild as ``TypeError: 'numpy.int64' object is not iterable`` from
    ``jsonable_encoder``) — cast explicitly rather than depend on the
    installed encoder's numpy support.
    """
    if isinstance(value, np.generic):
        return value.item()
    return value


def explain_row(model, record, lead_type_id, top_n=None):
    """Build the structured 'why' facts for one lead's win-probability score.

    Inputs
    ------
    model : fitted sklearn Pipeline or CalibratedClassifierCV
        Trained Anton model (LightGBM only — see ``_fitted_lgbm_estimators``).
    record : dict
        Raw lead attributes (same shape as ``BidRequest`` in ``predict.py``).
    lead_type_id : int
        6=auto, 1=home — selects the model feature schema.
    top_n : int | None
        How many top factors to keep; ``TOP_N_FACTORS`` (ini-configurable)
        when ``None``.

    Returns
    -------
    dict
        ``{"top_factors": [...], "base_win_rate": float}``. Each factor is
        ``{"feature", "value", "shap", "direction"}``, sorted by |shap|.
    """
    top_n = top_n or TOP_N_FACTORS
    numeric, categorical = config.feature_columns(lead_type_id)
    feature_cols = list(numeric) + list(categorical)

    frame = preprocessing.serving_frame([record], lead_type_id)
    shap_values, base_value = _shap_for_row(model, frame, feature_cols)

    ranked = sorted(shap_values.items(), key=lambda kv: abs(kv[1]), reverse=True)
    top_factors = [
        {
            "feature": name,
            "value": _to_native(frame.iloc[0][name]),
            "shap": round(value, 4),
            "direction": "increased" if value > 0 else "decreased",
        }
        for name, value in ranked[:top_n]
    ]
    return {"top_factors": top_factors, "base_win_rate": base_value}


def format_llm_prompt(facts: dict) -> str:
    """Render structured facts into a tightly-templated LLM prompt.

    Deliberately rigid (not "be creative") and explicit about not inventing
    numbers — this is a formatting task, not a reasoning task, so the prompt
    is written to minimize the model's room to hallucinate. An optional
    ``decision_note`` (from ``predict.decide_bid`` — e.g. "this was a
    scheduled exploration probe") is included as another fact the LLM may
    use, not something it has to guess at from the numbers alone.

    Two guardrails added after observing a small model's actual output on a
    real lead:
    - An explicit statement of the model's monotonic bid constraint, because
      without it the LLM sometimes reasoned backwards about the `bid`
      factor's SHAP sign (e.g. implying a *lower* bid would have won more
      often, which the model's design rules out).
    - An optional ``bid_curve`` (from ``predict.bid_curve_around``) — actual
      win-rate/profit numbers at nearby bids — so claims about "what a
      different bid would do" are grounded in real numbers instead of the
      LLM guessing from the single chosen bid alone.
    """
    lines = [
        "You explain a pricing model's decision in plain English for a "
        "business user.",
        "Use ONLY the facts below. Do not invent numbers. 2-3 sentences, no " "jargon.",
        "Model rule: predicted win rate never decreases as the bid rises "
        "(built into the model by design) -- never claim a lower bid would "
        "win more often than a higher one.",
        "",
        f"Recommended bid: ${facts['recommended_bid']:.2f}",
        f"Predicted win rate at this bid: {facts['predicted_win_rate']:.0%} "
        f"(vs. average {facts['base_win_rate']:.0%})",
        f"Expected profit: ${facts['expected_profit']:.2f}",
    ]
    if facts.get("decision_note"):
        lines += ["", f"Note: {facts['decision_note']}"]
    if facts.get("bid_curve"):
        lines += ["", "Nearby bids explored (bid -> predicted win rate):"]
        for point in facts["bid_curve"]:
            lines.append(f"- ${point['bid']:.2f} -> {point['predicted_win_rate']:.0%}")
    lines += ["", "Top factors:"]
    for f in facts["top_factors"]:
        lines.append(f"- {f['feature']}={f['value']}: {f['direction']} win likelihood")
    lines += ["", "Explanation:"]
    return "\n".join(lines)


def call_ollama(prompt, model=None, host=None, timeout=None):
    """Call a local Ollama model; returns the generated text.

    Best-effort: if Ollama isn't reachable (not installed / not running),
    this logs nothing scary and returns a clear fallback message instead of
    raising — an explanation feature must never break its caller, and it's
    not on the live bidding path anyway, so a degraded (but honest) response
    is the right failure mode.
    """
    import requests

    model = model or LLM_MODEL
    host = host or OLLAMA_HOST
    timeout = timeout or LLM_TIMEOUT_SECONDS
    try:
        resp = requests.post(
            f"{host}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=timeout,
        )
        resp.raise_for_status()
        return resp.json()["response"].strip()
    except Exception as exc:  # noqa: BLE001 - best-effort, never break the caller
        return (
            "(Explanation unavailable: could not reach the local LLM at "
            f"{host} — {exc}. The numeric factors above are still accurate.)"
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
        "expected_profit": decision["recommended_bid_expected_profit"],
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
