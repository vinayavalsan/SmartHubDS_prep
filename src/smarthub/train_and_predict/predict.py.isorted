"""Prediction and bid optimization for Anton.

The model predicts ``P(won | bid, lead features)``. The optimizer sweeps
candidate bids, predicts win rate at each, and picks the bid maximising
``expected_profit = P(win) * (expected_revenue - bid)``. ``expected_revenue`` is
NOT a model feature — it is used only in the objective.

Heavy/optional deps (joblib, mlflow, fastapi) are imported lazily so the pure
optimizer functions can be used and unit-tested without the full ``ml`` extra.
"""

from __future__ import annotations

import os
import warnings
from contextlib import contextmanager
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from . import config, preprocessing, registry


@contextmanager
def _quiet_feature_name_warning():
    """Silence sklearn's per-call "X does not have valid feature names" warning.

    Harmless (fit saw names, predict gets a transformed array), but the optimizer
    calls predict_proba thousands of times, so it floods the logs.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=".*does not have valid feature names.*"
        )
        yield


def resolve_model_uri(lead_type_id: int = 6) -> str:
    """Resolve which model artifact to load, for a given lead type.

    Priority (first that applies wins):
    1. The ``MODEL_URI`` env var — an explicit override (a ``.pkl`` path or an
       MLflow model URI), for pinning/emergency overrides regardless of what
       the registry says.
    2. ``smarthub.yaml`` ``prediction.active_model_version`` — an explicit
       version id (e.g. ``v3_2026-07-09T140501Z``) to pin this lead type to,
       without touching the serving pointer.
    3. The model **currently serving** this lead type
       (``data/models/<type>/current.json`` — see ``registry.py``). This is
       the normal case: whatever most recently passed the promotion gate in
       training.
    """
    env_override = os.getenv("MODEL_URI")
    if env_override:
        return env_override

    lead_type_name = config.lead_type_name(lead_type_id)

    pinned_version = config.active_model_version()
    if pinned_version:
        return str(registry.version_path(lead_type_name, pinned_version))

    path = registry.currently_serving_model_path(lead_type_name)
    if path is None:
        raise FileNotFoundError(
            f"Nothing is currently serving lead type '{lead_type_name}'. "
            "Train one first: python -m smarthub.train_and_predict.train "
            f"--lead-type-id {lead_type_id}"
        )
    return str(path)


def load_model(model_uri: str | None = None, lead_type_id: int = 6):
    """Load a trained Anton model from a local .pkl or an MLflow URI.

    ``model_uri`` is optional — if omitted, resolves via ``resolve_model_uri``
    (env override -> pinned version -> currently-serving model) for
    ``lead_type_id``.
    """
    uri = model_uri or resolve_model_uri(lead_type_id)
    if str(uri).endswith(".pkl"):
        import joblib

        return joblib.load(uri)

    import mlflow.sklearn

    return mlflow.sklearn.load_model(uri)


def load_model_and_manifest(lead_type_id: int = 6):
    """Resolve + load the serving model AND its manifest for one lead type.

    Same resolution priority as ``resolve_model_uri`` (env override -> pinned
    version -> currently-serving), but returns ``(None, None)`` instead of
    raising when nothing has ever been trained/promoted for this lead type —
    that's the true cold-start case ``decide_bid`` handles explicitly (a
    defined fallback bid), not an error condition. Same graceful-``None``
    convention as ``registry.load_currently_serving_model``.
    """
    lead_type_name = config.lead_type_name(lead_type_id)
    try:
        uri = resolve_model_uri(lead_type_id)
    except FileNotFoundError:
        return None, None

    model = load_model(uri)
    version = config.active_model_version() or registry.currently_serving_version(
        lead_type_name
    )
    manifest = None
    if version:
        try:
            manifest = registry.load_manifest(lead_type_name, version)
        except FileNotFoundError:
            manifest = None
    return model, manifest


def _empty_result(max_bid: float) -> dict:
    return {
        "recommended_bid": np.nan,
        "recommended_bid_predicted_win_rate": np.nan,
        "recommended_bid_expected_profit": np.nan,
        "max_bid": max_bid,
        "n_candidate_bids": 0,
    }


def optimize_bid_for_row(row, model, expected_revenue, target_cm, min_bid, bid_step):
    """Return the profit-maximising bid for one lead using the win model."""
    if pd.isna(expected_revenue) or expected_revenue <= 0:
        return _empty_result(np.nan)

    max_bid = expected_revenue * (1 - target_cm)
    if max_bid < min_bid:
        return _empty_result(max_bid)

    candidate_bids = np.arange(min_bid, max_bid + bid_step, bid_step)
    candidate_bids = candidate_bids[candidate_bids <= max_bid]
    if len(candidate_bids) == 0:
        return _empty_result(float(max_bid))

    candidate_rows = pd.DataFrame([row.to_dict()] * len(candidate_bids))
    candidate_rows["bid"] = candidate_bids

    with _quiet_feature_name_warning():
        predicted_win_rates = model.predict_proba(candidate_rows)[:, 1]
    expected_profits = predicted_win_rates * (expected_revenue - candidate_bids)
    best_idx = int(np.argmax(expected_profits))

    return {
        "recommended_bid": float(candidate_bids[best_idx]),
        "recommended_bid_predicted_win_rate": float(predicted_win_rates[best_idx]),
        "recommended_bid_expected_profit": float(expected_profits[best_idx]),
        "max_bid": float(max_bid),
        "n_candidate_bids": int(len(candidate_bids)),
    }


def bid_curve_around(
    row, model, expected_revenue, min_bid, max_bid, bid_step, center_bid, n_points=3
) -> list[dict]:
    """Predicted win rate + expected profit at a few bid points near ``center_bid``.

    Explanatory/offline use only (see ``explain.py``) — NOT part of the live
    `/recommend_bid` path, which only needs the single chosen bid. This is
    for showing a human "the shape of the market" Anton is exploring around
    the chosen bid (Kiran's framing, docs/CONTEXT.md §7: "probe *around its
    own optimum*... to gather real data on the shape of the market"), not
    just the one winning number — e.g. whether win rate is still climbing
    steeply near this bid, or has hit a "shelf" (docs/CONTEXT.md glossary:
    "a price point where win rate barely moves").

    Returns
    -------
    list[dict]
        ``[{"bid", "predicted_win_rate", "expected_profit"}, ...]``, sorted
        by bid, for up to ``2 * n_points + 1`` grid points spanning
        ``center_bid ± n_points * bid_step`` (clipped to ``[min_bid, max_bid]``
        and de-duplicated at the edges).
    """
    if max_bid < min_bid or pd.isna(center_bid):
        return []
    offsets = np.arange(-n_points, n_points + 1) * bid_step
    bids = np.clip(center_bid + offsets, min_bid, max_bid)
    bids = np.unique(np.round(bids, 6))

    candidate_rows = pd.DataFrame([row.to_dict()] * len(bids))
    candidate_rows["bid"] = bids
    with _quiet_feature_name_warning():
        win_rates = model.predict_proba(candidate_rows)[:, 1]
    profits = win_rates * (expected_revenue - bids)

    return [
        {
            "bid": float(b),
            "predicted_win_rate": float(w),
            "expected_profit": float(p),
        }
        for b, w, p in zip(bids, win_rates, profits)
    ]


# --- Cold-start / exploration bidding policy ---------------------------------
#
# Kiran's explainability + market-dynamics ask (docs/CONTEXT.md §3): "recent"
# must be an explicit, named config value, and when there's no recent data
# the bidding pattern must be "explicitly articulated... e.g. a defined
# exploration schedule or fallback bid" rather than emergent/chaotic
# behaviour. `decide_bid` below is the one bidding decision that ties the
# three cases together: a brand-new lead type with no model yet (cold-start
# fallback), a scheduled market-exploration probe around the optimum, or the
# normal profit-maximizing bid — always with an explicit, auditable
# `decision_path` + `decision_reason` on the result.


def _snap_to_bid_grid(
    bid: float, min_bid: float, max_bid: float, bid_step: float
) -> float:
    """Snap an arbitrary bid onto the same ``{min_bid, min_bid+step, ...}``
    grid ``optimize_bid_for_row`` sweeps, clipped to ``[min_bid, max_bid]``."""
    if max_bid < min_bid:
        return float(min_bid)
    steps = round((bid - min_bid) / bid_step)
    snapped = min_bid + steps * bid_step
    return float(min(max(snapped, min_bid), max_bid))


def cold_start_fallback_bid(
    expected_revenue, target_cm, min_bid, bid_step, fallback_pct=None
) -> dict:
    """A defined bid for a lead type with NO model ever trained/promoted yet.

    This is the true cold-start case — a brand-new partner/lead type, not
    just "the model is old" (see ``model_recency`` for that). Bids a fixed,
    configurable fraction of the way from ``min_bid`` to the CM-respecting
    ceiling (``[prediction] cold_start_fallback_bid_pct``, default 50%),
    snapped to the same bid grid the optimizer uses. Self-terminating: the
    first model trained for this lead type promotes unconditionally
    (``registry.decide_promotion``'s bootstrap case), so this path stops
    being reachable once that happens.
    """
    fallback_pct = (
        config.COLD_START_FALLBACK_BID_PCT if fallback_pct is None else fallback_pct
    )
    if pd.isna(expected_revenue) or expected_revenue <= 0:
        return _empty_result(np.nan)
    max_bid = expected_revenue * (1 - target_cm)
    if max_bid < min_bid:
        return _empty_result(max_bid)
    raw_bid = min_bid + (max_bid - min_bid) * fallback_pct
    bid = _snap_to_bid_grid(raw_bid, min_bid, max_bid, bid_step)
    return {
        "recommended_bid": bid,
        "recommended_bid_predicted_win_rate": None,
        "recommended_bid_expected_profit": None,
        "max_bid": float(max_bid),
        "n_candidate_bids": None,
    }


def _exploration_hour_of_week(created_dayofweek, created_hour) -> int:
    """0-167 index for the hour-of-week (``dayofweek*24 + hour``)."""
    return int(created_dayofweek) * 24 + int(created_hour)


def exploration_slot(created_dayofweek, created_hour, variance_pct=None):
    """Deterministic, auditable "explore vs. exploit" schedule.

    Kiran (docs/CONTEXT.md §3/§7): Anton must probe *around its own optimum*
    with "deliberate variability", on a **defined schedule** — not a random
    per-request coin flip, which can't be reproduced or explained after the
    fact. This buckets every lead by hour-of-week (0-167, from
    ``created_dayofweek``/``created_hour`` already on every request) and
    marks 1-in-``N`` buckets as scheduled explore slots, where
    ``N = round(1 / exploration_variance_pct)`` — so the ini's existing
    ``exploration_variance_pct`` (previously unused) both sizes the probe
    and sets the schedule density. Probe direction alternates (above/below
    the optimum) each time a bucket triggers, so exploration samples both
    sides of the price curve over time.

    Returns
    -------
    tuple[bool, int | None]
        ``(is_explore_slot, direction)`` — ``direction`` is ``+1`` (probe
        above the optimum) or ``-1`` (probe below); ``None`` when this isn't
        a scheduled explore slot.
    """
    variance_pct = (
        config.EXPLORATION_VARIANCE_PCT if variance_pct is None else variance_pct
    )
    if not variance_pct or variance_pct <= 0:
        return False, None
    n = max(1, round(1 / variance_pct))
    bucket = _exploration_hour_of_week(created_dayofweek, created_hour)
    if bucket % n != 0:
        return False, None
    direction = 1 if (bucket // n) % 2 == 0 else -1
    return True, direction


def model_recency(manifest, recency_window_days=None, now=None):
    """How stale is a model, per the explicitly-defined recency window?

    Kiran (docs/CONTEXT.md §3): "'Recent' must be explicitly defined as a
    configurable window... not buried in code." Reads the manifest's
    ``lineage.data_max_created_at`` (the newest row it was trained on, set in
    ``train.run_training``) and compares its age to
    ``[prediction] recency_window_days``. Informational — a stale model still
    bids normally, it's just flagged in the decision reason for
    monitoring/retraining-cadence calls.

    Returns
    -------
    tuple[float | None, bool]
        ``(age_days, is_stale)`` — ``(None, False)`` if there's no manifest or
        its lineage lacks a data date (e.g. an older artifact).
    """
    recency_window_days = (
        config.RECENCY_WINDOW_DAYS
        if recency_window_days is None
        else recency_window_days
    )
    if not manifest:
        return None, False
    data_max = (manifest.get("lineage") or {}).get("data_max_created_at")
    if not data_max:
        return None, False
    data_max_dt = pd.Timestamp(data_max)
    if data_max_dt.tzinfo is None:
        data_max_dt = data_max_dt.tz_localize("UTC")
    now_dt = pd.Timestamp(now or datetime.now(timezone.utc))
    age_days = (now_dt - data_max_dt).total_seconds() / 86400
    return round(age_days, 1), age_days > recency_window_days


def decide_bid(
    row,
    model,
    manifest,
    expected_revenue,
    target_cm,
    min_bid,
    bid_step,
    created_dayofweek=None,
    created_hour=None,
) -> dict:
    """The one bidding decision Anton makes for a lead.

    Cold-start fallback (no model yet) > scheduled exploration probe > the
    normal profit-maximizing bid — always with an explicit, auditable
    ``decision_path`` (``"cold_start_fallback" | "exploration" | "model"``)
    and ``decision_reason`` string on the result, so "why did a bid come out
    the way it did" is always answerable, not just a trusted score.

    ``model``/``manifest`` should come from ``load_model_and_manifest`` —
    ``model=None`` means nothing has ever been promoted for this lead type
    (true cold start; see ``cold_start_fallback_bid``).
    """
    if model is None:
        result = cold_start_fallback_bid(expected_revenue, target_cm, min_bid, bid_step)
        result["model_data_age_days"] = None
        result["decision_path"] = "cold_start_fallback"
        result["decision_reason"] = (
            "No model has ever been trained/promoted for this lead type yet "
            "(cold start). Bid set to "
            f"{config.COLD_START_FALLBACK_BID_PCT:.0%} of the way from the "
            "minimum bid to the CM-respecting ceiling, per the defined "
            "cold-start policy (config/smarthub.yaml "
            "prediction.cold_start_fallback_bid_pct)."
        )
        return result

    age_days, is_stale = model_recency(manifest)
    normal = optimize_bid_for_row(
        row, model, expected_revenue, target_cm, min_bid, bid_step
    )
    normal["model_data_age_days"] = age_days

    if pd.isna(normal["recommended_bid"]):
        normal["decision_path"] = "model"
        normal["decision_reason"] = (
            "No viable bid: expected revenue is too low to clear the target "
            "margin at or above the minimum bid."
        )
        return normal

    explore, direction = False, None
    if created_dayofweek is not None and created_hour is not None:
        explore, direction = exploration_slot(created_dayofweek, created_hour)

    if not explore:
        normal["decision_path"] = "model"
        stale_note = (
            " (note: currently-serving model's training data is "
            f"{age_days:.0f} days old, past the {config.RECENCY_WINDOW_DAYS}-"
            "day recency window — due for retraining)"
            if is_stale
            else ""
        )
        normal["decision_reason"] = (
            "Standard profit-maximizing bid from the currently-serving model."
            + stale_note
        )
        return normal

    optimum_bid = normal["recommended_bid"]
    variance_pct = config.EXPLORATION_VARIANCE_PCT
    perturbed_bid = _snap_to_bid_grid(
        optimum_bid * (1 + direction * variance_pct),
        min_bid, normal["max_bid"], bid_step,
    )
    candidate_row = pd.DataFrame([row.to_dict()])
    candidate_row["bid"] = perturbed_bid
    with _quiet_feature_name_warning():
        win_rate = float(model.predict_proba(candidate_row)[:, 1][0])
    profit = win_rate * (expected_revenue - perturbed_bid)

    result = dict(normal)
    result["recommended_bid"] = float(perturbed_bid)
    result["recommended_bid_predicted_win_rate"] = win_rate
    result["recommended_bid_expected_profit"] = float(profit)
    result["decision_path"] = "exploration"
    result["pre_exploration_optimum_bid"] = float(optimum_bid)
    result["decision_reason"] = (
        "Scheduled exploration probe: bid "
        f"{'above' if direction > 0 else 'below'} the profit-maximizing "
        f"optimum by {variance_pct:.0%} (optimum was ${optimum_bid:.2f}) to "
        "keep learning the market's shape at nearby price points, per the "
        "defined exploration schedule (config/smarthub.yaml "
        "prediction.exploration_variance_pct)."
    )
    return result


def run_bid_optimizer_evaluation(
    test_eval_df,
    model,
    feature_cols,
    target_cm=0.25,
    min_bid=0.25,
    bid_step=0.25,
):
    """Compare current-bid predictions vs. optimized candidate-bid choices.

    These are **offline predicted** metrics (model's predicted win rates), not
    measured production lift. Returns ``(eval_df, summary)`` or ``None`` when the
    data can't support the evaluation.
    """
    print("=" * 80)
    print("Bid Optimization Evaluation")

    if config.REVENUE_COL not in test_eval_df.columns:
        print(f"Skipped: {config.REVENUE_COL} is not available in this dataset.")
        return None

    eval_df = test_eval_df.dropna(subset=[config.REVENUE_COL, "bid"]).copy()
    eval_df = eval_df[eval_df[config.REVENUE_COL] > 0].copy()
    if eval_df.empty:
        print("Skipped: no rows with positive expected_revenue.")
        return None

    feature_cols = list(feature_cols)
    with _quiet_feature_name_warning():
        current_win_rate = model.predict_proba(eval_df[feature_cols])[:, 1]
    eval_df["current_bid_predicted_win_rate"] = current_win_rate
    eval_df["current_bid_expected_profit"] = current_win_rate * (
        eval_df[config.REVENUE_COL] - eval_df["bid"]
    )

    optimizer_rows = [
        optimize_bid_for_row(
            row=row,
            model=model,
            expected_revenue=row[config.REVENUE_COL],
            target_cm=target_cm,
            min_bid=min_bid,
            bid_step=bid_step,
        )
        for _, row in eval_df[feature_cols + [config.REVENUE_COL]].iterrows()
    ]
    optimizer_df = pd.DataFrame(optimizer_rows, index=eval_df.index)
    eval_df = pd.concat([eval_df, optimizer_df], axis=1)
    eval_df = eval_df.dropna(subset=["recommended_bid"]).copy()
    if eval_df.empty:
        print("Skipped: optimizer could not create candidate bids.")
        return None

    eval_df["expected_profit_lift"] = (
        eval_df["recommended_bid_expected_profit"]
        - eval_df["current_bid_expected_profit"]
    )
    eval_df["bid_change"] = eval_df["recommended_bid"] - eval_df["bid"]
    eval_df["predicted_win_rate_lift"] = (
        eval_df["recommended_bid_predicted_win_rate"]
        - eval_df["current_bid_predicted_win_rate"]
    )
    eval_df["recommended_bid_cm_if_won"] = (
        eval_df[config.REVENUE_COL] - eval_df["recommended_bid"]
    ) / eval_df[config.REVENUE_COL]

    eval_df["bid_change_direction"] = "unchanged"
    eval_df.loc[eval_df["bid_change"] > 0, "bid_change_direction"] = "increased"
    eval_df.loc[eval_df["bid_change"] < 0, "bid_change_direction"] = "decreased"

    direction_counts = (
        eval_df["bid_change_direction"]
        .value_counts()
        .reindex(["increased", "decreased", "unchanged"], fill_value=0)
    )

    current_total = float(eval_df["current_bid_expected_profit"].sum())
    recommended_total = float(eval_df["recommended_bid_expected_profit"].sum())
    n = len(eval_df)

    summary = {
        "optimizer_rows": int(n),
        "target_cm": float(target_cm),
        "min_bid": float(min_bid),
        "bid_step": float(bid_step),
        "current_bid_total_expected_profit": current_total,
        "recommended_bid_total_expected_profit": recommended_total,
        "expected_profit_lift_total": float(eval_df["expected_profit_lift"].sum()),
        "expected_profit_lift_pct": (
            (recommended_total - current_total) / current_total
            if current_total
            else float("nan")
        ),
        "avg_current_bid_predicted_win_rate": float(
            eval_df["current_bid_predicted_win_rate"].mean()
        ),
        "avg_recommended_bid_predicted_win_rate": float(
            eval_df["recommended_bid_predicted_win_rate"].mean()
        ),
        "avg_predicted_win_rate_lift": float(eval_df["predicted_win_rate_lift"].mean()),
        "bid_increase_count": int(direction_counts["increased"]),
        "bid_decrease_count": int(direction_counts["decreased"]),
        "bid_unchanged_count": int(direction_counts["unchanged"]),
        "bid_increase_pct": float(direction_counts["increased"] / n * 100),
        "bid_decrease_pct": float(direction_counts["decreased"] / n * 100),
        "bid_unchanged_pct": float(direction_counts["unchanged"] / n * 100),
        "avg_recommended_bid_cm_if_won": float(
            eval_df["recommended_bid_cm_if_won"].mean()
        ),
        "median_recommended_bid_cm_if_won": float(
            eval_df["recommended_bid_cm_if_won"].median()
        ),
    }

    print(f"Rows evaluated:                {summary['optimizer_rows']:,}")
    print(f"Target CM:                     {summary['target_cm']:.2%}")
    print(
        "Expected profit (current):     "
        f"{summary['current_bid_total_expected_profit']:.4f}"
    )
    print(
        "Expected profit (recommended): "
        f"{summary['recommended_bid_total_expected_profit']:.4f}"
    )
    print(f"Expected profit lift %:        {summary['expected_profit_lift_pct']:.2%}")
    print(
        "Bid up/down/same:              "
        f"{summary['bid_increase_count']:,} / "
        f"{summary['bid_decrease_count']:,} / "
        f"{summary['bid_unchanged_count']:,}"
    )
    print(
        "Avg recommended CM if won:     "
        f"{summary['avg_recommended_bid_cm_if_won']:.4f}"
    )
    return eval_df, summary


# --- Serving API (optional; requires the `ml` extra: fastapi + pydantic) -----
try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel, Field

    _FASTAPI_AVAILABLE = True
except ImportError:  # pragma: no cover - API is optional
    _FASTAPI_AVAILABLE = False


if _FASTAPI_AVAILABLE:

    class BidRequest(BaseModel):
        """Request schema for a bid recommendation."""

        expected_revenue: float = Field(..., gt=0)
        target_cm: float = Field(0.25, ge=0, lt=1)
        min_bid: float = Field(0.25, ge=0)
        bid_step: float = Field(0.25, gt=0)

        campaign_id: int
        account_id: int | None = None
        source_type_id: int | None = None
        lead_type_id: int = 6
        created_hour: int = Field(..., ge=0, le=23)
        created_dayofweek: int = Field(..., ge=0, le=6)

        state: str | None = None
        insured: str | None = None
        home_owner: str | None = None
        dui: str | None = None
        sr22_required: str | None = None
        military_affiliation: str | None = None
        gender: str | None = None
        marital_status: str | None = None

        num_vehicles: float | None = None
        num_drivers: float | None = None
        num_auto_violations: float | None = None
        num_auto_accidents: float | None = None
        continuous_coverage_months: float | None = None
        age: float | None = None

    app = FastAPI(title="Anton Bid Prediction API")

    @app.get("/health")
    def health(lead_type_id: int = 6):
        """Health check. Reports which model artifact would currently be served."""
        try:
            model_uri = resolve_model_uri(lead_type_id)
        except FileNotFoundError:
            model_uri = None
        return {
            "status": "ok",
            "lead_type_id": lead_type_id,
            "model_uri": model_uri,
        }

    @app.post("/recommend_bid")
    def recommend_bid(request: BidRequest):
        """Recommend the bid for one lead under Anton's full bidding policy.

        Applies ``decide_bid`` — the normal profit-maximizing bid, a defined
        cold-start fallback when no model has ever been promoted for this
        lead type, or a scheduled market-exploration probe — not just the
        raw optimizer. ``decision_path``/``decision_reason`` in the response
        say which applied and why.
        """
        model, manifest = load_model_and_manifest(request.lead_type_id)
        record = request.model_dump(
            exclude={"expected_revenue", "target_cm", "min_bid", "bid_step"}
        )
        record["bid"] = request.min_bid  # placeholder; optimizer sweeps bid
        frame = preprocessing.serving_frame([record], request.lead_type_id)
        row = frame.iloc[0]
        return decide_bid(
            row=row,
            model=model,
            manifest=manifest,
            expected_revenue=request.expected_revenue,
            target_cm=request.target_cm,
            min_bid=request.min_bid,
            bid_step=request.bid_step,
            created_dayofweek=request.created_dayofweek,
            created_hour=request.created_hour,
        )

    @app.post("/explain_bid")
    def explain_bid(request: BidRequest):
        """Explain, in plain English, why a lead got its recommended bid.

        Offline/on-demand only — NOT part of the live bidding path. Requires
        the `explain` extra (shap) and a running local Ollama server; the
        numeric factors still come back even if Ollama is unreachable (the
        prose `explanation` field just says so instead of raising).
        """
        try:
            from . import explain as explain_module
        except ImportError as exc:
            raise HTTPException(
                status_code=503,
                detail=(
                    "Explanation support not installed — "
                    f"pip install '.[explain,ml]' ({exc})"
                ),
            )

        model, manifest = load_model_and_manifest(request.lead_type_id)
        record = request.model_dump(
            exclude={"expected_revenue", "target_cm", "min_bid", "bid_step"}
        )
        try:
            return explain_module.explain_bid(
                model=model,
                record=record,
                lead_type_id=request.lead_type_id,
                expected_revenue=request.expected_revenue,
                manifest=manifest,
                target_cm=request.target_cm,
                min_bid=request.min_bid,
                bid_step=request.bid_step,
                created_dayofweek=request.created_dayofweek,
                created_hour=request.created_hour,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
