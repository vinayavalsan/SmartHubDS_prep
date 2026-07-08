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
    2. ``smarthub.ini [prediction] active_model_version`` — an explicit
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
    from fastapi import FastAPI
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
        """Recommend the bid that maximizes expected profit for one lead."""
        model = load_model(lead_type_id=request.lead_type_id)
        record = request.model_dump(
            exclude={"expected_revenue", "target_cm", "min_bid", "bid_step"}
        )
        record["bid"] = request.min_bid  # placeholder; optimizer sweeps bid
        frame = preprocessing.serving_frame([record], request.lead_type_id)
        row = frame.iloc[0]
        return optimize_bid_for_row(
            row=row,
            model=model,
            expected_revenue=request.expected_revenue,
            target_cm=request.target_cm,
            min_bid=request.min_bid,
            bid_step=request.bid_step,
        )
