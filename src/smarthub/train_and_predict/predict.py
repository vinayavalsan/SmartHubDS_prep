"""
Prediction and bid optimization utilities for Anton.

The trained model predicts:

    P(won | bid, lead features)

The bid optimizer:
1. Creates candidate bids.
2. Predicts win rate for each candidate bid.
3. Calculates expected profit for each candidate bid.
4. Returns the bid with the highest expected profit.

expected_revenue is NOT a model feature.
It is used only in the optimizer.
"""

from __future__ import annotations

import os

import joblib
import mlflow.sklearn
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

import config
import preprocessing


MODEL_URI = os.getenv("MODEL_URI", "models/anton_model.pkl")


def load_model(model_uri=MODEL_URI):
    """Load a trained Anton model from local pkl or MLflow URI."""
    if model_uri.endswith(".pkl"):
        return joblib.load(model_uri)

    return mlflow.sklearn.load_model(model_uri)


def optimize_bid_for_row(
    row,
    model,
    expected_revenue,
    target_cm,
    min_bid,
    bid_step,
):
    """Return the best bid for one lead using the trained win-probability model."""
    max_bid = expected_revenue * (1 - target_cm)

    if pd.isna(expected_revenue) or expected_revenue <= 0 or max_bid < min_bid:
        return {
            "recommended_bid": np.nan,
            "recommended_bid_predicted_win_rate": np.nan,
            "recommended_bid_expected_profit": np.nan,
            "max_bid": max_bid,
            "n_candidate_bids": 0,
        }

    candidate_bids = np.arange(min_bid, max_bid + bid_step, bid_step)
    candidate_bids = candidate_bids[candidate_bids <= max_bid]

    if len(candidate_bids) == 0:
        return {
            "recommended_bid": np.nan,
            "recommended_bid_predicted_win_rate": np.nan,
            "recommended_bid_expected_profit": np.nan,
            "max_bid": max_bid,
            "n_candidate_bids": 0,
        }

    candidate_rows = pd.DataFrame(
        [row[config.FEATURE_COLS].to_dict()] * len(candidate_bids)
    )
    candidate_rows["bid"] = candidate_bids

    predicted_win_rates = model.predict_proba(candidate_rows[config.FEATURE_COLS])[:, 1]
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
    target_cm=0.25,
    min_bid=0.25,
    bid_step=0.25,
):
    """Compare current-bid predictions against optimized candidate-bid choices."""
    print("=" * 80)
    print("Bid Optimization Evaluation")

    if "expected_revenue" not in test_eval_df.columns:
        print("Skipped: expected_revenue is not available in this dataset.")
        return None

    eval_df = test_eval_df.dropna(subset=["expected_revenue", "bid"]).copy()
    eval_df = eval_df[eval_df["expected_revenue"] > 0].copy()

    if eval_df.empty:
        print("Skipped: no rows with positive expected_revenue.")
        return None

    current_bid_predicted_win_rate = model.predict_proba(eval_df[config.FEATURE_COLS])[
        :, 1
    ]

    eval_df["current_bid_predicted_win_rate"] = current_bid_predicted_win_rate
    eval_df["current_bid_expected_profit"] = eval_df[
        "current_bid_predicted_win_rate"
    ] * (eval_df["expected_revenue"] - eval_df["bid"])

    optimizer_rows = []
    for _, row in eval_df.iterrows():
        optimizer_rows.append(
            optimize_bid_for_row(
                row=row,
                model=model,
                expected_revenue=row["expected_revenue"],
                target_cm=target_cm,
                min_bid=min_bid,
                bid_step=bid_step,
            )
        )

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
    eval_df["expected_profit_lift_pct"] = eval_df["expected_profit_lift"] / eval_df[
        "current_bid_expected_profit"
    ].replace(0, np.nan)
    eval_df["bid_change"] = eval_df["recommended_bid"] - eval_df["bid"]
    eval_df["bid_change_pct"] = eval_df["bid_change"] / eval_df["bid"].replace(
        0, np.nan
    )
    eval_df["predicted_win_rate_lift"] = (
        eval_df["recommended_bid_predicted_win_rate"]
        - eval_df["current_bid_predicted_win_rate"]
    )
    eval_df["predicted_win_rate_lift_pct"] = eval_df[
        "predicted_win_rate_lift"
    ] / eval_df["current_bid_predicted_win_rate"].replace(0, np.nan)
    eval_df["recommended_bid_cm_if_won"] = (
        eval_df["expected_revenue"] - eval_df["recommended_bid"]
    ) / eval_df["expected_revenue"]

    eval_df["bid_change_direction"] = "unchanged"
    eval_df.loc[eval_df["bid_change"] > 0, "bid_change_direction"] = "increased"
    eval_df.loc[eval_df["bid_change"] < 0, "bid_change_direction"] = "decreased"

    eval_df["selected_bid_percentile"] = eval_df["recommended_bid"] / eval_df[
        "max_bid"
    ].replace(0, np.nan)

    direction_counts = (
        eval_df["bid_change_direction"]
        .value_counts()
        .reindex(["increased", "decreased", "unchanged"], fill_value=0)
    )

    direction_summary = {}
    for direction in ["increased", "decreased", "unchanged"]:
        direction_df = eval_df[eval_df["bid_change_direction"] == direction]
        direction_summary[direction] = {
            "count": int(len(direction_df)),
            "percent": float(len(direction_df) / len(eval_df) * 100),
            "avg_bid_change": (
                float(direction_df["bid_change"].mean())
                if not direction_df.empty
                else 0.0
            ),
            "avg_predicted_win_rate_lift": (
                float(direction_df["predicted_win_rate_lift"].mean())
                if not direction_df.empty
                else 0.0
            ),
            "avg_expected_profit_lift": (
                float(direction_df["expected_profit_lift"].mean())
                if not direction_df.empty
                else 0.0
            ),
        }

    current_total_profit = eval_df["current_bid_expected_profit"].sum()
    recommended_total_profit = eval_df["recommended_bid_expected_profit"].sum()

    summary = {
        "optimizer_rows": int(len(eval_df)),
        "target_cm": float(target_cm),
        "min_bid": float(min_bid),
        "bid_step": float(bid_step),
        # Predicted performance comparison
        "current_bid_total_expected_profit": float(current_total_profit),
        "recommended_bid_total_expected_profit": float(recommended_total_profit),
        "expected_profit_lift_total": float(eval_df["expected_profit_lift"].sum()),
        "expected_profit_lift_pct": (
            float(
                (recommended_total_profit - current_total_profit) / current_total_profit
            )
            if current_total_profit != 0
            else np.nan
        ),
        "current_bid_avg_expected_profit": float(
            eval_df["current_bid_expected_profit"].mean()
        ),
        "recommended_bid_avg_expected_profit": float(
            eval_df["recommended_bid_expected_profit"].mean()
        ),
        "avg_current_bid_predicted_win_rate": float(
            eval_df["current_bid_predicted_win_rate"].mean()
        ),
        "avg_recommended_bid_predicted_win_rate": float(
            eval_df["recommended_bid_predicted_win_rate"].mean()
        ),
        "avg_predicted_win_rate_lift": float(eval_df["predicted_win_rate_lift"].mean()),
        "median_predicted_win_rate_lift": float(
            eval_df["predicted_win_rate_lift"].median()
        ),
        # Bid-change behavior
        "min_bid_change": float(eval_df["bid_change"].min()),
        "max_bid_change": float(eval_df["bid_change"].max()),
        "bid_increase_count": int(direction_counts["increased"]),
        "bid_decrease_count": int(direction_counts["decreased"]),
        "bid_unchanged_count": int(direction_counts["unchanged"]),
        "bid_increase_pct": float(direction_counts["increased"] / len(eval_df) * 100),
        "bid_decrease_pct": float(direction_counts["decreased"] / len(eval_df) * 100),
        "bid_unchanged_pct": float(direction_counts["unchanged"] / len(eval_df) * 100),
        # Optimizer behavior
        "median_candidate_bids_evaluated": float(eval_df["n_candidate_bids"].median()),
        "median_selected_bid_percentile": float(
            eval_df["selected_bid_percentile"].median()
        ),
        "avg_recommended_bid_cm_if_won": float(
            eval_df["recommended_bid_cm_if_won"].mean()
        ),
        "median_recommended_bid_cm_if_won": float(
            eval_df["recommended_bid_cm_if_won"].median()
        ),
        "p10_recommended_bid_cm_if_won": float(
            eval_df["recommended_bid_cm_if_won"].quantile(0.10)
        ),
        "p90_recommended_bid_cm_if_won": float(
            eval_df["recommended_bid_cm_if_won"].quantile(0.90)
        ),
        "bid_change_direction_summary": direction_summary,
    }

    print(f"Rows evaluated:                         {summary['optimizer_rows']:,}")
    print(f"Target CM:                              {summary['target_cm']:.2%}")
    print(
        f"Bid grid:                               {min_bid:.2f} to max_bid by {bid_step:.2f}"
    )
    print()
    print("Predicted Performance")
    print("-" * 80)
    print(
        "Expected Profit using Current Bid:      "
        f"{summary['current_bid_total_expected_profit']:.4f}"
    )
    print(
        "Expected Profit using Recommended Bid:  "
        f"{summary['recommended_bid_total_expected_profit']:.4f}"
    )
    print(
        f"Expected Profit Lift:                   {summary['expected_profit_lift_total']:.4f}"
    )
    print(
        f"Expected Profit Lift %:                 {summary['expected_profit_lift_pct']:.2%}"
    )
    print()
    print(
        "Avg Expected Profit using Current Bid:  "
        f"{summary['current_bid_avg_expected_profit']:.4f}"
    )
    print(
        "Avg Expected Profit using Recommended Bid: "
        f"{summary['recommended_bid_avg_expected_profit']:.4f}"
    )
    print()
    print(
        "Avg Predicted Win Rate using Current Bid:     "
        f"{summary['avg_current_bid_predicted_win_rate']:.4f}"
    )
    print(
        "Avg Predicted Win Rate using Recommended Bid: "
        f"{summary['avg_recommended_bid_predicted_win_rate']:.4f}"
    )
    print(
        "Avg Predicted Win Rate Lift:                  "
        f"{summary['avg_predicted_win_rate_lift']:.4f}"
    )
    print()
    print("Bid Changes")
    print("-" * 80)
    print(
        "Bid Increased / Decreased / Unchanged:  "
        f"{summary['bid_increase_count']:,} ({summary['bid_increase_pct']:.1f}%) / "
        f"{summary['bid_decrease_count']:,} ({summary['bid_decrease_pct']:.1f}%) / "
        f"{summary['bid_unchanged_count']:,} ({summary['bid_unchanged_pct']:.1f}%)"
    )
    print()
    print("Optimizer Behavior")
    print("-" * 80)
    print(
        "Average Recommended CM if Won:          "
        f"{summary['avg_recommended_bid_cm_if_won']:.4f}"
    )
    print(
        "Median Recommended CM if Won:           "
        f"{summary['median_recommended_bid_cm_if_won']:.4f}"
    )

    return eval_df, summary


class BidRequest(BaseModel):
    """Request schema for bid recommendation."""

    expected_revenue: float = Field(..., gt=0)
    target_cm: float = Field(0.25, ge=0, lt=1)
    min_bid: float = Field(0.25, ge=0)
    bid_step: float = Field(0.25, gt=0)

    campaign_id: int
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


def request_to_model_row(request):
    """Convert an API request into one cleaned model input row."""
    raw_row = pd.DataFrame(
        [
            {
                "bid": request.min_bid,
                "age": request.age,
                "num_vehicles": request.num_vehicles,
                "num_drivers": request.num_drivers,
                "num_auto_violations": request.num_auto_violations,
                "num_auto_accidents": request.num_auto_accidents,
                "continuous_coverage_months": request.continuous_coverage_months,
                "created_hour": request.created_hour,
                "created_dayofweek": request.created_dayofweek,
                "campaign_id": request.campaign_id,
                "lead_type_id": request.lead_type_id,
                "state": request.state,
                "insured": request.insured,
                "home_owner": request.home_owner,
                "dui": request.dui,
                "military_affiliation": request.military_affiliation,
                "gender": request.gender,
                "marital_status": request.marital_status,
            }
        ]
    )

    cleaned_df = preprocessing.clean_model_features(raw_row)
    return cleaned_df.iloc[0]


app = FastAPI(title="Anton Bid Prediction API")


@app.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "ok", "model_uri": MODEL_URI}


@app.post("/recommend_bid")
def recommend_bid(request: BidRequest):
    """Recommend the bid that maximizes expected profit."""
    model = load_model()
    row = request_to_model_row(request)

    return optimize_bid_for_row(
        row=row,
        model=model,
        expected_revenue=request.expected_revenue,
        target_cm=request.target_cm,
        min_bid=request.min_bid,
        bid_step=request.bid_step,
    )
