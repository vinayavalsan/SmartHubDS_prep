"""Offline evaluation for SmartHub bid optimization.

This module compares historical and recommended bids and summarizes results.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from . import config, optimizer

logger = logging.getLogger("smarthub.train_and_predict.evaluation")


@dataclass(frozen=True)
class OptimizerSummary:
    """Store the main offline bid-optimization metrics."""

    optimizer_rows: int
    target_cm: float
    current_bid_total_expected_profit: float
    recommended_bid_total_expected_profit: float
    expected_profit_lift_total: float
    expected_profit_lift_pct: float
    avg_current_bid_predicted_win_rate: float
    avg_recommended_bid_predicted_win_rate: float
    avg_bid_change: float
    median_bid_change: float
    bid_increase_pct: float
    bid_decrease_pct: float
    bid_unchanged_pct: float
    avg_recommended_bid_cm_if_won: float

    def to_dict(self) -> dict:
        """Return a serializable dictionary representation."""
        return asdict(self)

    def mlflow_metrics(self) -> dict:
        """Return flat numeric optimizer metrics for MLflow."""
        return {
            f"optimizer_{key}": value
            for key, value in self.to_dict().items()
            if isinstance(value, (int, float))
        }


def _safe_divide(numerator, denominator):
    """Divide values while treating zero denominators as missing.

    Inputs
    ------
    numerator : Any
        Numerator value or series.
    denominator : Any
        Denominator value or series.

    Returns
    -------
    Any
        Division result with zero denominators represented as missing.
    """
    if isinstance(denominator, pd.Series):
        return numerator / denominator.replace(0, np.nan)
    return numerator / denominator if denominator else float("nan")


def _prepare_frame(test_eval_df: pd.DataFrame) -> pd.DataFrame | None:
    """Filter rows to those valid for optimizer evaluation.

    Inputs
    ------
    test_eval_df : pandas.DataFrame
        Held-out evaluation dataframe.

    Returns
    -------
    pandas.DataFrame | None
        Filtered evaluation rows, or ``None`` when unavailable.
    """
    if config.REVENUE_COL not in test_eval_df.columns:
        logger.warning(
            "Skipping optimizer evaluation: %s is unavailable.",
            config.REVENUE_COL,
        )
        return None
    result = test_eval_df.dropna(subset=[config.REVENUE_COL, "bid"]).copy()
    result = result[result[config.REVENUE_COL] > 0].copy()
    return None if result.empty else result


def _score_current_bids(eval_df, model, feature_cols):
    """Score historical bids and calculate expected profit.

    Inputs
    ------
    eval_df : pandas.DataFrame
        Optimizer evaluation dataframe.
    model : Any
        Fitted model or model pipeline.
    feature_cols : list[str]
        Ordered model feature names.

    Returns
    -------
    pandas.DataFrame
        Evaluation rows with current-bid predictions.
    """
    with optimizer.quiet_feature_name_warning():
        win_rate = model.predict_proba(eval_df[feature_cols])[:, 1]
    result = eval_df.copy()
    result["current_bid_predicted_win_rate"] = win_rate
    result["current_bid_expected_profit"] = win_rate * (
        result[config.REVENUE_COL] - result["bid"]
    )
    return result


def _add_diagnostics(eval_df):
    """Add the row-level diagnostics needed by the retained summary metrics.

    Inputs
    ------
    eval_df : pandas.DataFrame
        Optimizer evaluation dataframe.

    Returns
    -------
    pandas.DataFrame
        Evaluation rows with retained optimizer diagnostics.
    """
    result = eval_df.copy()
    result["expected_profit_lift"] = (
        result["recommended_bid_expected_profit"]
        - result["current_bid_expected_profit"]
    )
    result["bid_change"] = result["recommended_bid"] - result["bid"]
    result["recommended_bid_cm_if_won"] = (
        result[config.REVENUE_COL] - result["recommended_bid"]
    ) / result[config.REVENUE_COL]
    return result


def summarize_results(eval_df, target_cm):
    """Aggregate the retained row-level optimizer metrics.

    Inputs
    ------
    eval_df : pandas.DataFrame
        Optimizer evaluation dataframe.
    target_cm : float
        Target contribution margin as a decimal.

    Returns
    -------
    OptimizerSummary
        Aggregate optimizer summary.
    """
    current_total = float(eval_df["current_bid_expected_profit"].sum())
    recommended_total = float(eval_df["recommended_bid_expected_profit"].sum())
    lift_total = recommended_total - current_total
    n_rows = len(eval_df)

    increased = int((eval_df["bid_change"] > 0).sum())
    decreased = int((eval_df["bid_change"] < 0).sum())
    unchanged = int((eval_df["bid_change"] == 0).sum())

    return OptimizerSummary(
        optimizer_rows=n_rows,
        target_cm=float(target_cm),
        current_bid_total_expected_profit=current_total,
        recommended_bid_total_expected_profit=recommended_total,
        expected_profit_lift_total=lift_total,
        expected_profit_lift_pct=_safe_divide(lift_total, current_total),
        avg_current_bid_predicted_win_rate=float(
            eval_df["current_bid_predicted_win_rate"].mean()
        ),
        avg_recommended_bid_predicted_win_rate=float(
            eval_df["recommended_bid_predicted_win_rate"].mean()
        ),
        avg_bid_change=float(eval_df["bid_change"].mean()),
        median_bid_change=float(eval_df["bid_change"].median()),
        bid_increase_pct=float(increased / n_rows * 100),
        bid_decrease_pct=float(decreased / n_rows * 100),
        bid_unchanged_pct=float(unchanged / n_rows * 100),
        avg_recommended_bid_cm_if_won=float(
            eval_df["recommended_bid_cm_if_won"].mean()
        ),
    )


def _format_percent(value):
    """Format a decimal value as a percentage."""
    return "—" if pd.isna(value) else f"{float(value):.2%}"


def log_summary(summary: OptimizerSummary, log=None):
    """Log aggregate optimizer evaluation metrics.

    Inputs
    ------
    summary : OptimizerSummary
        Optimizer summary to log.
    log : logging.Logger | None
        Optional logger for structured output.
    """
    log = log or logger
    log.info("Bid Optimizer Evaluation")
    log.info("  Rows evaluated                     : %s", f"{summary.optimizer_rows:,}")
    log.info("  Target CM                          : %.2f%%", summary.target_cm * 100)
    log.info(
        "  Expected profit, current/recommended: %.4f / %.4f",
        summary.current_bid_total_expected_profit,
        summary.recommended_bid_total_expected_profit,
    )
    log.info(
        "  Expected profit lift               : %.4f (%s)",
        summary.expected_profit_lift_total,
        _format_percent(summary.expected_profit_lift_pct),
    )
    log.info(
        "  Avg predicted win rate, current/reco: %.4f / %.4f",
        summary.avg_current_bid_predicted_win_rate,
        summary.avg_recommended_bid_predicted_win_rate,
    )
    log.info(
        "  Bid increased/decreased/unchanged  : %.2f%% / %.2f%% / %.2f%%",
        summary.bid_increase_pct,
        summary.bid_decrease_pct,
        summary.bid_unchanged_pct,
    )
    log.info(
        "  Avg / median bid change            : %.4f / %.4f",
        summary.avg_bid_change,
        summary.median_bid_change,
    )
    log.info(
        "  Avg recommended CM if won           : %.4f",
        summary.avg_recommended_bid_cm_if_won,
    )


def run_bid_optimizer_evaluation(
    test_eval_df,
    model,
    feature_cols,
    target_cm=0.25,
    min_bid=0.25,
    bid_step=0.25,
    log=None,
    log_summary_result=True,
):
    """Run offline bid optimization on held-out rows.

    Inputs
    ------
    test_eval_df : pandas.DataFrame
        Held-out evaluation dataframe.
    model : Any
        Fitted model or model pipeline.
    feature_cols : list[str]
        Ordered model feature names.
    target_cm : float
        Target contribution margin as a decimal.
    min_bid : float
        Minimum candidate bid.
    bid_step : float
        Increment between candidate bids.
    log : logging.Logger | None
        Optional logger for structured output.
    log_summary_result : bool
        Whether to log the aggregate optimizer summary.

    Returns
    -------
    tuple[pandas.DataFrame, OptimizerSummary] | None
        Row-level optimizer results and aggregate summary, or ``None``.
    """
    log = log or logger
    eval_df = _prepare_frame(test_eval_df)
    if eval_df is None:
        return None
    eval_df = _score_current_bids(eval_df, model, list(feature_cols))
    eval_df = optimizer.score_recommended_bids(
        eval_df,
        model,
        list(feature_cols),
        target_cm,
        min_bid,
        bid_step,
        log=log,
    )
    if eval_df is None:
        log.warning("Optimizer could not create candidate bids.")
        return None
    eval_df = _add_diagnostics(eval_df)
    summary = summarize_results(eval_df, target_cm)
    if log_summary_result:
        log_summary(summary, log)
    return eval_df, summary
