"""Offline evaluation for SmartHub bid optimization.

This module compares historical and recommended bids and summarizes results.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from smarthub.core.logging_utils import get_logger

from . import config, optimizer

logger = get_logger(__name__)


@dataclass(frozen=True)
class MonotonicitySummary:
    """Store bid-response monotonicity evaluation metrics."""

    enabled: bool
    checked_rows: int
    checked_steps: int
    violation_count: int
    violation_rate: float
    rows_with_violation_pct: float
    mean_violation_magnitude: float
    max_violation_magnitude: float
    max_allowed_violation_rate: float
    passed: bool | None

    def to_dict(self) -> dict:
        """Return a serializable dictionary representation."""
        return asdict(self)


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


def log_summary(summary: OptimizerSummary):
    """Log aggregate optimizer evaluation metrics.

    Inputs
    ------
    summary : OptimizerSummary
        Optimizer summary to log.
    """
    logger.info("Bid Optimizer Evaluation")
    logger.info(
        "  Rows evaluated                     : %s", f"{summary.optimizer_rows:,}"
    )
    logger.info(
        "  Target CM                          : %.2f%%", summary.target_cm * 100
    )
    logger.info(
        "  Expected profit, current/recommended: %.4f / %.4f",
        summary.current_bid_total_expected_profit,
        summary.recommended_bid_total_expected_profit,
    )
    logger.info(
        "  Expected profit lift               : %.4f (%s)",
        summary.expected_profit_lift_total,
        _format_percent(summary.expected_profit_lift_pct),
    )
    logger.info(
        "  Avg predicted win rate, current/reco: %.4f / %.4f",
        summary.avg_current_bid_predicted_win_rate,
        summary.avg_recommended_bid_predicted_win_rate,
    )
    logger.info(
        "  Bid increased/decreased/unchanged  : %.2f%% / %.2f%% / %.2f%%",
        summary.bid_increase_pct,
        summary.bid_decrease_pct,
        summary.bid_unchanged_pct,
    )
    logger.info(
        "  Avg / median bid change            : %.4f / %.4f",
        summary.avg_bid_change,
        summary.median_bid_change,
    )
    logger.info(
        "  Avg recommended CM if won           : %.4f",
        summary.avg_recommended_bid_cm_if_won,
    )


def _summarize_monotonicity(
    diagnostics: dict,
    *,
    enabled: bool,
    max_violation_rate: float,
) -> MonotonicitySummary:
    """Build aggregate monotonicity metrics from optimizer diagnostics."""
    if not enabled:
        return MonotonicitySummary(
            enabled=False,
            checked_rows=0,
            checked_steps=0,
            violation_count=0,
            violation_rate=0.0,
            rows_with_violation_pct=0.0,
            mean_violation_magnitude=0.0,
            max_violation_magnitude=0.0,
            max_allowed_violation_rate=float(max_violation_rate),
            passed=None,
        )

    checked_rows = int(diagnostics.get("checked_rows", 0))
    checked_steps = int(diagnostics.get("checked_steps", 0))
    violation_count = int(diagnostics.get("violation_count", 0))
    rows_with_violation = int(diagnostics.get("rows_with_violation", 0))
    magnitude_sum = float(diagnostics.get("violation_magnitude_sum", 0.0))
    violation_rate = float(violation_count / checked_steps) if checked_steps else 0.0
    rows_with_violation_pct = (
        float(rows_with_violation / checked_rows) if checked_rows else 0.0
    )
    mean_magnitude = float(magnitude_sum / violation_count) if violation_count else 0.0
    return MonotonicitySummary(
        enabled=True,
        checked_rows=checked_rows,
        checked_steps=checked_steps,
        violation_count=violation_count,
        violation_rate=violation_rate,
        rows_with_violation_pct=rows_with_violation_pct,
        mean_violation_magnitude=mean_magnitude,
        max_violation_magnitude=float(diagnostics.get("max_violation_magnitude", 0.0)),
        max_allowed_violation_rate=float(max_violation_rate),
        passed=violation_rate <= float(max_violation_rate),
    )


def _log_monotonicity_summary(summary: MonotonicitySummary) -> None:
    """Log bid-response monotonicity PASS/FAIL and diagnostics."""
    logger.info("Bid Monotonicity Evaluation")
    if not summary.enabled:
        logger.info("  Status                                : SKIPPED")
        logger.info("  Reason                                : disabled in config")
        return

    logger.info(
        "  Status                                : %s",
        "PASS" if summary.passed else "FAIL",
    )
    logger.info(
        "  Rows evaluated                        : %s",
        f"{summary.checked_rows:,}",
    )
    logger.info(
        "  Bid transitions checked               : %s",
        f"{summary.checked_steps:,}",
    )
    if summary.passed:
        return

    logger.info(
        "  Violation rate                        : %.6f%%",
        summary.violation_rate * 100.0,
    )
    logger.info(
        "  Violations                            : %s",
        f"{summary.violation_count:,}",
    )
    logger.info(
        "  Rows with >=1 violation               : %.6f%%",
        summary.rows_with_violation_pct * 100.0,
    )
    logger.info(
        "  Mean / max violation magnitude        : %.8f / %.8f",
        summary.mean_violation_magnitude,
        summary.max_violation_magnitude,
    )


def run_bid_optimizer_evaluation(
    test_eval_df,
    model,
    feature_cols,
    *,
    target_cm,
    min_bid,
    bid_step,
    chunk_size,
    monotonicity_enabled,
    monotonicity_tolerance,
    monotonicity_max_violation_rate,
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
    chunk_size : int
        Maximum rows processed per optimizer scoring chunk.
    monotonicity_enabled : bool
        Whether to evaluate bid-response monotonicity.
    monotonicity_tolerance : float
        Maximum probability decrease treated as numerical noise.
    monotonicity_max_violation_rate : float
        Maximum allowed violating bid-step fraction for a pass.
    log_summary_result : bool
        Whether to log the aggregate optimizer summary.

    Returns
    -------
    tuple[pandas.DataFrame, OptimizerSummary] | None
        Row-level optimizer results and aggregate summary, or ``None``.
    """
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
        chunk_size,
        monotonicity_tolerance=(
            monotonicity_tolerance if monotonicity_enabled else None
        ),
    )
    if eval_df is None:
        logger.warning("Optimizer could not create candidate bids.")
        return None

    monotonicity = _summarize_monotonicity(
        eval_df.attrs.get("monotonicity_diagnostics", {}),
        enabled=monotonicity_enabled,
        max_violation_rate=monotonicity_max_violation_rate,
    )
    eval_df = _add_diagnostics(eval_df)
    eval_df.attrs["monotonicity_summary"] = monotonicity.to_dict()

    summary = summarize_results(eval_df, target_cm)
    if log_summary_result:
        log_summary(summary)
        _log_monotonicity_summary(monotonicity)
    return eval_df, summary
