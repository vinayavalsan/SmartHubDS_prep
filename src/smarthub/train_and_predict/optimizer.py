"""Bid optimization for SmartHub model prediction.

This module scores candidate bids and selects the highest expected-profit bid.
"""

from __future__ import annotations

import warnings
from contextlib import contextmanager

import numpy as np
import pandas as pd

from smarthub.core.logging_utils import get_logger

from . import config

logger = get_logger(__name__)


def _empty_monotonicity_diagnostics() -> dict:
    """Return zeroed diagnostics for bid-response monotonicity."""
    return {
        "checked_rows": 0,
        "checked_steps": 0,
        "violation_count": 0,
        "rows_with_violation": 0,
        "violation_magnitude_sum": 0.0,
        "max_violation_magnitude": 0.0,
    }


def _merge_monotonicity_diagnostics(target: dict, source: dict) -> None:
    """Accumulate chunk-level monotonicity diagnostics in place."""
    for key in (
        "checked_rows",
        "checked_steps",
        "violation_count",
        "rows_with_violation",
    ):
        target[key] += int(source[key])
    target["violation_magnitude_sum"] += float(source["violation_magnitude_sum"])
    target["max_violation_magnitude"] = max(
        float(target["max_violation_magnitude"]),
        float(source["max_violation_magnitude"]),
    )


@contextmanager
def quiet_feature_name_warning():
    """Temporarily suppress harmless sklearn feature-name warnings."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=".*does not have valid feature names.*"
        )
        yield


def empty_result(max_bid: float) -> dict:
    """Build the standard result for an invalid candidate-bid set.

    Inputs
    ------
    max_bid : Any
        Input value.

    Returns
    -------
    dict
        Empty optimizer result dictionary.
    """
    return {
        "recommended_bid": np.nan,
        "recommended_bid_predicted_win_rate": np.nan,
        "recommended_bid_expected_profit": np.nan,
        "max_bid": max_bid,
        "n_candidate_bids": 0,
    }


def candidate_bids_for_revenue(
    expected_revenue: float,
    target_cm: float,
    min_bid: float,
    bid_step: float,
) -> tuple[np.ndarray, float]:
    """Generate valid candidate bids for expected revenue.

    Inputs
    ------
    expected_revenue : float
        Expected revenue if the lead is won.
    target_cm : float
        Target contribution margin as a decimal.
    min_bid : float
        Minimum candidate bid.
    bid_step : float
        Increment between candidate bids.

    Returns
    -------
    tuple[numpy.ndarray, float]
        Candidate bids followed by the maximum permitted bid.
    """
    if pd.isna(expected_revenue) or expected_revenue <= 0:
        return np.array([], dtype=float), float("nan")

    max_bid = float(expected_revenue * (1 - target_cm))
    if max_bid < min_bid:
        return np.array([], dtype=float), max_bid

    candidate_bids = np.arange(
        min_bid,
        max_bid + (bid_step / 2),
        bid_step,
    )
    candidate_bids = candidate_bids[candidate_bids <= max_bid]
    return candidate_bids.astype(float), max_bid


def optimize_bid_for_row(
    row,
    model,
    expected_revenue: float,
    target_cm: float,
    min_bid: float,
    bid_step: float,
    include_candidates: bool = False,
) -> dict:
    """Select the highest expected-profit bid for one lead.

    Inputs
    ------
    row : pandas.Series
        Single model-ready feature row.
    model : Any
        Fitted model or model pipeline.
    expected_revenue : float
        Expected revenue if the lead is won.
    target_cm : float
        Target contribution margin as a decimal.
    min_bid : float
        Minimum candidate bid.
    bid_step : float
        Increment between candidate bids.
    include_candidates : bool
        When ``True``, also return a ``candidate_evaluations`` list -- one entry
        per candidate bid evaluated by the sweep, each
        ``{bid, predicted_win_rate, expected_profit, selected}`` -- so the full
        optimizer evaluation history can be persisted for auditing/explainability
        (see docs/PREDICTION_LOG_SCHEMA.md). Off by default so the batch/eval
        callers pay nothing for it.

    Returns
    -------
    dict
        Recommended bid and predicted business metrics; plus
        ``candidate_evaluations`` when ``include_candidates`` is set.
    """
    candidate_bids, max_bid = candidate_bids_for_revenue(
        expected_revenue,
        target_cm,
        min_bid,
        bid_step,
    )
    if len(candidate_bids) == 0:
        return empty_result(max_bid)

    candidate_rows = pd.DataFrame([row.to_dict()] * len(candidate_bids))
    candidate_rows["bid"] = candidate_bids
    with quiet_feature_name_warning():
        predicted_win_rates = model.predict_proba(candidate_rows)[:, 1]
    expected_profits = predicted_win_rates * (expected_revenue - candidate_bids)
    best_idx = int(np.argmax(expected_profits))
    result = {
        "recommended_bid": float(candidate_bids[best_idx]),
        "recommended_bid_predicted_win_rate": float(predicted_win_rates[best_idx]),
        "recommended_bid_expected_profit": float(expected_profits[best_idx]),
        "max_bid": float(max_bid),
        "n_candidate_bids": int(len(candidate_bids)),
    }
    if include_candidates:
        # The full per-candidate sweep, ready to serialize. Rounded to keep the
        # JSON payload compact (win rate 6dp, money 4dp) without losing signal.
        result["candidate_evaluations"] = [
            {
                "bid": round(float(b), 4),
                "predicted_win_rate": round(float(w), 6),
                "expected_profit": round(float(p), 4),
                "selected": i == best_idx,
            }
            for i, (b, w, p) in enumerate(
                zip(candidate_bids, predicted_win_rates, expected_profits)
            )
        ]
    return result


def _score_chunk(
    chunk_df: pd.DataFrame,
    model,
    feature_cols: list[str],
    target_cm: float,
    min_bid: float,
    bid_step: float,
    monotonicity_tolerance: float | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Optimize one chunk using a single batched model call.

    Inputs
    ------
    chunk_df : pandas.DataFrame
        Subset of rows processed in one batch.
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

    monotonicity_tolerance : float | None
        Maximum probability decrease treated as numerical noise. When
        ``None``, monotonicity diagnostics are not collected.

    Returns
    -------
    tuple[pandas.DataFrame, dict]
        Recommended-bid results followed by monotonicity diagnostics.
    """
    candidate_frames = []
    empty_results = []

    columns = feature_cols + [config.REVENUE_COL]
    for row_index, row in chunk_df[columns].iterrows():
        expected_revenue = row[config.REVENUE_COL]
        candidate_bids, max_bid = candidate_bids_for_revenue(
            expected_revenue,
            target_cm,
            min_bid,
            bid_step,
        )
        if len(candidate_bids) == 0:
            result = empty_result(max_bid)
            result["_source_index"] = row_index
            empty_results.append(result)
            continue

        candidate_rows = pd.DataFrame(
            [row[feature_cols].to_dict()] * len(candidate_bids)
        )
        candidate_rows["bid"] = candidate_bids
        candidate_rows["_source_index"] = row_index
        candidate_rows["_candidate_bid"] = candidate_bids
        candidate_rows["_expected_revenue"] = expected_revenue
        candidate_rows["_max_bid"] = max_bid
        candidate_rows["_n_candidate_bids"] = len(candidate_bids)
        candidate_frames.append(candidate_rows)

    diagnostics = _empty_monotonicity_diagnostics()
    if not candidate_frames:
        result = pd.DataFrame(empty_results).set_index("_source_index")
        return result, diagnostics

    candidates = pd.concat(candidate_frames, ignore_index=True)
    with quiet_feature_name_warning():
        candidates["_predicted_win_rate"] = model.predict_proba(
            candidates[feature_cols]
        )[:, 1]
    candidates["_expected_profit"] = candidates["_predicted_win_rate"] * (
        candidates["_expected_revenue"] - candidates["_candidate_bid"]
    )

    if monotonicity_tolerance is not None:
        for _, group in candidates.groupby("_source_index", sort=False):
            probabilities = group["_predicted_win_rate"].to_numpy(dtype=float)
            if len(probabilities) < 2:
                continue
            differences = np.diff(probabilities)
            diagnostics["checked_rows"] += 1
            diagnostics["checked_steps"] += int(len(differences))

            violation_mask = differences < -float(monotonicity_tolerance)
            violation_count = int(np.sum(violation_mask))
            if not violation_count:
                continue

            magnitudes = -differences[violation_mask]
            diagnostics["violation_count"] += violation_count
            diagnostics["rows_with_violation"] += 1
            diagnostics["violation_magnitude_sum"] += float(np.sum(magnitudes))
            diagnostics["max_violation_magnitude"] = max(
                float(diagnostics["max_violation_magnitude"]),
                float(np.max(magnitudes)),
            )

    best_positions = candidates.groupby("_source_index")["_expected_profit"].idxmax()
    best = candidates.loc[best_positions].copy()
    best["recommended_bid"] = best["_candidate_bid"].astype(float)
    best["recommended_bid_predicted_win_rate"] = best["_predicted_win_rate"].astype(
        float
    )
    best["recommended_bid_expected_profit"] = best["_expected_profit"].astype(float)
    best["max_bid"] = best["_max_bid"].astype(float)
    best["n_candidate_bids"] = best["_n_candidate_bids"].astype(int)

    result_cols = [
        "_source_index",
        "recommended_bid",
        "recommended_bid_predicted_win_rate",
        "recommended_bid_expected_profit",
        "max_bid",
        "n_candidate_bids",
    ]
    result = best[result_cols].set_index("_source_index")
    if empty_results:
        empty_df = pd.DataFrame(empty_results).set_index("_source_index")
        result = pd.concat([result, empty_df], axis=0)
    return result.sort_index(), diagnostics


def score_recommended_bids(
    eval_df: pd.DataFrame,
    model,
    feature_cols: list[str],
    target_cm: float,
    min_bid: float,
    bid_step: float,
    chunk_size: int,
    monotonicity_tolerance: float | None = None,
) -> pd.DataFrame | None:
    """Attach recommended-bid outputs using chunked scoring.

    Inputs
    ------
    eval_df : pandas.DataFrame
        Optimizer evaluation dataframe.
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
        Maximum rows processed per scoring chunk.
    monotonicity_tolerance : float | None
        Maximum probability decrease treated as numerical noise. When set,
        monotonicity is measured from the same candidate predictions used for
        bid optimization.

    Returns
    -------
    pandas.DataFrame | None
        Evaluation rows with recommendation columns, or ``None``.
    """
    n_rows = len(eval_df)
    logger.info(
        "Optimizing bids for %s rows in chunks of %s",
        f"{n_rows:,}",
        f"{chunk_size:,}",
    )
    result_chunks = []
    monotonicity = _empty_monotonicity_diagnostics()
    for start in range(0, n_rows, chunk_size):
        stop = min(start + chunk_size, n_rows)
        chunk_result, chunk_monotonicity = _score_chunk(
            eval_df.iloc[start:stop],
            model,
            feature_cols,
            target_cm,
            min_bid,
            bid_step,
            monotonicity_tolerance=monotonicity_tolerance,
        )
        result_chunks.append(chunk_result)
        _merge_monotonicity_diagnostics(
            monotonicity,
            chunk_monotonicity,
        )
        if stop == n_rows or stop % (chunk_size * 10) == 0:
            logger.info("Optimized %s / %s rows", f"{stop:,}", f"{n_rows:,}")

    optimizer_df = pd.concat(result_chunks, axis=0)
    result = pd.concat([eval_df, optimizer_df], axis=1)
    result = result.dropna(subset=["recommended_bid"]).copy()
    if result.empty:
        return None
    result.attrs["monotonicity_diagnostics"] = monotonicity
    return result
