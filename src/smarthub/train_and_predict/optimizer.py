"""Bid optimization for SmartHub model prediction.

This module scores candidate bids and selects the highest expected-profit bid.
"""

from __future__ import annotations

import logging
import warnings
from contextlib import contextmanager

import numpy as np
import pandas as pd

from . import config

logger = logging.getLogger("smarthub.train_and_predict.optimizer")


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

    Returns
    -------
    dict
        Recommended bid and predicted business metrics.
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
    return {
        "recommended_bid": float(candidate_bids[best_idx]),
        "recommended_bid_predicted_win_rate": float(predicted_win_rates[best_idx]),
        "recommended_bid_expected_profit": float(expected_profits[best_idx]),
        "max_bid": float(max_bid),
        "n_candidate_bids": int(len(candidate_bids)),
    }


def _score_chunk(
    chunk_df: pd.DataFrame,
    model,
    feature_cols: list[str],
    target_cm: float,
    min_bid: float,
    bid_step: float,
) -> pd.DataFrame:
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

    Returns
    -------
    pandas.DataFrame
        Recommended-bid results indexed by source row.
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

    if not candidate_frames:
        return pd.DataFrame(empty_results).set_index("_source_index")

    candidates = pd.concat(candidate_frames, ignore_index=True)
    with quiet_feature_name_warning():
        candidates["_predicted_win_rate"] = model.predict_proba(
            candidates[feature_cols]
        )[:, 1]
    candidates["_expected_profit"] = candidates["_predicted_win_rate"] * (
        candidates["_expected_revenue"] - candidates["_candidate_bid"]
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
    return result.sort_index()


def score_recommended_bids(
    eval_df: pd.DataFrame,
    model,
    feature_cols: list[str],
    target_cm: float,
    min_bid: float,
    bid_step: float,
    chunk_size: int,
    log=None,
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
    log : logging.Logger | None
        Optional logger for structured output.

    Returns
    -------
    pandas.DataFrame | None
        Evaluation rows with recommendation columns, or ``None``.
    """
    log = log or logger
    n_rows = len(eval_df)
    log.info(
        "Optimizing bids for %s rows in chunks of %s",
        f"{n_rows:,}",
        f"{chunk_size:,}",
    )
    result_chunks = []
    for start in range(0, n_rows, chunk_size):
        stop = min(start + chunk_size, n_rows)
        result_chunks.append(
            _score_chunk(
                eval_df.iloc[start:stop],
                model,
                feature_cols,
                target_cm,
                min_bid,
                bid_step,
            )
        )
        if stop == n_rows or stop % (chunk_size * 10) == 0:
            log.info("Optimized %s / %s rows", f"{stop:,}", f"{n_rows:,}")

    optimizer_df = pd.concat(result_chunks, axis=0)
    result = pd.concat([eval_df, optimizer_df], axis=1)
    result = result.dropna(subset=["recommended_bid"]).copy()
    if result.empty:
        return None
    return result
