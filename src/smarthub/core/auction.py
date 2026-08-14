"""Shared auction-outcome semantics used across SmartHub stages.

This module centralizes the row-level business rules that define errored pings
and training-eligible auction observations. Data-pull validation and feature
engineering both depend on these helpers so the two workflows cannot silently
drift apart.
"""

from __future__ import annotations

import pandas as pd

_TRUE_TOKENS = frozenset({"true", "t", "1", "yes", "y"})


def _lower(series: pd.Series) -> pd.Series:
    """Normalize string-like values for warehouse boolean comparisons."""
    return series.astype("string").str.strip().str.lower()


def erred_mask(df: pd.DataFrame) -> pd.Series:
    """Return rows explicitly marked as errored.

    Inputs
    ------
    df : pandas.DataFrame
        Raw lead rows.

    Returns
    -------
    pandas.Series
        Boolean mask aligned to ``df.index``. Missing or unrecognized
        ``erred`` values are treated as non-erred.
    """
    if "erred" not in df.columns:
        return pd.Series(False, index=df.index, dtype=bool)
    return _lower(df["erred"]).isin(_TRUE_TOKENS)


def won_true_mask(df: pd.DataFrame) -> pd.Series:
    """Return rows explicitly marked as won."""
    if "won" not in df.columns:
        return pd.Series(False, index=df.index, dtype=bool)
    return _lower(df["won"]).eq("true").fillna(False)


def placed_bid_mask(df: pd.DataFrame, bid_column: str = "bid") -> pd.Series:
    """Return rows with a positive placed bid."""
    if bid_column not in df.columns:
        return pd.Series(False, index=df.index, dtype=bool)
    bid = pd.to_numeric(df[bid_column], errors="coerce")
    return bid.gt(0).fillna(False)


def auction_eligible_mask(
    df: pd.DataFrame,
    bid_column: str = "bid",
) -> pd.Series:
    """Return rows usable as observed auction outcomes for model training.

    A row is training-eligible when a positive bid was placed or the warehouse
    marks the row as won. This preserves anomalous ``won=true`` observations
    even when the bid is missing or non-positive, matching the existing
    SmartHub training-table behavior.
    """
    return placed_bid_mask(df, bid_column=bid_column) | won_true_mask(df)
