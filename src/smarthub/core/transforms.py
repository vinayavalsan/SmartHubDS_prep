"""Shared transforms used by the SmartHub data pipeline."""

from __future__ import annotations

from typing import Iterable

import pandas as pd

LEADS_DROP_COLS = ["lead_created_at", "excluded"]
_TRUE_TOKENS = {"true", "t", "1", "yes", "y"}


def normalize_won(series: pd.Series) -> pd.Series:
    """Convert a raw ``won`` column to nullable-int 0/1.

    Handles booleans, mixed case, surrounding whitespace and numeric
    strings; anything unrecognised (including blanks) becomes 0.

    Inputs
    ------
    series : pandas.Series
        The raw ``won`` column.

    Returns
    -------
    pandas.Series
        An ``Int64`` series of 0/1 values.
    """
    normalized = (
        series.astype("string")
        .str.strip()
        .str.lower()
        .map(lambda v: 1 if v in _TRUE_TOKENS else 0)
    )
    return normalized.astype("Int64")


def coerce_numeric(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """Coerce the given columns to numeric in place (errors become NaN).

    Inputs
    ------
    df : pandas.DataFrame
        Frame to modify in place.
    columns : Iterable[str]
        Columns to coerce; missing columns are skipped.

    Returns
    -------
    pandas.DataFrame
        The same frame, with the columns coerced.
    """
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def prepare_leads_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and enrich the raw leads dataframe.

    Drops unused columns, filters non-positive bids, normalises id/state
    columns, derives time features, and computes realized business metrics.

    Inputs
    ------
    df : pandas.DataFrame
        The raw leads frame.

    Returns
    -------
    pandas.DataFrame
        A cleaned, enriched copy.
    """
    out = df.drop(columns=LEADS_DROP_COLS, errors="ignore").copy()

    if "bid" in out.columns:
        out = out[out["bid"] > 0]

    id_cols = (
        "campaign_id",
        "lead_type_id",
        "account_id",
        "source_type_id",
        "bidding_strategy_id",
    )
    for id_col in id_cols:
        if id_col in out.columns:
            out[id_col] = out[id_col].astype("Int64")

    if "state" in out.columns:
        out["state"] = out["state"].fillna("NAvail")
        out.loc[out["state"].astype("string").str.strip() == "", "state"] = "NAvail"

    if "created_at" in out.columns:
        out["created_at"] = pd.to_datetime(out["created_at"])
        out["created_hour"] = out["created_at"].dt.hour
        out["created_dayofweek"] = out["created_at"].dt.dayofweek

    if "won" in out.columns:
        out["won"] = normalize_won(out["won"])

    if "rev" in out.columns:
        out["rev"] = pd.to_numeric(out["rev"], errors="coerce").fillna(0.0)
    if "accepted_listings" in out.columns:
        accepted_listings = pd.to_numeric(
            out["accepted_listings"], errors="coerce"
        ).fillna(0)
        out["sold"] = accepted_listings.gt(0).astype("Int64")
    if {"sold", "rev"}.issubset(out.columns):
        out["realized_revenue"] = out["sold"].astype("float64") * out["rev"]
    if {"sold", "bid"}.issubset(out.columns):
        bid = pd.to_numeric(out["bid"], errors="coerce").fillna(0.0)
        out["bid_cost"] = out["sold"].astype("float64") * bid
    if {"sold", "rev", "bid"}.issubset(out.columns):
        bid = pd.to_numeric(out["bid"], errors="coerce").fillna(0.0)
        out["realized_profit"] = out["sold"].astype("float64") * (out["rev"] - bid)

    return out
