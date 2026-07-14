"""Shared data transforms and metric definitions.

This is the single source of truth for how SmartHub metrics are computed
(profit, contribution margin, win rate). Both dashboards import from here so the
definitions cannot drift apart. See CONTEXT.md for the business meaning.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

# Columns dropped from the raw leads pull because they are unused downstream.
LEADS_DROP_COLS = ["lead_created_at", "excluded"]

# Values in the raw `won` column that should map to 1 (a won lead).
_TRUE_TOKENS = {"true", "t", "1", "yes", "y"}


def safe_divide(
    numerator: pd.Series, denominator: pd.Series, fill: float = 0.0
) -> pd.Series:
    """Element-wise division that avoids inf/NaN blow-ups.

    Inputs
    ------
    numerator : pandas.Series
        The dividend.
    denominator : pandas.Series
        The divisor; 0 and NaN map to ``fill`` in the result.
    fill : float
        Value used where the denominator is 0 or NaN.

    Returns
    -------
    pandas.Series
        The element-wise quotient with ``fill`` substituted.
    """
    denom = pd.Series(denominator).replace(0, np.nan)
    result = pd.Series(numerator).astype("float64") / denom
    return result.replace([np.inf, -np.inf], np.nan).fillna(fill)


def contribution_margin(profit: pd.Series, revenue: pd.Series) -> pd.Series:
    """Contribution margin, ``profit / revenue`` (0 when revenue is 0).

    Inputs
    ------
    profit : pandas.Series
        Profit per row.
    revenue : pandas.Series
        Revenue per row.

    Returns
    -------
    pandas.Series
        The contribution margin.
    """
    return safe_divide(profit, revenue)


def win_rate(won: pd.Series, count: pd.Series) -> pd.Series:
    """Win rate, ``won / count`` (0 when count is 0).

    Inputs
    ------
    won : pandas.Series
        Count of won leads.
    count : pandas.Series
        Total count of leads.

    Returns
    -------
    pandas.Series
        The win rate.
    """
    return safe_divide(won, count)


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


# ---------------------------------------------------------------------------
# Leads side (raw lead_pings -> leads.parquet)
# ---------------------------------------------------------------------------


def prepare_leads_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and enrich the raw leads dataframe.

    Drops unused columns, filters non-positive bids, normalises id/state
    columns, derives time features and computes ``payout`` and ``profit``.
    Pure function (operates on a copy) so it can be unit-tested without
    Streamlit or a Parquet file.

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
        out["rev"] = out["rev"].fillna(0.0)
    if {"won", "bid"}.issubset(out.columns):
        out["payout"] = out["won"].astype("float64") * out["bid"]
    if {"rev", "payout"}.issubset(out.columns):
        out["profit"] = out["rev"] - out["payout"]

    return out


def aggregate_leads(df: pd.DataFrame, aggregate_by: str) -> pd.DataFrame:
    """Aggregate leads by a single column, computing the headline metrics.

    Inputs
    ------
    df : pandas.DataFrame
        Prepared leads frame.
    aggregate_by : str
        Column to group by.

    Returns
    -------
    pandas.DataFrame
        One row per group with count, rev, bid, payout, won, profit, cm and
        winrate.
    """
    agg = (
        df.groupby(aggregate_by, dropna=False)
        .agg(
            count=("id", "count"),
            rev=("rev", "sum"),
            bid=("bid", "sum"),
            payout=("payout", "sum"),
            won=("won", "sum"),
        )
        .reset_index()
    )
    agg["profit"] = agg["rev"] - agg["payout"]
    agg["cm"] = contribution_margin(agg["profit"], agg["rev"])
    agg["winrate"] = win_rate(agg["won"], agg["count"])
    return agg


def build_metric_plot_data(
    df: pd.DataFrame, group_cols: list[str], metric_col: str
) -> pd.DataFrame:
    """Group ``df`` and produce a ``value`` column for the chosen metric.

    Shared by all of the leads plot types.

    Inputs
    ------
    df : pandas.DataFrame
        Prepared leads frame.
    group_cols : list[str]
        Columns to group by.
    metric_col : str
        Metric to compute (``count``, ``cm``, ``winrate`` or a summable
        column).

    Returns
    -------
    pandas.DataFrame
        Grouped rows with a ``value`` column, sorted by ``group_cols``.
    """
    grouped = df.groupby(group_cols, dropna=False)

    if metric_col == "count":
        plot_df = grouped.size().reset_index(name="value")
    elif metric_col == "cm":
        plot_df = grouped.agg(
            rev=("rev", "sum"), profit=("profit", "sum")
        ).reset_index()
        plot_df["value"] = contribution_margin(plot_df["profit"], plot_df["rev"])
    elif metric_col == "winrate":
        plot_df = grouped.agg(
            won=("won", "sum"), count=("id", "count")
        ).reset_index()
        plot_df["value"] = win_rate(plot_df["won"], plot_df["count"])
    else:
        plot_df = grouped[metric_col].sum().reset_index(name="value")

    return plot_df.sort_values(group_cols).reset_index(drop=True)


_CURVE_COLS = [
    "threshold",
    "winrate_below",
    "winrate_above",
    "winrate_delta",
    "n_below",
    "n_above",
]


def cumulative_winrate_curves(
    df: pd.DataFrame,
    bucket_size: float = 1.0,
    bid_col: str = "bid",
    won_col: str = "won",
) -> pd.DataFrame:
    """Win-rate "shelves" curves: win rate bidding at/under vs over a price.

    For each price ``threshold`` (stepped by ``bucket_size``):
    ``winrate_below`` is the win rate over pings with ``bid <= threshold``,
    ``winrate_above`` over pings with ``bid > threshold``, and
    ``winrate_delta`` is above minus below (the lift from bidding past the
    point). Used to spot artificial floors/ceilings and edges where a small
    bid change moves win rate a lot.

    Inputs
    ------
    df : pandas.DataFrame
        Leads frame; must contain ``bid_col`` and ``won_col``.
    bucket_size : float
        Price step between thresholds.
    bid_col : str
        Column holding the bid price.
    won_col : str
        Column holding the won flag.

    Returns
    -------
    pandas.DataFrame
        Columns ``threshold``, ``winrate_below``, ``winrate_above``,
        ``winrate_delta``, ``n_below`` and ``n_above``; empty if inputs are
        missing.
    """
    if not {bid_col, won_col}.issubset(df.columns):
        return pd.DataFrame(columns=_CURVE_COLS)

    sub = df[[bid_col, won_col]]
    sub = sub[pd.notna(sub[bid_col])]
    if sub.empty:
        return pd.DataFrame(columns=_CURVE_COLS)

    bids = sub[bid_col].astype("float64").to_numpy()
    won = pd.to_numeric(sub[won_col], errors="coerce").fillna(0).astype("float64")
    won = won.to_numpy()

    order = np.argsort(bids, kind="mergesort")
    bids, won = bids[order], won[order]
    total = bids.size
    total_won = float(won.sum())
    cum_won = np.cumsum(won)

    lo = np.floor(bids[0] / bucket_size) * bucket_size
    hi = np.ceil(bids[-1] / bucket_size) * bucket_size
    thresholds = np.round(np.arange(lo, hi + bucket_size, bucket_size), 6)

    rows = []
    for x in thresholds:
        idx = int(np.searchsorted(bids, x, side="right"))  # count with bid <= x
        won_below = float(cum_won[idx - 1]) if idx > 0 else 0.0
        n_above = total - idx
        won_above = total_won - won_below
        wr_below = won_below / idx if idx else np.nan
        wr_above = won_above / n_above if n_above else np.nan
        delta = (wr_above - wr_below) if (idx and n_above) else np.nan
        rows.append((x, wr_below, wr_above, delta, idx, n_above))

    return pd.DataFrame(rows, columns=_CURVE_COLS)


def funnel_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Build the accept/reject funnel stages as unambiguous counts.

    Exact accept/reject semantics (``accepted`` vs ``won`` vs
    ``accepted_listings``) are pending confirmation (CONTEXT §4 open
    questions), so these stages use only counts we can define cleanly.

    Inputs
    ------
    df : pandas.DataFrame
        Prepared leads frame.

    Returns
    -------
    pandas.DataFrame
        Columns ``stage`` and ``count``, one row per funnel stage.
    """
    stages: list[tuple[str, int]] = [("Pings", len(df))]
    if "won" in df.columns:
        won = pd.to_numeric(df["won"], errors="coerce").fillna(0)
        stages.append(("Won (partner accepted bid)", int((won == 1).sum())))
    if "accepted_listings" in df.columns:
        resold = pd.to_numeric(df["accepted_listings"], errors="coerce").fillna(0)
        stages.append(("Has accepted listing (resold)", int((resold > 0).sum())))
    return pd.DataFrame(stages, columns=["stage", "count"])


# ---------------------------------------------------------------------------
# Monitoring side (pre-aggregated ETL metrics)
# ---------------------------------------------------------------------------

MONITORING_NUMERIC_COLS = [
    "campaign_id",
    "revenue_measured",
    "revenue_expected",
    "payout",
    "profit",
    "cm_expected",
    "cm_measured",
    "num_opportunities",
    "num_won",
    "winrate",
]

MONITORING_SUM_COLS = [
    "revenue_measured",
    "revenue_expected",
    "payout",
    "num_opportunities",
    "num_won",
]


def leads_to_monitoring_base(df: pd.DataFrame) -> pd.DataFrame:
    """Turn cleaned per-ping leads into the per-row monitoring base.

    Produces the columns ``aggregate_monitoring`` expects (one row per
    ping): ``datetime_min`` (from ``created_at``), ``state``,
    ``campaign_id`` and the sum bases ``num_opportunities``, ``num_won``,
    ``revenue_measured``, ``revenue_expected`` and ``payout``. Profit / CM /
    win rate are derived after resampling.

    Inputs
    ------
    df : pandas.DataFrame
        Prepared leads frame (from ``io.load_leads``).

    Returns
    -------
    pandas.DataFrame
        The per-row monitoring base, with rows lacking a datetime dropped.
    """
    out = pd.DataFrame(index=df.index)
    out["datetime_min"] = pd.to_datetime(df["created_at"], errors="coerce")
    out["state"] = df.get("state")
    out["campaign_id"] = df.get("campaign_id")
    won = pd.to_numeric(df.get("won"), errors="coerce").fillna(0)
    out["num_opportunities"] = 1
    out["num_won"] = won.astype("int64")
    out["revenue_measured"] = pd.to_numeric(df.get("rev"), errors="coerce").fillna(0.0)
    if "expected_revenue" in df.columns:
        out["revenue_expected"] = pd.to_numeric(
            df["expected_revenue"], errors="coerce"
        ).fillna(0.0)
    else:
        out["revenue_expected"] = 0.0
    out["payout"] = pd.to_numeric(df.get("payout"), errors="coerce").fillna(0.0)
    return out.dropna(subset=["datetime_min"]).reset_index(drop=True)


def add_monitoring_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Recompute profit, contribution margins and win rate for monitoring data.

    Inputs
    ------
    df : pandas.DataFrame
        Monitoring base or aggregated sums.

    Returns
    -------
    pandas.DataFrame
        A copy with ``profit``, ``cm_expected``, ``cm_measured`` and
        ``winrate`` added.
    """
    out = df.copy()
    out["profit"] = out["revenue_measured"] - out["payout"]
    out["cm_expected"] = contribution_margin(
        out["revenue_expected"] - out["payout"], out["revenue_expected"]
    )
    out["cm_measured"] = contribution_margin(
        out["revenue_measured"] - out["payout"], out["revenue_measured"]
    )
    out["winrate"] = win_rate(out["num_won"], out["num_opportunities"])
    return out


def aggregate_monitoring(
    df: pd.DataFrame, freq: str, group_col: str | None = None
) -> pd.DataFrame:
    """Resample monitoring data, optionally grouped, and derive metrics.

    Resamples to ``freq`` (summing the base columns) then recomputes the
    derived metrics on the aggregated sums.

    Inputs
    ------
    df : pandas.DataFrame
        Monitoring base with a ``datetime_min`` column.
    freq : str
        Pandas resample frequency (e.g. ``D``, ``H``).
    group_col : str | None
        Optional column to group by before resampling.

    Returns
    -------
    pandas.DataFrame
        The resampled frame with derived metric columns.
    """
    indexed = df.set_index("datetime_min")
    if group_col is None:
        agg = (
            indexed[MONITORING_SUM_COLS]
            .resample(freq, label="left", closed="left")
            .sum()
            .reset_index()
            .sort_values("datetime_min")
        )
    else:
        agg = (
            indexed.groupby(group_col)[MONITORING_SUM_COLS]
            .resample(freq, label="left", closed="left")
            .sum()
            .reset_index()
            .sort_values([group_col, "datetime_min"])
        )
    return add_monitoring_derived_columns(agg)
