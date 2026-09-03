"""Data preparation and metric transforms for SmartHub monitoring.

This module prepares dashboard data and computes shared Leads and Performance
metrics and aggregations.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from smarthub.core.transforms import normalize_won

MONITORING_NUMERIC_COLS = [
    "campaign_id",
    "realized_revenue",
    "expected_revenue",
    "bid_cost",
    "realized_profit",
    "cm_expected",
    "cm_measured",
    "num_opportunities",
    "num_won",
    "winrate",
]
MONITORING_SUM_COLS = [
    "realized_revenue",
    "expected_revenue",
    "bid_cost",
    "num_opportunities",
    "num_won",
]
_CURVE_COLS = [
    "threshold",
    "winrate_below",
    "winrate_above",
    "winrate_delta",
    "n_below",
    "n_above",
]


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
        One row per group with count, realized_revenue, bid,
        bid_cost, won, realized_profit, cm and winrate.
    """
    agg = (
        df.groupby(aggregate_by, dropna=False)
        .agg(
            count=("id", "count"),
            realized_revenue=("realized_revenue", "sum"),
            bid=("bid", "sum"),
            bid_cost=("bid_cost", "sum"),
            won=("won", "sum"),
        )
        .reset_index()
    )
    agg["realized_profit"] = agg["realized_revenue"] - agg["bid_cost"]
    agg["cm"] = contribution_margin(agg["realized_profit"], agg["realized_revenue"])
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
            realized_revenue=("realized_revenue", "sum"),
            realized_profit=("realized_profit", "sum"),
        ).reset_index()
        plot_df["value"] = contribution_margin(
            plot_df["realized_profit"], plot_df["realized_revenue"]
        )
    elif metric_col == "winrate":
        plot_df = grouped.agg(won=("won", "sum"), count=("id", "count")).reset_index()
        plot_df["value"] = win_rate(plot_df["won"], plot_df["count"])
    else:
        plot_df = grouped[metric_col].sum().reset_index(name="value")

    return plot_df.sort_values(group_cols).reset_index(drop=True)


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
    """Build opportunity, won, and sold funnel counts.

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
        sold = pd.to_numeric(df["accepted_listings"], errors="coerce").fillna(0)
        stages.append(("Sold", int((sold > 0).sum())))
    return pd.DataFrame(stages, columns=["stage", "count"])


def leads_to_monitoring_base(df: pd.DataFrame) -> pd.DataFrame:
    """Turn cleaned per-ping leads into the per-row monitoring base.

    Produces the columns ``aggregate_monitoring`` expects (one row per
    ping): ``datetime_min`` (from ``created_at``), ``state``,
    ``campaign_id`` and the sum bases ``num_opportunities``, ``num_won``,
    ``realized_revenue``, ``expected_revenue`` and ``bid_cost``. Profit / CM /
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
    realized_source = df["rev"] if "rev" in df.columns else df.get("realized_revenue")
    out["realized_revenue"] = pd.to_numeric(realized_source, errors="coerce").fillna(
        0.0
    )
    if "expected_revenue" in df.columns:
        out["expected_revenue"] = pd.to_numeric(
            df["expected_revenue"], errors="coerce"
        ).fillna(0.0)
    else:
        out["expected_revenue"] = 0.0
    if "bid_cost" in df.columns:
        out["bid_cost"] = pd.to_numeric(df["bid_cost"], errors="coerce").fillna(0.0)
    elif "bid" in df.columns:
        bid = pd.to_numeric(df["bid"], errors="coerce").fillna(0.0)
        if "accepted_listings" in df.columns:
            accepted_listings = pd.to_numeric(
                df["accepted_listings"], errors="coerce"
            ).fillna(0)
            sold = accepted_listings.gt(0).astype("float64")
        else:
            sold = pd.Series(0.0, index=df.index, dtype="float64")
        out["bid_cost"] = sold * bid
    else:
        out["bid_cost"] = 0.0
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
        A copy with ``realized_profit``, ``cm_expected``, ``cm_measured`` and
        ``winrate`` added.
    """
    out = df.copy()
    out["realized_profit"] = out["realized_revenue"] - out["bid_cost"]
    out["cm_expected"] = contribution_margin(
        out["expected_revenue"] - out["bid_cost"], out["expected_revenue"]
    )
    out["cm_measured"] = contribution_margin(
        out["realized_profit"], out["realized_revenue"]
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


def resolve_model_expected_revenue(df: pd.DataFrame) -> pd.Series:
    """Resolve the expected-revenue series used by monitoring metrics.

    Inputs
    ------
    df : pandas.DataFrame
        Data containing ``exp_rev`` and/or ``expected_revenue``.

    Returns
    -------
    pandas.Series
        Expected revenue with positive ``exp_rev`` preferred over the fallback
        ``expected_revenue`` value.
    """
    fallback = (
        pd.to_numeric(df["expected_revenue"], errors="coerce")
        if "expected_revenue" in df.columns
        else pd.Series(np.nan, index=df.index, dtype="float64")
    )
    if "exp_rev" not in df.columns:
        return fallback
    backend = pd.to_numeric(df["exp_rev"], errors="coerce")
    return backend.where(backend > 0, fallback)


def add_historical_business_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Add canonical per-lead historical business metrics.

    Inputs
    ------
    df : pandas.DataFrame
        Lead-level data containing historical outcomes and revenue fields.

    Returns
    -------
    pandas.DataFrame
        Copy containing normalized win, sold, expected, realized, cost, and profit
        fields used by monitoring.
    """
    out = df.copy()
    if "created_at" in out.columns:
        out["datetime_min"] = pd.to_datetime(
            out["created_at"], errors="coerce", utc=True
        )

    won = (
        normalize_won(out["won"]).astype("float64")
        if "won" in out.columns
        else pd.Series(0.0, index=out.index, dtype="float64")
    )
    bid = (
        pd.to_numeric(out["bid"], errors="coerce").fillna(0.0)
        if "bid" in out.columns
        else pd.Series(0.0, index=out.index, dtype="float64")
    )
    raw_revenue = (
        pd.to_numeric(out["rev"], errors="coerce").fillna(0.0)
        if "rev" in out.columns
        else pd.to_numeric(out.get("realized_revenue", 0.0), errors="coerce").fillna(
            0.0
        )
    )
    accepted_listings = (
        pd.to_numeric(out["accepted_listings"], errors="coerce").fillna(0.0)
        if "accepted_listings" in out.columns
        else pd.Series(0.0, index=out.index, dtype="float64")
    )
    sold = accepted_listings.gt(0).astype("float64")
    expected_revenue = resolve_model_expected_revenue(out)

    out["won_numeric"] = won
    out["sold"] = sold
    out["num_opportunities"] = 1
    out["num_won"] = won
    out["expected_revenue"] = expected_revenue
    out["expected_profit"] = expected_revenue - bid

    # Realized economics exist only when the lead was sold downstream.
    out["realized_revenue"] = sold * raw_revenue
    out["bid_cost"] = sold * bid
    out["realized_profit"] = sold * (raw_revenue - bid)
    return out


def add_recommended_bid_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Add probability-weighted economics at the recommended bid.

    Inputs
    ------
    df : pandas.DataFrame
        Data containing recommended bid, predicted win rate, and expected revenue.

    Returns
    -------
    pandas.DataFrame
        Copy containing predicted revenue, bid cost, and profit for the recommended
        bid.
    """
    out = df.copy()
    p = pd.to_numeric(out.get("recommended_bid_predicted_win_rate"), errors="coerce")
    bid = pd.to_numeric(out.get("recommended_bid"), errors="coerce")
    if "prediction_expected_revenue" in out.columns:
        revenue = pd.to_numeric(out["prediction_expected_revenue"], errors="coerce")
    elif "expected_revenue" in out.columns:
        revenue = pd.to_numeric(out["expected_revenue"], errors="coerce")
    else:
        revenue = resolve_model_expected_revenue(out)

    out["recommended_bid_predicted_revenue"] = p * revenue
    out["recommended_bid_predicted_bid_cost"] = p * bid
    calculated = p * (revenue - bid)
    if "recommended_bid_predicted_profit" in out.columns:
        stored = pd.to_numeric(out["recommended_bid_predicted_profit"], errors="coerce")
        out["recommended_bid_predicted_profit"] = stored.where(
            stored.notna(), calculated
        )
    else:
        out["recommended_bid_predicted_profit"] = calculated
    return out


def _add_outcome_conditioned_expected_metrics(work: pd.DataFrame) -> pd.DataFrame:
    """Add temporary columns used only while aggregating Performance metrics."""
    out = work.copy()
    revenue = pd.to_numeric(out["expected_revenue"], errors="coerce").fillna(0.0)
    profit = pd.to_numeric(out["expected_profit"], errors="coerce").fillna(0.0)
    won = pd.to_numeric(out["won_numeric"], errors="coerce").fillna(0.0)
    sold = pd.to_numeric(out["sold"], errors="coerce").fillna(0.0)
    out["_expected_revenue_on_wins"] = won * revenue
    out["_expected_profit_on_wins"] = won * profit
    out["_expected_revenue_on_sold"] = sold * revenue
    out["_expected_profit_on_sold"] = sold * profit
    return out


def aggregate_historical_business_metrics(
    df: pd.DataFrame, *, freq: str, group_col: str | None = None
) -> pd.DataFrame:
    """Aggregate historical business metrics over time and an optional group.

    Inputs
    ------
    df : pandas.DataFrame
        Per-lead historical business metrics.
    freq : str
        Pandas frequency used to bucket ``datetime_min``.
    group_col : str | None
        Optional dimension to aggregate independently.

    Returns
    -------
    pandas.DataFrame
        Aggregated historical counts, economics, win rate, and contribution margins.
    """
    work = df[df["datetime_min"].notna()].copy()
    if work.empty:
        return pd.DataFrame()
    work["datetime_min"] = work["datetime_min"].dt.floor(freq)
    work = _add_outcome_conditioned_expected_metrics(work)
    keys = ["datetime_min"] + ([group_col] if group_col else [])
    agg = (
        work.groupby(keys, dropna=False, observed=False)
        .agg(
            num_opportunities=("num_opportunities", "sum"),
            num_won=("num_won", "sum"),
            realized_revenue=("realized_revenue", "sum"),
            bid_cost=("bid_cost", "sum"),
            realized_profit=("realized_profit", "sum"),
            expected_revenue=("expected_revenue", "sum"),
            expected_profit=("expected_profit", "sum"),
            expected_revenue_on_sold=("_expected_revenue_on_sold", "sum"),
            expected_profit_on_sold=("_expected_profit_on_sold", "sum"),
            expected_revenue_on_wins=("_expected_revenue_on_wins", "sum"),
            expected_profit_on_wins=("_expected_profit_on_wins", "sum"),
        )
        .reset_index()
    )
    agg["measured_winrate"] = win_rate(agg["num_won"], agg["num_opportunities"])
    agg["realized_cm"] = contribution_margin(
        agg["realized_profit"], agg["realized_revenue"]
    )
    agg["expected_cm_on_wins"] = contribution_margin(
        agg["expected_profit_on_wins"], agg["expected_revenue_on_wins"]
    )
    return agg


def recommended_bid_metrics_available_mask(df: pd.DataFrame) -> pd.Series:
    """Identify rows with complete recommended-bid comparison metrics.

    Inputs
    ------
    df : pandas.DataFrame
        Data containing recommended-bid prediction fields.

    Returns
    -------
    pandas.Series
        Boolean mask marking rows with all required ML comparison metrics.
    """
    required = (
        "recommended_bid_predicted_win_rate",
        "recommended_bid_predicted_revenue",
        "recommended_bid_predicted_bid_cost",
        "recommended_bid_predicted_profit",
    )
    if any(column not in df.columns for column in required):
        return pd.Series(False, index=df.index, dtype="bool")
    mask = pd.Series(True, index=df.index, dtype="bool")
    for column in required:
        mask &= pd.to_numeric(df[column], errors="coerce").notna()
    return mask


def aggregate_recommended_bid_comparison(
    df: pd.DataFrame, *, freq: str, group_col: str | None = None
) -> pd.DataFrame:
    """Aggregate historical and recommended-bid metrics for comparable rows.

    Inputs
    ------
    df : pandas.DataFrame
        Per-lead historical and recommended-bid metrics.
    freq : str
        Pandas frequency used to bucket ``datetime_min``.
    group_col : str | None
        Optional dimension to aggregate independently.

    Returns
    -------
    pandas.DataFrame
        Aggregated historical and ML economics, win rates, and contribution margins.
    """
    ml_mask = recommended_bid_metrics_available_mask(df)
    work = df[ml_mask & df["datetime_min"].notna()].copy()
    if work.empty:
        return pd.DataFrame()
    work["datetime_min"] = work["datetime_min"].dt.floor(freq)
    work = _add_outcome_conditioned_expected_metrics(work)
    keys = ["datetime_min"] + ([group_col] if group_col else [])
    agg = (
        work.groupby(keys, dropna=False, observed=False)
        .agg(
            num_opportunities=("id", "size"),
            num_won=("won_numeric", "sum"),
            realized_revenue=("realized_revenue", "sum"),
            bid_cost=("bid_cost", "sum"),
            realized_profit=("realized_profit", "sum"),
            expected_revenue_on_wins=("_expected_revenue_on_wins", "sum"),
            expected_profit_on_wins=("_expected_profit_on_wins", "sum"),
            recommended_bid_predicted_revenue=(
                "recommended_bid_predicted_revenue",
                "sum",
            ),
            recommended_bid_predicted_bid_cost=(
                "recommended_bid_predicted_bid_cost",
                "sum",
            ),
            recommended_bid_predicted_profit=(
                "recommended_bid_predicted_profit",
                "sum",
            ),
            recommended_bid_predicted_win_rate=(
                "recommended_bid_predicted_win_rate",
                "mean",
            ),
        )
        .reset_index()
    )
    agg["measured_winrate"] = win_rate(agg["num_won"], agg["num_opportunities"])
    agg["realized_cm"] = contribution_margin(
        agg["realized_profit"], agg["realized_revenue"]
    )
    agg["expected_cm_on_wins"] = contribution_margin(
        agg["expected_profit_on_wins"], agg["expected_revenue_on_wins"]
    )
    agg["recommended_bid_predicted_cm"] = contribution_margin(
        agg["recommended_bid_predicted_profit"],
        agg["recommended_bid_predicted_revenue"],
    )
    return agg
