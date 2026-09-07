"""SmartHub Performance dashboard.

This module loads historical and prediction-monitoring data, computes business and
ML performance metrics, and renders interactive performance views.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from smarthub.core import io
from smarthub.monitoring import transforms

_BIN_MAP = {"1hr": "1h", "6hr": "6h", "12hr": "12h", "day": "D", "week": "W-MON"}

_FILTER_DIMENSIONS = [
    "campaign_id",
    "bidding_strategy_id",
    "traffic_tier",
    "source_type_id",
    "state",
]

PREDICTION_MONITORING_PATH = Path(
    "data/raw_datasets/monitoring_datasets/prediction_monitoring.parquet"
)

HISTORICAL_METRIC_GROUPS = {
    "All": None,
    "Sold Lead Revenue": [
        "bid_cost",
        "realized_revenue",
        "expected_revenue_on_sold",
    ],
    "Sold Lead Profit": [
        "realized_profit",
        "expected_profit_on_sold",
    ],
    "ML Revenue Comparison": [
        "recommended_bid_predicted_revenue",
        "expected_revenue_on_wins",
    ],
    "ML Profit Comparison": [
        "recommended_bid_predicted_profit",
        "expected_profit_on_wins",
    ],
    "Win Rate": ["measured_winrate"],
    "Profit Realization Rate": ["profit_realization_rate"],
    "P(Sold | Won)": ["p_sold_given_won"],
    "CM": ["realized_cm", "expected_cm_on_wins"],
    "Number of opportunities, won, and sold leads": [
        "num_opportunities",
        "num_won",
        "num_sold",
    ],
}

_STYLE_MAP = {
    # Revenue family: same color, source shown by line style.
    "realized_revenue": dict(color="#1f77b4", dash="solid"),
    "expected_revenue_on_sold": dict(color="#1f77b4", dash="dot"),
    "recommended_bid_predicted_revenue_on_sold": dict(color="#1f77b4", dash="dash"),
    # Bid-cost family.
    "bid_cost": dict(color="#000000", dash="solid"),
    "recommended_bid_predicted_bid_cost": dict(color="#000000", dash="dash"),
    # Profit family.
    "realized_profit": dict(color="#ff7f0e", dash="solid"),
    "expected_profit_on_sold": dict(color="#ff7f0e", dash="dot"),
    "recommended_bid_predicted_profit_on_sold": dict(color="#ff7f0e", dash="dash"),
    "expected_revenue_on_wins": dict(color="#1f77b4", dash="dot"),
    "recommended_bid_predicted_revenue": dict(color="#1f77b4", dash="dash"),
    "expected_profit_on_wins": dict(color="#ff7f0e", dash="dot"),
    "recommended_bid_predicted_profit": dict(color="#ff7f0e", dash="dash"),
    # Win-rate family.
    "measured_winrate": dict(color="#2ca02c", dash="solid"),
    "recommended_bid_predicted_win_rate": dict(color="#2ca02c", dash="dash"),
    "profit_realization_rate": dict(color="#9467bd", dash="solid"),
    "p_sold_given_won": dict(color="#ff7f0e", dash="solid"),
    # CM family.
    "realized_cm": dict(color="#9467bd", dash="solid"),
    "expected_cm_on_wins": dict(color="#9467bd", dash="dot"),
    "recommended_bid_predicted_cm": dict(color="#9467bd", dash="dash"),
    # Volume metrics.
    "num_opportunities": dict(color="#7f7f7f", dash="solid"),
    "num_won": dict(color="#2ca02c", dash="solid"),
    "num_sold": dict(color="#ff7f0e", dash="solid"),
}

_METRIC_LABELS = {
    "realized_revenue": "Realized Revenue",
    "expected_revenue_on_sold": "Expected Revenue",
    "recommended_bid_predicted_revenue_on_sold": "ML Expected Revenue",
    "bid_cost": "Bid Cost",
    "recommended_bid_predicted_bid_cost": "ML Predicted Bid Cost",
    "realized_profit": "Realized Profit",
    "expected_profit_on_sold": "Expected Profit — Sold Leads",
    "expected_revenue_on_wins": "Expected Revenue — Won Leads",
    "expected_profit_on_wins": "Observed Expected Profit — Won Leads",
    "recommended_bid_predicted_revenue": "ML Predicted Revenue — All Leads",
    "recommended_bid_predicted_profit": "ML Predicted Profit — Recommended Bid",
    "recommended_bid_predicted_profit_on_sold": "ML Expected Profit — Sold Leads",
    "measured_winrate": "Measured Win Rate",
    "recommended_bid_predicted_win_rate": "ML Predicted Win Rate",
    "profit_realization_rate": "Profit Realization Rate",
    "p_sold_given_won": "P(Sold | Won)",
    "realized_cm": "Realized CM",
    "expected_cm_on_wins": "Expected CM on Wins",
    "recommended_bid_predicted_cm": "ML Predicted CM",
    "num_opportunities": "Opportunities",
    "num_won": "Won",
    "num_sold": "Sold",
}

_FEATURE_EXCLUDE = {
    "id",
    "created_at",
    "lead_created_at",
    "lead_type_id",
    "campaign_id",
    "bid",
    "rev",
    "won",
    "accepted",
    "erred",
    "error_reason_id",
    "response_ms",
    "expected_revenue",
    "exp_rev",
    "model_expected_revenue",
    "realized_payout",
    "num_selected_listings",
    "realized_revenue",
    "sold",
    "bid_cost",
    "realized_profit",
    "expected_revenue_on_sold",
    "expected_profit_on_sold",
    "expected_revenue_on_wins",
    "expected_profit_on_wins",
    "measured_winrate",
    "p_sold_given_won",
    "profit_realization_rate",
    "realized_cm",
    "expected_cm_on_wins",
    "num_opportunities",
    "num_won",
    "num_sold",
    "recommended_bid",
    "recommended_bid_predicted_win_rate",
    "recommended_bid_predicted_revenue",
    "recommended_bid_predicted_bid_cost",
    "recommended_bid_predicted_profit",
    "recommended_bid_predicted_revenue_on_sold",
    "recommended_bid_predicted_profit_on_sold",
    "recommended_bid_predicted_win_rate",
    "recommended_bid_predicted_cm",
}


@st.cache_data
def load_leads(days: int):
    """Load a recent window of historical leads.

    Inputs
    ------
    days : int
        Number of recent days to load.

    Returns
    -------
    pandas.DataFrame
        Historical lead rows within the requested window.
    """
    return io.load_leads_window(days)


@st.cache_data
def load_prediction_monitoring(days: int) -> pd.DataFrame:
    """Load recent prediction-monitoring rows from local parquet storage.

    Inputs
    ------
    days : int
        Number of recent days to retain after loading the dataset.

    Returns
    -------
    pandas.DataFrame
        Successful prediction-monitoring rows with usable lead identifiers.

    Raises
    ------
    io.DataNotFoundError
        Raised when the prediction-monitoring parquet dataset does not exist.
    """
    if not PREDICTION_MONITORING_PATH.exists():
        raise io.DataNotFoundError(
            f"Prediction monitoring dataset not found: {PREDICTION_MONITORING_PATH}"
        )

    df = pd.read_parquet(PREDICTION_MONITORING_PATH)
    if df.empty:
        return df

    for column in ("created_at", "served_at"):
        if column in df.columns:
            df[column] = pd.to_datetime(df[column], errors="coerce", utc=True)

    numeric_columns = (
        "lead_ping_id",
        "lead_type_id",
        "campaign_id",
        "source_type_id",
        "expected_revenue",
        "recommended_bid",
        "recommended_bid_predicted_win_rate",
        "recommended_bid_predicted_profit",
        "recommended_bid_predicted_cm",
        "tat_seconds",
        "lead_rev",
        "lead_exp_rev",
    )
    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    time_col = "served_at" if "served_at" in df.columns else "created_at"
    if time_col in df.columns:
        cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=int(days))
        df = df[df[time_col].ge(cutoff)].copy()

    if "status" in df.columns:
        df = df[
            df["status"].astype("string").str.lower().isin({"success", "ok"})
        ].copy()

    if "lead_ping_id" not in df.columns:
        return pd.DataFrame()

    df = df[df["lead_ping_id"].notna()].copy()
    sort_col = "served_at" if "served_at" in df.columns else "created_at"
    if sort_col in df.columns:
        df = df.sort_values(sort_col, ascending=False, na_position="last")

    return df.drop_duplicates("lead_ping_id", keep="first").reset_index(drop=True)


def attach_prediction_monitoring(
    leads_df: pd.DataFrame,
    prediction_df: pd.DataFrame,
) -> pd.DataFrame:
    """Attach prediction-monitoring metrics to historical lead rows.

    Inputs
    ------
    leads_df : pandas.DataFrame
        Historical leads to enrich.
    prediction_df : pandas.DataFrame
        Prediction-monitoring rows keyed by lead ping identifier.

    Returns
    -------
    pandas.DataFrame
        Historical leads with the latest matching prediction metrics and derived ML
        economics.
    """
    if prediction_df.empty or "id" not in leads_df.columns:
        return leads_df.copy()

    left = leads_df.copy()
    left["_lead_ping_id"] = pd.to_numeric(left["id"], errors="coerce").astype("Int64")

    right = prediction_df.copy()
    right["_lead_ping_id"] = pd.to_numeric(
        right["lead_ping_id"], errors="coerce"
    ).astype("Int64")

    keep = [
        column
        for column in (
            "_lead_ping_id",
            "prediction_id",
            "served_at",
            "model_name",
            "model_version",
            "model_type",
            "training_table_version",
            "decision_path",
            "status",
            "expected_revenue",
            "recommended_bid",
            "recommended_bid_predicted_win_rate",
            "recommended_bid_predicted_profit",
            "recommended_bid_predicted_cm",
            "tat_seconds",
            "lead_won",
            "lead_rev",
            "lead_exp_rev",
        )
        if column in right.columns
    ]
    right = right[keep].rename(
        columns={"expected_revenue": "prediction_expected_revenue"}
    )

    merged = left.merge(
        right,
        on="_lead_ping_id",
        how="left",
        validate="many_to_one",
    ).drop(columns="_lead_ping_id")

    return transforms.add_recommended_bid_metrics(merged)


def _feature_columns(frame: pd.DataFrame) -> list[str]:
    """Return useful live feature/context columns for feature analysis."""
    columns = []
    for column in frame.columns:
        if column in _FEATURE_EXCLUDE:
            continue
        if frame[column].notna().sum() == 0:
            continue
        columns.append(column)
    return sorted(columns)


def _feature_kind(frame: pd.DataFrame, feature: str) -> str:
    """Classify a live feature as categorical, discrete, or continuous."""
    if feature == "id" or feature.endswith("_id"):
        return "categorical"
    series = frame[feature]
    if not pd.api.types.is_numeric_dtype(series):
        return "categorical"
    unique_count = pd.to_numeric(series, errors="coerce").nunique(dropna=True)
    if unique_count <= 20:
        return "discrete"
    return "continuous"


def _feature_buckets(
    frame: pd.DataFrame,
    feature: str,
    *,
    bins: int,
    binning: str,
    top_n: int,
    min_support: int,
) -> pd.Series:
    """Build feature buckets using the same broad rules as model diagnostics."""
    series = frame[feature]
    kind = _feature_kind(frame, feature)

    if kind == "continuous":
        numeric = pd.to_numeric(series, errors="coerce")
        if numeric.dropna().nunique() <= 1:
            return numeric.astype("string").fillna("<NA>")
        if binning == "quantile":
            try:
                bucketed = pd.qcut(numeric, q=bins, duplicates="drop")
            except ValueError:
                bucketed = pd.cut(numeric, bins=bins, duplicates="drop")
        else:
            bucketed = pd.cut(numeric, bins=bins, duplicates="drop")
        return bucketed.astype("string").fillna("<NA>")

    values = series.astype("string").fillna("<NA>").replace("", "<EMPTY>")
    counts = values.value_counts(dropna=False)
    keep = counts[counts >= min_support].index.tolist()
    if top_n > 0:
        keep = keep[:top_n]
    return values.where(values.isin(keep), "Other")


def _with_outcome_expected_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    """Add temporary outcome-conditioned expected metrics for aggregation only.

    The canonical lead-level frame keeps one value per business concept. These
    underscore-prefixed columns exist only inside Performance aggregations.
    """
    working = frame.copy()
    revenue = pd.to_numeric(working.get("expected_revenue"), errors="coerce").fillna(
        0.0
    )
    profit = pd.to_numeric(working.get("expected_profit"), errors="coerce").fillna(0.0)
    won = pd.to_numeric(working.get("won_numeric"), errors="coerce").fillna(0.0)
    sold = pd.to_numeric(working.get("sold"), errors="coerce").fillna(0.0)

    working["_expected_revenue_on_sold"] = sold * revenue
    working["_expected_profit_on_sold"] = sold * profit
    working["_expected_revenue_on_wins"] = won * revenue
    working["_expected_profit_on_wins"] = won * profit
    return working


def _build_feature_summary(
    frame: pd.DataFrame,
    feature: str,
    *,
    bins: int,
    binning: str,
    top_n: int,
    min_support: int,
) -> pd.DataFrame:
    """Aggregate realized production economics by feature value/bucket."""
    working = _with_outcome_expected_metrics(frame)
    working["feature_bucket"] = _feature_buckets(
        working,
        feature,
        bins=bins,
        binning=binning,
        top_n=top_n,
        min_support=min_support,
    )

    summary = (
        working.groupby("feature_bucket", observed=False)
        .agg(
            leads=("feature_bucket", "size"),
            wins=("won_numeric", "sum"),
            win_rate=("won_numeric", "mean"),
            avg_bid=("bid", "mean"),
            median_bid=("bid", "median"),
            realized_revenue=("realized_revenue", "sum"),
            bid_cost=("bid_cost", "sum"),
            realized_profit=("realized_profit", "sum"),
            expected_revenue_on_wins=("_expected_revenue_on_wins", "sum"),
            expected_profit_on_wins=("_expected_profit_on_wins", "sum"),
        )
        .reset_index()
    )
    summary = summary[summary["leads"] >= min_support].copy()
    if summary.empty:
        return summary

    total_rows = float(summary["leads"].sum())
    summary["fraction_of_filtered_rows"] = summary["leads"] / total_rows
    summary["realized_revenue_per_lead"] = (
        summary["realized_revenue"] / summary["leads"]
    )
    summary["realized_profit_per_lead"] = summary["realized_profit"] / summary["leads"]
    summary["realized_cm"] = np.where(
        summary["realized_revenue"] != 0.0,
        summary["realized_profit"] / summary["realized_revenue"],
        np.nan,
    )
    summary["expected_cm_on_wins"] = np.where(
        summary["expected_revenue_on_wins"] != 0.0,
        summary["expected_profit_on_wins"] / summary["expected_revenue_on_wins"],
        np.nan,
    )

    if _feature_kind(frame, feature) == "continuous":
        sort_key = pd.to_numeric(
            summary["feature_bucket"]
            .astype(str)
            .str.extract(
                r"^[\[(]\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
                expand=False,
            ),
            errors="coerce",
        )
        summary = (
            summary.assign(_sort_key=sort_key)
            .sort_values("_sort_key", kind="stable", na_position="last")
            .drop(columns="_sort_key")
            .reset_index(drop=True)
        )
    return summary


def _feature_win_rate_chart(summary: pd.DataFrame, feature: str) -> go.Figure:
    """Plot observed win rate with lead support by feature."""
    labels = summary["feature_bucket"].astype(str).tolist()
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_bar(
        x=labels,
        y=summary["leads"],
        name="Leads",
        opacity=0.20,
        secondary_y=True,
    )
    fig.add_trace(
        go.Scatter(
            x=labels,
            y=summary["win_rate"],
            mode="lines+markers",
            name="Measured win rate",
        ),
        secondary_y=False,
    )
    fig.update_layout(
        title=f"Measured Win Rate by {feature}",
        hovermode="x unified",
        legend=dict(orientation="h", x=0, xanchor="left", y=-0.22, yanchor="top"),
        margin=dict(l=20, r=20, t=60, b=120),
    )
    fig.update_yaxes(title_text="Win rate", range=[0, 1], secondary_y=False)
    fig.update_yaxes(title_text="Number of leads", secondary_y=True)
    fig.update_xaxes(title_text=feature)
    return fig


def _feature_bid_chart(summary: pd.DataFrame, feature: str) -> go.Figure:
    """Plot average and median actual bid by feature."""
    labels = summary["feature_bucket"].astype(str).tolist()
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=labels,
            y=summary["avg_bid"],
            mode="lines+markers",
            name="Average bid",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=labels,
            y=summary["median_bid"],
            mode="lines+markers",
            name="Median bid",
        )
    )
    fig.update_layout(
        title=f"Historical Bid by {feature}",
        xaxis_title=feature,
        yaxis_title="Bid ($)",
        hovermode="x unified",
        legend=dict(orientation="h", x=0, xanchor="left", y=-0.22, yanchor="top"),
        margin=dict(l=20, r=20, t=60, b=120),
    )
    return fig


def _feature_economics_chart(summary: pd.DataFrame, feature: str) -> go.Figure:
    """Plot canonical realized economics by feature."""
    labels = summary["feature_bucket"].astype(str).tolist()
    fig = go.Figure()
    for column, name in (
        ("realized_revenue", "Realized revenue"),
        ("bid_cost", "Realized bid cost"),
        ("realized_profit", "Realized profit"),
    ):
        fig.add_trace(go.Bar(x=labels, y=summary[column], name=name))
    fig.update_layout(
        title=f"Realized Economics by {feature}",
        xaxis_title=feature,
        yaxis_title="Amount ($)",
        barmode="group",
        hovermode="x unified",
        legend=dict(orientation="h", x=0, xanchor="left", y=-0.22, yanchor="top"),
        margin=dict(l=20, r=20, t=60, b=120),
    )
    return fig


def _feature_cm_chart(summary: pd.DataFrame, feature: str) -> go.Figure:
    """Plot realized contribution margin with lead support by feature."""
    labels = summary["feature_bucket"].astype(str).tolist()
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_bar(
        x=labels,
        y=summary["leads"],
        name="Leads",
        opacity=0.20,
        secondary_y=True,
    )
    fig.add_trace(
        go.Scatter(
            x=labels,
            y=summary["realized_cm"],
            mode="lines+markers",
            name="Realized CM",
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=labels,
            y=summary["expected_cm_on_wins"],
            mode="lines+markers",
            name="Expected CM on wins",
        ),
        secondary_y=False,
    )
    fig.update_layout(
        title=f"CM by {feature}",
        hovermode="x unified",
        legend=dict(orientation="h", x=0, xanchor="left", y=-0.22, yanchor="top"),
        margin=dict(l=20, r=20, t=60, b=120),
    )
    fig.update_yaxes(title_text="CM", secondary_y=False)
    fig.update_yaxes(title_text="Number of leads", secondary_y=True)
    fig.update_xaxes(title_text=feature)
    return fig


def plot_metric_block(
    df_plot: pd.DataFrame,
    columns: list[str],
    title: str,
    y_label: str,
    *,
    x_col: str = "datetime_min",
    x_label: str | None = None,
) -> None:
    """Render an overlaid line chart for a selected metric group.

    Inputs
    ------
    df_plot : pandas.DataFrame
        Aggregated metric data to plot.
    columns : list[str]
        Metric columns to include in the chart.
    title : str
        Chart title.
    y_label : str
        Label for the y-axis.
    """
    available = [c for c in columns if c in df_plot.columns]
    if df_plot.empty or not available or x_col not in df_plot.columns:
        st.info(f"No data available for {title}.")
        return

    plot_df = df_plot[[x_col] + available].melt(
        id_vars=x_col,
        var_name="metric",
        value_name="value",
    )
    fig = px.line(
        plot_df,
        x=x_col,
        y="value",
        color="metric",
        markers=True,
    )
    for trace in fig.data:
        metric_key = trace.name
        style = _STYLE_MAP.get(metric_key)
        if style:
            trace.line.color = style["color"]
            trace.line.dash = style["dash"]
            trace.line.width = 2.5
        trace.name = _METRIC_LABELS.get(metric_key, metric_key)

    fig.update_layout(
        title=title,
        hovermode="x unified",
        legend_title_text="",
        legend=dict(orientation="h", x=0, y=1.02, xanchor="left", yanchor="bottom"),
        margin=dict(t=80),
    )
    fig.update_xaxes(showgrid=True, title_text=x_label or x_col)
    fig.update_yaxes(showgrid=True, title_text=y_label)
    st.plotly_chart(fig, width="stretch")


def _add_volume_lead_counts(
    historical_agg: pd.DataFrame,
    df: pd.DataFrame,
    *,
    freq: str,
    group_col: str | None,
) -> pd.DataFrame:
    """Add sold-lead count per time bucket."""
    if historical_agg.empty:
        return historical_agg

    work = df.copy()
    work["realized_revenue"] = pd.to_numeric(
        work.get("realized_revenue"), errors="coerce"
    )
    work["has_realized_revenue"] = work["realized_revenue"].gt(0).astype(int)
    work["is_sold"] = (
        pd.to_numeric(work.get("sold"), errors="coerce").fillna(0).gt(0).astype(int)
    )

    groupers = [pd.Grouper(key="datetime_min", freq=freq)]
    if group_col:
        groupers.append(group_col)

    volume_counts = (
        work.groupby(groupers, observed=False)
        .agg(
            num_sold=("is_sold", "sum"),
            num_realized_revenue=("has_realized_revenue", "sum"),
        )
        .reset_index()
    )
    merge_keys = ["datetime_min"] + ([group_col] if group_col else [])
    return historical_agg.merge(
        volume_counts, on=merge_keys, how="left", validate="one_to_one"
    )


def _assign_count_bins(
    df: pd.DataFrame,
    *,
    count_per_bin: int,
    count_type: str,
    group_col: str | None,
) -> pd.DataFrame:
    """Assign sequential bins containing approximately ``won_per_bin`` wins.

    Bins are chronological. Each complete bin contains exactly ``won_per_bin``
    observed won leads; the final bin may contain fewer. When ``group_col`` is
    set, bins are built independently within each comparison value.
    """
    work = df.copy()
    work["datetime_min"] = pd.to_datetime(work["datetime_min"], errors="coerce")
    sort_cols = ([group_col] if group_col else []) + ["datetime_min"]
    if "id" in work.columns:
        sort_cols.append("id")
    # Build bins from the most recent rows backwards so the newest observations
    # always form the first, complete count bin. This avoids leaving the current
    # end of the monitoring window as a small/incomplete trailing bin.
    ascending = [True] * len(sort_cols)
    ascending[-1] = False
    if len(sort_cols) > 1 and sort_cols[-2] == "datetime_min":
        ascending[-2] = False
    work = work.sort_values(sort_cols, ascending=ascending, kind="stable").copy()

    if count_type == "won":
        increment = (
            pd.to_numeric(work.get("won_numeric"), errors="coerce")
            .fillna(0.0)
            .gt(0)
            .astype("int64")
        )
        bin_col = "won_count_bin"
    else:
        increment = pd.Series(1, index=work.index, dtype="int64")
        bin_col = "opportunity_count_bin"

    work["_count_bin_increment"] = increment
    if group_col:
        cumulative_count = work.groupby(group_col, dropna=False)[
            "_count_bin_increment"
        ].cumsum()
    else:
        cumulative_count = work["_count_bin_increment"].cumsum()

    work[bin_col] = (
        ((cumulative_count.clip(lower=1) - 1) // int(count_per_bin)) + 1
    ).astype("int64")

    # Restore chronological row order for downstream aggregation/plotting.
    chronological_cols = ([group_col] if group_col else []) + ["datetime_min"]
    if "id" in work.columns:
        chronological_cols.append("id")
    return work.sort_values(chronological_cols, kind="stable").drop(
        columns="_count_bin_increment"
    )


def _aggregate_historical_by_count(
    df: pd.DataFrame,
    *,
    count_per_bin: int,
    count_type: str,
    group_col: str | None,
) -> pd.DataFrame:
    """Aggregate canonical historical metrics into equal-win-count bins."""
    work = _assign_count_bins(
        df,
        count_per_bin=count_per_bin,
        count_type=count_type,
        group_col=group_col,
    )
    work = _with_outcome_expected_metrics(work)
    bin_col = "won_count_bin" if count_type == "won" else "opportunity_count_bin"
    keys = [bin_col] + ([group_col] if group_col else [])

    agg = (
        work.groupby(keys, dropna=False, observed=False)
        .agg(
            bin_start=("datetime_min", "min"),
            bin_end=("datetime_min", "max"),
            datetime_min=("datetime_min", "max"),
            num_opportunities=("num_opportunities", "sum"),
            num_won=("num_won", "sum"),
            num_sold=("sold", "sum"),
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
    agg["measured_winrate"] = transforms.win_rate(
        agg["num_won"],
        agg["num_opportunities"],
    )
    agg["p_sold_given_won"] = np.where(
        pd.to_numeric(agg["num_won"], errors="coerce") > 0,
        pd.to_numeric(agg["num_sold"], errors="coerce")
        / pd.to_numeric(agg["num_won"], errors="coerce"),
        np.nan,
    )
    agg["profit_realization_rate"] = np.where(
        pd.to_numeric(agg["expected_profit_on_sold"], errors="coerce") > 0,
        pd.to_numeric(agg["realized_profit"], errors="coerce")
        / pd.to_numeric(agg["expected_profit_on_sold"], errors="coerce"),
        np.nan,
    )
    agg["realized_cm"] = transforms.contribution_margin(
        agg["realized_profit"],
        agg["realized_revenue"],
    )
    agg["expected_cm_on_wins"] = transforms.contribution_margin(
        agg["expected_profit_on_wins"],
        agg["expected_revenue_on_wins"],
    )
    return agg


def _aggregate_ml_by_count(
    df: pd.DataFrame,
    *,
    count_per_bin: int,
    count_type: str,
    group_col: str | None,
) -> pd.DataFrame:
    """Aggregate ML expected metrics using the same historical win-count bins."""
    work = _assign_count_bins(
        df,
        count_per_bin=count_per_bin,
        count_type=count_type,
        group_col=group_col,
    )
    mask = transforms.recommended_bid_metrics_available_mask(work)
    work = work.loc[mask].copy()
    if work.empty:
        return pd.DataFrame()

    bin_col = "won_count_bin" if count_type == "won" else "opportunity_count_bin"
    keys = [bin_col] + ([group_col] if group_col else [])
    agg = (
        work.groupby(keys, dropna=False, observed=False)
        .agg(
            recommended_bid_predicted_revenue=(
                "recommended_bid_predicted_revenue",
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
    agg["recommended_bid_predicted_cm"] = transforms.contribution_margin(
        agg["recommended_bid_predicted_profit"],
        agg["recommended_bid_predicted_revenue"],
    )
    return agg


def _filter_value_options(df: pd.DataFrame, column: str) -> list[str]:
    """Return display values for one optional monitoring filter."""
    if column == "None":
        return ["None"]

    values = df[column].dropna().unique().tolist()
    if column in {"campaign_id", "bidding_strategy_id", "source_type_id"}:
        values = sorted(int(value) for value in values)
        return ["All"] + [str(value) for value in values]
    return ["All"] + sorted(str(value) for value in values)


def _apply_optional_filter(
    df: pd.DataFrame,
    column: str,
    value: str,
) -> pd.DataFrame:
    """Apply one optional exact-match filter."""
    if column == "None" or value in {"None", "All"}:
        return df
    return df[df[column].astype(str) == value].copy()


def _render_historical_kpis(df: pd.DataFrame) -> None:
    """Render aggregate historical KPIs for the currently filtered cohort."""
    realized_revenue = float(df["realized_revenue"].sum())
    bid_cost = float(df["bid_cost"].sum())
    realized_profit = float(df["realized_profit"].sum())
    measured_winrate = float(df["won_numeric"].mean()) if len(df) else 0.0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Realized Revenue", f"${realized_revenue:,.2f}")
    c2.metric("Realized Bid Cost", f"${bid_cost:,.2f}")
    c3.metric("Realized Profit", f"${realized_profit:,.2f}")
    c4.metric("Measured Win Rate", f"{measured_winrate:.3%}")

    sold = pd.to_numeric(df["sold"], errors="coerce").fillna(0.0).eq(1)
    expected_revenue_on_sold = float(
        pd.to_numeric(df.loc[sold, "expected_revenue"], errors="coerce")
        .fillna(0.0)
        .sum()
    )
    expected_profit_on_sold = float(
        pd.to_numeric(df.loc[sold, "expected_profit"], errors="coerce")
        .fillna(0.0)
        .sum()
    )
    realized_cm = (
        realized_profit / realized_revenue if realized_revenue != 0.0 else np.nan
    )
    expected_cm = (
        expected_profit_on_sold / expected_revenue_on_sold
        if expected_revenue_on_sold != 0.0
        else np.nan
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Expected Revenue — Sold Leads", f"${expected_revenue_on_sold:,.2f}")
    c2.metric("Expected Profit — Sold Leads", f"${expected_profit_on_sold:,.2f}")
    c3.metric("Realized CM", "—" if np.isnan(realized_cm) else f"{realized_cm:.3%}")
    c4.metric(
        "Expected CM — Sold Leads",
        "—" if np.isnan(expected_cm) else f"{expected_cm:.3%}",
    )


def _render_ml_kpis(df: pd.DataFrame) -> None:
    """Render production prediction-vs-observed KPIs."""
    mask = transforms.recommended_bid_metrics_available_mask(df)
    cohort = df.loc[mask].copy()
    if cohort.empty:
        st.info("No ML metrics are available for the selected filters.")
        return

    predicted_profit = float(cohort["recommended_bid_predicted_profit"].sum())
    won = pd.to_numeric(cohort["won_numeric"], errors="coerce").fillna(0.0).eq(1)
    observed_expected_profit = float(
        pd.to_numeric(cohort.loc[won, "expected_profit"], errors="coerce")
        .fillna(0.0)
        .sum()
    )
    predicted_win_rate = float(cohort["recommended_bid_predicted_win_rate"].mean())
    measured_win_rate = float(cohort["won_numeric"].mean()) if len(cohort) else np.nan

    profit_prediction_error = (
        (predicted_profit - observed_expected_profit) / abs(observed_expected_profit)
        if observed_expected_profit != 0.0
        else np.nan
    )
    profit_prediction_error_dollars = predicted_profit - observed_expected_profit
    win_rate_prediction_error_pp = (
        100.0 * (predicted_win_rate - measured_win_rate)
        if not np.isnan(measured_win_rate)
        else np.nan
    )

    st.markdown("#### Production ML Prediction Performance")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "Profit Prediction Error",
        "—" if np.isnan(profit_prediction_error) else f"{profit_prediction_error:+.2%}",
        delta=(
            None
            if np.isnan(profit_prediction_error)
            else f"{profit_prediction_error_dollars:+,.2f}"
        ),
        help=(
            "Profit Prediction Error = (ML Predicted Profit - "
            "Observed Expected Profit) "
            "/ |Observed Expected Profit|.\n\n"
            "ML Predicted Profit = sum[p(win) x "
            "(expected_revenue - recommended_bid)]. recommended_bid comes from "
            "the prediction log.\n\n"
            "Observed Expected Profit = sum[won x (expected_revenue - bid)]. "
            "bid comes from the historical/raw lead data.\n\n"
            "A positive value means ML predicted profit was higher than the "
            "observed expected profit; a negative value means it was lower. "
            "Values closer to zero indicate better agreement."
        ),
    )
    c2.metric(
        "Win Rate Prediction Error",
        (
            "—"
            if np.isnan(win_rate_prediction_error_pp)
            else f"{win_rate_prediction_error_pp:+.2f} pp"
        ),
        delta=(
            None
            if np.isnan(measured_win_rate)
            else f"{predicted_win_rate:.3%} predicted vs "
            f"{measured_win_rate:.3%} measured"
        ),
        help=(
            "Win Rate Prediction Error (pp) = 100 x "
            "(Predicted Win Rate - Measured Win Rate).\n\n"
            "Predicted Win Rate = mean[p(win)] at the production recommended bids.\n"
            "Measured Win Rate = won / opportunities.\n\n"
            "Values closer to zero indicate better agreement."
        ),
    )
    c3.metric(
        "ML Predicted Profit",
        f"${predicted_profit:,.2f}",
        help=(
            "sum[p(win) x (expected_revenue - recommended_bid)]. "
            "recommended_bid is taken from the prediction log."
        ),
    )
    c4.metric(
        "Observed Expected Profit",
        f"${observed_expected_profit:,.2f}",
        help=(
            "sum[won x (expected_revenue - bid)]. "
            "bid is taken from the historical/raw lead data."
        ),
    )


def main():
    """Render the SmartHub Performance dashboard."""
    st.title("SmartHub Performance")
    st.caption(
        "Realized metrics use actual outcomes. ML metrics are probability-weighted "
        "recommended-bid quantities loaded from the monitoring dataset."
    )

    history_col, bin_type_col, bin_size_col = st.columns(3)
    with history_col:
        days = st.number_input(
            "History window (days)",
            min_value=1,
            max_value=60,
            value=30,
            step=1,
            key="mon_days",
        )
    with bin_type_col:
        bin_type = st.selectbox(
            "Bin type",
            options=["Time", "Won count", "Opportunity count"],
            index=0,
            key="mon_bin_type",
        )
    with bin_size_col:
        if bin_type == "Time":
            bin_size = st.selectbox(
                "Bin size",
                options=list(_BIN_MAP.keys()),
                index=3,
                key="mon_bin_size",
            )
            count_per_bin = None
        else:
            count_per_bin = st.number_input(
                (
                    "Won leads per bin"
                    if bin_type == "Won count"
                    else "Opportunities per bin"
                ),
                min_value=10,
                max_value=2000,
                value=500 if bin_type == "Won count" else 1000,
                step=10,
                key=(
                    "mon_won_per_bin"
                    if bin_type == "Won count"
                    else "mon_opportunities_per_bin"
                ),
            )
            bin_size = None

    try:
        leads_df = load_leads(int(days))
    except io.DataNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    lead_type_options = sorted(
        leads_df["lead_type_id"].dropna().astype(int).unique().tolist()
    )
    if not lead_type_options:
        st.info("No lead types available for the selected history window.")
        st.stop()

    lead_type_col, _, _ = st.columns(3)
    with lead_type_col:
        selected_lead_type = st.selectbox(
            "lead_type_id",
            options=lead_type_options,
            index=lead_type_options.index(6) if 6 in lead_type_options else 0,
            key="mon_lead_type_id",
        )

    leads_df = leads_df[
        leads_df["lead_type_id"].astype(int) == selected_lead_type
    ].copy()

    filter1_col, filter1_value_col, _ = st.columns(3)
    with filter1_col:
        filter1_dimension = st.selectbox(
            "Filter 1",
            options=["None"] + _FILTER_DIMENSIONS,
            index=0,
            key="mon_filter1_dimension",
        )

    with filter1_value_col:
        filter1_value = st.selectbox(
            "Filter 1 value",
            options=_filter_value_options(leads_df, filter1_dimension),
            index=0,
            key="mon_filter1_value",
        )

    leads_df = _apply_optional_filter(
        leads_df,
        filter1_dimension,
        filter1_value,
    )

    filter2_dimensions = [
        dimension for dimension in _FILTER_DIMENSIONS if dimension != filter1_dimension
    ]
    filter2_col, filter2_value_col, _ = st.columns(3)
    with filter2_col:
        filter2_dimension = st.selectbox(
            "Filter 2",
            options=["None"] + filter2_dimensions,
            index=0,
            key="mon_filter2_dimension",
        )

    with filter2_value_col:
        filter2_value = st.selectbox(
            "Filter 2 value",
            options=_filter_value_options(leads_df, filter2_dimension),
            index=0,
            key="mon_filter2_value",
        )

    leads_df = _apply_optional_filter(
        leads_df,
        filter2_dimension,
        filter2_value,
    )

    if leads_df.empty:
        st.info("No data for the selected filters.")
        st.stop()

    df = transforms.add_historical_business_metrics(leads_df)

    st.markdown("---")
    overview_tab, feature_tab = st.tabs(["Overview", "Feature Analysis"])

    with overview_tab:
        st.subheader("Business Performance")
        _render_historical_kpis(df)

        metric_col, compare_col = st.columns(2)
        with metric_col:
            selected_metric = st.selectbox(
                "Metric",
                options=list(HISTORICAL_METRIC_GROUPS.keys()),
                index=0,
                key="mon_metric",
            )

        group_col = None
        with compare_col:
            if selected_metric != "All":
                compare_options = [
                    "None",
                    *[
                        dimension
                        for dimension in _FILTER_DIMENSIONS
                        if dimension not in {filter1_dimension, filter2_dimension}
                        and dimension in df.columns
                    ],
                ]
                selected_compare_by = st.selectbox(
                    "Compare by",
                    options=compare_options,
                    index=0,
                    key="mon_compare_by",
                    help=(
                        "Show the selected metric for every value of this dimension "
                        "after Filter 1 and Filter 2 are applied."
                    ),
                )
                group_col = (
                    None if selected_compare_by == "None" else selected_compare_by
                )
            else:
                st.selectbox(
                    "Compare by",
                    options=["None"],
                    index=0,
                    key="mon_compare_by_all",
                    disabled=True,
                )

        checkbox1, checkbox2, _ = st.columns([1, 1, 2])
        with checkbox1:
            show_table = st.checkbox(
                "Show table",
                value=False,
                key="mon_show_table",
            )
        with checkbox2:
            show_ml_metrics = st.checkbox(
                "Show ML Metrics",
                value=True,
                key="mon_show_ml_metrics",
            )

        plot_df = df
        if show_ml_metrics:
            try:
                prediction_df = load_prediction_monitoring(int(days))
            except io.DataNotFoundError as exc:
                st.warning(str(exc))
                prediction_df = pd.DataFrame()

            if not prediction_df.empty:
                prediction_df = prediction_df[
                    prediction_df["lead_type_id"].eq(int(selected_lead_type))
                ].copy()

                for dimension, value in (
                    (filter1_dimension, filter1_value),
                    (filter2_dimension, filter2_value),
                ):
                    if (
                        dimension != "None"
                        and value not in {"None", "All"}
                        and dimension in prediction_df.columns
                    ):
                        prediction_df = prediction_df[
                            prediction_df[dimension].astype(str) == value
                        ].copy()

            plot_df = attach_prediction_monitoring(df, prediction_df)

        # Always aggregate historical metrics on the full filtered lead cohort.
        # ML metrics remain a sparse overlay on the same bins.
        if bin_type == "Time":
            x_col = "datetime_min"
            x_label = "Time"
            historical_agg = transforms.aggregate_historical_business_metrics(
                df,
                freq=_BIN_MAP[bin_size],
                group_col=group_col,
            )
            historical_agg = _add_volume_lead_counts(
                historical_agg,
                df,
                freq=_BIN_MAP[bin_size],
                group_col=group_col,
            )
            historical_agg["p_sold_given_won"] = np.where(
                pd.to_numeric(historical_agg["num_won"], errors="coerce") > 0,
                pd.to_numeric(historical_agg["num_sold"], errors="coerce")
                / pd.to_numeric(historical_agg["num_won"], errors="coerce"),
                np.nan,
            )
            historical_agg["profit_realization_rate"] = np.where(
                pd.to_numeric(
                    historical_agg["expected_profit_on_sold"],
                    errors="coerce",
                )
                > 0,
                pd.to_numeric(historical_agg["realized_profit"], errors="coerce")
                / pd.to_numeric(
                    historical_agg["expected_profit_on_sold"],
                    errors="coerce",
                ),
                np.nan,
            )
        else:
            x_col = "datetime_min"
            x_label = "Time"
            count_type = "won" if bin_type == "Won count" else "opportunity"
            historical_agg = _aggregate_historical_by_count(
                df,
                count_per_bin=int(count_per_bin),
                count_type=count_type,
                group_col=group_col,
            )

        agg = historical_agg

        if show_ml_metrics:
            if bin_type == "Time":
                ml_agg = transforms.aggregate_recommended_bid_comparison(
                    plot_df,
                    freq=_BIN_MAP[bin_size],
                    group_col=group_col,
                )
                merge_keys = ["datetime_min"] + ([group_col] if group_col else [])
            else:
                count_type = "won" if bin_type == "Won count" else "opportunity"
                ml_agg = _aggregate_ml_by_count(
                    plot_df,
                    count_per_bin=int(count_per_bin),
                    count_type=count_type,
                    group_col=group_col,
                )
                count_bin_col = (
                    "won_count_bin" if count_type == "won" else "opportunity_count_bin"
                )
                merge_keys = [count_bin_col] + ([group_col] if group_col else [])

            if ml_agg.empty:
                st.info(
                    "ML metrics are not available for the selected filters; "
                    "showing historical metrics only."
                )
                show_ml_metrics = False
            else:
                ml_columns = merge_keys + [
                    column
                    for column in (
                        "recommended_bid_predicted_revenue",
                        "recommended_bid_predicted_profit",
                        "recommended_bid_predicted_win_rate",
                        "recommended_bid_predicted_cm",
                    )
                    if column in ml_agg.columns
                ]
                agg = historical_agg.merge(
                    ml_agg[ml_columns],
                    on=merge_keys,
                    how="left",
                    validate="one_to_one",
                )

        def _plot_group(cols, title, y_label):
            if group_col is None:
                plot_metric_block(
                    agg,
                    cols,
                    title,
                    y_label,
                    x_col=x_col,
                    x_label=x_label,
                )
                return

            available = [column for column in cols if column in agg.columns]
            if agg.empty or not available:
                st.info(f"No data available for {title}.")
                return

            group_values = sorted(
                agg[group_col].dropna().unique().tolist(),
                key=lambda value: str(value),
            )
            if not group_values:
                st.info(f"No {group_col} values are available for {title}.")
                return

            for group_value in group_values:
                sub = agg[agg[group_col] == group_value].copy()
                if sub.empty:
                    continue

                st.markdown(f"### {group_col} = {group_value}")
                plot_metric_block(
                    sub,
                    available,
                    title,
                    y_label,
                    x_col=x_col,
                    x_label=x_label,
                )

        sold_revenue_cols = [
            "bid_cost",
            "realized_revenue",
            "expected_revenue_on_sold",
        ]
        sold_profit_cols = [
            "realized_profit",
            "expected_profit_on_sold",
        ]
        ml_revenue_comparison_cols = [
            "recommended_bid_predicted_revenue",
            "expected_revenue_on_wins",
        ]
        winrate_cols = ["measured_winrate"]
        cm_cols = ["realized_cm", "expected_cm_on_wins"]

        if show_ml_metrics:
            winrate_cols.append("recommended_bid_predicted_win_rate")
            cm_cols.append("recommended_bid_predicted_cm")

        if selected_metric == "All":
            business_tab, ml_tab = st.tabs(["Business Performance", "ML Performance"])

            with business_tab:
                st.caption(
                    "Realized business outcomes and downstream performance for "
                    "the selected production cohort."
                )
                _plot_group(
                    sold_revenue_cols,
                    "Sold Leads — Bid Cost, Realized Revenue, and Expected Revenue",
                    "Amount ($)",
                )
                _plot_group(
                    sold_profit_cols,
                    "Sold Leads — Realized Profit and Expected Profit",
                    "Amount ($)",
                )
                _plot_group(
                    ["measured_winrate"],
                    "Measured Win Rate",
                    "Win rate",
                )
                _plot_group(
                    ["realized_cm", "expected_cm_on_wins"],
                    "Contribution Margin",
                    "CM",
                )
                _plot_group(
                    ["num_opportunities", "num_won", "num_sold"],
                    "Number of Opportunities, Won, and Sold",
                    "Count",
                )
                _plot_group(
                    ["profit_realization_rate"],
                    "Profit Realization Rate = Realized Profit / Expected Profit "
                    "on Sold Leads",
                    "Fraction",
                )
                _plot_group(
                    ["p_sold_given_won"],
                    "P(Sold | Won) = Sold Leads / Won Leads",
                    "Probability",
                )

            with ml_tab:
                st.caption(
                    "Production ML prediction performance for the selected cohort: "
                    "predicted values compared with observed auction outcomes."
                )
                if show_ml_metrics:
                    _render_ml_kpis(plot_df)
                    _plot_group(
                        [
                            "measured_winrate",
                            "recommended_bid_predicted_win_rate",
                        ],
                        "Win Rate: ML Predicted vs Measured",
                        "Win rate",
                    )
                    _plot_group(
                        ml_revenue_comparison_cols,
                        "Revenue: ML Predicted on All Leads vs Expected on Won Leads",
                        "Amount ($)",
                    )
                    _plot_group(
                        [
                            "recommended_bid_predicted_profit",
                            "expected_profit_on_wins",
                        ],
                        "Profit: ML Predicted on All Leads vs Expected on Won Leads",
                        "Amount ($)",
                    )
                    if "recommended_bid_predicted_cm" in agg.columns:
                        _plot_group(
                            [
                                "expected_cm_on_wins",
                                "recommended_bid_predicted_cm",
                            ],
                            "Contribution Margin: ML Predicted vs "
                            "Expected on Won Leads",
                            "CM",
                        )
                else:
                    st.info("Enable Show ML Metrics to display ML performance plots.")

        elif selected_metric == "Sold Lead Revenue":
            _plot_group(
                sold_revenue_cols,
                "Sold Leads — Bid Cost, Realized Revenue, and Expected Revenue",
                "Amount ($)",
            )
        elif selected_metric == "Sold Lead Profit":
            _plot_group(
                sold_profit_cols,
                "Sold Leads — Realized Profit and Expected Profit",
                "Amount ($)",
            )
        elif selected_metric == "ML Revenue Comparison":
            if show_ml_metrics:
                _plot_group(
                    ml_revenue_comparison_cols,
                    "Revenue: ML Predicted on All Leads vs Expected on Won Leads",
                    "Amount ($)",
                )
            else:
                st.info("Enable Show ML Metrics to display the ML revenue comparison.")
        elif selected_metric == "ML Profit Comparison":
            if show_ml_metrics:
                _plot_group(
                    [
                        "recommended_bid_predicted_profit",
                        "expected_profit_on_wins",
                    ],
                    "Profit: ML Predicted on All Leads vs Expected on Won Leads",
                    "Amount ($)",
                )
            else:
                st.info("Enable Show ML Metrics to display the ML profit comparison.")
        elif selected_metric == "Win Rate":
            _plot_group(winrate_cols, "Win Rate", "Rate")
        elif selected_metric == "P(Sold | Won)":
            _plot_group(
                ["p_sold_given_won"],
                "P(Sold | Won) = Sold Leads / Won Leads",
                "Probability",
            )
        elif selected_metric == "CM":
            _plot_group(cm_cols, "Contribution Margin", "Rate")
        else:
            _plot_group(
                HISTORICAL_METRIC_GROUPS[selected_metric],
                selected_metric,
                "Count",
            )

        if show_table:
            st.markdown("## Aggregated Data")
            st.dataframe(
                agg.sort_values("datetime_min", ascending=False).reset_index(drop=True),
                width="stretch",
                hide_index=True,
            )

    with feature_tab:
        st.caption(
            "Realized production performance by feature value or numeric bucket. "
            "The global filters above apply to this analysis."
        )
        monitoring_feature_frame = df.copy()

        features = _feature_columns(monitoring_feature_frame)
        if not features:
            st.info("No feature columns are available for the selected data.")
            st.stop()

        feature_col, support_col = st.columns(2)
        with feature_col:
            feature = st.selectbox(
                "Feature",
                options=features,
                key="mon_feature_analysis_feature",
            )

        kind = _feature_kind(monitoring_feature_frame, feature)
        with support_col:
            min_support = st.number_input(
                "Minimum leads per value / bucket",
                min_value=1,
                value=100,
                step=50,
                key="mon_feature_min_support",
            )

        if kind == "continuous":
            bin_col_1, bin_col_2 = st.columns(2)
            with bin_col_1:
                binning = st.selectbox(
                    "Numeric binning",
                    options=["quantile", "fixed"],
                    key="mon_feature_binning",
                )
            with bin_col_2:
                bins = st.slider(
                    "Number of feature bins",
                    min_value=3,
                    max_value=30,
                    value=10,
                    key="mon_feature_bins",
                )
            top_n = 20
        else:
            binning = "quantile"
            bins = 10
            top_n = st.slider(
                "Top categories / values",
                min_value=5,
                max_value=50,
                value=20,
                key="mon_feature_top_n",
            )

        summary = _build_feature_summary(
            monitoring_feature_frame,
            feature,
            bins=int(bins),
            binning=binning,
            top_n=int(top_n),
            min_support=int(min_support),
        )
        if summary.empty:
            st.info(
                "No feature values / buckets remain after the current "
                "minimum-support setting."
            )
            st.stop()

        left, right = st.columns(2)
        with left:
            st.plotly_chart(
                _feature_win_rate_chart(summary, feature),
                width="stretch",
            )
        with right:
            st.plotly_chart(
                _feature_bid_chart(summary, feature),
                width="stretch",
            )

        left, right = st.columns(2)
        with left:
            st.plotly_chart(
                _feature_economics_chart(summary, feature),
                width="stretch",
            )
        with right:
            st.plotly_chart(
                _feature_cm_chart(summary, feature),
                width="stretch",
            )

        st.subheader("Feature Summary")
        display = summary.copy()
        display["fraction_of_filtered_rows"] = (
            display["fraction_of_filtered_rows"] * 100.0
        )
        st.dataframe(
            display,
            width="stretch",
            hide_index=True,
        )


if __name__ == "__main__":
    st.set_page_config(page_title="SmartHub Performance", layout="wide")
    main()
