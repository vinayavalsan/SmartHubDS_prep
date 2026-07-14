"""Streamlit dashboard for monitoring DS performance over time.

Run with:
    streamlit run src/smarthub/monitoring/monitoring_app.py
"""

from __future__ import annotations

import plotly.express as px
import streamlit as st

from smarthub.core import io
from smarthub.core.transforms import aggregate_monitoring, leads_to_monitoring_base

# set_page_config is called by the entry (app.py or the __main__ guard below).

METRIC_GROUPS = {
    "All": None,
    "Revenue + Payout breakdown": ["revenue_measured", "payout", "revenue_expected"],
    "Profit": ["profit"],
    "Win Rate": ["winrate"],
    "CM": ["cm_measured", "cm_expected"],
    "Number of opportunities and won": ["num_opportunities", "num_won"],
}

_STYLE_MAP = {
    "revenue_measured": dict(color="black", dash="solid"),
    "revenue_expected": dict(color="black", dash="dash"),
    "payout": dict(color="black", dash="dot"),
    "profit": dict(color="orange", dash="solid"),
    "winrate": dict(color="black", dash="solid"),
    "cm_measured": dict(color="black", dash="solid"),
    "cm_expected": dict(color="black", dash="dash"),
    "num_opportunities": dict(color="black", dash="dash"),
    "num_won": dict(color="black", dash="solid"),
}

_BIN_MAP = {"1hr": "1h", "6hr": "6h", "12hr": "12h", "day": "D", "week": "W-MON"}


@st.cache_data
def load_data():
    """Build the monitoring base from the real accumulated leads."""
    return leads_to_monitoring_base(io.load_leads())


def y_label_for(metric: str) -> str:
    """Return the y-axis label appropriate to a metric group.

    Inputs
    ------
    metric : str
        The selected metric-group name.

    Returns
    -------
    str
        The matching axis label.
    """
    if metric in ("Win Rate", "CM"):
        return "Rate"
    if metric in ("Revenue + Payout breakdown", "Profit"):
        return "Amount ($)"
    if metric == "Number of opportunities and won":
        return "Count"
    return "Value"


def plot_metric_block(df_plot, columns, title, y_label=None):
    """Plot one metric group as an overlaid time-series line chart.

    Inputs
    ------
    df_plot : pd.DataFrame
        Aggregated data with a ``datetime_min`` column.
    columns : list[str]
        Metric columns to overlay.
    title : str
        Chart title.
    y_label : str, optional
        Y-axis label; defaults to ``"value"``.
    """
    plot_df = df_plot[["datetime_min"] + columns].melt(
        id_vars="datetime_min", var_name="metric", value_name="value"
    )
    fig = px.line(plot_df, x="datetime_min", y="value", color="metric", markers=False)

    for trace in fig.data:
        style = _STYLE_MAP.get(trace.name)
        if style:
            trace.line.color = style["color"]
            trace.line.dash = style["dash"]

    fig.update_layout(
        title=title,
        hovermode="x unified",
        legend_title_text="",
        legend=dict(orientation="h", x=0, y=1.02, xanchor="left", yanchor="bottom"),
        margin=dict(t=80),
    )
    fig.update_xaxes(showgrid=True, title_text="datetime_min")
    fig.update_yaxes(showgrid=True, title_text=y_label or "value")
    st.plotly_chart(fig, use_container_width=True)


def _resolve_group_col(selected_campaign, selected_state, selected_metric):
    """Decide which dimension to split by, given the active filters.

    Inputs
    ------
    selected_campaign : str
        Selected campaign filter, or ``"All"``.
    selected_state : str
        Selected state filter, or ``"All"``.
    selected_metric : str
        Selected metric-group name.

    Returns
    -------
    str or None
        The column to group by, or None for no split.
    """
    if selected_metric == "All":
        return None
    if selected_campaign == "All" and selected_state == "All":
        return "campaign_id"
    if selected_campaign != "All" and selected_state == "All":
        return "state"
    return None


def _render_controls(df):
    """Render the top control row and return the selected settings.

    Inputs
    ------
    df : pd.DataFrame
        Monitoring base used to populate filter options.

    Returns
    -------
    tuple
        ``(bin_size, selected_state, selected_campaign, selected_metric,
        show_table)``.
    """
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        bin_size = st.selectbox("Bin size", options=list(_BIN_MAP.keys()), index=3)
    with c2:
        state_options = ["All"] + sorted(df["state"].dropna().unique().tolist())
        selected_state = st.selectbox("state", options=state_options, index=0)
    with c3:
        campaign_options = ["All"] + [
            str(x)
            for x in sorted(df["campaign_id"].dropna().astype(int).unique().tolist())
        ]
        selected_campaign = st.selectbox(
            "campaign_id", options=campaign_options, index=0
        )
    with c4:
        selected_metric = st.selectbox(
            "Metric", options=list(METRIC_GROUPS.keys()), index=0
        )
    with c5:
        show_table = st.checkbox("Show table", value=True)
    return bin_size, selected_state, selected_campaign, selected_metric, show_table


def main():
    """Run the Monitoring dashboard page (load, filter, aggregate, plot)."""
    st.title("SmartHub Monitoring")

    try:
        df = load_data()
    except io.DataNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    bin_size, sel_state, sel_campaign, sel_metric, show_table = _render_controls(df)
    st.markdown("---")

    filtered = df.copy()
    if sel_state != "All":
        filtered = filtered[filtered["state"] == sel_state]
    if sel_campaign != "All":
        filtered = filtered[filtered["campaign_id"].astype(str) == sel_campaign]
    if filtered.empty:
        st.info("No data for the selected filters.")
        st.stop()

    group_col = _resolve_group_col(sel_campaign, sel_state, sel_metric)
    agg = aggregate_monitoring(filtered, freq=_BIN_MAP[bin_size], group_col=group_col)

    if sel_metric == "All":
        plot_metric_block(
            agg, METRIC_GROUPS["Revenue + Payout breakdown"],
            "Revenue + Payout breakdown", "Amount ($)",
        )
        plot_metric_block(agg, ["profit"], "Profit", "Amount ($)")
        plot_metric_block(agg, ["winrate"], "Win Rate", "Rate")
        plot_metric_block(agg, ["cm_measured", "cm_expected"], "CM", "Margin")
        plot_metric_block(
            agg, ["num_opportunities", "num_won"],
            "Number of opportunities and won", "Count",
        )
    else:
        cols = METRIC_GROUPS[sel_metric]
        y_label = y_label_for(sel_metric)
        if group_col is None:
            plot_metric_block(agg, cols, sel_metric, y_label)
        else:
            for group_value in sorted(agg[group_col].dropna().unique().tolist()):
                sub = agg[agg[group_col] == group_value].copy()
                if sub.empty:
                    continue
                st.markdown(f"### {group_col} = {group_value}")
                plot_metric_block(sub, cols, sel_metric, y_label)

    if show_table:
        st.markdown("## Aggregated Data")
        st.dataframe(
            agg.sort_values("datetime_min", ascending=False).reset_index(drop=True),
            use_container_width=True,
        )


if __name__ == "__main__":
    st.set_page_config(page_title="SmartHub Monitoring", layout="wide")
    main()
