"""Streamlit dashboard for exploring raw lead-ping data.

Run with:
    streamlit run src/smarthub/dashboards/leads_app.py
"""

from __future__ import annotations

import plotly.express as px
import streamlit as st

from smarthub import io
from smarthub.transforms import (
    aggregate_leads,
    build_metric_plot_data,
    cumulative_winrate_curves,
    funnel_counts,
)
from smarthub.dashboards import _ui

st.set_page_config(page_title="SmartHub Leads", layout="wide")


@st.cache_data
def load_data():
    return io.load_leads()


def ordered_state_list(df) -> list[str]:
    states = sorted([s for s in df["state"].dropna().unique() if s != "NAvail"])
    if "NAvail" in df["state"].values:
        states.append("NAvail")
    return states


# ---------------------------------------------------------------------------
# Plot Type 1 - any feature on the x-axis
# ---------------------------------------------------------------------------


def _build_plot_type_1_data(df, feature_col, metric_col, legend_col):
    group_cols = [feature_col]
    if legend_col != "None" and legend_col != feature_col:
        group_cols.append(legend_col)

    plot_df = build_metric_plot_data(df, group_cols, metric_col)
    plot_df[feature_col] = plot_df[feature_col].astype(str)
    if legend_col != "None" and legend_col in plot_df.columns:
        plot_df[legend_col] = plot_df[legend_col].astype(str)
    return plot_df


def _figure_type_1(plot_df, feature_col, metric_col, legend_col):
    use_legend = legend_col != "None" and legend_col in plot_df.columns
    fig = px.line(
        plot_df,
        x=feature_col,
        y="value",
        color=legend_col if use_legend else None,
        markers=True,
        title=(
            f"{metric_col} vs {feature_col} split by {legend_col}"
            if use_legend
            else f"{metric_col} by {feature_col}"
        ),
    )
    fig.update_layout(
        xaxis_title=feature_col,
        yaxis_title=metric_col,
        legend_title=legend_col if use_legend else None,
    )
    return _ui.style_figure(fig)


def display_plot_type_1(df):
    st.markdown("### Plot Type 1")
    col1, col2, col3 = st.columns(3)

    feature_options = sorted(df.columns.tolist())
    metric_options = _ui.get_plot_metric_options(df)
    legend_options = _ui.get_legend_options(df)
    default_feature = (
        "num_auto_violations"
        if "num_auto_violations" in feature_options
        else feature_options[0]
    )

    with col1:
        feature_col = st.selectbox(
            "x-axis",
            options=feature_options,
            index=_ui.get_default_option(feature_options, default_feature),
        )
    with col2:
        metric_cols = st.multiselect(
            "y-axis", options=metric_options, default=["winrate"]
        )
    with col3:
        legend_col = st.selectbox(
            "Legend divisions",
            options=legend_options,
            index=_ui.get_default_option(legend_options, "None"),
        )

    if not metric_cols:
        st.info("Select at least one y-axis metric to display the plot.")
        return

    for metric_col in metric_cols:
        plot_df = _build_plot_type_1_data(df, feature_col, metric_col, legend_col)
        st.plotly_chart(
            _figure_type_1(plot_df, feature_col, metric_col, legend_col),
            use_container_width=True,
        )


# ---------------------------------------------------------------------------
# Plot Type 2 - time series
# ---------------------------------------------------------------------------

_FREQ_MAP = {"1 hr": "h", "1 day": "D"}


def _build_plot_type_2_data(df, freq_label, metric_col, legend_col):
    source = df.copy()
    source["created_at_bucket"] = source["created_at"].dt.floor(_FREQ_MAP[freq_label])
    group_cols = ["created_at_bucket"]
    if legend_col != "None":
        group_cols.append(legend_col)

    plot_df = build_metric_plot_data(source, group_cols, metric_col)
    if legend_col != "None" and legend_col in plot_df.columns:
        plot_df[legend_col] = plot_df[legend_col].astype(str)
    return plot_df


def _figure_type_2(plot_df, metric_col, legend_col, freq_label):
    use_legend = legend_col != "None" and legend_col in plot_df.columns
    fig = px.line(
        plot_df,
        x="created_at_bucket",
        y="value",
        color=legend_col if use_legend else None,
        markers=True,
        title=(
            f"{metric_col} over time ({freq_label}) split by {legend_col}"
            if use_legend
            else f"{metric_col} over time ({freq_label})"
        ),
    )
    fig.update_layout(
        xaxis_title="created_at",
        yaxis_title=metric_col,
        legend_title=legend_col if use_legend else None,
    )
    return _ui.style_figure(fig)


def display_plot_type_2(df):
    st.markdown("### Plot Type 2 - Time Series")
    if "created_at" not in df.columns:
        st.info("created_at column is required for Plot Type 2.")
        return

    col1, col2, col3 = st.columns(3)
    metric_options = _ui.get_plot_metric_options(df)
    legend_options = _ui.get_legend_options(df)

    with col1:
        freq_label = st.selectbox("Frequency", options=list(_FREQ_MAP.keys()), index=0)
    with col2:
        metric_cols = st.multiselect(
            "y-axis metrics", options=metric_options, default=["winrate"],
            key="plot_type_2_metrics",
        )
    with col3:
        legend_col = st.selectbox(
            "Legend divisions",
            options=legend_options,
            index=_ui.get_default_option(legend_options, "None"),
            key="plot_type_2_legend",
        )

    if not metric_cols:
        st.info("Select at least one y-axis metric to display the plot.")
        return

    for metric_col in metric_cols:
        plot_df = _build_plot_type_2_data(df, freq_label, metric_col, legend_col)
        st.plotly_chart(
            _figure_type_2(plot_df, metric_col, legend_col, freq_label),
            use_container_width=True,
        )


# ---------------------------------------------------------------------------
# Plot Type 3 - dollar-value bins
# ---------------------------------------------------------------------------

_X_AXIS_MAP = {"profit": "profit", "bid": "bid", "payout": "payout", "revenue": "rev"}
_BUCKET_OPTIONS = {"$0.50": 0.5, "$1": 1, "$2": 2, "$5": 5, "$10": 10}


def _build_plot_type_3_data(df, x_col, bucket_size, metric_col, legend_col):
    source = df.dropna(subset=[x_col]).copy()
    if source.empty:
        return source, "bucket_upper"

    source["bucket_lower"] = (
        source[x_col].astype(float).floordiv(bucket_size) * bucket_size
    )
    source["bucket_upper"] = (source["bucket_lower"] + bucket_size).round(6)

    group_cols = ["bucket_upper"]
    if legend_col != "None":
        group_cols.append(legend_col)

    plot_df = build_metric_plot_data(source, group_cols, metric_col)
    if legend_col != "None" and legend_col in plot_df.columns:
        plot_df[legend_col] = plot_df[legend_col].astype(str)
    return plot_df, "bucket_upper"


def _figure_type_3(
    plot_df, x_label, bucket_upper_col, metric_col, legend_col, size_lbl
):
    use_legend = legend_col != "None" and legend_col in plot_df.columns
    fig = px.line(
        plot_df,
        x=bucket_upper_col,
        y="value",
        color=legend_col if use_legend else None,
        markers=True,
        title=(
            f"{metric_col} by {x_label} bucket ({size_lbl}) split by {legend_col}"
            if use_legend
            else f"{metric_col} by {x_label} bucket ({size_lbl})"
        ),
    )
    fig.update_layout(
        xaxis_title=f"{x_label} bucket upper bound",
        yaxis_title=metric_col,
        legend_title=legend_col if use_legend else None,
    )
    fig.update_xaxes(type="linear")
    return _ui.style_figure(fig)


def display_plot_type_3(df):
    st.markdown("### Plot Type 3 - Metric Value Series")
    x_axis_options = [lbl for lbl, col in _X_AXIS_MAP.items() if col in df.columns]
    if not x_axis_options:
        st.info("Plot Type 3 requires one of profit, bid, payout, or revenue.")
        return

    col1, col2, col3, col4 = st.columns(4)
    metric_options = _ui.get_plot_metric_options(df)
    legend_options = _ui.get_legend_options(df)

    with col1:
        x_label = st.selectbox(
            "x-axis metric",
            options=x_axis_options,
            index=_ui.get_default_option(x_axis_options, "profit"),
            key="plot_type_3_x_axis",
        )
    with col2:
        size_lbl = st.selectbox(
            "Bin width", options=list(_BUCKET_OPTIONS.keys()), index=1,
            key="plot_type_3_frequency",
        )
    with col3:
        metric_cols = st.multiselect(
            "y-axis metrics", options=metric_options, default=["winrate"],
            key="plot_type_3_metrics",
        )
    with col4:
        legend_col = st.selectbox(
            "Legend divisions",
            options=legend_options,
            index=_ui.get_default_option(legend_options, "None"),
            key="plot_type_3_legend",
        )

    if not metric_cols:
        st.info("Select at least one y-axis metric to display the plot.")
        return

    x_col = _X_AXIS_MAP[x_label]
    bucket_size = _BUCKET_OPTIONS[size_lbl]

    for metric_col in metric_cols:
        plot_df, bucket_upper_col = _build_plot_type_3_data(
            df, x_col, bucket_size, metric_col, legend_col
        )
        if plot_df.empty:
            st.info(f"No data available for {metric_col} by {x_label} buckets.")
            continue
        st.plotly_chart(
            _figure_type_3(
                plot_df, x_label, bucket_upper_col, metric_col, legend_col, size_lbl
            ),
            use_container_width=True,
        )


# ---------------------------------------------------------------------------
# Plot Type 4 - cumulative win-rate "shelves" curves
# ---------------------------------------------------------------------------


def _winrate_curve_figure(curves, title, show_delta):
    value_cols = ["winrate_below", "winrate_above"]
    if show_delta:
        value_cols.append("winrate_delta")
    long = curves.melt(
        id_vars="threshold",
        value_vars=value_cols,
        var_name="curve",
        value_name="value",
    )
    fig = px.line(
        long, x="threshold", y="value", color="curve", markers=True, title=title
    )
    fig.update_layout(xaxis_title="bid threshold ($)", yaxis_title="win rate")
    return _ui.style_figure(fig)


def display_plot_type_4(df):
    st.markdown("### Plot Type 4 - Cumulative win-rate curves")
    st.caption(
        "Win rate when bidding at/under (winrate_below) vs over (winrate_above) "
        "each price — find floors, ceilings and edges."
    )
    if not {"bid", "won"}.issubset(df.columns):
        st.info("bid and won columns are required for Plot Type 4.")
        return

    legend_options = _ui.get_legend_options(df)
    col1, col2, col3 = st.columns(3)
    with col1:
        size_lbl = st.selectbox(
            "Bin width", options=list(_BUCKET_OPTIONS.keys()), index=1, key="plot4_bin"
        )
    with col2:
        legend_col = st.selectbox(
            "Split by",
            options=legend_options,
            index=_ui.get_default_option(legend_options, "None"),
            key="plot4_legend",
        )
    with col3:
        show_delta = st.checkbox("Show win-rate delta", value=False, key="plot4_delta")

    bucket = _BUCKET_OPTIONS[size_lbl]

    if legend_col == "None":
        curves = cumulative_winrate_curves(df, bucket)
        if curves.empty:
            st.info("No data available.")
            return
        st.plotly_chart(
            _winrate_curve_figure(curves, f"Win rate vs bid ({size_lbl})", show_delta),
            use_container_width=True,
        )
    else:
        for value in sorted(df[legend_col].dropna().unique().tolist()):
            curves = cumulative_winrate_curves(df[df[legend_col] == value], bucket)
            if curves.empty:
                continue
            st.plotly_chart(
                _winrate_curve_figure(
                    curves, f"{legend_col} = {value} ({size_lbl})", show_delta
                ),
                use_container_width=True,
            )


def display_funnel(df):
    st.markdown("### Accept / reject funnel")
    funnel = funnel_counts(df)
    fig = px.funnel(funnel, x="count", y="stage")
    fig.update_layout(yaxis_title=None)
    st.plotly_chart(_ui.style_figure(fig), use_container_width=True)
    st.caption(
        "Stages use unambiguous counts; exact accept/reject semantics are pending "
        "confirmation (CONTEXT §4)."
    )


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------


def _render_filters(df):
    st.sidebar.header("Filters")

    lead_types = sorted(df["lead_type_id"].dropna().unique())
    default_lead_type = 6
    default_index = (
        lead_types.index(default_lead_type) if default_lead_type in lead_types else 0
    )
    selected_lead_type = st.sidebar.selectbox(
        "lead_type_id", options=lead_types, index=default_index
    )

    lead_df = df[df["lead_type_id"] == selected_lead_type]
    campaigns = sorted(lead_df["campaign_id"].dropna().unique())
    selected_campaign = st.sidebar.selectbox(
        "campaign_id", options=["All campaign_ids"] + campaigns, index=0
    )

    # Additional dimension filters (Kiran: partner / bidding strategy / insured)
    for col, label in (
        ("account_id", "account_id (partner)"),
        ("bidding_strategy_id", "bidding_strategy_id"),
        ("insured", "insured"),
    ):
        if col in lead_df.columns:
            options = ["All"] + sorted(lead_df[col].dropna().unique().tolist())
            choice = st.sidebar.selectbox(
                label, options=options, index=0, key=f"flt_{col}"
            )
            if choice != "All":
                lead_df = lead_df[lead_df[col] == choice]

    available_states = ordered_state_list(lead_df)
    if "selected_states" not in st.session_state:
        st.session_state.selected_states = available_states.copy()
    if set(st.session_state.selected_states) - set(available_states):
        st.session_state.selected_states = available_states.copy()

    st.sidebar.subheader("States")
    col1, col2 = st.sidebar.columns(2)
    if col1.button("Select All", use_container_width=True):
        st.session_state.selected_states = available_states.copy()
    if col2.button("Deselect All", use_container_width=True):
        st.session_state.selected_states = []

    selected_states = st.sidebar.multiselect(
        label="states", options=available_states, key="selected_states",
        label_visibility="collapsed",
    )
    if selected_states:
        lead_df = lead_df[lead_df["state"].isin(selected_states)]

    if selected_campaign != "All campaign_ids":
        lead_df = lead_df[lead_df["campaign_id"] == selected_campaign]
    return lead_df


def _render_metrics(df):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Unique ids", df["id"].nunique())
    c3.metric("Campaigns", df["campaign_id"].nunique())
    c4.metric("States", df["state"].nunique())

    c1, c2, c3, c4 = st.columns(4)
    rev_sum = df["rev"].sum()
    c1.metric("Revenue", round(rev_sum, 2))
    c2.metric("Profit", round(df["profit"].sum(), 2))
    c3.metric("CM", round(df["profit"].sum() / rev_sum, 2) if rev_sum else 0)
    c4.metric("Win Rate", round(df["won"].mean(), 2) if len(df) else 0)


def main():
    st.title("SmartHub Leads")
    if st.button("🔄 Reload Data"):
        st.cache_data.clear()
        st.rerun()

    try:
        df = load_data()
    except io.DataNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    filtered_df = _render_filters(df)
    if filtered_df.empty:
        st.info("No data available for the selected filters.")
        return

    _render_metrics(filtered_df)

    st.sidebar.subheader("Aggregated Data")
    aggregate_by = st.sidebar.selectbox(
        "Aggregate by", options=["state", "created_hour", "created_dayofweek"]
    )

    st.subheader("Aggregated Data")
    st.dataframe(aggregate_leads(filtered_df, aggregate_by), use_container_width=True)

    st.subheader("Raw Data")
    st.dataframe(filtered_df.head(5000), use_container_width=True)

    st.subheader("Funnel")
    display_funnel(filtered_df)

    st.subheader("Plots")
    display_plot_type_1(filtered_df)
    display_plot_type_2(filtered_df)
    display_plot_type_3(filtered_df)
    display_plot_type_4(filtered_df)


if __name__ == "__main__":
    main()
