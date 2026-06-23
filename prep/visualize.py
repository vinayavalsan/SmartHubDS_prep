import pandas as pd
import plotly.express as px
import streamlit as st


st.set_page_config(page_title="Smarthub Anton Dashboard", layout="wide")


# --------------------
# Data helpers
# --------------------


@st.cache_data
def load_data():
    df = pd.read_parquet("leads.parquet")

    # Drop unused columns
    cols_to_drop = [
        "lead_created_at",
        "excluded",
    ]
    # df = df[df["created_at"] >= "2026-06-14 00:00:00"]

    df = df.drop(columns=cols_to_drop, errors="ignore")

    df = df[df["bid"] > 0]

    # Convert IDs from float to int where appropriate
    if "campaign_id" in df.columns:
        df["campaign_id"] = df["campaign_id"].astype("Int64")

    if "lead_type_id" in df.columns:
        df["lead_type_id"] = df["lead_type_id"].astype("Int64")

    # Replace blank states with NAvail
    df["state"] = df["state"].fillna("NAvail")
    df.loc[df["state"].str.strip() == "", "state"] = "NAvail"

    # Create date and time related columns
    df["created_at"] = pd.to_datetime(df["created_at"])
    df["created_hour"] = df["created_at"].dt.hour
    df["created_dayofweek"] = df["created_at"].dt.dayofweek

    # Replace blank won with false and convert string true/false to 1/0
    df["won"] = df["won"].fillna("false")
    df["won"] = df["won"].map({"true": 1, "false": 0}).astype("Int64")

    # Replace blank rev with 0
    df["rev"] = df["rev"].fillna(0.0)
    df["payout"] = df["won"] * df["bid"]
    df["profit"] = df["rev"] - df["payout"]

    return df


def ordered_state_list(dataframe):
    states = sorted([s for s in dataframe["state"].dropna().unique() if s != "NAvail"])

    if "NAvail" in dataframe["state"].values:
        states.append("NAvail")

    return states


def aggregate_data(dataframe, aggregate_by):
    agg_df = (
        dataframe.groupby(aggregate_by, dropna=False)
        .agg(
            count=("id", "count"),
            rev=("rev", "sum"),
            bid=("bid", "sum"),
            payout=("payout", "sum"),
            won=("won", "sum"),
        )
        .reset_index()
    )

    agg_df["profit"] = agg_df["rev"] - agg_df["payout"]
    agg_df["cm"] = agg_df["profit"] / agg_df["rev"]
    agg_df.loc[agg_df["rev"] == 0, "cm"] = 0
    agg_df["winrate"] = agg_df["won"] / agg_df["count"]
    agg_df.loc[agg_df["count"] == 0, "winrate"] = 0

    return agg_df


# --------------------
# Plot helpers
# --------------------


def get_plot_metric_options(dataframe):
    metric_options = [
        "count",
        "rev",
        "bid",
        "payout",
        "profit",
        "cm",
        "winrate",
    ]

    required_cols_by_metric = {
        "count": ["id"],
        "rev": ["rev"],
        "bid": ["bid"],
        "payout": ["payout"],
        "profit": ["profit"],
        "cm": ["profit", "rev"],
        "winrate": ["won"],
    }

    return [
        metric
        for metric in metric_options
        if all(col in dataframe.columns for col in required_cols_by_metric[metric])
    ]


def get_legend_options(dataframe):
    excluded_cols = {
        "bid",
        "age",
        "created_at",
        "payout",
        "profit",
        "rev",
        "bid",
        "zip",
    }

    legend_options = ["None"] + sorted(
        [c for c in dataframe.columns if c not in excluded_cols]
    )
    return legend_options


def get_default_option(options, preferred_value):
    return options.index(preferred_value) if preferred_value in options else 0


def style_figure(fig):
    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor="lightgray",
    )

    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor="lightgray",
    )

    return fig


def build_metric_plot_data(dataframe, group_cols, metric_col):
    grouped = dataframe.groupby(group_cols, dropna=False)

    if metric_col == "count":
        plot_df = grouped.size().reset_index(name="value")

    elif metric_col == "cm":
        plot_df = grouped.agg(
            rev=("rev", "sum"),
            profit=("profit", "sum"),
        ).reset_index()
        plot_df["value"] = plot_df["profit"] / plot_df["rev"]
        plot_df.loc[plot_df["rev"] == 0, "value"] = 0

    elif metric_col == "winrate":
        plot_df = grouped.agg(
            won=("won", "sum"),
            count=("id", "count"),
        ).reset_index()
        plot_df["value"] = plot_df["won"] / plot_df["count"]
        plot_df.loc[plot_df["count"] == 0, "value"] = 0

    else:
        plot_df = grouped[metric_col].sum().reset_index(name="value")

    return plot_df.sort_values(group_cols).copy()


def build_plot_type_1_data(dataframe, feature_col, metric_col, legend_col):
    group_cols = [feature_col]

    if legend_col != "None" and legend_col != feature_col:
        group_cols.append(legend_col)

    plot_df = build_metric_plot_data(dataframe, group_cols, metric_col)
    plot_df[feature_col] = plot_df[feature_col].astype(str)

    if legend_col != "None" and legend_col in plot_df.columns:
        plot_df[legend_col] = plot_df[legend_col].astype(str)

    return plot_df


def build_plot_type_1_figure(plot_df, feature_col, metric_col, legend_col):
    if legend_col != "None" and legend_col in plot_df.columns:
        fig = px.line(
            plot_df,
            x=feature_col,
            y="value",
            color=legend_col,
            markers=True,
            title=f"{metric_col} vs {feature_col} split by {legend_col}",
        )
    else:
        fig = px.line(
            plot_df,
            x=feature_col,
            y="value",
            markers=True,
            title=f"{metric_col} by {feature_col}",
        )

    fig.update_layout(
        xaxis_title=feature_col,
        yaxis_title=metric_col,
        legend_title=legend_col if legend_col != "None" else None,
    )

    return style_figure(fig)


def display_plot_type_1(dataframe):
    st.markdown("### Plot Type 1")

    plot_col1, plot_col2, plot_col3 = st.columns(3)

    feature_options = sorted(dataframe.columns.tolist())
    metric_options = get_plot_metric_options(dataframe)

    legend_options = get_legend_options(dataframe)

    default_feature = (
        "num_auto_violations"
        if "num_auto_violations" in feature_options
        else feature_options[0]
    )
    default_legend = "None"

    with plot_col1:
        feature_col = st.selectbox(
            "x-axis",
            options=feature_options,
            index=get_default_option(feature_options, default_feature),
        )

    with plot_col2:
        metric_cols = st.multiselect(
            "y-axis",
            options=metric_options,
            default=["winrate"],
        )

    with plot_col3:
        legend_col = st.selectbox(
            "Legend divisions",
            options=legend_options,
            index=get_default_option(legend_options, default_legend),
        )

    if not metric_cols:
        st.info("Select at least one y-axis metric to display the plot.")
        return

    for metric_col in metric_cols:
        plot_df = build_plot_type_1_data(
            dataframe=dataframe,
            feature_col=feature_col,
            metric_col=metric_col,
            legend_col=legend_col,
        )

        fig = build_plot_type_1_figure(
            plot_df=plot_df,
            feature_col=feature_col,
            metric_col=metric_col,
            legend_col=legend_col,
        )

        st.plotly_chart(fig, use_container_width=True)


def build_plot_type_2_data(dataframe, freq_label, metric_col, legend_col):
    freq_map = {
        "1 hr": "h",
        "1 day": "D",
    }

    freq = freq_map[freq_label]
    plot_df_source = dataframe.copy()
    plot_df_source["created_at_bucket"] = plot_df_source["created_at"].dt.floor(freq)

    group_cols = ["created_at_bucket"]

    if legend_col != "None":
        group_cols.append(legend_col)

    plot_df = build_metric_plot_data(plot_df_source, group_cols, metric_col)

    if legend_col != "None" and legend_col in plot_df.columns:
        plot_df[legend_col] = plot_df[legend_col].astype(str)

    return plot_df


def build_plot_type_2_figure(plot_df, metric_col, legend_col, freq_label):
    if legend_col != "None" and legend_col in plot_df.columns:
        fig = px.line(
            plot_df,
            x="created_at_bucket",
            y="value",
            color=legend_col,
            markers=True,
            title=f"{metric_col} over time ({freq_label}) split by {legend_col}",
        )
    else:
        fig = px.line(
            plot_df,
            x="created_at_bucket",
            y="value",
            markers=True,
            title=f"{metric_col} over time ({freq_label})",
        )

    fig.update_layout(
        xaxis_title="created_at",
        yaxis_title=metric_col,
        legend_title=legend_col if legend_col != "None" else None,
    )

    return style_figure(fig)


def display_plot_type_2(dataframe):
    st.markdown("### Plot Type 2 - Time Series")

    if "created_at" not in dataframe.columns:
        st.info("created_at column is required for Plot Type 2.")
        return

    plot_col1, plot_col2, plot_col3 = st.columns(3)

    freq_options = ["1 hr", "1 day"]
    metric_options = get_plot_metric_options(dataframe)
    legend_options = get_legend_options(dataframe)
    default_legend = "None"

    with plot_col1:
        freq_label = st.selectbox(
            "Frequency",
            options=freq_options,
            index=0,
        )

    with plot_col2:
        metric_cols = st.multiselect(
            "y-axis metrics",
            options=metric_options,
            default=["winrate"],
            key="plot_type_2_metrics",
        )

    with plot_col3:
        legend_col = st.selectbox(
            "Legend divisions",
            options=legend_options,
            index=get_default_option(legend_options, default_legend),
            key="plot_type_2_legend",
        )

    if not metric_cols:
        st.info("Select at least one y-axis metric to display the plot.")
        return

    for metric_col in metric_cols:
        plot_df = build_plot_type_2_data(
            dataframe=dataframe,
            freq_label=freq_label,
            metric_col=metric_col,
            legend_col=legend_col,
        )

        fig = build_plot_type_2_figure(
            plot_df=plot_df,
            metric_col=metric_col,
            legend_col=legend_col,
            freq_label=freq_label,
        )

        st.plotly_chart(fig, use_container_width=True)


def build_plot_type_3_data(dataframe, x_col, bucket_size, metric_col, legend_col):
    """
    Build Plot Type 3 data.

    This mirrors Plot Type 2, but instead of time buckets it creates dollar-value
    buckets from the selected x-axis metric. The x-axis value is the upper bound
    of each bucket.

    Example with bucket_size = 1:
        bid values in [0, 1) -> x value 1
        bid values in [1, 2) -> x value 2
        bid values in [2, 3) -> x value 3
    """
    plot_df_source = dataframe.dropna(subset=[x_col]).copy()

    if plot_df_source.empty:
        return pd.DataFrame(), "bucket_upper"

    bucket_upper_col = "bucket_upper"

    # Floor each row to the lower bin edge, then use lower + bucket_size as
    # the plotted x value. This gives a clean numeric x-axis sorted like time.
    plot_df_source["bucket_lower"] = (
        plot_df_source[x_col].astype(float).floordiv(bucket_size) * bucket_size
    )
    plot_df_source[bucket_upper_col] = plot_df_source["bucket_lower"] + bucket_size
    plot_df_source[bucket_upper_col] = plot_df_source[bucket_upper_col].round(6)

    group_cols = [bucket_upper_col]

    if legend_col != "None":
        group_cols.append(legend_col)

    plot_df = build_metric_plot_data(plot_df_source, group_cols, metric_col)
    plot_df = plot_df.sort_values(group_cols).copy()

    if legend_col != "None" and legend_col in plot_df.columns:
        plot_df[legend_col] = plot_df[legend_col].astype(str)

    return plot_df, bucket_upper_col


def build_plot_type_3_figure(
    plot_df,
    x_label,
    bucket_upper_col,
    metric_col,
    legend_col,
    bucket_size_label,
):
    if legend_col != "None" and legend_col in plot_df.columns:
        fig = px.line(
            plot_df,
            x=bucket_upper_col,
            y="value",
            color=legend_col,
            markers=True,
            title=f"{metric_col} by {x_label} bucket ({bucket_size_label}) "
            f"split by {legend_col}",
        )
    else:
        fig = px.line(
            plot_df,
            x=bucket_upper_col,
            y="value",
            markers=True,
            title=f"{metric_col} by {x_label} bucket ({bucket_size_label})",
        )

    fig.update_layout(
        xaxis_title=f"{x_label} bucket upper bound",
        yaxis_title=metric_col,
        legend_title=legend_col if legend_col != "None" else None,
    )

    fig.update_xaxes(type="linear")

    return style_figure(fig)


def display_plot_type_3(dataframe):
    st.markdown("### Plot Type 3 - Metric Value Series")

    x_axis_map = {
        "profit": "profit",
        "bid": "bid",
        "payout": "payout",
        "revenue": "rev",
    }
    x_axis_options = [
        label for label, col in x_axis_map.items() if col in dataframe.columns
    ]

    if not x_axis_options:
        st.info("Plot Type 3 requires at least one of profit, bid, payout, or revenue.")
        return

    plot_col1, plot_col2, plot_col3, plot_col4 = st.columns(4)

    bucket_options = {
        "$0.50": 0.5,
        "$1": 1,
        "$2": 2,
        "$5": 5,
        "$10": 10,
    }

    metric_options = get_plot_metric_options(dataframe)
    legend_options = get_legend_options(dataframe)
    default_legend = "None"

    with plot_col1:
        x_label = st.selectbox(
            "x-axis metric",
            options=x_axis_options,
            index=get_default_option(x_axis_options, "profit"),
            key="plot_type_3_x_axis",
        )

    with plot_col2:
        bucket_size_label = st.selectbox(
            "Bin width",
            options=list(bucket_options.keys()),
            index=1,
            key="plot_type_3_frequency",
        )

    with plot_col3:
        metric_cols = st.multiselect(
            "y-axis metrics",
            options=metric_options,
            default=["winrate"],
            key="plot_type_3_metrics",
        )

    with plot_col4:
        legend_col = st.selectbox(
            "Legend divisions",
            options=legend_options,
            index=get_default_option(legend_options, default_legend),
            key="plot_type_3_legend",
        )

    if not metric_cols:
        st.info("Select at least one y-axis metric to display the plot.")
        return

    x_col = x_axis_map[x_label]
    bucket_size = bucket_options[bucket_size_label]

    for metric_col in metric_cols:
        plot_df, bucket_upper_col = build_plot_type_3_data(
            dataframe=dataframe,
            x_col=x_col,
            bucket_size=bucket_size,
            metric_col=metric_col,
            legend_col=legend_col,
        )

        if plot_df.empty:
            st.info(f"No data available for {metric_col} by {x_label} buckets.")
            continue

        fig = build_plot_type_3_figure(
            plot_df=plot_df,
            x_label=x_label,
            bucket_upper_col=bucket_upper_col,
            metric_col=metric_col,
            legend_col=legend_col,
            bucket_size_label=bucket_size_label,
        )

        st.plotly_chart(fig, use_container_width=True)


# --------------------
# App
# --------------------


def main():
    st.title("Smarthub Anton Dashboard")
    if st.button("🔄 Reload Data"):
        st.cache_data.clear()
        st.rerun()

    df = load_data()

    # Filters
    st.sidebar.header("Filters")

    lead_types = sorted(df["lead_type_id"].dropna().unique())
    default_lead_type = 6

    default_index = (
        lead_types.index(default_lead_type) if default_lead_type in lead_types else 0
    )

    selected_lead_type = st.sidebar.selectbox(
        "lead_type_id", options=lead_types, index=default_index
    )

    # Campaigns available for selected lead type
    lead_df = df[df["lead_type_id"] == selected_lead_type]

    campaigns = sorted(lead_df["campaign_id"].dropna().unique())

    selected_campaign = st.sidebar.selectbox(
        "campaign_id", options=["All campaign_ids"] + campaigns, index=0
    )

    available_states = ordered_state_list(lead_df)

    # Initialize state selection to all available states
    if "selected_states" not in st.session_state:
        st.session_state.selected_states = available_states.copy()

    # Reset selection if available states change due to other filters
    if set(st.session_state.selected_states) - set(available_states):
        st.session_state.selected_states = available_states.copy()

    st.sidebar.subheader("States")

    col1, col2 = st.sidebar.columns(2)

    if col1.button("Select All", use_container_width=True):
        st.session_state.selected_states = available_states.copy()

    if col2.button("Deselect All", use_container_width=True):
        st.session_state.selected_states = []

    selected_states = st.sidebar.multiselect(
        label="", options=available_states, key="selected_states"
    )

    # Apply state filter only when at least one state is selected
    if selected_states:
        lead_df = lead_df[lead_df["state"].isin(selected_states)]

    # Apply filters
    filtered_df = lead_df

    if selected_campaign != "All campaign_ids":
        filtered_df = filtered_df[filtered_df["campaign_id"] == selected_campaign]

    if filtered_df.empty:
        st.info("No data available for the selected filters.")
        return

    # --------------------
    # Main Page
    # --------------------

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Rows", f"{len(filtered_df):,}")
    col2.metric("Unique ids", filtered_df["id"].nunique())
    col3.metric("Campaigns", filtered_df["campaign_id"].nunique())
    col4.metric("States", filtered_df["state"].nunique())

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Revenue", round(filtered_df["rev"].sum(), 2))
    col2.metric("Profit", round(filtered_df["profit"].sum(), 2))

    cm = (
        filtered_df["profit"].sum() / filtered_df["rev"].sum()
        if filtered_df["rev"].sum()
        else 0
    )
    col3.metric("CM", round(cm, 2))

    winrate = filtered_df["won"].mean() if len(filtered_df) else 0
    col4.metric("Win Rate", round(winrate, 2))

    st.sidebar.subheader("Aggregated Data")

    aggregate_by = st.sidebar.selectbox(
        "Aggregate by", options=["state", "created_hour", "created_dayofweek"]
    )

    agg_df = aggregate_data(filtered_df, aggregate_by)

    st.subheader("Aggregated Data")
    st.dataframe(agg_df, use_container_width=True)

    st.subheader("Raw Data")
    st.dataframe(filtered_df.head(5000), use_container_width=True)

    st.subheader("Plots")
    display_plot_type_1(filtered_df)
    display_plot_type_2(filtered_df)
    display_plot_type_3(filtered_df)


if __name__ == "__main__":
    main()
