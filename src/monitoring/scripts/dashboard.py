import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px


st.set_page_config(layout="wide", page_title="SmartHub DS Performance")


METRIC_GROUPS = {
    "All": None,
    "Revenue + Payout breakdown": [
        "revenue_measured",
        "payout",
        "revenue_expected",
    ],
    "Profit": ["profit"],
    "Win Rate": ["winrate"],
    "CM": ["cm_measured", "cm_expected"],
    "Number of opportunities and won": ["num_opportunities", "num_won"],
}


def load_data(uploaded_file=None, default_path="../../../data/etl/sample_data.csv"):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_csv(default_path)

    df["datetime_min"] = pd.to_datetime(df["datetime_min"])
    df["datetime_max"] = pd.to_datetime(df["datetime_max"])

    numeric_cols = [
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
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["profit"] = out["revenue_measured"] - out["payout"]

    rev_exp = out["revenue_expected"].replace(0, np.nan)
    rev_meas = out["revenue_measured"].replace(0, np.nan)
    opps = out["num_opportunities"].replace(0, np.nan)

    out["cm_expected"] = (out["revenue_expected"] - out["payout"]) / rev_exp
    out["cm_measured"] = (out["revenue_measured"] - out["payout"]) / rev_meas
    out["winrate"] = out["num_won"] / opps

    return out


def aggregate_df(
    df: pd.DataFrame, freq: str, group_col: str | None = None
) -> pd.DataFrame:
    sum_cols = [
        "revenue_measured",
        "revenue_expected",
        "payout",
        "num_opportunities",
        "num_won",
    ]

    if group_col is None:
        agg = (
            df.set_index("datetime_min")[sum_cols]
            .resample(freq, label="left", closed="left")
            .sum()
            .reset_index()
            .sort_values("datetime_min")
        )
    else:
        agg = (
            df.set_index("datetime_min")
            .groupby(group_col)[sum_cols]
            .resample(freq, label="left", closed="left")
            .sum()
            .reset_index()
            .sort_values([group_col, "datetime_min"])
        )

    agg = add_derived_columns(agg)
    return agg


def plot_metric_block(
    df_plot: pd.DataFrame, columns: list[str], title: str, y_label: str | None = None
):
    style_map = {
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

    plot_df = df_plot[["datetime_min"] + columns].melt(
        id_vars="datetime_min",
        var_name="metric",
        value_name="value",
    )

    fig = px.line(
        plot_df,
        x="datetime_min",
        y="value",
        color="metric",
        markers=False,
    )

    # --- apply styles ---
    for trace in fig.data:
        metric_name = trace.name
        if metric_name in style_map:
            trace.line.color = style_map[metric_name]["color"]
            trace.line.dash = style_map[metric_name]["dash"]

    fig.update_layout(
        title=title,
        hovermode="x unified",
        legend_title_text="",
        legend=dict(
            orientation="h",
            x=0,
            y=1.02,
            xanchor="left",
            yanchor="bottom",
        ),
        margin=dict(t=80),
    )

    fig.update_xaxes(showgrid=True, title_text="datetime_min")
    fig.update_yaxes(showgrid=True, title_text=y_label or "value")

    st.plotly_chart(fig, use_container_width=True)


def main():
    st.title("SmartHub DS Performance")

    df = load_data("../../../data/etl/sample_data.csv")

    df = add_derived_columns(df)

    bin_map = {
        "1hr": "1h",
        "6hr": "6h",
        "12hr": "12h",
        "day": "D",
        "week": "W-MON",
    }

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        bin_size = st.selectbox("Bin size", options=list(bin_map.keys()), index=3)

    with col2:
        state_options = ["All"] + sorted(df["state"].dropna().unique().tolist())
        selected_state = st.selectbox("state", options=state_options, index=0)

    with col3:
        campaign_options = ["All"] + [
            str(x)
            for x in sorted(df["campaign_id"].dropna().astype(int).unique().tolist())
        ]
        selected_campaign = st.selectbox(
            "campaign_id", options=campaign_options, index=0
        )

    with col4:
        metric_options = list(METRIC_GROUPS.keys())
        selected_metric = st.selectbox("Metric", options=metric_options, index=0)

    with col5:
        show_table = st.checkbox("Show table", value=True)

    st.markdown("---")

    filtered = df.copy()
    if selected_state != "All":
        filtered = filtered[filtered["state"] == selected_state]
    if selected_campaign != "All":
        filtered = filtered[filtered["campaign_id"].astype(str) == selected_campaign]

    if filtered.empty:
        st.info("No data for the selected filters.")
        st.stop()

    group_col = None
    if (
        selected_campaign == "All"
        and selected_state == "All"
        and selected_metric != "All"
    ):
        group_col = "campaign_id"
    elif (
        selected_campaign != "All"
        and selected_state == "All"
        and selected_metric != "All"
    ):
        group_col = "state"

    agg = aggregate_df(filtered, freq=bin_map[bin_size], group_col=group_col)

    if selected_metric == "All":

        plot_metric_block(
            agg,
            ["revenue_measured", "payout", "revenue_expected"],
            "Revenue + Payout breakdown",
            "Amount ($)",
        )

        plot_metric_block(agg, ["profit"], "Profit", "Amount ($)")
        plot_metric_block(agg, ["winrate"], "Win Rate", "Rate")
        plot_metric_block(agg, ["cm_measured", "cm_expected"], "CM", "Margin")

        plot_metric_block(
            agg,
            ["num_opportunities", "num_won"],
            "Number of opportunities and won",
            "Count",
        )

    else:
        cols = METRIC_GROUPS[selected_metric]

        if group_col is None:
            y_label = "Rate" if selected_metric in ["Win Rate", "CM"] else "Value"
            if selected_metric in ["Revenue + Payout breakdown", "Profit"]:
                y_label = "Amount ($)"
            elif selected_metric == "Number of opportunities and won":
                y_label = "Count"

            plot_metric_block(agg, cols, selected_metric, y_label)
        else:
            for group_value in sorted(agg[group_col].dropna().unique().tolist()):
                sub = agg[agg[group_col] == group_value].copy()
                if sub.empty:
                    continue

                st.markdown(f"### {group_col} = {group_value}")

                y_label = "Rate" if selected_metric in ["Win Rate", "CM"] else "Value"
                if selected_metric in ["Revenue + Payout breakdown", "Profit"]:
                    y_label = "Amount ($)"
                elif selected_metric == "Number of opportunities and won":
                    y_label = "Count"

                plot_metric_block(sub, cols, selected_metric, y_label)

    if show_table:
        st.markdown("## Aggregated Data")
        st.dataframe(
            agg.sort_values("datetime_min", ascending=False).reset_index(drop=True),
            use_container_width=True,
        )


if __name__ == "__main__":
    main()
