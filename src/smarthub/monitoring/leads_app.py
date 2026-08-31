"""SmartHub Leads dashboard.

This module loads eligible lead data, adds derived features and business metrics,
and renders lead-level summaries, filters, funnels, and diagnostic plots.
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from smarthub.core import auction, io, storage
from smarthub.core import transforms as core_transforms
from smarthub.core.config import StorageSettings
from smarthub.core.lead_types import lead_type_name
from smarthub.feature_engineering.feature_registry import FEATURES
from smarthub.monitoring import _ui
from smarthub.monitoring.transforms import (
    add_historical_business_metrics,
    aggregate_leads,
    build_metric_plot_data,
    cumulative_winrate_curves,
    funnel_counts,
)

# set_page_config is called by the entry (app.py or the __main__ guard below).


# The Leads page is intentionally model-agnostic. Prediction outputs and other
# ML-derived fields belong on Performance / Predictions, never here.
_ML_COLUMNS = {
    "model_expected_revenue",
    "prediction_expected_revenue",
    "recommended_bid",
    "recommended_bid_predicted_win_rate",
    "recommended_bid_predicted_revenue",
    "recommended_bid_predicted_bid_cost",
    "recommended_bid_predicted_profit",
    "recommended_bid_predicted_cm",
    "recommended_bid_predicted_revenue_on_sold",
    "recommended_bid_predicted_profit_on_sold",
    "ml_predicted_revenue",
    "ml_predicted_bid_cost",
    "ml_predicted_profit",
    "prediction_id",
    "model_name",
    "model_version",
    "model_type",
    "training_table_version",
    "decision_path",
    "shap_explanation",
}


def _registry_derived_feature_names(df=None):
    """Return registry-derived names, optionally limited to dataframe columns."""
    names = [
        spec.name
        for spec in FEATURES.values()
        if spec.source == "derived" and spec.derive is not None
    ]
    if df is None:
        return names
    return [name for name in names if name in df.columns]


def _plot_dimension_options(df):
    """Return plot dimensions with registry-derived features guaranteed present."""
    base = list(_ui.get_legend_options(df))
    derived = _registry_derived_feature_names(df)
    options = ["None"] + derived + [value for value in base if value != "None"]
    return list(dict.fromkeys(options))


def _is_numeric_plot_feature(df, column):
    """Return whether a column should be treated as numeric on plot axes."""
    spec = FEATURES.get(column)
    if spec is not None and spec.kind in {
        "numeric",
        "numeric_continuous",
        "numeric_discrete",
        "binary",
    }:
        return True
    if column in {
        "bid",
        "rev",
        "realized_revenue",
        "bid_cost",
        "realized_profit",
        "expected_revenue",
        "expected_profit",
    }:
        return True
    return column in df.columns and pd.api.types.is_numeric_dtype(df[column])


def _numeric_plot_axis_options(df):
    """Return numeric plot axes, including registry-derived numeric features."""
    preferred_names = (
        "realized_profit",
        "realized_revenue",
        "bid_cost",
        "expected_profit",
        "expected_revenue",
        "bid",
        "rev",
    )
    preferred = [column for column in preferred_names if column in df.columns]
    registry_names = [
        spec.name
        for spec in FEATURES.values()
        if spec.name in df.columns
        and spec.kind in {"numeric", "numeric_continuous", "numeric_discrete", "binary"}
    ]
    detected = [column for column in df.columns if _is_numeric_plot_feature(df, column)]
    return list(dict.fromkeys(preferred + registry_names + detected))


def _add_registry_derived_features(df):
    """Add every registry-defined derived feature without filtering rows.

    Derived columns are generated directly from ``FEATURES`` so monitoring
    stays aligned with the feature registry automatically as new derivations
    are added. Lead-type-specific derivations are only populated for rows to
    which the registry says they apply.
    """
    out = df.copy()
    derived_specs = [
        spec
        for spec in FEATURES.values()
        if spec.source == "derived" and spec.derive is not None
    ]
    for spec in derived_specs:
        if spec.name not in out.columns:
            out[spec.name] = pd.NA

    if "lead_type_id" not in out.columns:
        return out

    lead_type_ids = pd.to_numeric(out["lead_type_id"], errors="coerce")
    for raw_lead_type_id in lead_type_ids.dropna().unique():
        current_id = int(raw_lead_type_id)
        current_name = lead_type_name(current_id)
        mask = lead_type_ids.eq(current_id)

        for spec in derived_specs:
            if spec.lead_types and current_name not in spec.lead_types:
                continue
            source = out.loc[mask].copy()
            derived = spec.derive(source)
            out.loc[mask, spec.name] = derived

    return out


def _prepare_monitoring_leads_frame(df):
    """Prepare already-eligible lead rows for dashboard display.

    Row filtering is intentionally handled in ``load_data`` using the same
    shared auction rules as feature engineering. This helper normalizes display
    fields, adds every registry-derived feature, and computes dashboard metrics
    without removing any additional rows.
    """
    out = df.drop(
        columns=list(core_transforms.LEADS_DROP_COLS)
        + ["realized_payout", "payout", "profit"],
        errors="ignore",
    ).copy()

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
        blank_state = out["state"].astype("string").str.strip() == ""
        out.loc[blank_state, "state"] = "NAvail"

    out = _add_registry_derived_features(out)

    if "created_at" in out.columns:
        out["created_at"] = pd.to_datetime(out["created_at"], utc=True)

    if "won" in out.columns:
        out["won"] = core_transforms.normalize_won(out["won"])

    # Add canonical per-lead business metrics.
    out = add_historical_business_metrics(out)
    out = out.drop(columns=list(_ML_COLUMNS), errors="ignore")

    return out


@st.cache_data
def load_data(days: int):
    """Load recent eligible leads for dashboard analysis.

    Inputs
    ------
    days : int
        Number of recent days to load from raw storage.

    Returns
    -------
    pandas.DataFrame
        Eligible lead rows with display fields, derived features, and canonical
        business metrics.

    Raises
    ------
    io.DataNotFoundError
        Raised when the requested raw data window cannot be loaded.
    """
    try:
        raw = storage.load_window_raw(StorageSettings.from_env(), days)
    except storage.StorageError as exc:
        raise io.DataNotFoundError(str(exc)) from exc

    eligible = raw.loc[~auction.erred_mask(raw)].copy()
    eligible = eligible.loc[auction.auction_eligible_mask(eligible)].copy()
    return _prepare_monitoring_leads_frame(eligible)


def ordered_state_list(df) -> list[str]:
    """Return distinct states with ``"NAvail"`` ordered last.

    Inputs
    ------
    df : pandas.DataFrame
        Leads data containing a ``state`` column.

    Returns
    -------
    list[str]
        Sorted state values with ``"NAvail"`` appended when present.
    """
    states = sorted([s for s in df["state"].dropna().unique() if s != "NAvail"])
    if "NAvail" in df["state"].values:
        states.append("NAvail")
    return states


# ---------------------------------------------------------------------------
# Plot Type 1 - any feature on the x-axis
# ---------------------------------------------------------------------------


def _build_plot_type_1_data(df, feature_col, metric_col, legend_col):
    """Aggregate a metric by a feature (optionally split by a legend column).

    Inputs
    ------
    df : pd.DataFrame
        Filtered leads data.
    feature_col : str
        Column placed on the x-axis.
    metric_col : str
        Metric to aggregate.
    legend_col : str
        Column to split by, or ``"None"``.

    Returns
    -------
    pd.DataFrame
        Plot-ready data with string-cast grouping columns.
    """
    group_cols = [feature_col]
    if legend_col != "None" and legend_col != feature_col:
        group_cols.append(legend_col)

    plot_df = build_metric_plot_data(df, group_cols, metric_col)
    if _is_numeric_plot_feature(df, feature_col):
        plot_df[feature_col] = pd.to_numeric(plot_df[feature_col], errors="coerce")
        plot_df = plot_df.sort_values(feature_col)
    else:
        plot_df[feature_col] = plot_df[feature_col].astype(str)
    if legend_col != "None" and legend_col in plot_df.columns:
        plot_df[legend_col] = plot_df[legend_col].astype(str)
    return plot_df


def _figure_type_1(plot_df, feature_col, metric_col, legend_col):
    """Build the Plot Type 1 line figure from aggregated data.

    Inputs
    ------
    plot_df : pd.DataFrame
        Aggregated plot data.
    feature_col : str
        Column on the x-axis.
    metric_col : str
        Metric on the y-axis.
    legend_col : str
        Column used for color, or ``"None"``.

    Returns
    -------
    plotly.graph_objects.Figure
        The styled line figure.
    """
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
    """Render metric-by-feature plots for the filtered leads.

    Inputs
    ------
    df : pandas.DataFrame
        Filtered leads data to plot.
    """
    st.markdown("### Plot Type 1")
    col1, col2, col3 = st.columns(3)

    derived_features = _registry_derived_feature_names(df)
    other_features = sorted(
        column for column in df.columns if column not in derived_features
    )
    feature_options = derived_features + other_features
    metric_options = _ui.get_plot_metric_options(df)
    legend_options = _plot_dimension_options(df)
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
            width="stretch",
        )


# ---------------------------------------------------------------------------
# Plot Type 2 - time series
# ---------------------------------------------------------------------------

_FREQ_MAP = {"1 hr": "h", "1 day": "D"}


def _build_plot_type_2_data(df, freq_label, metric_col, legend_col):
    """Aggregate a metric over time buckets (optionally split by a legend).

    Inputs
    ------
    df : pd.DataFrame
        Filtered leads data with ``created_at``.
    freq_label : str
        Bucket size label (key of ``_FREQ_MAP``).
    metric_col : str
        Metric to aggregate.
    legend_col : str
        Column to split by, or ``"None"``.

    Returns
    -------
    pd.DataFrame
        Time-bucketed, plot-ready data.
    """
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
    """Build the Plot Type 2 time-series line figure.

    Inputs
    ------
    plot_df : pd.DataFrame
        Aggregated time-series data.
    metric_col : str
        Metric on the y-axis.
    legend_col : str
        Column used for color, or ``"None"``.
    freq_label : str
        Bucket size label shown in the title.

    Returns
    -------
    plotly.graph_objects.Figure
        The styled line figure.
    """
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
    """Render time-series metric plots for the filtered leads.

    Inputs
    ------
    df : pandas.DataFrame
        Filtered leads data containing ``created_at``.
    """
    st.markdown("### Plot Type 2 - Time Series")
    if "created_at" not in df.columns:
        st.info("created_at column is required for Plot Type 2.")
        return

    col1, col2, col3 = st.columns(3)
    metric_options = _ui.get_plot_metric_options(df)
    legend_options = _plot_dimension_options(df)

    with col1:
        freq_label = st.selectbox("Frequency", options=list(_FREQ_MAP.keys()), index=0)
    with col2:
        metric_cols = st.multiselect(
            "y-axis metrics",
            options=metric_options,
            default=["winrate"],
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
            width="stretch",
        )


# ---------------------------------------------------------------------------
# Plot Type 3 - numeric feature bins
# ---------------------------------------------------------------------------

_NUMERIC_BUCKET_OPTIONS = {"0.5": 0.5, "1": 1, "2": 2, "5": 5, "10": 10}
_BID_BUCKET_OPTIONS = {"$0.50": 0.5, "$1": 1, "$2": 2, "$5": 5, "$10": 10}


def _build_plot_type_3_data(df, x_col, bucket_size, metric_col, legend_col):
    """Bin a numeric column and aggregate a metric per bucket.

    Inputs
    ------
    df : pd.DataFrame
        Filtered leads data.
    x_col : str
        Numeric column to bucket.
    bucket_size : float
        Width of each bucket.
    metric_col : str
        Metric to aggregate.
    legend_col : str
        Column to split by, or ``"None"``.

    Returns
    -------
    tuple[pd.DataFrame, str]
        The plot-ready data and the bucket-upper column name.
    """
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
    """Build the Plot Type 3 bucketed line figure.

    Inputs
    ------
    plot_df : pd.DataFrame
        Aggregated, bucketed data.
    x_label : str
        Human-readable name of the bucketed metric.
    bucket_upper_col : str
        Column holding each bucket's upper bound.
    metric_col : str
        Metric on the y-axis.
    legend_col : str
        Column used for color, or ``"None"``.
    size_lbl : str
        Bucket-width label shown in the title.

    Returns
    -------
    plotly.graph_objects.Figure
        The styled line figure.
    """
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
    """Render numeric-feature bucket plots for the filtered leads.

    Inputs
    ------
    df : pandas.DataFrame
        Filtered leads data containing numeric features.
    """
    st.markdown("### Plot Type 3 - Numeric Feature Series")
    x_axis_options = _numeric_plot_axis_options(df)
    if not x_axis_options:
        st.info("Plot Type 3 requires at least one numeric feature.")
        return

    col1, col2, col3, col4 = st.columns(4)
    metric_options = _ui.get_plot_metric_options(df)
    legend_options = _plot_dimension_options(df)

    with col1:
        x_label = st.selectbox(
            "x-axis metric",
            options=x_axis_options,
            index=_ui.get_default_option(x_axis_options, "profit"),
            key="plot_type_3_x_axis",
        )
    with col2:
        size_lbl = st.selectbox(
            "Bin width",
            options=list(_NUMERIC_BUCKET_OPTIONS.keys()),
            index=1,
            key="plot_type_3_frequency",
        )
    with col3:
        metric_cols = st.multiselect(
            "y-axis metrics",
            options=metric_options,
            default=["winrate"],
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

    x_col = x_label
    bucket_size = _NUMERIC_BUCKET_OPTIONS[size_lbl]

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
            width="stretch",
        )


# ---------------------------------------------------------------------------
# Plot Type 4 - cumulative win-rate "shelves" curves
# ---------------------------------------------------------------------------


def _winrate_curve_figure(curves, title, show_delta):
    """Build the cumulative win-rate curve figure.

    Inputs
    ------
    curves : pd.DataFrame
        Win-rate-vs-threshold curves.
    title : str
        Figure title.
    show_delta : bool
        Whether to include the win-rate delta curve.

    Returns
    -------
    plotly.graph_objects.Figure
        The styled line figure.
    """
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
    """Render cumulative win-rate curves across bid thresholds.

    Inputs
    ------
    df : pandas.DataFrame
        Filtered leads data containing ``bid`` and ``won``.
    """
    st.markdown("### Plot Type 4 - Cumulative win-rate curves")
    st.caption(
        "Win rate when bidding at/under (winrate_below) vs over (winrate_above) "
        "each price — find floors, ceilings and edges."
    )
    if not {"bid", "won"}.issubset(df.columns):
        st.info("bid and won columns are required for Plot Type 4.")
        return

    legend_options = _plot_dimension_options(df)
    col1, col2, col3 = st.columns(3)
    with col1:
        size_lbl = st.selectbox(
            "Bin width",
            options=list(_BID_BUCKET_OPTIONS.keys()),
            index=1,
            key="plot4_bin",
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

    bucket = _BID_BUCKET_OPTIONS[size_lbl]

    if legend_col == "None":
        curves = cumulative_winrate_curves(df, bucket)
        if curves.empty:
            st.info("No data available.")
            return
        st.plotly_chart(
            _winrate_curve_figure(curves, f"Win rate vs bid ({size_lbl})", show_delta),
            width="stretch",
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
                width="stretch",
            )


def display_funnel(df):
    """Render opportunity, won, and sold funnel counts.

    Inputs
    ------
    df : pandas.DataFrame
        Filtered leads data to summarize.
    """
    st.markdown("### Accept / reject funnel")
    funnel = funnel_counts(df)
    fig = px.funnel(funnel, x="count", y="stage")
    fig.update_layout(yaxis_title=None)
    st.plotly_chart(_ui.style_figure(fig), width="stretch")
    st.caption("Lead volume from auction opportunity through win and downstream sale.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------


def _render_filters(df):
    """Render the sidebar filters and return the filtered leads.

    Inputs
    ------
    df : pd.DataFrame
        Full leads data to filter.

    Returns
    -------
    pd.DataFrame
        The subset matching the active sidebar selections.
    """
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

    # Additional dimension filters.
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
    if col1.button("Select All", width="stretch"):
        st.session_state.selected_states = available_states.copy()
    if col2.button("Deselect All", width="stretch"):
        st.session_state.selected_states = []

    selected_states = st.sidebar.multiselect(
        label="states",
        options=available_states,
        key="selected_states",
        label_visibility="collapsed",
    )
    if selected_states:
        lead_df = lead_df[lead_df["state"].isin(selected_states)]

    if selected_campaign != "All campaign_ids":
        lead_df = lead_df[lead_df["campaign_id"] == selected_campaign]
    return lead_df


def _render_metrics(df):
    """Render the headline metric tiles for the filtered leads.

    Inputs
    ------
    df : pd.DataFrame
        Filtered leads data to summarize.
    """
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Unique ids", df["id"].nunique())
    c3.metric("Campaigns", df["campaign_id"].nunique())
    c4.metric("States", df["state"].nunique())

    # Historical realized business performance.
    realized_revenue = (
        float(df["realized_revenue"].sum()) if "realized_revenue" in df.columns else 0.0
    )
    bid_cost = float(df["bid_cost"].sum()) if "bid_cost" in df.columns else 0.0
    realized_profit = (
        float(df["realized_profit"].sum()) if "realized_profit" in df.columns else 0.0
    )
    measured_win_rate = (
        float(df["won"].mean()) if len(df) and "won" in df.columns else 0.0
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Realized Revenue", round(realized_revenue, 2))
    c2.metric("Bid Cost", round(bid_cost, 2))
    c3.metric("Realized Profit", round(realized_profit, 2))
    c4.metric(
        "Realized CM",
        round(realized_profit / realized_revenue, 4) if realized_revenue else 0.0,
    )

    expected_revenue = (
        float(pd.to_numeric(df["expected_revenue"], errors="coerce").fillna(0.0).sum())
        if "expected_revenue" in df.columns
        else 0.0
    )
    expected_profit = (
        float(pd.to_numeric(df["expected_profit"], errors="coerce").fillna(0.0).sum())
        if "expected_profit" in df.columns
        else 0.0
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Expected Revenue", round(expected_revenue, 2))
    c2.metric("Expected Profit", round(expected_profit, 2))
    c3.metric("Measured Win Rate", round(measured_win_rate, 4))


def main():
    """Render the SmartHub Leads dashboard."""
    st.title("SmartHub Leads")
    cw1, cw2 = st.columns([1, 3])
    with cw1:
        days = st.number_input(
            "History window (days)",
            min_value=1,
            max_value=21,
            value=3,
            step=1,
        )
    if st.button("🔄 Reload Data"):
        st.cache_data.clear()
        st.rerun()

    try:
        df = load_data(int(days))
    except io.DataNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    filtered_df = _render_filters(df)
    if filtered_df.empty:
        st.info("No data available for the selected filters.")
        return

    _render_metrics(filtered_df)

    st.sidebar.subheader("Aggregated Data")
    aggregate_options = [
        option for option in _plot_dimension_options(filtered_df) if option != "None"
    ]
    aggregate_by = st.sidebar.selectbox(
        "Aggregate by",
        options=aggregate_options,
        index=_ui.get_default_option(aggregate_options, "state"),
    )

    st.subheader("Aggregated Data")
    st.dataframe(aggregate_leads(filtered_df, aggregate_by), width="stretch")

    st.subheader("Lead-Level Data")
    st.dataframe(filtered_df.head(5000), width="stretch")

    st.subheader("Funnel")
    display_funnel(filtered_df)

    st.subheader("Plots")
    display_plot_type_1(filtered_df)
    display_plot_type_2(filtered_df)
    display_plot_type_3(filtered_df)
    display_plot_type_4(filtered_df)


if __name__ == "__main__":
    st.set_page_config(page_title="SmartHub Leads", layout="wide")
    main()
