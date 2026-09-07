"""Interactive SmartHub post-training model diagnostics.

Standalone Streamlit app for reviewing model-evaluation artifacts after a
production training run — kept separate from the regular SmartHub monitoring
dashboards (``smarthub.monitoring``). Evaluation artifacts are loaded per MLflow
``run_id`` (with a local ``data/model_evaluations`` fallback for dev). Never
reruns the model.

Run from the repository root with:

    streamlit run src/smarthub/model_diagnostics/app.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from smarthub.model_diagnostics.diagnostics import (
    FeatureDiagnosticConfig,
    apply_feature_bucket,
    apply_outcome_filter,
    apply_recommendation_filter,
    available_feature_columns,
    build_feature_buckets,
    build_feature_diagnostic,
    build_subset_diagnostics,
    resolve_feature_analysis_kind,
    validate_optimizer_frame,
)

DEFAULT_MODEL_EVALUATION_ROOT = Path("data/model_evaluations")

_ACCENT_COLOR = "#6A3D9A"


@st.cache_data(show_spinner=False)
def discover_evaluation_artifacts(root: str) -> list[str]:
    """Find optimizer evaluation CSV artifacts recursively."""
    base = Path(root).expanduser()
    if not base.exists():
        return []
    return sorted(
        str(path)
        for path in base.rglob("bid_optimizer_test_rows.csv")
        if path.is_file()
    )


@st.cache_data(show_spinner=False)
def load_evaluation_artifact(path: str) -> pd.DataFrame:
    """Load one optimizer evaluation artifact."""
    frame = pd.read_csv(path)
    validate_optimizer_frame(frame)
    return frame


# --- MLflow-run source ------------------------------------------------------
# Evaluation artifacts are addressed by MLflow run_id (see mlflow_runs.py), so a
# single hosted app covers every training run and the artifact store stays
# transparent. A local-folder mode is kept as a dev fallback.


@st.cache_data(show_spinner=False, ttl=60)
def _cached_run_options() -> list[tuple[str, str]]:
    """Return ``[(label, run_id), ...]`` for the MLflow run dropdown (newest first)."""
    from smarthub.model_diagnostics import mlflow_runs

    return [(r.label, r.run_id) for r in mlflow_runs.list_runs()]


@st.cache_data(show_spinner="Downloading run artifacts…")
def _cached_optimizer_csv(run_id: str) -> str:
    """Download + locate a run's optimizer evaluation CSV (cached per run_id)."""
    from smarthub.model_diagnostics import mlflow_runs

    return mlflow_runs.optimizer_csv_path(run_id)


def _query_run_id() -> str | None:
    """Read ``?run_id=`` from the URL so an MLflow deep-link preselects that run."""
    try:
        value = st.query_params.get("run_id")
    except Exception:  # pragma: no cover - older Streamlit fallback
        value = (st.experimental_get_query_params().get("run_id") or [None])[0]
    return value or None


def _select_from_mlflow() -> pd.DataFrame | None:
    """Sidebar: pick an MLflow run and load its optimizer evaluation frame."""
    try:
        options = _cached_run_options()
    except Exception as exc:  # noqa: BLE001 - surface, don't crash the page
        st.sidebar.error(f"Could not reach MLflow: {exc}")
        return None
    if not options:
        st.sidebar.warning("No MLflow training runs found.")
        return None

    run_ids = [run_id for _label, run_id in options]
    requested = _query_run_id()
    default_index = run_ids.index(requested) if requested in run_ids else 0

    with st.sidebar:
        choice = st.selectbox(
            "MLflow run",
            options=list(range(len(options))),
            index=default_index,
            format_func=lambda i: options[i][0],
            help="Choose a training run. Opened from MLflow, the ?run_id= "
            "run is preselected.",
        )
    run_id = run_ids[choice]
    if requested and requested not in run_ids:
        st.sidebar.info(
            f"Run `{requested}` from the link wasn't found — showing the "
            "latest instead."
        )
    st.caption(f"MLflow run: `{run_id}`")

    try:
        csv_path = _cached_optimizer_csv(run_id)
        return load_evaluation_artifact(csv_path)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Unable to load evaluation artifact for run `{run_id}`: {exc}")
        return None


def _select_from_local() -> pd.DataFrame | None:
    """Sidebar: pick a local optimizer artifact (dev fallback)."""
    with st.sidebar:
        root = st.text_input(
            "Model evaluation folder",
            value=str(DEFAULT_MODEL_EVALUATION_ROOT),
        )
        artifacts = discover_evaluation_artifacts(root)
        if not artifacts:
            st.warning(
                "No bid_optimizer_test_rows.csv files were found under "
                f"{Path(root).expanduser()}."
            )
            return None
        artifact = st.selectbox(
            "Evaluation artifact",
            options=artifacts,
            index=len(artifacts) - 1,
        )
    try:
        return load_evaluation_artifact(artifact)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Unable to load evaluation artifact: {exc}")
        return None


def _ordered_labels(summary: pd.DataFrame) -> list[str]:
    return summary["feature_bucket"].astype(str).tolist()


def win_rate_chart(
    summary: pd.DataFrame, feature_name: str, analysis_kind: str
) -> go.Figure:
    """Build observed/current/recommended win-rate chart with row counts."""
    labels = _ordered_labels(summary)
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_bar(
        x=labels,
        y=summary["leads"],
        name="Leads",
        opacity=0.20,
        secondary_y=True,
    )
    for column, name in (
        ("observed_win_rate", "Observed win rate"),
        ("predicted_win_rate_existing", "Predicted WR @ existing bid"),
        ("predicted_win_rate_recommended", "Predicted WR @ recommended bid"),
    ):
        fig.add_trace(
            go.Scatter(
                x=labels,
                y=summary[column],
                mode="markers" if analysis_kind == "categorical" else "lines+markers",
                name=name,
            ),
            secondary_y=False,
        )

    fig.update_layout(
        title=f"Win rate by {feature_name}",
        hovermode="x unified",
        legend=dict(orientation="h", x=0, xanchor="left", y=-0.22, yanchor="top"),
        margin=dict(l=20, r=20, t=60, b=120),
    )
    fig.update_yaxes(
        title_text="Win rate",
        range=[0, 1],
        showgrid=True,
        secondary_y=False,
    )
    fig.update_yaxes(
        title_text="Number of leads",
        showgrid=False,
        secondary_y=True,
    )
    fig.update_xaxes(
        title_text=feature_name,
        type="category" if analysis_kind == "categorical" else None,
        categoryorder="array" if analysis_kind == "categorical" else None,
        categoryarray=labels if analysis_kind == "categorical" else None,
    )
    return fig


def bid_chart(
    summary: pd.DataFrame, feature_name: str, analysis_kind: str
) -> go.Figure:
    """Build existing/recommended average bid chart."""
    labels = _ordered_labels(summary)
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=labels,
            y=summary["avg_existing_bid"],
            mode="markers" if analysis_kind == "categorical" else "lines+markers",
            name="Average existing bid",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=labels,
            y=summary["avg_recommended_bid"],
            mode="markers" if analysis_kind == "categorical" else "lines+markers",
            name="Average recommended bid",
        )
    )
    fig.add_trace(
        go.Bar(
            x=labels,
            y=summary["avg_bid_change"],
            name="Average bid change",
            opacity=0.25,
        )
    )

    fig.update_layout(
        title=f"Bid behavior by {feature_name}",
        xaxis_title=feature_name,
        yaxis_title="Bid",
        hovermode="x unified",
        legend=dict(orientation="h", x=0, xanchor="left", y=-0.22, yanchor="top"),
        margin=dict(l=20, r=20, t=60, b=120),
    )
    if analysis_kind == "categorical":
        fig.update_xaxes(
            type="category",
            categoryorder="array",
            categoryarray=labels,
        )
    return fig


def _density_heatmap(
    frame: pd.DataFrame,
    *,
    x_column: str,
    y_column: str,
    x_label: str,
    y_label: str,
    title: str,
    bins: int,
    minimum_count: int,
    diagonal: bool = False,
    horizontal_reference: float | None = None,
    equal_axes: bool = False,
) -> go.Figure | None:
    """Build a 2D count-density heatmap for two numeric columns."""
    if x_column not in frame.columns or y_column not in frame.columns:
        return None

    x = pd.to_numeric(frame[x_column], errors="coerce")
    y = pd.to_numeric(frame[y_column], errors="coerce")
    valid = x.notna() & y.notna()
    if not valid.any():
        return None

    x = x.loc[valid].to_numpy(dtype=float)
    y = y.loc[valid].to_numpy(dtype=float)
    x_min, x_max = float(np.min(x)), float(np.max(x))
    y_min, y_max = float(np.min(y)), float(np.max(y))
    if equal_axes:
        shared_min = min(x_min, y_min)
        shared_max = max(x_max, y_max)
        x_min = y_min = shared_min
        x_max = y_max = shared_max
    if x_min == x_max:
        x_min -= 0.5
        x_max += 0.5
    if y_min == y_max:
        y_min -= 0.5
        y_max += 0.5

    counts, x_edges, y_edges = np.histogram2d(
        x,
        y,
        bins=bins,
        range=((x_min, x_max), (y_min, y_max)),
    )
    counts[counts < minimum_count] = np.nan
    counts[counts <= 0] = np.nan
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2.0
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2.0

    # Match training_artifacts PowerNorm(gamma=0.5). Plotly does not provide
    # PowerNorm directly, so transform the displayed density while retaining
    # the original counts for hover text and colorbar labels.
    density_gamma = 0.25
    max_count = float(np.nanmax(counts)) if np.isfinite(counts).any() else 1.0
    transformed_counts = np.power(counts, density_gamma)
    tick_candidates = [
        10,
        25,
        50,
        100,
        250,
        500,
        1_000,
        2_500,
        5_000,
        10_000,
        25_000,
        50_000,
    ]
    colorbar_ticks = [
        value for value in tick_candidates if minimum_count <= value <= max_count
    ]
    if minimum_count <= max_count and (
        not colorbar_ticks or colorbar_ticks[0] > minimum_count
    ):
        colorbar_ticks.insert(0, minimum_count)
    if not colorbar_ticks:
        colorbar_ticks = [minimum_count]
    if colorbar_ticks[-1] < max_count and max_count / colorbar_ticks[-1] >= 1.5:
        colorbar_ticks.append(max_count)

    fig = go.Figure(
        data=go.Heatmap(
            x=x_centers,
            y=y_centers,
            z=transformed_counts.T,
            customdata=counts.T,
            colorscale=[
                [0.00, "#ffffff"],
                [0.18, "#fffadc"],
                [0.38, "#e6c800"],
                [0.58, "#a05a00"],
                [0.78, "#780064"],
                [0.92, "#1e0064"],
                [1.00, "#000000"],
            ],
            # Keep the density normalization anchored at zero so 0 leads maps
            # to pure white and the maximum density maps to black. The visible
            # colorbar starts at the minimum displayed density instead of zero.
            zmin=0.0,
            zmax=max(max_count**density_gamma, 1.0),
            colorbar=(
                dict(
                    title=dict(text="Leads", side="right"),
                    orientation="v",
                    x=0.91,
                    xanchor="left",
                    y=0.5,
                    yanchor="middle",
                    len=0.82,
                    thickness=14,
                    tickvals=[value**density_gamma for value in colorbar_ticks],
                    ticktext=[f"{value:,.0f}" for value in colorbar_ticks],
                )
                if equal_axes
                else dict(
                    title="Leads",
                    tickvals=[value**density_gamma for value in colorbar_ticks],
                    ticktext=[f"{value:,.0f}" for value in colorbar_ticks],
                )
            ),
            hovertemplate=(
                f"{x_label}: %{{x:.3f}}<br>"
                f"{y_label}: %{{y:.3f}}<br>"
                "Leads: %{customdata:,.0f}<extra></extra>"
            ),
        )
    )

    if diagonal:
        lower = min(x_min, y_min)
        upper = max(x_max, y_max)
        fig.add_trace(
            go.Scatter(
                x=[lower, upper],
                y=[lower, upper],
                mode="lines",
                line=dict(dash="dash"),
                name="No change",
            )
        )

    if horizontal_reference is not None:
        fig.add_hline(
            y=horizontal_reference,
            line_dash="dash",
            annotation_text=f"Reference = {horizontal_reference:.2f}",
        )

    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        margin=dict(
            l=55 if equal_axes else 20,
            r=55 if equal_axes else 20,
            t=60,
            b=55 if equal_axes else 20,
        ),
        height=520 if equal_axes else None,
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(128,128,128,0.25)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(128,128,128,0.25)")
    if equal_axes:
        fig.update_xaxes(range=[x_min, x_max], constrain="domain")
        fig.update_yaxes(
            range=[y_min, y_max],
            scaleanchor="x",
            scaleratio=1,
            constrain="domain",
        )
    return fig


def _derived_diagnostic_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Add analysis-only derived columns used by interactive plots."""
    result = frame.copy()
    existing_bid = pd.to_numeric(result["bid"], errors="coerce")
    recommended_bid = pd.to_numeric(result["recommended_bid"], errors="coerce")
    current_wr = pd.to_numeric(
        result["current_bid_predicted_win_rate"], errors="coerce"
    )
    recommended_wr = pd.to_numeric(
        result["recommended_bid_predicted_win_rate"], errors="coerce"
    )
    result["analysis_bid_change"] = recommended_bid - existing_bid
    result["analysis_win_rate_change"] = recommended_wr - current_wr
    return result


def recommended_bid_distribution(frame: pd.DataFrame) -> go.Figure:
    """Build the recommended-bid distribution histogram."""
    values = pd.to_numeric(frame["recommended_bid"], errors="coerce").dropna()
    fig = go.Figure(
        go.Histogram(
            x=values,
            nbinsx=60,
            marker=dict(color=_ACCENT_COLOR),
        )
    )
    fig.update_layout(
        title="ML Recommended Bid Distribution",
        xaxis_title="Recommended bid",
        yaxis_title="Number of leads",
        margin=dict(l=20, r=20, t=60, b=20),
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(128,128,128,0.25)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(128,128,128,0.25)")
    return fig


def _candidate_bid_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Build candidate-search diagnostics from saved optimizer rows."""
    result = _derived_diagnostic_frame(frame)

    if "n_candidate_bids" in result.columns:
        result["analysis_candidate_bid_count"] = pd.to_numeric(
            result["n_candidate_bids"], errors="coerce"
        )
    else:
        result["analysis_candidate_bid_count"] = np.nan

    maximum_column = next(
        (
            name
            for name in ("maximum_candidate_bid", "max_bid")
            if name in result.columns
        ),
        None,
    )
    if maximum_column is not None:
        result["analysis_maximum_candidate_bid"] = pd.to_numeric(
            result[maximum_column], errors="coerce"
        )
    else:
        result["analysis_maximum_candidate_bid"] = np.nan

    minimum_column = next(
        (
            name
            for name in ("minimum_candidate_bid", "min_bid")
            if name in result.columns
        ),
        None,
    )
    if minimum_column is not None:
        result["analysis_minimum_candidate_bid"] = pd.to_numeric(
            result[minimum_column], errors="coerce"
        )
    else:
        # The training artifact normally stores candidate count and the maximum
        # candidate bid, but not the common minimum candidate bid.  When that is
        # the case, recover the candidate grid from max = min + (n - 1) * step.
        count = result["analysis_candidate_bid_count"]
        maximum = result["analysis_maximum_candidate_bid"]
        valid = count.notna() & maximum.notna() & (count >= 2)
        if valid.sum() >= 2 and count.loc[valid].nunique() >= 2:
            x = count.loc[valid].to_numpy(dtype=float) - 1.0
            y = maximum.loc[valid].to_numpy(dtype=float)
            step, minimum = np.polyfit(x, y, 1)
            if np.isfinite(step) and np.isfinite(minimum) and step > 0:
                result["analysis_minimum_candidate_bid"] = float(minimum)
            else:
                result["analysis_minimum_candidate_bid"] = np.nan
        else:
            result["analysis_minimum_candidate_bid"] = np.nan

    minimum = pd.to_numeric(result["analysis_minimum_candidate_bid"], errors="coerce")
    maximum = pd.to_numeric(result["analysis_maximum_candidate_bid"], errors="coerce")
    recommended = pd.to_numeric(result["recommended_bid"], errors="coerce")
    width = maximum - minimum
    valid_position = (
        minimum.notna() & maximum.notna() & recommended.notna() & (width > 0)
    )
    result["analysis_candidate_position"] = np.nan
    result.loc[valid_position, "analysis_candidate_position"] = (
        (recommended.loc[valid_position] - minimum.loc[valid_position])
        / width.loc[valid_position]
    ).clip(0.0, 1.0)
    return result


def _single_histogram(
    frame: pd.DataFrame,
    column: str,
    title: str,
    x_label: str,
    *,
    bins: int = 50,
    x_range: tuple[float, float] | None = None,
) -> go.Figure | None:
    """Build one uniformly styled histogram for an available numeric column."""
    if column not in frame.columns:
        return None
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    if values.empty:
        return None
    trace_kwargs = dict(x=values, nbinsx=bins, marker=dict(color=_ACCENT_COLOR))
    if x_range is not None:
        trace_kwargs["xbins"] = dict(
            start=float(x_range[0]),
            end=float(x_range[1]),
            size=(float(x_range[1]) - float(x_range[0])) / float(bins),
        )
        trace_kwargs.pop("nbinsx", None)
    fig = go.Figure(go.Histogram(**trace_kwargs))
    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title="Number of leads",
        margin=dict(l=20, r=20, t=60, b=20),
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(128,128,128,0.25)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(128,128,128,0.25)")
    if x_range is not None:
        fig.update_xaxes(range=list(x_range))
    return fig


def _render_candidate_bid_diagnostics(
    frame: pd.DataFrame,
    heatmap_args: dict[str, int],
    *,
    key_prefix: str | None = None,
) -> None:
    """Render the four candidate-search diagnostics for one row subset."""
    candidate_frame = _candidate_bid_frame(frame)

    def chart_key(suffix: str) -> str | None:
        return f"{key_prefix}_{suffix}" if key_prefix else None

    first, second = st.columns(2)
    with first:
        figure = _single_histogram(
            candidate_frame,
            "analysis_candidate_bid_count",
            "Candidate Bid Count Distribution",
            "Number of candidate bids evaluated",
            bins=50,
        )
        if figure is None:
            st.info("Candidate-bid count is unavailable in this artifact.")
        else:
            st.plotly_chart(
                figure,
                use_container_width=True,
                key=chart_key("candidate_count_hist"),
            )

    with second:
        figure = _single_histogram(
            candidate_frame,
            "analysis_candidate_position",
            "Recommended Bid Position Within Candidate Range",
            "Recommended Bid Position (0 = Min, 0.5 = Middle, 1 = Max)",
            bins=40,
            x_range=(0.0, 1.0),
        )
        if figure is None:
            st.info(
                "Candidate-range position is unavailable because this artifact "
                "does not contain enough candidate-range information."
            )
        else:
            st.plotly_chart(
                figure,
                use_container_width=True,
                key=chart_key("candidate_position_hist"),
            )

    first, second = st.columns(2)
    with first:
        figure = _density_heatmap(
            candidate_frame,
            x_column="analysis_candidate_bid_count",
            y_column="analysis_bid_change",
            x_label="Number of candidate bids evaluated",
            y_label="Bid change: recommended - existing",
            title="Candidate Bid Count vs Bid Change",
            **heatmap_args,
        )
        if figure is None:
            st.info("Candidate-bid count vs bid change is unavailable.")
        else:
            st.plotly_chart(
                figure,
                use_container_width=True,
                key=chart_key("candidate_count_vs_change"),
            )

    with second:
        figure = _density_heatmap(
            candidate_frame,
            x_column="analysis_candidate_bid_count",
            y_column="analysis_candidate_position",
            x_label="Number of candidate bids evaluated",
            y_label="Recommended Bid Position (0 = Min, 0.5 = Middle, 1 = Max)",
            title="Candidate Bid Count vs Recommended-Bid Position",
            **heatmap_args,
        )
        if figure is None:
            st.info(
                "Candidate-bid count vs range position is unavailable because "
                "this artifact does not contain enough candidate-range information."
            )
        else:
            st.plotly_chart(
                figure,
                use_container_width=True,
                key=chart_key("candidate_count_vs_position"),
            )


def _show_subset_metrics(diagnostics: dict[str, float | int | None]) -> None:
    """Render compact selected-subset metrics."""
    if not diagnostics:
        return

    row1 = st.columns(4)
    row1[0].metric("Leads", f"{diagnostics['leads']:,}")
    share = diagnostics.get("share_of_filtered_rows")
    row1[1].metric("Share of outcome-filtered rows", f"{share:.2%}")
    row1[2].metric(
        "Historical win rate",
        f"{diagnostics['observed_win_rate']:.2%}",
    )
    maximum_share = diagnostics.get("at_maximum_candidate_bid")
    row1[3].metric(
        "At maximum candidate bid",
        "n/a" if maximum_share is None else f"{maximum_share:.2%}",
    )

    row2 = st.columns(4)
    row2[0].metric(
        "Median existing bid",
        f"${diagnostics['median_existing_bid']:.2f}",
    )
    row2[1].metric(
        "Median recommended bid",
        f"${diagnostics['median_recommended_bid']:.2f}",
    )
    row2[2].metric(
        "Median bid change",
        f"${diagnostics['median_bid_change']:+.2f}",
    )
    profit_lift = diagnostics.get("median_expected_profit_lift")
    row2[3].metric(
        "Median expected-profit lift",
        "n/a" if profit_lift is None else f"${profit_lift:+.2f}",
    )

    row3 = st.columns(3)
    row3[0].metric(
        "Avg predicted WR @ existing",
        f"{diagnostics['avg_current_predicted_win_rate']:.3f}",
    )
    row3[1].metric(
        "Avg predicted WR @ recommended",
        f"{diagnostics['avg_recommended_predicted_win_rate']:.3f}",
    )
    row3[2].metric(
        "Avg predicted WR change",
        f"{diagnostics['avg_predicted_win_rate_change']:+.3f}",
    )


def main() -> None:
    st.set_page_config(
        page_title="SmartHub Feature Diagnostics",
        layout="wide",
    )
    st.title("SmartHub Feature Diagnostics")
    st.caption(
        "Explore saved held-out optimizer evaluation artifacts. "
        "This page does not rerun the model."
    )

    with st.sidebar:
        st.header("Evaluation source")
        source = st.radio(
            "Load artifacts from",
            options=["MLflow run", "Local folder"],
            index=0,
            help="MLflow: pick a training run by run_id. "
            "Local: read data/model_evaluations (dev).",
        )

    if source == "MLflow run":
        frame = _select_from_mlflow()
    else:
        frame = _select_from_local()
    if frame is None:
        st.stop()

    features = available_feature_columns(frame)
    if not features:
        st.error("No candidate feature columns were found in this artifact.")
        st.stop()

    with st.sidebar:
        feature = st.selectbox("Feature", options=features)
        outcome = st.selectbox("Historical outcome", ["All", "Won", "Lost"])
        recommendation_filter = st.selectbox(
            "Optimizer recommendation",
            [
                "All",
                "Minimum bid",
                "Interior",
                "Maximum candidate bid",
                "Large bid increase",
                "Large bid decrease",
            ],
        )

        analysis_kind = resolve_feature_analysis_kind(frame, feature)
        if analysis_kind == "continuous":
            binning = st.selectbox("Numeric binning", ["quantile", "fixed"])
            bins = st.slider("Number of feature bins", 3, 30, 10)
            top_n = 20
            group_rare = True
        else:
            binning = "quantile"
            bins = 10
            top_n = st.slider("Top categories / values", 5, 50, 20)
            group_rare = st.checkbox("Group remaining values as Other", True)

        min_support = st.number_input(
            "Minimum leads per bucket/category",
            min_value=1,
            value=100,
            step=50,
        )
        density_bins = st.slider("2D heatmap bins", 20, 100, 50, step=5)
        density_min_count = st.number_input(
            "Minimum leads per heatmap cell",
            min_value=1,
            value=1,
            step=1,
        )

    outcome_filtered = apply_outcome_filter(frame, outcome)
    try:
        recommendation_filtered = apply_recommendation_filter(
            outcome_filtered, recommendation_filter
        )
    except ValueError as exc:
        st.error(str(exc))
        st.stop()
    if recommendation_filtered.empty:
        st.warning("No rows remain after the optimizer recommendation filter.")
        st.stop()
    config = FeatureDiagnosticConfig(
        feature=feature,
        bins=int(bins),
        binning=binning,
        top_n=int(top_n),
        min_support=int(min_support),
        group_rare_as_other=group_rare,
    )

    try:
        summary = build_feature_diagnostic(recommendation_filtered, config)
        buckets = build_feature_buckets(recommendation_filtered, config)
    except Exception as exc:
        st.error(f"Unable to build feature diagnostics: {exc}")
        st.stop()

    if summary.empty:
        st.warning("No feature buckets remain after the current filters.")
        st.stop()

    bucket_counts = buckets.value_counts(dropna=True)
    bucket_options = [
        value
        for value in summary["feature_bucket"].astype(str).tolist()
        if bucket_counts.get(value, 0) >= int(min_support)
    ]
    with st.sidebar:
        selected_bucket = st.selectbox(
            "Feature value / bucket",
            options=["All", *bucket_options],
        )
        show_all_buckets = st.checkbox(
            "Show plots for all feature values / buckets",
            value=False,
            help=(
                "Render the diagnostic plots for every visible feature value or "
                "bucket one after another, in addition to the "
                "currently selected bucket."
            ),
        )

    selected = apply_feature_bucket(
        recommendation_filtered,
        config,
        selected_bucket,
    )
    if selected.empty:
        st.warning("No rows remain for the selected feature bucket.")
        st.stop()

    diagnostics = build_subset_diagnostics(
        selected,
        reference_rows=len(recommendation_filtered),
    )

    st.write(
        f"**Rows after outcome filter:** {len(outcome_filtered):,}  \n"
        f"**Rows after recommendation filter:** {len(recommendation_filtered):,}  \n"
        f"**Optimizer recommendation:** `{recommendation_filter}`  \n"
        f"**Selected feature bucket:** `{selected_bucket}`"
    )
    _show_subset_metrics(diagnostics)

    summary_tab, bid_tab, economics_tab, candidate_tab = st.tabs(
        ["Summary", "Bid Diagnostics", "Economics", "Candidate Bids"]
    )

    with summary_tab:
        left, right = st.columns(2)
        with left:
            st.plotly_chart(
                win_rate_chart(summary, feature, analysis_kind),
                use_container_width=True,
            )
        with right:
            st.plotly_chart(
                bid_chart(summary, feature, analysis_kind),
                use_container_width=True,
            )

        st.subheader("Feature summary")
        display = summary.copy()
        display["fraction_of_filtered_rows"] = (
            display["fraction_of_filtered_rows"] * 100.0
        )
        st.dataframe(
            display,
            use_container_width=True,
            hide_index=True,
        )

        csv = summary.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download current summary CSV",
            data=csv,
            file_name=f"feature_diagnostic_{feature}.csv",
            mime="text/csv",
        )

    heatmap_args = {
        "bins": int(density_bins),
        "minimum_count": int(density_min_count),
    }

    # When all-bucket plotting is enabled, the main plots use the complete
    # recommendation-filtered population. The selected feature bucket remains
    # available for the Summary tab and for normal single-bucket exploration.
    main_plot_source = recommendation_filtered if show_all_buckets else selected
    main_plot_frame = _derived_diagnostic_frame(main_plot_source)

    with bid_tab:
        if show_all_buckets:
            st.subheader("All Feature Values Combined")

        first, second, third = st.columns(3)
        with first:
            figure = _density_heatmap(
                main_plot_frame,
                x_column="bid",
                y_column="recommended_bid",
                x_label="Existing bid",
                y_label="ML recommended bid",
                title="Existing Bid vs ML Recommended Bid",
                diagonal=True,
                equal_axes=True,
                **heatmap_args,
            )
            if figure is not None:
                st.plotly_chart(figure, use_container_width=True)
        with second:
            figure = _density_heatmap(
                main_plot_frame,
                x_column="analysis_bid_change",
                y_column="analysis_win_rate_change",
                x_label="Bid change: recommended - existing",
                y_label="Predicted win-rate change",
                title="Bid Change vs Predicted Win-Rate Change",
                **heatmap_args,
            )
            if figure is not None:
                st.plotly_chart(figure, use_container_width=True)
        with third:
            st.plotly_chart(
                recommended_bid_distribution(main_plot_frame),
                use_container_width=True,
            )

        if show_all_buckets:
            st.divider()
            st.subheader("All Feature Values")
            st.caption(
                "Each section below applies the current outcome, "
                "optimizer recommendation, support, and density settings "
                "to one feature value or bucket."
            )
            for bucket_value in bucket_options:
                bucket_frame = apply_feature_bucket(
                    recommendation_filtered,
                    config,
                    bucket_value,
                )
                if bucket_frame.empty:
                    continue

                st.markdown(f"#### {feature} = {bucket_value}")
                st.caption(f"Rows: {len(bucket_frame):,}")
                bucket_plot_frame = _derived_diagnostic_frame(bucket_frame)

                first, second, third = st.columns(3)
                with first:
                    figure = _density_heatmap(
                        bucket_plot_frame,
                        x_column="bid",
                        y_column="recommended_bid",
                        x_label="Existing bid",
                        y_label="ML recommended bid",
                        title="Existing Bid vs ML Recommended Bid",
                        diagonal=True,
                        equal_axes=True,
                        **heatmap_args,
                    )
                    if figure is not None:
                        st.plotly_chart(
                            figure,
                            use_container_width=True,
                            key=f"all_bid_bid_{feature}_{bucket_value}",
                        )

                with second:
                    figure = _density_heatmap(
                        bucket_plot_frame,
                        x_column="analysis_bid_change",
                        y_column="analysis_win_rate_change",
                        x_label="Bid change: recommended - existing",
                        y_label="Predicted win-rate change",
                        title="Bid Change vs Predicted Win-Rate Change",
                        **heatmap_args,
                    )
                    if figure is not None:
                        st.plotly_chart(
                            figure,
                            use_container_width=True,
                            key=f"all_bid_wr_{feature}_{bucket_value}",
                        )

                with third:
                    st.plotly_chart(
                        recommended_bid_distribution(bucket_plot_frame),
                        use_container_width=True,
                        key=f"all_bid_hist_{feature}_{bucket_value}",
                    )
                st.divider()

    with candidate_tab:
        st.caption(
            "Candidate-search diagnostics use the same active outcome, optimizer "
            "recommendation, support, density, and feature-bucket settings as the "
            "other diagnostic tabs."
        )
        st.markdown(
            "**Recommended Bid Position** = "
            "(Recommended Bid - Minimum Candidate Bid) / "
            "(Maximum Candidate Bid - Minimum Candidate Bid), where "
            "**0 = minimum, 0.5 = middle, and 1 = maximum candidate bid**."
        )

        if show_all_buckets:
            st.subheader("All Feature Values Combined")

        _render_candidate_bid_diagnostics(
            main_plot_source,
            heatmap_args,
            key_prefix=(
                "candidate_combined" if show_all_buckets else "candidate_selected"
            ),
        )

        if show_all_buckets:
            st.divider()
            st.subheader("All Feature Values")
            st.caption(
                "Each section below applies the current outcome, "
                "optimizer recommendation, support, and density settings "
                "to one feature value or bucket."
            )

            for bucket_value in bucket_options:
                bucket_frame = apply_feature_bucket(
                    recommendation_filtered,
                    config,
                    bucket_value,
                )
                if bucket_frame.empty:
                    continue

                st.markdown(f"#### {feature} = {bucket_value}")
                st.caption(f"Rows: {len(bucket_frame):,}")
                safe_bucket = str(bucket_value).replace(" ", "_").replace("/", "_")
                _render_candidate_bid_diagnostics(
                    bucket_frame,
                    heatmap_args,
                    key_prefix=f"candidate_{feature}_{safe_bucket}",
                )
                st.divider()

    with economics_tab:
        if show_all_buckets:
            st.subheader("All Feature Values Combined")

        first, second = st.columns(2)
        with first:
            figure = _density_heatmap(
                main_plot_frame,
                x_column="analysis_bid_change",
                y_column="expected_profit_lift",
                x_label="Bid change: recommended - existing",
                y_label="Probability-weighted expected-profit lift",
                title="Bid Change vs Expected-Profit Lift",
                **heatmap_args,
            )
            if figure is None:
                st.info("Expected-profit lift is unavailable in this artifact.")
            else:
                st.plotly_chart(figure, use_container_width=True)

        with second:
            figure = _density_heatmap(
                main_plot_frame,
                x_column="analysis_bid_change",
                y_column="recommended_bid_cm_if_won",
                x_label="Bid change: recommended - existing",
                y_label="Recommended CM if won",
                title="Bid Change vs Recommended CM",
                **heatmap_args,
            )
            if figure is None:
                st.info("Recommended CM is unavailable in this artifact.")
            else:
                st.plotly_chart(figure, use_container_width=True)

        if show_all_buckets:
            st.divider()
            st.subheader("All Feature Values")
            st.caption(
                "Each section below applies the current outcome, "
                "optimizer recommendation, support, and density settings "
                "to one feature value or bucket."
            )
            for bucket_value in bucket_options:
                bucket_frame = apply_feature_bucket(
                    recommendation_filtered,
                    config,
                    bucket_value,
                )
                if bucket_frame.empty:
                    continue

                st.markdown(f"#### {feature} = {bucket_value}")
                st.caption(f"Rows: {len(bucket_frame):,}")
                bucket_plot_frame = _derived_diagnostic_frame(bucket_frame)

                first, second = st.columns(2)
                with first:
                    figure = _density_heatmap(
                        bucket_plot_frame,
                        x_column="analysis_bid_change",
                        y_column="expected_profit_lift",
                        x_label="Bid change: recommended - existing",
                        y_label="Probability-weighted expected-profit lift",
                        title="Bid Change vs Expected-Profit Lift",
                        **heatmap_args,
                    )
                    if figure is None:
                        st.info("Expected-profit lift is unavailable in this artifact.")
                    else:
                        st.plotly_chart(
                            figure,
                            use_container_width=True,
                            key=f"all_profit_{feature}_{bucket_value}",
                        )

                with second:
                    figure = _density_heatmap(
                        bucket_plot_frame,
                        x_column="analysis_bid_change",
                        y_column="recommended_bid_cm_if_won",
                        x_label="Bid change: recommended - existing",
                        y_label="Recommended CM if won",
                        title="Bid Change vs Recommended CM",
                        **heatmap_args,
                    )
                    if figure is None:
                        st.info("Recommended CM is unavailable in this artifact.")
                    else:
                        st.plotly_chart(
                            figure,
                            use_container_width=True,
                            key=f"all_cm_{feature}_{bucket_value}",
                        )

                st.divider()


if __name__ == "__main__":
    main()
