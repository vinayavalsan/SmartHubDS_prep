"""Training reports and artifacts for SmartHub model runs.

This module persists training reports, plots, evaluation outputs, comparison
datasets, metadata, and deterministic test-set identifiers.
"""

import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, PowerNorm
from sklearn.calibration import calibration_curve
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve

from smarthub.core.logging_utils import get_logger

logger = get_logger(__name__)

_DENSITY_MIN_COUNT = 10
_DENSITY_GAMMA = 0.25
_DENSITY_CMAP = LinearSegmentedColormap.from_list(
    "smarthub_inverse_electric",
    [
        "#ffffff",
        "#fffadc",
        "#e6c800",
        "#a05a00",
        "#780064",
        "#1e0064",
        "#000000",
    ],
)


def _density_colorbar_ticks(max_count):
    """Return readable count ticks for a density colorbar."""
    candidates = [
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
    ticks = [value for value in candidates if value <= max_count]
    if not ticks:
        return [max(_DENSITY_MIN_COUNT, int(round(max_count)))]
    if ticks[-1] < max_count and max_count / ticks[-1] >= 1.5:
        ticks.append(int(round(max_count)))
    return ticks


def _format_density_tick(count, total_rows):
    """Format one density tick as count and fraction of test leads."""
    fraction_pct = (float(count) / float(total_rows) * 100.0) if total_rows else 0.0
    return f"{int(round(count)):,} ({fraction_pct:.3f}%)"


def _limit_density_colorbar(colorbar, max_count):
    """Display only the retained density range while preserving 0 = white."""
    if max_count >= _DENSITY_MIN_COUNT:
        colorbar.ax.set_ylim(float(_DENSITY_MIN_COUNT), float(max_count))


def save_feature_summary_files(
    report_dir,
    feature_summary_df,
    feature_counts_df,
):
    """Write feature-summary tables to CSV files.

    Inputs
    ------
    report_dir : str | pathlib.Path
        Directory for report artifacts.
    feature_summary_df : pandas.DataFrame
        Feature-level summary dataframe.
    feature_counts_df : pandas.DataFrame
        Feature-value count dataframe.
    """
    report_dir = Path(report_dir)
    report_dir.mkdir(exist_ok=True)

    feature_summary_df.to_csv(report_dir / "feature_summary.csv", index=False)
    feature_counts_df.to_csv(report_dir / "feature_value_counts.csv", index=False)


def _plot_histogram(report_dir, df, column, filename, xlabel, title, bins=40):
    """Save a histogram for an available optimizer diagnostic.

    Inputs
    ------
    report_dir : str | pathlib.Path
        Directory for report artifacts.
    df : pandas.DataFrame
        Input dataframe.
    column : str
        Column to process.
    filename : str
        Output filename.
    xlabel : str
        Horizontal-axis label.
    title : str
        Plot title.
    bins : int
        Number of histogram bins.

    Returns
    -------
    pathlib.Path | None
        Saved plot path, or ``None`` when no valid values exist.
    """
    if column not in df.columns:
        return None
    values = pd.to_numeric(df[column], errors="coerce").dropna()
    if values.empty:
        return None

    plt.figure(figsize=(8, 6))
    plt.hist(values, bins=bins)
    plt.xlabel(xlabel)
    plt.ylabel("Number of rows")
    plt.title(title)
    plt.grid(True, alpha=0.25, linewidth=0.6)
    plt.tight_layout()
    path = Path(report_dir) / filename
    plt.savefig(path)
    plt.close()
    return path


def _save_calibration_by_bucket(report_dir, y_test, pred):
    """Plot observed win rate against mean predicted win rate by bucket."""
    actual = pd.to_numeric(pd.Series(y_test), errors="coerce").reset_index(drop=True)
    predicted = pd.to_numeric(pd.Series(pred), errors="coerce").reset_index(drop=True)

    if len(actual) != len(predicted):
        raise ValueError(
            "Calibration inputs must have the same number of rows: "
            f"y_test={len(actual):,}, pred={len(predicted):,}."
        )

    calibration_df = pd.DataFrame(
        {
            "actual": actual,
            "predicted": predicted,
        }
    ).dropna()
    if calibration_df.empty:
        return None

    bucket_edges = [value / 10 for value in range(11)]
    bucket_labels = [
        f"{bucket_edges[index]:.1f}-{bucket_edges[index + 1]:.1f}"
        for index in range(10)
    ]
    calibration_df["bucket"] = pd.cut(
        calibration_df["predicted"],
        bins=bucket_edges,
        labels=bucket_labels,
        include_lowest=True,
        right=False,
    )
    calibration_df.loc[
        calibration_df["predicted"] == 1.0,
        "bucket",
    ] = bucket_labels[-1]

    bucket_summary = (
        calibration_df.groupby("bucket", observed=False)
        .agg(
            predicted_win_rate=("predicted", "mean"),
            actual_win_rate=("actual", "mean"),
            row_count=("actual", "size"),
        )
        .reset_index()
    )
    bucket_summary = bucket_summary[bucket_summary["row_count"] > 0]
    if bucket_summary.empty:
        return None

    fig, calibration_ax = plt.subplots(figsize=(10, 7))
    count_ax = calibration_ax.twinx()

    count_ax.bar(
        bucket_summary["predicted_win_rate"],
        bucket_summary["row_count"],
        width=0.08,
        alpha=0.20,
        label="Rows in bucket",
    )

    calibration_ax.plot(
        bucket_summary["predicted_win_rate"],
        bucket_summary["actual_win_rate"],
        marker="o",
        linewidth=2,
        label="Observed win rate",
    )
    calibration_ax.plot(
        [0.0, 1.0],
        [0.0, 1.0],
        linestyle="--",
        linewidth=1,
        label="Perfect calibration",
    )

    calibration_ax.set_xlabel("Mean predicted win rate within bucket")
    calibration_ax.set_ylabel("Observed win rate within bucket")
    count_ax.set_ylabel("Number of leads in bucket")
    calibration_ax.set_title("Observed vs Predicted Win Rate by Probability Bucket")
    calibration_ax.set_xlim(0.0, 1.0)
    calibration_ax.set_ylim(0.0, 1.0)
    calibration_ax.grid(True, alpha=0.25, linewidth=0.6)
    calibration_ax.legend(loc="upper left")

    fig.tight_layout()
    path = Path(report_dir) / "calibration_by_bucket.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def _save_policy_profit_comparison(report_dir, optimizer_eval_df):
    """Compare policy revenue, bid spend, profit, and aggregate CM.

    Inputs
    ------
    report_dir : str | pathlib.Path
        Directory for report artifacts.
    optimizer_eval_df : pandas.DataFrame
        Row-level optimizer evaluation results.

    Returns
    -------
    pathlib.Path | None
        Saved plot path, or ``None`` when required columns are unavailable.
    """
    required = {
        "observed_policy_expected_revenue",
        "observed_policy_bid_cost",
        "observed_policy_expected_profit",
        "current_bid_predicted_win_rate",
        "recommended_bid_predicted_win_rate",
        "current_bid_expected_profit",
        "recommended_bid_expected_profit",
        "bid",
        "recommended_bid",
    }
    if not required.issubset(optimizer_eval_df.columns):
        return None

    revenue_col = None
    for candidate in ("expected_revenue", "revenue"):
        if candidate in optimizer_eval_df.columns:
            revenue_col = candidate
            break
    if revenue_col is None:
        return None

    current_win_rate = pd.to_numeric(
        optimizer_eval_df["current_bid_predicted_win_rate"], errors="coerce"
    )
    recommended_win_rate = pd.to_numeric(
        optimizer_eval_df["recommended_bid_predicted_win_rate"], errors="coerce"
    )
    revenue = pd.to_numeric(optimizer_eval_df[revenue_col], errors="coerce")
    current_bid = pd.to_numeric(optimizer_eval_df["bid"], errors="coerce")
    recommended_bid = pd.to_numeric(
        optimizer_eval_df["recommended_bid"], errors="coerce"
    )

    observed_revenue = float(
        pd.to_numeric(
            optimizer_eval_df["observed_policy_expected_revenue"], errors="coerce"
        ).sum()
    )
    observed_bid_spend = float(
        pd.to_numeric(
            optimizer_eval_df["observed_policy_bid_cost"], errors="coerce"
        ).sum()
    )
    observed_profit = float(
        pd.to_numeric(
            optimizer_eval_df["observed_policy_expected_profit"], errors="coerce"
        ).sum()
    )

    current_revenue = float((current_win_rate * revenue).sum())
    current_bid_spend = float((current_win_rate * current_bid).sum())
    current_profit = float(
        pd.to_numeric(
            optimizer_eval_df["current_bid_expected_profit"], errors="coerce"
        ).sum()
    )

    recommended_revenue = float((recommended_win_rate * revenue).sum())
    recommended_bid_spend = float((recommended_win_rate * recommended_bid).sum())
    recommended_profit = float(
        pd.to_numeric(
            optimizer_eval_df["recommended_bid_expected_profit"], errors="coerce"
        ).sum()
    )

    policy_values = [
        (observed_revenue, observed_bid_spend, observed_profit),
        (current_revenue, current_bid_spend, current_profit),
        (recommended_revenue, recommended_bid_spend, recommended_profit),
    ]
    cms = [
        observed_profit / observed_revenue if observed_revenue else float("nan"),
        current_profit / current_revenue if current_revenue else float("nan"),
        (
            recommended_profit / recommended_revenue
            if recommended_revenue
            else float("nan")
        ),
    ]

    labels = [
        "Existing policy\n(observed)",
        "ML @ existing bids\n(probability-weighted)",
        "ML optimized\n(probability-weighted)",
    ]

    revenue_values = [row[0] for row in policy_values]
    bid_spend_values = [row[1] for row in policy_values]
    profit_values = [row[2] for row in policy_values]

    x = np.arange(len(labels))
    width = 0.24

    fig, ax = plt.subplots(figsize=(11, 7))
    revenue_bars = ax.bar(x - width, revenue_values, width, label="Revenue")
    bid_spend_bars = ax.bar(x, bid_spend_values, width, label="Bid spend")
    profit_bars = ax.bar(x + width, profit_values, width, label="Profit")

    ax.set_ylabel("Total dollars")
    ax.set_title("Policy Economics: Revenue, Bid Spend, Profit, and CM")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.6)
    ax.legend()

    max_value = max(revenue_values + bid_spend_values + profit_values)
    ax.set_ylim(0.0, max_value * 1.22 if max_value > 0 else 1.0)

    for bars, values in (
        (revenue_bars, revenue_values),
        (bid_spend_bars, bid_spend_values),
        (profit_bars, profit_values),
    ):
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max_value * 0.01,
                f"${value:,.0f}",
                ha="center",
                va="bottom",
                rotation=90,
                fontsize=9,
            )

    for index, cm in enumerate(cms):
        if pd.notna(cm):
            ax.text(
                x[index],
                max_value * 1.11,
                f"CM = {cm:.1%}",
                ha="center",
                va="center",
                fontweight="bold",
            )

    fig.tight_layout()
    path = Path(report_dir) / "policy_profit_comparison.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def _save_policy_win_rate_comparison(report_dir, optimizer_eval_df):
    """Compare observed and model-predicted win rates.

    Inputs
    ------
    report_dir : str | pathlib.Path
        Directory for report artifacts.
    optimizer_eval_df : pandas.DataFrame
        Row-level optimizer evaluation results.

    Returns
    -------
    pathlib.Path | None
        Saved plot path, or ``None`` when required columns are unavailable.
    """
    required = {
        "current_bid_predicted_win_rate",
        "recommended_bid_predicted_win_rate",
    }
    if not required.issubset(optimizer_eval_df.columns):
        return None

    target_col = None
    for candidate in ("won_flag", "won"):
        if candidate in optimizer_eval_df.columns:
            target_col = candidate
            break
    if target_col is None:
        return None

    observed = pd.to_numeric(
        optimizer_eval_df[target_col],
        errors="coerce",
    ).mean()
    current = pd.to_numeric(
        optimizer_eval_df["current_bid_predicted_win_rate"],
        errors="coerce",
    ).mean()
    recommended = pd.to_numeric(
        optimizer_eval_df["recommended_bid_predicted_win_rate"],
        errors="coerce",
    ).mean()
    values = [float(observed), float(current), float(recommended)]
    labels = [
        "Existing policy\n(observed)",
        "ML @ existing bids\n(predicted)",
        "ML optimized\n(predicted)",
    ]

    plt.figure(figsize=(9, 6))
    bars = plt.bar(labels, values)
    plt.ylabel("Win rate")
    plt.title("Existing Policy vs ML Win Rate")
    plt.grid(True, axis="y", alpha=0.25, linewidth=0.6)
    plt.ylim(0.0, max(values) * 1.2 if max(values) > 0 else 1.0)
    for bar, value in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.4f}",
            ha="center",
            va="bottom",
        )
    plt.tight_layout()
    path = Path(report_dir) / "policy_win_rate_comparison.png"
    plt.savefig(path)
    plt.close()
    return path


def _save_expected_profit_by_max_win_probability(report_dir, optimizer_eval_df):
    """Plot probability-weighted expected profit by win-rate threshold.

    Inputs
    ------
    report_dir : str | pathlib.Path
        Directory for report artifacts.
    optimizer_eval_df : pandas.DataFrame
        Row-level optimizer evaluation results.

    Returns
    -------
    pathlib.Path | None
        Saved plot path, or ``None`` when required columns are unavailable.
    """
    required = {
        "recommended_bid_predicted_win_rate",
        "recommended_bid_expected_profit",
    }
    if not required.issubset(optimizer_eval_df.columns):
        return None

    win_rate = pd.to_numeric(
        optimizer_eval_df["recommended_bid_predicted_win_rate"],
        errors="coerce",
    )
    expected_profit = pd.to_numeric(
        optimizer_eval_df["recommended_bid_expected_profit"],
        errors="coerce",
    )
    valid = win_rate.notna() & expected_profit.notna()
    if not valid.any():
        return None

    win_rate = win_rate[valid]
    expected_profit = expected_profit[valid]
    total_profit = float(expected_profit.sum())
    if total_profit == 0.0:
        return None

    thresholds = [value / 100 for value in range(0, 101)]
    cumulative_profit = [
        float(expected_profit[win_rate <= threshold].sum()) for threshold in thresholds
    ]
    cumulative_fraction = [value / total_profit for value in cumulative_profit]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(thresholds, cumulative_fraction)

    ax.set_xlabel("Maximum predicted win rate")
    ax.set_ylabel("Fraction of total probability-weighted expected profit")
    ax.set_title("Probability-Weighted Expected Profit Below Win-Rate Threshold")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, alpha=0.25, linewidth=0.6)

    fig.tight_layout()
    path = Path(report_dir) / "expected_profit_by_max_win_probability.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def _save_bid_change_vs_win_probability_change(
    report_dir,
    optimizer_eval_df,
):
    """Plot 2D count-density heatmaps for bid and win-probability changes.

    Historical wins and losses are shown in separate panels with shared axes
    and one shared power-normalized count scale. Rectangular cells show the
    number of test leads in each region.
    """
    required = {
        "bid",
        "recommended_bid",
        "current_bid_predicted_win_rate",
        "recommended_bid_predicted_win_rate",
    }
    if not required.issubset(optimizer_eval_df.columns):
        return None

    target_col = None
    for candidate in ("won_flag", "won"):
        if candidate in optimizer_eval_df.columns:
            target_col = candidate
            break
    if target_col is None:
        return None

    total_test_leads = len(optimizer_eval_df)

    frame = pd.DataFrame(
        {
            "bid_change": (
                pd.to_numeric(
                    optimizer_eval_df["recommended_bid"],
                    errors="coerce",
                )
                - pd.to_numeric(optimizer_eval_df["bid"], errors="coerce")
            ),
            "win_probability_change": (
                pd.to_numeric(
                    optimizer_eval_df["recommended_bid_predicted_win_rate"],
                    errors="coerce",
                )
                - pd.to_numeric(
                    optimizer_eval_df["current_bid_predicted_win_rate"],
                    errors="coerce",
                )
            ),
            "observed_outcome": pd.to_numeric(
                optimizer_eval_df[target_col],
                errors="coerce",
            ),
        }
    ).dropna()

    if frame.empty:
        return None

    won = frame[frame["observed_outcome"] == 1]
    lost = frame[frame["observed_outcome"] == 0]
    if won.empty and lost.empty:
        return None

    x_min = float(frame["bid_change"].min())
    x_max = float(frame["bid_change"].max())
    y_min = float(frame["win_probability_change"].min())
    y_max = float(frame["win_probability_change"].max())

    x_pad = max((x_max - x_min) * 0.03, 0.1)
    y_pad = max((y_max - y_min) * 0.03, 0.01)
    x_range = (x_min - x_pad, x_max + x_pad)
    y_range = (y_min - y_pad, y_max + y_pad)

    bins = 50

    max_count = float(_DENSITY_MIN_COUNT)
    histograms = {}
    for name, subset in (("won", won), ("lost", lost)):
        if subset.empty:
            continue

        counts, x_edges, y_edges = np.histogram2d(
            subset["bid_change"],
            subset["win_probability_change"],
            bins=bins,
            range=(x_range, y_range),
        )
        counts[counts < _DENSITY_MIN_COUNT] = np.nan
        counts[counts <= 0] = np.nan
        histograms[name] = (counts, x_edges, y_edges)

        finite_counts = counts[np.isfinite(counts)]
        if finite_counts.size:
            max_count = max(max_count, float(finite_counts.max()))

    norm = PowerNorm(
        gamma=_DENSITY_GAMMA,
        vmin=1.0,
        vmax=max_count,
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(15, 6),
        sharex=True,
        sharey=True,
    )

    panels = (
        (axes[0], "won", won, "Existing Bid Won"),
        (axes[1], "lost", lost, "Existing Bid Lost"),
    )

    heatmap = None
    for ax, key, subset, title in panels:
        if key in histograms:
            counts, x_edges, y_edges = histograms[key]
            heatmap = ax.pcolormesh(
                x_edges,
                y_edges,
                counts.T,
                cmap=_DENSITY_CMAP,
                norm=norm,
                shading="auto",
            )

        ax.axvline(0.0, linestyle="--", linewidth=1)
        ax.axhline(0.0, linestyle="--", linewidth=1)
        ax.set_title(f"{title} (n={len(subset):,})")
        ax.set_xlabel("Bid change: ML recommended - existing bid")
        ax.set_xlim(*x_range)
        ax.set_ylim(*y_range)
        ax.grid(True, alpha=0.25, linewidth=0.6)

    axes[0].set_ylabel(
        "Predicted win-probability change: ML recommended - existing bid"
    )

    if heatmap is not None:
        fig.subplots_adjust(
            left=0.07,
            right=0.86,
            bottom=0.12,
            top=0.88,
            wspace=0.06,
        )
        colorbar_ax = fig.add_axes([0.89, 0.14, 0.020, 0.70])
        colorbar = fig.colorbar(heatmap, cax=colorbar_ax)
        ticks = _density_colorbar_ticks(max_count)
        colorbar.set_ticks(ticks)
        colorbar.set_ticklabels(
            [_format_density_tick(value, total_test_leads) for value in ticks]
        )
        _limit_density_colorbar(colorbar, max_count)
        colorbar.set_label("Lead density: count (% of test leads)")

    fig.suptitle("Bid Change vs Predicted Win-Probability Change - 2D Count Heatmap")
    path = Path(report_dir) / "bid_change_vs_win_probability_change.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _save_bid_distribution_by_outcome(report_dir, optimizer_eval_df):
    """Compare existing and recommended bid distributions by historical outcome.

    Historical wins and losses are shown in separate panels. Each histogram is
    normalized within its own outcome group so the y-axis represents the
    fraction of leads in that group. Both panels use identical bid bins and
    x-axis limits for direct comparison.

    Inputs
    ------
    report_dir : str | pathlib.Path
        Directory for report artifacts.
    optimizer_eval_df : pandas.DataFrame
        Row-level optimizer evaluation results.

    Returns
    -------
    pathlib.Path | None
        Saved plot path, or ``None`` when required columns are unavailable.
    """
    required = {"bid", "recommended_bid"}
    if not required.issubset(optimizer_eval_df.columns):
        return None

    target_col = None
    for candidate in ("won_flag", "won"):
        if candidate in optimizer_eval_df.columns:
            target_col = candidate
            break
    if target_col is None:
        return None

    frame = pd.DataFrame(
        {
            "existing_bid": pd.to_numeric(
                optimizer_eval_df["bid"],
                errors="coerce",
            ),
            "recommended_bid": pd.to_numeric(
                optimizer_eval_df["recommended_bid"],
                errors="coerce",
            ),
            "observed_outcome": pd.to_numeric(
                optimizer_eval_df[target_col],
                errors="coerce",
            ),
        }
    ).dropna()

    frame = frame[frame["observed_outcome"].isin([0, 1])].copy()
    if frame.empty:
        return None

    all_bids = pd.concat(
        [frame["existing_bid"], frame["recommended_bid"]],
        ignore_index=True,
    )
    bid_min = float(all_bids.min())
    bid_max = float(all_bids.max())
    if bid_min == bid_max:
        bid_min -= 0.5
        bid_max += 0.5

    bins = np.linspace(bid_min, bid_max, 41)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14, 6),
        sharex=True,
        sharey=True,
    )

    panels = (
        (axes[0], 1, "Historical Bid Won"),
        (axes[1], 0, "Historical Bid Lost"),
    )

    for ax, outcome, title in panels:
        subset = frame[frame["observed_outcome"] == outcome]
        if subset.empty:
            ax.set_title(f"{title} (n=0)")
            ax.set_xlabel("Bid")
            continue

        existing = subset["existing_bid"].to_numpy(dtype=float)
        recommended = subset["recommended_bid"].to_numpy(dtype=float)

        existing_weights = np.full(
            len(existing),
            1.0 / len(existing),
            dtype=float,
        )
        recommended_weights = np.full(
            len(recommended),
            1.0 / len(recommended),
            dtype=float,
        )

        ax.hist(
            existing,
            bins=bins,
            weights=existing_weights,
            alpha=0.55,
            label="Existing bid",
        )
        ax.hist(
            recommended,
            bins=bins,
            weights=recommended_weights,
            alpha=0.55,
            label="ML recommended bid",
        )

        existing_median = float(np.median(existing))
        recommended_median = float(np.median(recommended))
        ax.axvline(
            existing_median,
            linestyle="--",
            linewidth=1.5,
            label=f"Existing median = ${existing_median:.2f}",
        )
        ax.axvline(
            recommended_median,
            linestyle=":",
            linewidth=1.5,
            label=f"ML median = ${recommended_median:.2f}",
        )

        ax.set_title(f"{title} (n={len(subset):,})")
        ax.set_xlabel("Bid")
        ax.set_xlim(bid_min, bid_max)
        ax.grid(True, alpha=0.25, linewidth=0.6)
        ax.legend()

    axes[0].set_ylabel("Fraction of leads within historical outcome")
    fig.suptitle("Existing vs ML Recommended Bid Distribution by Historical Outcome")
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    path = Path(report_dir) / "bid_distribution_by_outcome.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _optimizer_diagnostic_frame(
    optimizer_eval_df,
    target_cm,
    min_bid,
    bid_step,
):
    """Build numeric optimizer diagnostics, including per-row bid ceiling."""
    required = {"bid", "recommended_bid", "expected_profit_lift"}
    if not required.issubset(optimizer_eval_df.columns):
        return pd.DataFrame()

    revenue_col = next(
        (
            name
            for name in ("expected_revenue", "revenue")
            if name in optimizer_eval_df.columns
        ),
        None,
    )
    if revenue_col is None or target_cm is None or min_bid is None or bid_step is None:
        return pd.DataFrame()

    frame = pd.DataFrame(
        {
            "existing_bid": pd.to_numeric(optimizer_eval_df["bid"], errors="coerce"),
            "recommended_bid": pd.to_numeric(
                optimizer_eval_df["recommended_bid"], errors="coerce"
            ),
            "expected_profit_lift": pd.to_numeric(
                optimizer_eval_df["expected_profit_lift"],
                errors="coerce",
            ),
            "expected_revenue": pd.to_numeric(
                optimizer_eval_df[revenue_col], errors="coerce"
            ),
        }
    )

    for source, destination in (
        ("current_bid_predicted_win_rate", "current_predicted_win_rate"),
        (
            "recommended_bid_predicted_win_rate",
            "recommended_predicted_win_rate",
        ),
    ):
        if source in optimizer_eval_df.columns:
            frame[destination] = pd.to_numeric(
                optimizer_eval_df[source], errors="coerce"
            )

    outcome_col = next(
        (name for name in ("won_flag", "won") if name in optimizer_eval_df.columns),
        None,
    )
    if outcome_col is not None:
        raw_outcome = optimizer_eval_df[outcome_col]
        outcome = pd.to_numeric(raw_outcome, errors="coerce")
        if outcome.isna().any():
            normalized = raw_outcome.astype("string").str.strip().str.lower()
            outcome = outcome.fillna(normalized.map({"true": 1.0, "false": 0.0}))
        frame["observed_outcome"] = outcome

    frame = frame.dropna(
        subset=[
            "existing_bid",
            "recommended_bid",
            "expected_profit_lift",
            "expected_revenue",
        ]
    )
    if frame.empty:
        return frame

    step = float(bid_step)
    minimum = float(min_bid)
    cm_ceiling = frame["expected_revenue"] * (1.0 - float(target_cm))
    frame["maximum_candidate_bid"] = (
        np.floor((cm_ceiling - minimum) / step + 1e-12) * step + minimum
    )
    frame["maximum_candidate_bid"] = frame["maximum_candidate_bid"].clip(lower=minimum)
    frame["bid_change"] = frame["recommended_bid"] - frame["existing_bid"]
    return frame


def log_optimizer_recommendation_diagnostics(
    optimizer_eval_df,
    target_cm,
    min_bid,
    bid_step,
):
    """Log optimizer boundary usage and large bid movements."""
    frame = _optimizer_diagnostic_frame(optimizer_eval_df, target_cm, min_bid, bid_step)
    if frame.empty:
        return

    tolerance = max(float(bid_step) * 1e-6, 1e-9)
    at_min = np.isclose(
        frame["recommended_bid"],
        float(min_bid),
        atol=tolerance,
        rtol=0.0,
    )
    at_max = np.isclose(
        frame["recommended_bid"],
        frame["maximum_candidate_bid"],
        atol=tolerance,
        rtol=0.0,
    )
    interior = ~(at_min | at_max)
    large_increase = frame["bid_change"] >= 10.0
    large_decrease = frame["bid_change"] <= -10.0
    total = len(frame)

    logger.info("Optimizer Recommendation Diagnostics")
    for label, mask in (
        ("At minimum bid", at_min),
        ("At maximum candidate bid", at_max),
        ("Interior recommendation", interior),
        ("Large bid increase >= $10", large_increase),
        ("Large bid decrease <= -$10", large_decrease),
    ):
        count = int(mask.sum())
        logger.info(
            "  %-38s: %s (%.2f%%)",
            label,
            f"{count:,}",
            count / total * 100.0,
        )

    boundary = frame.loc[at_max]
    if boundary.empty:
        return

    logger.info("  Maximum-Candidate Bid Rows")
    logger.info(
        "    %-36s: $%.2f",
        "Average existing bid",
        boundary["existing_bid"].mean(),
    )
    logger.info(
        "    %-36s: $%.2f",
        "Average recommended bid",
        boundary["recommended_bid"].mean(),
    )
    logger.info(
        "    %-36s: $%.2f",
        "Average expected revenue",
        boundary["expected_revenue"].mean(),
    )
    logger.info(
        "    %-36s: $%.2f",
        "Average expected profit lift",
        boundary["expected_profit_lift"].mean(),
    )

    for column, label in (
        ("current_predicted_win_rate", "Average current predicted win rate"),
        (
            "recommended_predicted_win_rate",
            "Average recommended predicted win rate",
        ),
        ("observed_outcome", "Historical win rate"),
    ):
        if column in boundary.columns:
            values = boundary[column].dropna()
            if not values.empty:
                logger.info(
                    "    %-36s: %.4f",
                    label,
                    values.mean(),
                )


def _save_optimizer_boundary_diagnostics(
    report_dir, optimizer_eval_df, target_cm, min_bid, bid_step
):
    """Plot the share of recommendations at the lower/upper bid boundary."""
    frame = _optimizer_diagnostic_frame(optimizer_eval_df, target_cm, min_bid, bid_step)
    if frame.empty:
        return None
    tolerance = max(float(bid_step) * 1e-6, 1e-9)
    at_min = np.isclose(
        frame["recommended_bid"], float(min_bid), atol=tolerance, rtol=0.0
    )
    at_max = np.isclose(
        frame["recommended_bid"],
        frame["maximum_candidate_bid"],
        atol=tolerance,
        rtol=0.0,
    )
    counts = [int(at_min.sum()), int((~(at_min | at_max)).sum()), int(at_max.sum())]
    labels = ["Minimum bid", "Interior", "Maximum candidate bid"]
    total = len(frame)
    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.bar(labels, counts)
    ax.set_ylabel("Number of rows")
    ax.set_title("Optimizer Recommendation Boundary Diagnostics")
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.6)
    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{count:,} ({count / total:.1%})",
            ha="center",
            va="bottom",
        )
    fig.tight_layout()
    path = Path(report_dir) / "optimizer_boundary_diagnostics.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def _save_existing_vs_recommended_bid_heatmap(report_dir, optimizer_eval_df):
    """Plot existing bid against ML recommended bid as a 2D count heatmap."""
    if not {"bid", "recommended_bid"}.issubset(optimizer_eval_df.columns):
        return None
    x = pd.to_numeric(optimizer_eval_df["bid"], errors="coerce")
    y = pd.to_numeric(optimizer_eval_df["recommended_bid"], errors="coerce")
    valid = x.notna() & y.notna()
    if not valid.any():
        return None
    x, y = x[valid], y[valid]
    low, high = float(min(x.min(), y.min())), float(max(x.max(), y.max()))
    if low == high:
        low, high = low - 0.5, high + 0.5
    counts, x_edges, y_edges = np.histogram2d(
        x, y, bins=60, range=((low, high), (low, high))
    )
    counts[counts < _DENSITY_MIN_COUNT] = np.nan
    counts[counts <= 0] = np.nan
    finite = counts[np.isfinite(counts)]
    max_count = (
        max(float(_DENSITY_MIN_COUNT), float(finite.max()))
        if finite.size
        else float(_DENSITY_MIN_COUNT)
    )
    fig, ax = plt.subplots(figsize=(9, 7))
    heatmap = ax.pcolormesh(
        x_edges,
        y_edges,
        counts.T,
        cmap=_DENSITY_CMAP,
        norm=PowerNorm(gamma=_DENSITY_GAMMA, vmin=1.0, vmax=max_count),
        shading="auto",
    )
    ax.plot([low, high], [low, high], linestyle="--")
    ax.set_xlabel("Existing bid")
    ax.set_ylabel("ML recommended bid")
    ax.set_title("Existing Bid vs ML Recommended Bid - 2D Count Heatmap")
    ax.grid(True, alpha=0.25, linewidth=0.6)
    colorbar = fig.colorbar(heatmap, ax=ax)
    ticks = _density_colorbar_ticks(max_count)
    colorbar.set_ticks(ticks)
    colorbar.set_ticklabels(
        [_format_density_tick(v, len(optimizer_eval_df)) for v in ticks]
    )
    _limit_density_colorbar(colorbar, max_count)
    colorbar.set_label("Lead density: count (% of test leads)")
    fig.tight_layout()
    path = Path(report_dir) / "existing_vs_recommended_bid_heatmap.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _save_bid_change_vs_expected_profit_lift_heatmap(report_dir, optimizer_eval_df):
    """Plot bid change against probability-weighted expected-profit lift."""
    if not {"bid", "recommended_bid", "expected_profit_lift"}.issubset(
        optimizer_eval_df.columns
    ):
        return None
    x = pd.to_numeric(
        optimizer_eval_df["recommended_bid"], errors="coerce"
    ) - pd.to_numeric(optimizer_eval_df["bid"], errors="coerce")
    y = pd.to_numeric(optimizer_eval_df["expected_profit_lift"], errors="coerce")
    valid = x.notna() & y.notna()
    if not valid.any():
        return None
    x, y = x[valid], y[valid]
    counts, x_edges, y_edges = np.histogram2d(x, y, bins=60)
    counts[counts < _DENSITY_MIN_COUNT] = np.nan
    counts[counts <= 0] = np.nan
    finite = counts[np.isfinite(counts)]
    max_count = (
        max(float(_DENSITY_MIN_COUNT), float(finite.max()))
        if finite.size
        else float(_DENSITY_MIN_COUNT)
    )
    fig, ax = plt.subplots(figsize=(9, 7))
    heatmap = ax.pcolormesh(
        x_edges,
        y_edges,
        counts.T,
        cmap=_DENSITY_CMAP,
        norm=PowerNorm(gamma=_DENSITY_GAMMA, vmin=1.0, vmax=max_count),
        shading="auto",
    )
    ax.axvline(0.0, linestyle="--", linewidth=1)
    ax.axhline(0.0, linestyle="--", linewidth=1)
    ax.set_xlabel("Bid change: ML recommended - existing bid")
    ax.set_ylabel("Probability-weighted expected profit lift")
    ax.set_title("Bid Change vs Expected Profit Lift - 2D Count Heatmap")
    ax.grid(True, alpha=0.25, linewidth=0.6)
    colorbar = fig.colorbar(heatmap, ax=ax)
    ticks = _density_colorbar_ticks(max_count)
    colorbar.set_ticks(ticks)
    colorbar.set_ticklabels(
        [_format_density_tick(v, len(optimizer_eval_df)) for v in ticks]
    )
    _limit_density_colorbar(colorbar, max_count)
    colorbar.set_label("Lead density: count (% of test leads)")
    fig.tight_layout()
    path = Path(report_dir) / "bid_change_vs_expected_profit_lift_heatmap.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _save_bid_change_vs_recommended_cm_heatmap(
    report_dir, optimizer_eval_df, target_cm
):
    """Plot bid change against recommended CM, with the target CM marked."""
    if target_cm is None:
        return None
    if not {"bid", "recommended_bid", "recommended_bid_cm_if_won"}.issubset(
        optimizer_eval_df.columns
    ):
        return None
    x = pd.to_numeric(
        optimizer_eval_df["recommended_bid"], errors="coerce"
    ) - pd.to_numeric(optimizer_eval_df["bid"], errors="coerce")
    y = pd.to_numeric(optimizer_eval_df["recommended_bid_cm_if_won"], errors="coerce")
    valid = x.notna() & y.notna()
    if not valid.any():
        return None
    x, y = x[valid], y[valid]
    counts, x_edges, y_edges = np.histogram2d(x, y, bins=60)
    counts[counts < _DENSITY_MIN_COUNT] = np.nan
    counts[counts <= 0] = np.nan
    finite = counts[np.isfinite(counts)]
    max_count = (
        max(float(_DENSITY_MIN_COUNT), float(finite.max()))
        if finite.size
        else float(_DENSITY_MIN_COUNT)
    )
    fig, ax = plt.subplots(figsize=(9, 7))
    heatmap = ax.pcolormesh(
        x_edges,
        y_edges,
        counts.T,
        cmap=_DENSITY_CMAP,
        norm=PowerNorm(gamma=_DENSITY_GAMMA, vmin=1.0, vmax=max_count),
        shading="auto",
    )
    ax.axhline(
        float(target_cm),
        linestyle="--",
        linewidth=1,
        label=f"Target CM = {float(target_cm):.1%}",
    )
    ax.set_xlabel("Bid change: ML recommended - existing bid")
    ax.set_ylabel("Recommended CM if won")
    ax.set_title("Bid Change vs Recommended CM - 2D Count Heatmap")
    ax.legend()
    ax.grid(True, alpha=0.25, linewidth=0.6)
    colorbar = fig.colorbar(heatmap, ax=ax)
    ticks = _density_colorbar_ticks(max_count)
    colorbar.set_ticks(ticks)
    colorbar.set_ticklabels(
        [_format_density_tick(v, len(optimizer_eval_df)) for v in ticks]
    )
    _limit_density_colorbar(colorbar, max_count)
    colorbar.set_label("Lead density: count (% of test leads)")
    fig.tight_layout()
    path = Path(report_dir) / "bid_change_vs_recommended_cm_heatmap.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _save_optimizer_plots(
    report_dir, optimizer_eval_df, target_cm=None, min_bid=None, bid_step=None
):
    """Write the retained optimizer diagnostic plots.

    Inputs
    ------
    report_dir : str | pathlib.Path
        Directory for report artifacts.
    optimizer_eval_df : pandas.DataFrame | None
        Row-level optimizer evaluation results.

    Returns
    -------
    list[str]
        Generated optimizer plot filenames.
    """
    files = []
    specs = [
        (
            "expected_profit_lift",
            "optimizer_expected_profit_lift.png",
            "Probability-weighted expected profit lift: recommended - current",
            "Probability-Weighted Expected Profit Lift from Recommended Bid",
        ),
        (
            "n_candidate_bids",
            "candidate_bid_count_distribution.png",
            "Number of candidate bids evaluated",
            "Candidate Bid Count Distribution",
        ),
        (
            "bid_change",
            "recommended_bid_change.png",
            "Recommended bid - current bid",
            "Recommended Bid Change from Current Bid",
        ),
        (
            "recommended_bid_cm_if_won",
            "recommended_cm_distribution.png",
            "Recommended CM if won",
            "Recommended CM Distribution",
        ),
    ]
    for column, filename, xlabel, title in specs:
        path = _plot_histogram(
            report_dir,
            optimizer_eval_df,
            column,
            filename,
            xlabel,
            title,
        )
        if path is not None:
            files.append(filename)

    for plot_builder in (
        _save_policy_profit_comparison,
        _save_policy_win_rate_comparison,
        _save_expected_profit_by_max_win_probability,
        _save_bid_change_vs_win_probability_change,
        _save_bid_distribution_by_outcome,
        _save_existing_vs_recommended_bid_heatmap,
        _save_bid_change_vs_expected_profit_lift_heatmap,
    ):
        path = plot_builder(report_dir, optimizer_eval_df)
        if path is not None:
            files.append(path.name)

    for path in (
        _save_optimizer_boundary_diagnostics(
            report_dir, optimizer_eval_df, target_cm, min_bid, bid_step
        ),
        _save_bid_change_vs_recommended_cm_heatmap(
            report_dir, optimizer_eval_df, target_cm
        ),
    ):
        if path is not None:
            files.append(path.name)

    log_optimizer_recommendation_diagnostics(
        optimizer_eval_df, target_cm, min_bid, bid_step
    )

    if {
        "current_bid_predicted_win_rate",
        "recommended_bid_predicted_win_rate",
    }.issubset(optimizer_eval_df.columns):
        current_win_rate = pd.to_numeric(
            optimizer_eval_df["current_bid_predicted_win_rate"],
            errors="coerce",
        )
        recommended_win_rate = pd.to_numeric(
            optimizer_eval_df["recommended_bid_predicted_win_rate"],
            errors="coerce",
        )
        valid = current_win_rate.notna() & recommended_win_rate.notna()

        if valid.any():
            total_test_leads = len(optimizer_eval_df)
            bins = 60

            counts, x_edges, y_edges = np.histogram2d(
                current_win_rate[valid],
                recommended_win_rate[valid],
                bins=bins,
                range=((0.0, 1.0), (0.0, 1.0)),
            )
            counts[counts < _DENSITY_MIN_COUNT] = np.nan
            counts[counts <= 0] = np.nan

            finite_counts = counts[np.isfinite(counts)]
            max_count = (
                max(float(_DENSITY_MIN_COUNT), float(finite_counts.max()))
                if finite_counts.size
                else float(_DENSITY_MIN_COUNT)
            )

            fig, ax = plt.subplots(figsize=(9, 6))
            heatmap = ax.pcolormesh(
                x_edges,
                y_edges,
                counts.T,
                cmap=_DENSITY_CMAP,
                norm=PowerNorm(
                    gamma=_DENSITY_GAMMA,
                    vmin=1.0,
                    vmax=max_count,
                ),
                shading="auto",
            )

            ax.plot([0, 1], [0, 1], linestyle="--")
            ax.set_xlabel("Predicted win rate using current bid")
            ax.set_ylabel("Predicted win rate using recommended bid")
            ax.set_title("Current vs Recommended Predicted Win Rate - 2D Count Heatmap")
            ax.set_xlim(0.0, 1.0)
            ax.set_ylim(0.0, 1.0)
            ax.grid(True, alpha=0.25, linewidth=0.6)

            fig.subplots_adjust(
                left=0.10,
                right=0.82,
                bottom=0.14,
                top=0.88,
            )
            colorbar_ax = fig.add_axes([0.85, 0.14, 0.025, 0.70])
            colorbar = fig.colorbar(heatmap, cax=colorbar_ax)
            ticks = _density_colorbar_ticks(max_count)
            colorbar.set_ticks(ticks)
            colorbar.set_ticklabels(
                [_format_density_tick(value, total_test_leads) for value in ticks]
            )
            _limit_density_colorbar(colorbar, max_count)
            colorbar.set_label("Lead density: count (% of test leads)")

            fig.savefig(
                Path(report_dir) / "current_vs_recommended_win_rate.png",
                bbox_inches="tight",
            )
            plt.close(fig)
            files.append("current_vs_recommended_win_rate.png")

    return files


def save_performance_plots(
    report_dir,
    y_test,
    pred,
    pred_class,
    roc_auc,
    pr_auc,
    accuracy,
    precision,
    recall,
    f1,
    f2,
    optimizer_eval_df=None,
    target_cm=None,
    min_bid=None,
    bid_step=None,
):
    """Write classifier and optimizer diagnostic plots.

    Inputs
    ------
    report_dir : str | pathlib.Path
        Directory for report artifacts.
    y_test : pandas.Series | numpy.ndarray
        Held-out target values.
    pred : numpy.ndarray
        Predicted positive-class probabilities.
    pred_class : numpy.ndarray
        Predicted binary classes.
    roc_auc : float
        ROC AUC displayed in the report.
    pr_auc : float
        PR AUC displayed in the report.
    accuracy : float
        Classification accuracy.
    precision : float
        Classification precision.
    recall : float
        Classification recall.
    f1 : float
        F1 score.
    f2 : float
        F2 score, weighting recall more heavily than precision.
    optimizer_eval_df : pandas.DataFrame | None
        Row-level optimizer evaluation results.
    """
    report_dir = Path(report_dir)
    report_dir.mkdir(exist_ok=True)

    fpr, tpr, _ = roc_curve(y_test, pred)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.grid(True, alpha=0.25, linewidth=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(report_dir / "roc_curve.png")
    plt.close()

    precision_curve, recall_curve, _ = precision_recall_curve(y_test, pred)

    plt.figure(figsize=(8, 6))
    plt.plot(recall_curve, precision_curve, label=f"PR AUC = {pr_auc:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.grid(True, alpha=0.25, linewidth=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(report_dir / "precision_recall_curve.png")
    plt.close()

    prob_true, prob_pred = calibration_curve(
        y_test,
        pred,
        n_bins=10,
        strategy="quantile",
    )

    plt.figure(figsize=(8, 6))
    plt.plot(prob_pred, prob_true, marker="o", label="Model")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")
    plt.xlabel("Average predicted win probability")
    plt.ylabel("Actual win rate")
    plt.title("Calibration Curve")
    plt.grid(True, alpha=0.25, linewidth=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(report_dir / "calibration_curve.png")
    plt.close()

    _save_calibration_by_bucket(
        report_dir=report_dir,
        y_test=y_test,
        pred=pred,
    )

    plt.figure(figsize=(8, 6))
    plt.hist(pred[y_test == 0], bins=30, alpha=0.6, label="Lost")
    plt.hist(pred[y_test == 1], bins=30, alpha=0.6, label="Won")
    plt.xlabel("Predicted win probability")
    plt.ylabel("Number of rows")
    plt.title("Predicted Probability Distribution")
    plt.grid(True, alpha=0.25, linewidth=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(report_dir / "probability_histogram.png")
    plt.close()

    cm = confusion_matrix(y_test, pred_class)

    labels = [["TN", "FP"], ["FN", "TP"]]
    total = cm.sum()

    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) else 0
    negative_predictive_value = tn / (tn + fn) if (tn + fn) else 0

    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(cm)

    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Lost", "Won"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Lost", "Won"])

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            count = cm[i, j]
            pct = count / total * 100 if total else 0
            text_color = "white" if count < cm.max() / 2 else "black"

            ax.text(
                j,
                i,
                f"{labels[i][j]}\n{count:,}\n{pct:.1f}%",
                ha="center",
                va="center",
                color=text_color,
                fontsize=14,
                fontweight="bold",
            )

    fig.colorbar(im, ax=ax)

    metrics_text = (
        f"Accuracy: {accuracy:.1%}    "
        f"Precision: {precision:.1%}    "
        f"Recall: {recall:.1%}\n"
        f"F1 Score: {f1:.1%}    F2 Score: {f2:.1%}    "
        f"Specificity: {specificity:.1%}    "
        f"NPV: {negative_predictive_value:.1%}"
    )

    fig.text(
        0.5,
        0.02,
        metrics_text,
        ha="center",
        va="bottom",
        fontsize=11,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
    )

    plt.tight_layout(rect=[0, 0.10, 1, 1])
    plt.savefig(report_dir / "confusion_matrix.png")
    plt.close()

    if optimizer_eval_df is not None:
        _save_optimizer_plots(
            report_dir, optimizer_eval_df, target_cm, min_bid, bid_step
        )


def build_optimizer_scenario_summary(optimizer_eval_df):
    """Summarize historical outcome versus ML bid-direction scenarios.

    The four directional scenarios quantify whether ML recommends a higher or
    lower bid for historically won and lost leads. Rows whose recommended bid
    is unchanged are reported separately so they are not forced into a
    directional scenario.

    Inputs
    ------
    optimizer_eval_df : pandas.DataFrame
        Row-level optimizer evaluation results.

    Returns
    -------
    pandas.DataFrame
        Scenario-level counts, shares, and bid / predicted-win-rate changes.
    """
    required = {
        "bid",
        "recommended_bid",
        "current_bid_predicted_win_rate",
        "recommended_bid_predicted_win_rate",
    }
    if not required.issubset(optimizer_eval_df.columns):
        return pd.DataFrame()

    target_col = None
    for candidate in ("won_flag", "won"):
        if candidate in optimizer_eval_df.columns:
            target_col = candidate
            break
    if target_col is None:
        return pd.DataFrame()

    raw_outcome = optimizer_eval_df[target_col]
    numeric_outcome = pd.to_numeric(raw_outcome, errors="coerce")
    if numeric_outcome.isna().any():
        normalized = raw_outcome.astype("string").str.strip().str.lower()
        mapped = normalized.map({"true": 1.0, "false": 0.0})
        numeric_outcome = numeric_outcome.fillna(mapped)

    frame = pd.DataFrame(
        {
            "observed_outcome": numeric_outcome,
            "bid_change": (
                pd.to_numeric(optimizer_eval_df["recommended_bid"], errors="coerce")
                - pd.to_numeric(optimizer_eval_df["bid"], errors="coerce")
            ),
            "win_probability_change": (
                pd.to_numeric(
                    optimizer_eval_df["recommended_bid_predicted_win_rate"],
                    errors="coerce",
                )
                - pd.to_numeric(
                    optimizer_eval_df["current_bid_predicted_win_rate"],
                    errors="coerce",
                )
            ),
        }
    ).dropna()
    frame = frame[frame["observed_outcome"].isin([0, 1])].copy()
    if frame.empty:
        return pd.DataFrame()

    total_rows = len(frame)
    won_rows = int((frame["observed_outcome"] == 1).sum())
    lost_rows = int((frame["observed_outcome"] == 0).sum())
    tolerance = 1e-12

    scenario_specs = [
        (
            "Won - ML bid higher",
            (frame["observed_outcome"] == 1) & (frame["bid_change"] > tolerance),
            won_rows,
        ),
        (
            "Won - ML bid lower",
            (frame["observed_outcome"] == 1) & (frame["bid_change"] < -tolerance),
            won_rows,
        ),
        (
            "Lost - ML bid higher",
            (frame["observed_outcome"] == 0) & (frame["bid_change"] > tolerance),
            lost_rows,
        ),
        (
            "Lost - ML bid lower",
            (frame["observed_outcome"] == 0) & (frame["bid_change"] < -tolerance),
            lost_rows,
        ),
        (
            "Bid unchanged",
            frame["bid_change"].abs() <= tolerance,
            total_rows,
        ),
    ]

    rows = []
    for scenario, mask, outcome_rows in scenario_specs:
        subset = frame.loc[mask]
        count = len(subset)
        rows.append(
            {
                "scenario": scenario,
                "lead_count": int(count),
                "pct_of_all_optimizer_rows": float(count / total_rows * 100.0),
                "pct_within_outcome_group": (
                    float(count / outcome_rows * 100.0) if outcome_rows else 0.0
                ),
                "avg_bid_change": (
                    float(subset["bid_change"].mean()) if count else float("nan")
                ),
                "median_bid_change": (
                    float(subset["bid_change"].median()) if count else float("nan")
                ),
                "avg_predicted_win_probability_change": (
                    float(subset["win_probability_change"].mean())
                    if count
                    else float("nan")
                ),
                "median_predicted_win_probability_change": (
                    float(subset["win_probability_change"].median())
                    if count
                    else float("nan")
                ),
            }
        )

    return pd.DataFrame(rows)


def log_optimizer_scenario_summary(summary_df):
    """Log the historical-outcome / ML-bid scenario summary."""
    if summary_df is None or summary_df.empty:
        return

    logger.info("Historical Outcome vs ML Bid Recommendation")
    for row in summary_df.itertuples(index=False):
        logger.info("  %s", row.scenario)
        logger.info(
            "    Leads / share of all               : %s / %.2f%%",
            f"{int(row.lead_count):,}",
            float(row.pct_of_all_optimizer_rows),
        )
        logger.info(
            "    Share within outcome group         : %.2f%%",
            float(row.pct_within_outcome_group),
        )
        logger.info(
            "    Avg / median bid change             : %+.4f / %+.4f",
            float(row.avg_bid_change),
            float(row.median_bid_change),
        )
        logger.info(
            "    Avg / median predicted win-rate chg : %+.4f / %+.4f",
            float(row.avg_predicted_win_probability_change),
            float(row.median_predicted_win_probability_change),
        )


def save_evaluation_summary(
    report_dir,
    evaluation_summary,
    optimizer_eval_df=None,
):
    """Write evaluation summaries and row-level optimizer results.

    Inputs
    ------
    report_dir : str | pathlib.Path
        Directory for report artifacts.
    evaluation_summary : dict
        Combined model and optimizer evaluation summary.
    optimizer_eval_df : pandas.DataFrame | None
        Row-level optimizer evaluation results.
    """
    report_dir = Path(report_dir)
    report_dir.mkdir(exist_ok=True)

    summary_payload = dict(evaluation_summary)

    if optimizer_eval_df is not None:
        optimizer_eval_df.to_csv(
            report_dir / "bid_optimizer_test_rows.csv",
            index=False,
        )

        scenario_summary = build_optimizer_scenario_summary(optimizer_eval_df)
        if not scenario_summary.empty:
            scenario_summary.to_csv(
                report_dir / "optimizer_scenario_summary.csv",
                index=False,
            )
            summary_payload["optimizer_scenarios"] = scenario_summary.where(
                pd.notna(scenario_summary), None
            ).to_dict(orient="records")
            log_optimizer_scenario_summary(scenario_summary)

    summary_json_path = report_dir / "model_evaluation_summary.json"
    summary_json_path.write_text(json.dumps(summary_payload, indent=2))

    return summary_json_path


def log_saved_report_files(report_dir, optimizer_eval_df=None):
    """Log the report artifacts written by the training run.

    Inputs
    ------
    report_dir : str | pathlib.Path
        Directory for report artifacts.
    optimizer_eval_df : pandas.DataFrame | None
        Row-level optimizer evaluation results.
    """
    report_dir = Path(report_dir)

    files = [
        "feature_summary.csv",
        "feature_value_counts.csv",
        "roc_curve.png",
        "precision_recall_curve.png",
        "calibration_curve.png",
        "calibration_by_bucket.png",
        "probability_histogram.png",
        "confusion_matrix.png",
        "model_evaluation_summary.json",
    ]

    if optimizer_eval_df is not None:
        files.extend(
            [
                "optimizer_expected_profit_lift.png",
                "candidate_bid_count_distribution.png",
                "recommended_bid_change.png",
                "recommended_cm_distribution.png",
                "current_vs_recommended_win_rate.png",
                "policy_profit_comparison.png",
                "policy_win_rate_comparison.png",
                "expected_profit_by_max_win_probability.png",
                "bid_change_vs_win_probability_change.png",
                "bid_distribution_by_outcome.png",
                "optimizer_boundary_diagnostics.png",
                "existing_vs_recommended_bid_heatmap.png",
                "bid_change_vs_expected_profit_lift_heatmap.png",
                "bid_change_vs_recommended_cm_heatmap.png",
                "optimizer_scenario_summary.csv",
                "bid_optimizer_test_rows.csv",
            ]
        )

    existing = [filename for filename in files if (report_dir / filename).exists()]
    logger.info("Saved Training Reports")
    logger.info("  Report directory                      : %s", report_dir)
    logger.info("  Files generated                       : %s", f"{len(existing):,}")
    for filename in existing:
        logger.info("    %s", report_dir / filename)


def print_saved_report_files(report_dir, optimizer_eval_df=None):
    """Log report artifacts using the module logger.

    Inputs
    ------
    report_dir : str | pathlib.Path
        Directory for report artifacts.
    optimizer_eval_df : pandas.DataFrame | None
        Row-level optimizer evaluation results.
    """
    log_saved_report_files(report_dir, optimizer_eval_df)


def build_test_set_id(
    test_df: pd.DataFrame,
    *,
    training_table_version: str,
    split_settings: dict[str, Any],
) -> str:
    """Build a deterministic identifier for the exact held-out dataset."""
    row_hashes = pd.util.hash_pandas_object(
        test_df,
        index=True,
        categorize=True,
    ).to_numpy()
    digest = hashlib.sha256()
    digest.update(str(training_table_version).encode("utf-8"))
    digest.update(
        json.dumps(split_settings, sort_keys=True, default=str).encode("utf-8")
    )
    digest.update(row_hashes.tobytes())
    return digest.hexdigest()


def save_comparison_artifacts(
    *,
    output_dir: str | Path,
    evaluation_df: pd.DataFrame,
    optimizer_df: pd.DataFrame,
    metadata: dict[str, Any],
) -> dict[str, str]:
    """Write the complete comparison artifact set to a temporary directory."""
    artifact_dir = Path(output_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    evaluation_path = artifact_dir / "evaluation_dataset.parquet"
    optimizer_path = artifact_dir / "optimizer_results.parquet"
    metadata_path = artifact_dir / "evaluation_metadata.json"

    evaluation_df.to_parquet(evaluation_path, index=False)
    optimizer_df.to_parquet(optimizer_path, index=False)
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    return {
        "artifact_dir": str(artifact_dir),
        "evaluation_dataset": str(evaluation_path),
        "optimizer_results": str(optimizer_path),
        "metadata": str(metadata_path),
    }
