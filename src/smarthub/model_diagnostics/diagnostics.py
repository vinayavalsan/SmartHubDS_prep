"""Feature-level diagnostics for SmartHub optimizer evaluation artifacts.

The helpers in this module are intentionally analysis-only. They read the
row-level optimizer evaluation output produced by model training and summarize
how observed/predicted win rate and bid behavior vary across feature values.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = {
    "bid",
    "recommended_bid",
    "current_bid_predicted_win_rate",
    "recommended_bid_predicted_win_rate",
}


@dataclass(frozen=True)
class FeatureDiagnosticConfig:
    """Configuration for one feature-level diagnostic summary."""

    feature: str
    bins: int = 10
    binning: str = "quantile"
    top_n: int = 20
    min_support: int = 1
    group_rare_as_other: bool = True


def resolve_target_column(frame: pd.DataFrame) -> str:
    """Return the historical outcome column used by the evaluation artifact."""
    for candidate in ("won_flag", "won"):
        if candidate in frame.columns:
            return candidate
    raise ValueError("Evaluation artifact does not contain 'won_flag' or 'won'.")


def validate_optimizer_frame(frame: pd.DataFrame) -> None:
    """Require the columns needed for feature-level optimizer diagnostics."""
    missing = sorted(REQUIRED_COLUMNS.difference(frame.columns))
    if missing:
        raise ValueError(
            "Optimizer evaluation artifact is missing required columns: "
            + ", ".join(missing)
        )
    resolve_target_column(frame)


def _is_identifier_feature(feature: str) -> bool:
    """Return whether a feature name represents an unordered identifier."""
    return feature == "id" or feature.endswith("_id")


def resolve_feature_analysis_kind(
    frame: pd.DataFrame,
    feature: str,
) -> str:
    """Resolve how a feature should be grouped in analysis.

    Registry metadata is preferred over pandas storage dtype. Identifier columns
    are always categorical because their numeric values are labels, not ordered
    measurements. Discrete numeric features keep their individual values, while
    only continuous numeric features are bucketed into ranges.
    """
    if feature not in frame.columns:
        raise ValueError(f"Feature {feature!r} is not present in the artifact.")

    if _is_identifier_feature(feature):
        return "categorical"

    try:
        from smarthub.feature_engineering.feature_registry import FEATURES

        spec = FEATURES.get(feature)
    except ImportError:
        spec = None

    if spec is not None:
        if spec.kind == "numeric_continuous":
            return "continuous"
        if spec.kind == "numeric_discrete":
            return "discrete"
        return "categorical"

    try:
        from smarthub.data_pull.field_registry import RAW_FIELD_REGISTRY

        raw_spec = RAW_FIELD_REGISTRY.get(feature)
    except ImportError:
        raw_spec = None

    if raw_spec is not None:
        kind = raw_spec.validation.kind
        if kind == "numeric_continuous":
            return "continuous"
        if kind == "numeric_discrete":
            return "discrete"
        return "categorical"

    if pd.api.types.is_numeric_dtype(frame[feature]):
        return "continuous"
    return "categorical"


def _numeric_bucket(
    series: pd.Series,
    *,
    bins: int,
    binning: str,
) -> pd.Series:
    """Bucket a numeric feature with quantile or equal-width bins."""
    numeric = pd.to_numeric(series, errors="coerce")

    if numeric.dropna().nunique() <= 1:
        return numeric.astype("string").fillna("<NA>")

    if binning == "quantile":
        try:
            bucketed = pd.qcut(numeric, q=bins, duplicates="drop")
        except ValueError:
            bucketed = pd.cut(numeric, bins=bins, duplicates="drop")
    elif binning == "fixed":
        bucketed = pd.cut(numeric, bins=bins, duplicates="drop")
    else:
        raise ValueError("binning must be either 'quantile' or 'fixed'.")

    return bucketed.astype("string").fillna("<NA>")


def _categorical_bucket(
    series: pd.Series,
    *,
    top_n: int,
    min_support: int,
    group_rare_as_other: bool,
) -> pd.Series:
    """Normalize and optionally collapse low-support categorical values."""
    values = series.astype("string").fillna("<NA>").replace("", "<EMPTY>")
    counts = values.value_counts(dropna=False)

    keep = counts[counts >= min_support].index.tolist()
    if top_n > 0:
        keep = keep[:top_n]

    if group_rare_as_other:
        return values.where(values.isin(keep), "Other")
    return values.where(values.isin(keep))


def build_feature_buckets(
    frame: pd.DataFrame,
    config: FeatureDiagnosticConfig,
) -> pd.Series:
    """Return one analysis bucket/category label per input row."""
    if config.feature not in frame.columns:
        raise ValueError(f"Feature {config.feature!r} is not present in the artifact.")

    feature_series = frame[config.feature]
    analysis_kind = resolve_feature_analysis_kind(frame, config.feature)
    if analysis_kind == "continuous":
        return _numeric_bucket(
            feature_series,
            bins=config.bins,
            binning=config.binning,
        )

    return _categorical_bucket(
        feature_series,
        top_n=config.top_n,
        min_support=config.min_support,
        group_rare_as_other=config.group_rare_as_other,
    )


def apply_feature_bucket(
    frame: pd.DataFrame,
    config: FeatureDiagnosticConfig,
    feature_bucket: str,
) -> pd.DataFrame:
    """Filter a frame to one configured feature bucket/category."""
    if feature_bucket == "All":
        return frame.copy()

    buckets = build_feature_buckets(frame, config)
    mask = buckets.astype("string") == str(feature_bucket)
    return frame.loc[mask.fillna(False)].copy()


def build_feature_diagnostic(
    frame: pd.DataFrame,
    config: FeatureDiagnosticConfig,
) -> pd.DataFrame:
    """Aggregate win-rate and bid behavior by one feature."""
    validate_optimizer_frame(frame)

    target_col = resolve_target_column(frame)
    working = frame.copy()
    working["feature_bucket"] = build_feature_buckets(working, config)

    working["observed_outcome"] = pd.to_numeric(working[target_col], errors="coerce")
    working["existing_bid"] = pd.to_numeric(working["bid"], errors="coerce")
    working["recommended_bid_numeric"] = pd.to_numeric(
        working["recommended_bid"], errors="coerce"
    )
    working["pred_wr_existing"] = pd.to_numeric(
        working["current_bid_predicted_win_rate"], errors="coerce"
    )
    working["pred_wr_recommended"] = pd.to_numeric(
        working["recommended_bid_predicted_win_rate"], errors="coerce"
    )
    working["bid_change"] = working["recommended_bid_numeric"] - working["existing_bid"]
    working["predicted_wr_change"] = (
        working["pred_wr_recommended"] - working["pred_wr_existing"]
    )

    valid = working["feature_bucket"].notna()
    working = working.loc[valid].copy()
    if working.empty:
        return pd.DataFrame()

    summary = (
        working.groupby("feature_bucket", observed=False)
        .agg(
            leads=("feature_bucket", "size"),
            observed_win_rate=("observed_outcome", "mean"),
            predicted_win_rate_existing=("pred_wr_existing", "mean"),
            predicted_win_rate_recommended=("pred_wr_recommended", "mean"),
            avg_existing_bid=("existing_bid", "mean"),
            median_existing_bid=("existing_bid", "median"),
            avg_recommended_bid=("recommended_bid_numeric", "mean"),
            median_recommended_bid=("recommended_bid_numeric", "median"),
            avg_bid_change=("bid_change", "mean"),
            median_bid_change=("bid_change", "median"),
            avg_predicted_win_rate_change=("predicted_wr_change", "mean"),
            median_predicted_win_rate_change=("predicted_wr_change", "median"),
        )
        .reset_index()
    )

    summary = summary[summary["leads"] >= config.min_support].copy()
    if summary.empty:
        return summary

    summary["fraction_of_filtered_rows"] = summary["leads"] / summary["leads"].sum()

    # Preserve the natural feature order for plotting. Numeric-valued feature
    # labels should always progress from low to high, even if registry metadata
    # is unavailable and the feature was resolved as categorical. Identifier
    # fields remain categorical because their numeric values are labels only.
    analysis_kind = resolve_feature_analysis_kind(frame, config.feature)
    if analysis_kind == "continuous":
        sort_key = pd.to_numeric(
            summary["feature_bucket"]
            .astype(str)
            .str.extract(
                r"^[\[(]\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
                expand=False,
            ),
            errors="coerce",
        )
    elif not _is_identifier_feature(config.feature):
        sort_key = pd.to_numeric(summary["feature_bucket"], errors="coerce")
    else:
        sort_key = pd.Series(np.nan, index=summary.index, dtype=float)

    if sort_key.notna().any():
        summary = (
            summary.assign(_feature_sort_key=sort_key)
            .sort_values("_feature_sort_key", kind="stable", na_position="last")
            .drop(columns="_feature_sort_key")
            .reset_index(drop=True)
        )

    return summary


def build_subset_diagnostics(
    frame: pd.DataFrame,
    reference_rows: int,
) -> dict[str, float | int | None]:
    """Return compact optimizer diagnostics for the selected row subset."""
    validate_optimizer_frame(frame)
    target_col = resolve_target_column(frame)

    if frame.empty:
        return {}

    existing_bid = pd.to_numeric(frame["bid"], errors="coerce")
    recommended_bid = pd.to_numeric(frame["recommended_bid"], errors="coerce")
    current_wr = pd.to_numeric(frame["current_bid_predicted_win_rate"], errors="coerce")
    recommended_wr = pd.to_numeric(
        frame["recommended_bid_predicted_win_rate"], errors="coerce"
    )
    outcome = pd.to_numeric(frame[target_col], errors="coerce")

    result: dict[str, float | int | None] = {
        "leads": int(len(frame)),
        "share_of_filtered_rows": (
            float(len(frame) / reference_rows) if reference_rows else None
        ),
        "observed_win_rate": float(outcome.mean()),
        "median_existing_bid": float(existing_bid.median()),
        "median_recommended_bid": float(recommended_bid.median()),
        "median_bid_change": float((recommended_bid - existing_bid).median()),
        "avg_current_predicted_win_rate": float(current_wr.mean()),
        "avg_recommended_predicted_win_rate": float(recommended_wr.mean()),
        "avg_predicted_win_rate_change": float((recommended_wr - current_wr).mean()),
    }

    if "expected_profit_lift" in frame.columns:
        expected_profit_lift = pd.to_numeric(
            frame["expected_profit_lift"], errors="coerce"
        )
        result["median_expected_profit_lift"] = float(expected_profit_lift.median())
    else:
        result["median_expected_profit_lift"] = None

    if "max_bid" in frame.columns:
        maximum_bid = pd.to_numeric(frame["max_bid"], errors="coerce")
        valid = recommended_bid.notna() & maximum_bid.notna()
        if valid.any():
            at_maximum = np.isclose(
                recommended_bid.loc[valid].to_numpy(dtype=float),
                maximum_bid.loc[valid].to_numpy(dtype=float),
                rtol=0.0,
                atol=1e-9,
            )
            result["at_maximum_candidate_bid"] = float(at_maximum.mean())
        else:
            result["at_maximum_candidate_bid"] = None
    else:
        result["at_maximum_candidate_bid"] = None

    return result


def available_feature_columns(frame: pd.DataFrame) -> list[str]:
    """Return candidate feature columns while excluding diagnostic outputs."""
    excluded = {
        "won",
        "won_flag",
        "predicted_class",
        "predicted_win_probability",
        "current_bid_predicted_win_rate",
        "current_bid_expected_profit",
        "recommended_bid",
        "recommended_bid_predicted_win_rate",
        "recommended_bid_expected_profit",
        "expected_profit_lift",
        "bid_change",
        "recommended_bid_cm_if_won",
        "observed_policy_expected_revenue",
        "observed_policy_bid_cost",
        "observed_policy_expected_profit",
        "max_bid",
        "n_candidate_bids",
    }
    return sorted(column for column in frame.columns if column not in excluded)


def apply_recommendation_filter(
    frame: pd.DataFrame, recommendation_filter: str
) -> pd.DataFrame:
    """Filter rows by optimizer recommendation behavior."""
    if recommendation_filter == "All":
        return frame.copy()

    existing = pd.to_numeric(frame["bid"], errors="coerce")
    recommended = pd.to_numeric(frame["recommended_bid"], errors="coerce")
    change = recommended - existing
    tolerance = 1e-9

    if recommendation_filter == "Minimum bid":
        valid = recommended.dropna()
        if valid.empty:
            return frame.iloc[0:0].copy()
        minimum = float(valid.min())
        mask = np.isclose(recommended, minimum, rtol=0.0, atol=tolerance)
    elif recommendation_filter == "Large bid increase":
        mask = change >= 10.0
    elif recommendation_filter == "Large bid decrease":
        mask = change <= -10.0
    elif recommendation_filter in {"Interior", "Maximum candidate bid"}:
        if "maximum_candidate_bid" in frame.columns:
            maximum = pd.to_numeric(frame["maximum_candidate_bid"], errors="coerce")
        elif "max_bid" in frame.columns:
            maximum = pd.to_numeric(frame["max_bid"], errors="coerce")
        else:
            raise ValueError(
                "Maximum-candidate filtering requires 'maximum_candidate_bid' "
                "or 'max_bid' in the saved artifact."
            )
        at_max = np.isclose(recommended, maximum, rtol=0.0, atol=tolerance)
        minimum = float(recommended.dropna().min())
        at_min = np.isclose(recommended, minimum, rtol=0.0, atol=tolerance)
        mask = (
            at_max
            if recommendation_filter == "Maximum candidate bid"
            else ~(at_min | at_max)
        )
    else:
        raise ValueError(
            f"Unsupported recommendation filter: {recommendation_filter!r}"
        )

    return frame.loc[pd.Series(mask, index=frame.index).fillna(False)].copy()


def apply_outcome_filter(frame: pd.DataFrame, outcome: str) -> pd.DataFrame:
    """Filter to all, historically won, or historically lost rows."""
    if outcome == "All":
        return frame.copy()

    target_col = resolve_target_column(frame)
    target = pd.to_numeric(frame[target_col], errors="coerce")
    if outcome == "Won":
        return frame.loc[target == 1].copy()
    if outcome == "Lost":
        return frame.loc[target == 0].copy()

    raise ValueError(f"Unsupported outcome filter: {outcome!r}")
