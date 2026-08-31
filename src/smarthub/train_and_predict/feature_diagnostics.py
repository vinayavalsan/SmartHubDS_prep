"""Feature diagnostics shared by SmartHub training and HPO.

This module owns descriptive diagnostics about model inputs and their relationship
to the target: dataset summaries, value counts, zero-variance checks, train/eval
coverage, and pre-split feature-to-target association. Diagnostics are reporting
only; they do not remove features or otherwise change the model feature schema.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif

from smarthub.core.logging_utils import get_logger

logger = get_logger(__name__)


def _numeric_values(series: pd.Series) -> np.ndarray:
    """Return one numeric feature as a finite two-dimensional array."""
    values = pd.to_numeric(series, errors="coerce")
    median = values.median()
    fill_value = float(median) if pd.notna(median) else 0.0
    return values.fillna(fill_value).to_numpy(dtype=float).reshape(-1, 1)


def _categorical_values(series: pd.Series) -> np.ndarray:
    """Return one categorical feature as integer category codes."""
    values = series.astype("string").fillna("<NA>")
    codes, _ = pd.factorize(values, sort=True)
    return codes.reshape(-1, 1)


def build_feature_target_association(
    frame: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
    target_column: str,
    random_seed: int,
) -> list[dict[str, Any]]:
    """Calculate mutual information for every configured model feature.

    Inputs
    ------
    frame : pandas.DataFrame
        Complete prepared dataset before any HPO data split.
    numeric_features : list[str]
        Numeric model feature names.
    categorical_features : list[str]
        Categorical model feature names.
    target_column : str
        Binary target column.
    random_seed : int
        Random seed used by the continuous mutual-information estimator.

    Returns
    -------
    list[dict[str, Any]]
        Feature association rows sorted by decreasing mutual information.
    """
    target = pd.to_numeric(frame[target_column], errors="coerce")
    valid_target = target.notna()
    target_values = target.loc[valid_target].astype(int).to_numpy()

    rows: list[dict[str, Any]] = []
    categorical_set = set(categorical_features)
    features = list(numeric_features) + list(categorical_features)

    for feature in features:
        if feature not in frame.columns:
            continue

        series = frame.loc[valid_target, feature]
        is_categorical = feature in categorical_set
        values = (
            _categorical_values(series) if is_categorical else _numeric_values(series)
        )

        if np.unique(values).size <= 1:
            mutual_information = 0.0
        else:
            mutual_information = float(
                mutual_info_classif(
                    values,
                    target_values,
                    discrete_features=is_categorical,
                    random_state=random_seed,
                )[0]
            )

        rows.append(
            {
                "feature": feature,
                "feature_type": "categorical" if is_categorical else "numeric",
                "mutual_information": mutual_information,
            }
        )

    rows.sort(
        key=lambda row: (-row["mutual_information"], row["feature"]),
    )
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank

    return rows


def log_feature_target_association(
    diagnostics: list[dict[str, Any]],
) -> None:
    """Log a compact feature-target mutual-information ranking."""
    logger.info("Feature-Target Association")
    if not diagnostics:
        logger.info("  No feature-target association diagnostics available.")
        return

    table = pd.DataFrame(diagnostics)[
        ["rank", "feature", "feature_type", "mutual_information"]
    ].copy()
    table["mutual_information"] = table["mutual_information"].round(6)
    logger.info("\n%s", table.to_string(index=False))


def build_feature_summary_dataframe(
    df,
    continuous_features,
    discrete_features,
    categorical_features,
):
    """Build one summary row for each configured feature.

    Inputs
    ------
    df : pandas.DataFrame
        Input dataframe.
    continuous_features : list[str]
        Continuous feature names.
    discrete_features : list[str]
        Discrete feature names.
    categorical_features : list[str]
        Categorical feature names.

    Returns
    -------
    pandas.DataFrame
        Feature-level summary table.
    """
    rows = []

    feature_types = {
        **{feature: "continuous" for feature in continuous_features},
        **{feature: "discrete" for feature in discrete_features},
        **{feature: "categorical" for feature in categorical_features},
    }

    for feature, feature_type in feature_types.items():
        if feature not in df.columns:
            continue

        series = df[feature]
        missing_count = (
            series.isna().sum() + (series.astype(str).str.strip() == "").sum()
        )
        missing_pct = missing_count / len(df) * 100 if len(df) else 0

        mode_values = series.dropna().replace("", pd.NA).dropna().mode()
        mode_value = mode_values.iloc[0] if not mode_values.empty else pd.NA

        row = {
            "feature": feature,
            "type": feature_type,
            "missing_count": int(missing_count),
            "missing_pct": round(missing_pct, 2),
            "unique_values": int(series.nunique(dropna=True)),
            "mode": mode_value,
            "mean": pd.NA,
            "median": pd.NA,
            "min": pd.NA,
            "max": pd.NA,
            "std": pd.NA,
        }

        if feature_type == "continuous":
            numeric_series = pd.to_numeric(series, errors="coerce")
            row.update(
                {
                    "mean": round(numeric_series.mean(), 4),
                    "median": round(numeric_series.median(), 4),
                    "min": round(numeric_series.min(), 4),
                    "max": round(numeric_series.max(), 4),
                    "std": round(numeric_series.std(), 4),
                }
            )

        rows.append(row)

    return pd.DataFrame(rows)


def build_feature_value_counts_dataframe(df, features, top_n_per_feature=30):
    """Build long-format value counts for selected features.

    Inputs
    ------
    df : pandas.DataFrame
        Input dataframe.
    features : list[str]
        Feature names to summarize.
    top_n_per_feature : int
        Maximum values retained per feature.

    Returns
    -------
    pandas.DataFrame
        Long-format value-count table.
    """
    long_df = df[features].copy()

    for col in features:
        long_df[col] = (
            long_df[col].astype("string").fillna("<NA>").replace("", "<EMPTY>")
        )

    long_df = long_df.melt(
        var_name="feature",
        value_name="feature_value",
    )

    counts_df = (
        long_df.groupby(["feature", "feature_value"]).size().reset_index(name="count")
    )

    counts_df["percent"] = (
        counts_df["count"]
        / counts_df.groupby("feature")["count"].transform("sum")
        * 100
    )
    counts_df["percent"] = counts_df["percent"].round(2)

    counts_df = counts_df.sort_values(
        ["feature", "count", "feature_value"],
        ascending=[True, False, True],
    )

    counts_df["rank"] = counts_df.groupby("feature")["count"].rank(
        method="first",
        ascending=False,
    )
    counts_df = counts_df[counts_df["rank"] <= top_n_per_feature].copy()
    counts_df = counts_df.drop(columns=["rank"])

    return counts_df


def build_training_data_summary(
    df,
    continuous_features,
    discrete_features,
    categorical_features,
):
    """Build feature summaries and value-count tables.

    Inputs
    ------
    df : pandas.DataFrame
        Input dataframe.
    continuous_features : list[str]
        Continuous feature names.
    discrete_features : list[str]
        Discrete feature names.
    categorical_features : list[str]
        Categorical feature names.

    Returns
    -------
    tuple[pandas.DataFrame, pandas.DataFrame]
        Feature summary followed by feature-value counts.
    """
    feature_summary_df = build_feature_summary_dataframe(
        df=df,
        continuous_features=continuous_features,
        discrete_features=discrete_features,
        categorical_features=categorical_features,
    )
    count_features = list(discrete_features) + list(categorical_features)
    feature_counts_df = build_feature_value_counts_dataframe(
        df=df,
        features=count_features,
        top_n_per_feature=30,
    )
    return feature_summary_df, feature_counts_df


def log_training_data_summary(
    df,
    feature_summary_df,
    feature_counts_df,
    target_col=None,
):
    """Log a compact summary of the prepared training data.

    Inputs
    ------
    df : pandas.DataFrame
        Input dataframe.
    feature_summary_df : pandas.DataFrame
        Feature-level summary dataframe.
    feature_counts_df : pandas.DataFrame
        Feature-value count dataframe.
    target_col : str | None
        Optional target column name.
    """
    logger.info("Dataset Summary")
    logger.info("  Total rows                            : %s", f"{len(df):,}")

    if target_col is not None and target_col in df.columns:
        target = pd.to_numeric(df[target_col], errors="coerce")
        if target.notna().any():
            logger.info("  Target win rate                       : %.4f", target.mean())

    if not feature_summary_df.empty:
        total_missing = int(feature_summary_df["missing_count"].sum())
        missing_features = int((feature_summary_df["missing_count"] > 0).sum())
        logger.info(
            "  Features summarized                   : %s",
            f"{len(feature_summary_df):,}",
        )
        logger.info(
            "  Features with missing values          : %s",
            f"{missing_features:,}",
        )
        logger.info(
            "  Total missing feature values          : %s",
            f"{total_missing:,}",
        )

        top_missing = feature_summary_df.sort_values(
            "missing_pct", ascending=False
        ).head(8)
        top_missing = top_missing[top_missing["missing_count"] > 0]
        if not top_missing.empty:
            logger.info("  Top missing features")
            for row in top_missing.itertuples(index=False):
                logger.info(
                    "    %-35s %8s (%6.2f%%)",
                    row.feature,
                    f"{int(row.missing_count):,}",
                    float(row.missing_pct),
                )

    if feature_counts_df is not None and not feature_counts_df.empty:
        logger.info(
            "  Feature-value count rows saved        : %s",
            f"{len(feature_counts_df):,}",
        )


def print_training_data_summary(
    df,
    continuous_features,
    discrete_features,
    categorical_features,
    target_col=None,
):
    """Build and log training-data summaries.

    Inputs
    ------
    df : pandas.DataFrame
        Input dataframe.
    continuous_features : list[str]
        Continuous feature names.
    discrete_features : list[str]
        Discrete feature names.
    categorical_features : list[str]
        Categorical feature names.
    target_col : str | None
        Optional target column name.

    Returns
    -------
    tuple[pandas.DataFrame, pandas.DataFrame]
        Feature summary followed by feature-value counts.
    """
    feature_summary_df, feature_counts_df = build_training_data_summary(
        df=df,
        continuous_features=continuous_features,
        discrete_features=discrete_features,
        categorical_features=categorical_features,
    )
    log_training_data_summary(
        df=df,
        feature_summary_df=feature_summary_df,
        feature_counts_df=feature_counts_df,
        target_col=target_col,
    )
    return feature_summary_df, feature_counts_df


def coverage_features(
    frame: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
) -> list[str]:
    """Return categorical features plus true binary numeric features.

    Inputs
    ------
    frame : pandas.DataFrame
        Dataframe used to identify binary numeric features.
    numeric_features : list[str]
        Numeric model feature names.
    categorical_features : list[str]
        Categorical model feature names.

    Returns
    -------
    list[str]
        Features included in categorical/binary coverage diagnostics.
    """
    binary_numeric = []
    for column in numeric_features:
        if column not in frame.columns:
            continue
        values = set(frame[column].dropna().unique().tolist())
        if values and values.issubset({0, 1, False, True}):
            binary_numeric.append(column)
    return list(categorical_features) + binary_numeric


def feature_coverage_rows(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    features: list[str],
    partition: str,
) -> list[dict]:
    """Summarize feature support from fitting data into evaluation data.

    Inputs
    ------
    train_df : pandas.DataFrame
        Model-fitting partition.
    eval_df : pandas.DataFrame
        Held-out validation or test partition.
    features : list[str]
        Categorical and binary features to inspect.
    partition : str
        Human-readable evaluation partition name.

    Returns
    -------
    list[dict]
        Feature-level coverage diagnostic rows.
    """
    rows = []
    eval_row_count = len(eval_df)

    for column in features:
        if column not in train_df.columns or column not in eval_df.columns:
            continue

        train_counts = train_df[column].dropna().value_counts()
        eval_values = eval_df[column].dropna()
        eval_unique_values = set(eval_values.unique().tolist())
        train_unique_values = set(train_counts.index.tolist())
        unseen_values = eval_unique_values - train_unique_values

        unseen_rows = (
            int(eval_df[column].isin(unseen_values).sum()) if unseen_values else 0
        )
        train_support = [
            int(train_counts[value])
            for value in eval_unique_values
            if value in train_counts.index
        ]
        min_train_support = (
            0 if unseen_values else min(train_support) if train_support else 0
        )

        rows.append(
            {
                "partition": partition,
                "feature": column,
                "train_unique": int(len(train_unique_values)),
                "eval_unique": int(len(eval_unique_values)),
                "unseen_eval_unique": int(len(unseen_values)),
                "eval_rows_unseen": unseen_rows,
                "eval_pct_unseen": (
                    float(unseen_rows / eval_row_count * 100.0)
                    if eval_row_count
                    else 0.0
                ),
                "min_train_support_for_eval_values": int(min_train_support),
            }
        )

    return rows


def find_zero_variance_features(
    frame,
    numeric_features,
    categorical_features,
) -> list[str]:
    """Return model features with no observed variation.

    Zero-variance features are diagnostic information only. They remain in
    the model feature schema so temporary lack of variation in a training
    window does not change the model contract.

    Inputs
    ------
    frame : pandas.DataFrame
        Input dataframe used to fit models.
    numeric_features : list[str]
        Numeric feature names.
    categorical_features : list[str]
        Categorical feature names.

    Returns
    -------
    list[str]
        Feature names with at most one observed non-null value.
    """
    return [
        column
        for column in list(numeric_features) + list(categorical_features)
        if column in frame.columns and frame[column].nunique(dropna=True) <= 1
    ]


def log_zero_variance_features(
    features: list[str],
    population: str,
) -> None:
    """Log the count and names of zero-variance features for one population."""
    logger.info(
        "Zero-Variance Features, %-16s : %s (%s)",
        population,
        f"{len(features):,}",
        ", ".join(features) if features else "none",
    )


def feature_coverage_differences(diagnostics: list[dict[str, Any]]) -> pd.DataFrame:
    """Return only train/evaluation boundaries with a coverage difference."""
    table = pd.DataFrame(diagnostics)
    if table.empty:
        return table
    return table[
        (table["train_unique"] != table["eval_unique"])
        | (table["unseen_eval_unique"] > 0)
    ].copy()


def log_feature_coverage_diagnostics(
    diagnostics: list[dict[str, Any]],
    *,
    evaluation_label: str = "evaluation",
    include_partition: bool = True,
) -> pd.DataFrame:
    """Log coverage differences and return the displayed diagnostic rows."""
    report = feature_coverage_differences(diagnostics)
    logger.info("Feature Coverage Diagnostics")
    if report.empty:
        logger.info(
            "  No train/%s coverage differences detected.",
            evaluation_label,
        )
        return report

    display = report.copy()
    if not include_partition and "partition" in display.columns:
        display = display.drop(columns=["partition"])
    if evaluation_label != "eval":
        display = display.rename(
            columns={
                "eval_unique": f"{evaluation_label}_unique",
                "unseen_eval_unique": f"unseen_{evaluation_label}_unique",
                "eval_rows_unseen": f"{evaluation_label}_rows_unseen",
                "eval_pct_unseen": f"{evaluation_label}_pct_unseen",
                "min_train_support_for_eval_values": (
                    f"min_train_support_for_{evaluation_label}_values"
                ),
            }
        )
        pct_column = f"{evaluation_label}_pct_unseen"
    else:
        pct_column = "eval_pct_unseen"
    if pct_column in display.columns:
        display[pct_column] = display[pct_column].round(2)
    logger.info("\n%s", display.to_string(index=False))
    return report


def find_binary_variance_loss(
    full_frame: pd.DataFrame,
    coverage_diagnostics: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return binary features that vary overall but are constant in fitting data."""
    result = []
    for row in coverage_diagnostics:
        column = row["feature"]
        if column not in full_frame.columns:
            continue
        overall_unique = full_frame[column].nunique(dropna=True)
        if overall_unique == 2 and row["train_unique"] == 1 and row["eval_unique"] == 2:
            result.append(row)
    return result
