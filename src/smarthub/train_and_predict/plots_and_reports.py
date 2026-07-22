"""Training reports and plots for SmartHub models.

This module summarizes data and writes model and optimizer report artifacts.
"""

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve

logger = logging.getLogger("smarthub.train_and_predict.plots_and_reports")


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

    # for col in features:
    #    long_df[col] = long_df[col].fillna("<NA>").replace("", "<EMPTY>").astype(str)
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
    log=None,
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
    log : logging.Logger | None
        Optional logger for structured output.
    """
    log = log or logger
    log.info("Dataset Summary")
    log.info("  Total rows                            : %s", f"{len(df):,}")

    if target_col is not None and target_col in df.columns:
        target = pd.to_numeric(df[target_col], errors="coerce")
        if target.notna().any():
            log.info("  Target win rate                       : %.4f", target.mean())

    if not feature_summary_df.empty:
        total_missing = int(feature_summary_df["missing_count"].sum())
        missing_features = int((feature_summary_df["missing_count"] > 0).sum())
        log.info(
            "  Features summarized                   : %s",
            f"{len(feature_summary_df):,}",
        )
        log.info(
            "  Features with missing values          : %s",
            f"{missing_features:,}",
        )
        log.info(
            "  Total missing feature values          : %s",
            f"{total_missing:,}",
        )

        top_missing = feature_summary_df.sort_values(
            "missing_pct", ascending=False
        ).head(8)
        top_missing = top_missing[top_missing["missing_count"] > 0]
        if not top_missing.empty:
            log.info("  Top missing features")
            for row in top_missing.itertuples(index=False):
                log.info(
                    "    %-35s %8s (%6.2f%%)",
                    row.feature,
                    f"{int(row.missing_count):,}",
                    float(row.missing_pct),
                )

    if feature_counts_df is not None and not feature_counts_df.empty:
        log.info(
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
    plt.tight_layout()
    path = Path(report_dir) / filename
    plt.savefig(path)
    plt.close()
    return path


def _save_optimizer_plots(report_dir, optimizer_eval_df):
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
            "Expected profit lift: recommended bid - current bid",
            "Expected Profit Lift from Recommended Bid",
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

    if {
        "current_bid_predicted_win_rate",
        "recommended_bid_predicted_win_rate",
    }.issubset(optimizer_eval_df.columns):
        plt.figure(figsize=(8, 6))
        plt.scatter(
            optimizer_eval_df["current_bid_predicted_win_rate"],
            optimizer_eval_df["recommended_bid_predicted_win_rate"],
            alpha=0.3,
        )
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.xlabel("Predicted win rate using current bid")
        plt.ylabel("Predicted win rate using recommended bid")
        plt.title("Current vs Recommended Predicted Win Rate")
        plt.tight_layout()
        plt.savefig(Path(report_dir) / "current_vs_recommended_win_rate.png")
        plt.close()
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
    plt.legend()
    plt.tight_layout()
    plt.savefig(report_dir / "calibration_curve.png")
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.hist(pred[y_test == 0], bins=30, alpha=0.6, label="Lost")
    plt.hist(pred[y_test == 1], bins=30, alpha=0.6, label="Won")
    plt.xlabel("Predicted win probability")
    plt.ylabel("Number of rows")
    plt.title("Predicted Probability Distribution")
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
        _save_optimizer_plots(report_dir, optimizer_eval_df)


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

    summary_json_path = report_dir / "model_evaluation_summary.json"
    summary_json_path.write_text(json.dumps(evaluation_summary, indent=2))

    if optimizer_eval_df is not None:
        optimizer_eval_df.to_csv(
            report_dir / "bid_optimizer_test_rows.csv",
            index=False,
        )

    return summary_json_path


def log_saved_report_files(report_dir, optimizer_eval_df=None, log=None):
    """Log the report artifacts written by the training run.

    Inputs
    ------
    report_dir : str | pathlib.Path
        Directory for report artifacts.
    optimizer_eval_df : pandas.DataFrame | None
        Row-level optimizer evaluation results.
    log : logging.Logger | None
        Optional logger for structured output.
    """
    log = log or logger
    report_dir = Path(report_dir)

    files = [
        "feature_summary.csv",
        "feature_value_counts.csv",
        "roc_curve.png",
        "precision_recall_curve.png",
        "calibration_curve.png",
        "probability_histogram.png",
        "confusion_matrix.png",
        "model_evaluation_summary.json",
    ]

    if optimizer_eval_df is not None:
        files.extend(
            [
                "optimizer_expected_profit_lift.png",
                "recommended_bid_change.png",
                "recommended_cm_distribution.png",
                "current_vs_recommended_win_rate.png",
                "bid_optimizer_test_rows.csv",
            ]
        )

    existing = [filename for filename in files if (report_dir / filename).exists()]
    log.info("Saved Training Reports")
    log.info("  Report directory                      : %s", report_dir)
    log.info("  Files generated                       : %s", f"{len(existing):,}")
    for filename in existing:
        log.info("    %s", report_dir / filename)


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
