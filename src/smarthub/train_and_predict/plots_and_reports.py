"""
Plots and report utilities for Anton training.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve


def build_feature_summary_dataframe(
    df,
    continuous_features,
    discrete_features,
    categorical_features,
):
    """Create one compact summary row per feature."""
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
    """Create long-format counts and percentages per feature value."""
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


def print_training_data_summary(
    df,
    continuous_features,
    discrete_features,
    categorical_features,
    target_col=None,
):
    """Print compact feature summary tables and return them as DataFrames."""
    print()
    print("=" * 80)
    print("Training Data Feature Summary")
    print("=" * 80)
    print(f"Total Rows: {len(df):,}")

    if target_col is not None and target_col in df.columns:
        target = pd.to_numeric(df[target_col], errors="coerce")
        if target.notna().any():
            print(f"Target Win Rate: {target.mean():.4f}")

    feature_summary_df = build_feature_summary_dataframe(
        df=df,
        continuous_features=continuous_features,
        discrete_features=discrete_features,
        categorical_features=categorical_features,
    )

    print()
    print("=" * 80)
    print("Feature Summary")
    print("=" * 80)
    print(feature_summary_df.to_string(index=False))

    count_features = discrete_features + categorical_features

    feature_counts_df = build_feature_value_counts_dataframe(
        df=df,
        features=count_features,
        top_n_per_feature=30,
    )

    print()
    print("=" * 80)
    print("Discrete and Categorical Feature Value Counts")
    print("=" * 80)
    print(feature_counts_df.to_string(index=False))

    return feature_summary_df, feature_counts_df


def save_feature_summary_files(
    report_dir,
    feature_summary_df,
    feature_counts_df,
):
    """Save feature summary outputs to CSV files."""
    report_dir = Path(report_dir)
    report_dir.mkdir(exist_ok=True)

    feature_summary_df.to_csv(report_dir / "feature_summary.csv", index=False)
    feature_counts_df.to_csv(report_dir / "feature_value_counts.csv", index=False)


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
    optimizer_eval_df=None,
):
    """Save model-performance and optimizer diagnostic plots."""
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
        f"F1 Score: {f1:.1%}    "
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
        plt.figure(figsize=(8, 6))
        plt.hist(optimizer_eval_df["expected_profit_lift"], bins=40)
        plt.xlabel("Expected profit lift: recommended bid - current bid")
        plt.ylabel("Number of rows")
        plt.title("Expected Profit Lift from Recommended Bid")
        plt.tight_layout()
        plt.savefig(report_dir / "optimizer_expected_profit_lift.png")
        plt.close()

        plt.figure(figsize=(8, 6))
        plt.hist(optimizer_eval_df["bid_change"], bins=40)
        plt.xlabel("Recommended bid - current bid")
        plt.ylabel("Number of rows")
        plt.title("Recommended Bid Change from Current Bid")
        plt.tight_layout()
        plt.savefig(report_dir / "recommended_bid_change.png")
        plt.close()

        plt.figure(figsize=(8, 6))
        plt.hist(optimizer_eval_df["predicted_win_rate_lift"], bins=40)
        plt.xlabel("Predicted win-rate lift: recommended bid - current bid")
        plt.ylabel("Number of rows")
        plt.title("Predicted Win-Rate Lift from Recommended Bid")
        plt.tight_layout()
        plt.savefig(report_dir / "predicted_win_rate_lift.png")
        plt.close()

        plt.figure(figsize=(8, 6))
        plt.hist(optimizer_eval_df["recommended_bid_cm_if_won"], bins=40)
        plt.xlabel("Recommended CM if won")
        plt.ylabel("Number of rows")
        plt.title("Recommended CM Distribution")
        plt.tight_layout()
        plt.savefig(report_dir / "recommended_cm_distribution.png")
        plt.close()

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
        plt.savefig(report_dir / "current_vs_recommended_win_rate.png")
        plt.close()

        plt.figure(figsize=(8, 6))
        plt.hist(optimizer_eval_df["bid_change"], bins=40)
        plt.xlabel("Recommended bid - current bid")
        plt.ylabel("Number of rows")
        plt.title("Recommended Bid Change from Current Bid")
        plt.tight_layout()
        plt.savefig(report_dir / "recommended_bid_change.png")
        plt.close()


def save_evaluation_summary(
    report_dir,
    evaluation_summary,
    optimizer_eval_df=None,
):
    """Save evaluation summary JSON and optional optimizer row-level CSV."""
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


def print_saved_report_files(report_dir, optimizer_eval_df=None):
    """Print files generated in the training report."""
    report_dir = Path(report_dir)

    print()
    print("=" * 80)
    print("Saved Training Report Files")
    print("=" * 80)

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
                "predicted_win_rate_lift.png",
                "recommended_cm_distribution.png",
                "current_vs_recommended_win_rate.png",
                "bid_optimizer_test_rows.csv",
            ]
        )

    for filename in files:
        print(report_dir / filename)
