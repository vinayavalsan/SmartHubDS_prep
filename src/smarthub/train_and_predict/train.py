"""
Train Anton's first ML model.

Model:
    P(won | bid, lead features)

This script intentionally orchestrates the workflow and keeps implementation
details in helper modules.
"""

import argparse
from pathlib import Path

import joblib
import numpy as np
from sklearn.model_selection import train_test_split

import config
import metrics
import mlflow_utils
import models
import plots_and_reports
import predict
import preprocessing
import smarthub.io


np.random.seed(config.RANDOM_SEED)


# =============================================================================
# Load and clean data and summarize
# =============================================================================

raw_df = smarthub.io.load_leads()

feature_summary_df, feature_counts_df = plots_and_reports.print_training_data_summary(
    df=raw_df,
    continuous_features=config.CONTINUOUS_FEATURES,
    discrete_features=config.DISCRETE_FEATURES,
    categorical_features=config.CATEGORICAL_FEATURES,
    target_col=config.TARGET_COL,
)


df, cleaning_summary = preprocessing.clean_training_data(
    raw_df,
    lead_type_id=config.LEAD_TYPE_ID,
)

print(f"Lead Type ID         : {config.LEAD_TYPE_ID}")
print(f"Rows Before Cleaning : {cleaning_summary['raw_rows']:,}")
print(f"Rows After Cleaning  : {cleaning_summary['training_rows']:,}")
print(
    f"Rows Dropped         : {cleaning_summary['dropped_rows']:,} "
    f"({cleaning_summary['dropped_pct']:.2f}%)"
)
print(f"Continuous Features  : {len(config.CONTINUOUS_FEATURES):,}")
print(f"Discrete Features    : {len(config.DISCRETE_FEATURES):,}")
print(f"Categorical Features : {len(config.CATEGORICAL_FEATURES):,}")


# =============================================================================
# Split
# =============================================================================

X = df[config.FEATURE_COLS]
y = df[config.TARGET_COL]

train_idx, test_idx = train_test_split(
    df.index,
    test_size=0.20,
    random_state=config.RANDOM_SEED,
)

X_train = X.loc[train_idx]
X_test = X.loc[test_idx]
y_train = y.loc[train_idx]
y_test = y.loc[test_idx]

test_eval_df = df.loc[test_idx].copy()


# =============================================================================
# Train model
# =============================================================================

model = models.build_logistic_regression_model(
    random_seed=config.RANDOM_SEED, model_params=config.LOGISTIC_REGRESSION_PARAMS
)
model.fit(X_train, y_train)


# =============================================================================
# Evaluate model
# =============================================================================

pred, pred_class, model_metrics = metrics.evaluate_model(
    model=model,
    X_test=X_test,
    y_test=y_test,
)

# =============================================================================
# Bid optimization evaluation
# =============================================================================

optimizer_result = predict.run_bid_optimizer_evaluation(
    test_eval_df=test_eval_df,
    model=model,
    target_cm=config.TARGET_CM,
    min_bid=config.MIN_BID,
    bid_step=config.BID_STEP,
)


if optimizer_result is not None:
    optimizer_eval_df, optimizer_summary = optimizer_result
    model_metrics["predicted_recommended_bid_win_rate"] = optimizer_summary[
        "avg_recommended_bid_predicted_win_rate"
    ]
else:
    optimizer_eval_df, optimizer_summary = None, {}
    model_metrics["predicted_recommended_bid_win_rate"] = float("nan")

metrics.print_model_evaluation(
    metrics=model_metrics,
    rows_trained=len(df),
    train_rows=len(X_train),
    test_rows=len(X_test),
)

# =============================================================================
# Save report
# =============================================================================

report_dir = config.REPORT_DIR
Path(report_dir).mkdir(exist_ok=True)

plots_and_reports.save_feature_summary_files(
    report_dir=report_dir,
    feature_summary_df=feature_summary_df,
    feature_counts_df=feature_counts_df,
)

plots_and_reports.save_performance_plots(
    report_dir=report_dir,
    y_test=y_test,
    pred=pred,
    pred_class=pred_class,
    roc_auc=model_metrics["roc_auc"],
    pr_auc=model_metrics["pr_auc"],
    accuracy=model_metrics["accuracy"],
    precision=model_metrics["precision"],
    recall=model_metrics["recall"],
    f1=model_metrics["f1"],
    optimizer_eval_df=optimizer_eval_df,
)

evaluation_summary = {
    "rows_trained": int(len(df)),
    "train_rows": int(len(X_train)),
    "test_rows": int(len(X_test)),
    **model_metrics,
    "bid_optimizer": optimizer_summary,
}

plots_and_reports.save_evaluation_summary(
    report_dir=report_dir,
    evaluation_summary=evaluation_summary,
    optimizer_eval_df=optimizer_eval_df,
)

plots_and_reports.print_saved_report_files(
    report_dir=report_dir,
    optimizer_eval_df=optimizer_eval_df,
)


# =============================================================================
# Save model locally
# =============================================================================

Path(config.MODEL_DIR).mkdir(exist_ok=True)
joblib.dump(model, config.LOCAL_MODEL_PATH)
print(f"Saved model to {config.LOCAL_MODEL_PATH}")


# =============================================================================
# MLflow
# =============================================================================

mlflow_utils.log_training_run(
    model=model,
    model_params=config.LOGISTIC_REGRESSION_PARAMS,
    feature_cols=config.FEATURE_COLS,
    metrics=model_metrics,
    report_dir=report_dir,
    experiment_name=config.MLFLOW_EXPERIMENT_NAME,
    run_name=config.MLFLOW_RUN_NAME,
    registered_model_name=config.MLFLOW_REGISTERED_MODEL_NAME,
)

print("Done.")
