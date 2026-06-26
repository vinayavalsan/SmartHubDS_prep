"""
Train Anton's first ML model.

Model:
    P(won | bid, lead features)

This script intentionally orchestrates the workflow and keeps implementation
details in helper modules.
"""

from pathlib import Path

import joblib
import numpy as np
from sklearn.model_selection import train_test_split

from config import (
    CATEGORICAL_FEATURES,
    CONTINUOUS_FEATURES,
    DISCRETE_FEATURES,
    FEATURE_COLS,
    LOCAL_MODEL_PATH,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_REGISTERED_MODEL_NAME,
    MLFLOW_RUN_NAME,
    MIN_BID,
    MODEL_DIR,
    RANDOM_SEED,
    REPORT_DIR,
    TARGET_CM,
    TARGET_COL,
    BID_STEP,
)
from metrics import evaluate_model, print_model_evaluation
from mlflow_utils import log_training_run
from models import build_logistic_regression_model
from plots_and_reports import (
    print_saved_report_files,
    print_training_data_summary,
    save_evaluation_summary,
    save_feature_summary_files,
    save_performance_plots,
)
from predict import run_bid_optimizer_evaluation
from preprocessing import clean_training_data
from smarthub import io


np.random.seed(RANDOM_SEED)


# =============================================================================
# Load and clean data
# =============================================================================

df = io.load_leads()
print(sorted(df.keys()))

df, cleaning_summary = clean_training_data(df)

print(f"Loaded {cleaning_summary['raw_rows']:,} rows")
print(f"Training rows: {cleaning_summary['training_rows']:,}")


# =============================================================================
# Data summary
# =============================================================================

print()
print("=" * 80)
print("Dataset")
print("=" * 80)
print(f"Rows Before Cleaning : {cleaning_summary['raw_rows']:,}")
print(f"Rows After Cleaning  : {cleaning_summary['training_rows']:,}")
print(
    f"Rows Dropped         : {cleaning_summary['dropped_rows']:,} "
    f"({cleaning_summary['dropped_pct']:.2f}%)"
)
print(f"Continuous Features  : {len(CONTINUOUS_FEATURES):,}")
print(f"Discrete Features    : {len(DISCRETE_FEATURES):,}")
print(f"Categorical Features : {len(CATEGORICAL_FEATURES):,}")

feature_summary_df, feature_counts_df = print_training_data_summary(
    df=df,
    continuous_features=CONTINUOUS_FEATURES,
    discrete_features=DISCRETE_FEATURES,
    categorical_features=CATEGORICAL_FEATURES,
    target_col=TARGET_COL,
)


# =============================================================================
# Split
# =============================================================================

X = df[FEATURE_COLS]
y = df[TARGET_COL]

train_idx, test_idx = train_test_split(
    df.index,
    test_size=0.20,
    random_state=RANDOM_SEED,
)

X_train = X.loc[train_idx]
X_test = X.loc[test_idx]
y_train = y.loc[train_idx]
y_test = y.loc[test_idx]

test_eval_df = df.loc[test_idx].copy()


# =============================================================================
# Train model
# =============================================================================

model = build_logistic_regression_model(random_seed=RANDOM_SEED)
model.fit(X_train, y_train)


# =============================================================================
# Evaluate model
# =============================================================================

pred, pred_class, model_metrics = evaluate_model(
    model=model,
    X_test=X_test,
    y_test=y_test,
)

print_model_evaluation(
    metrics=model_metrics,
    rows_trained=len(df),
    train_rows=len(X_train),
    test_rows=len(X_test),
)


# =============================================================================
# Bid optimization evaluation
# =============================================================================

optimizer_result = run_bid_optimizer_evaluation(
    test_eval_df=test_eval_df,
    model=model,
    target_cm=TARGET_CM,
    min_bid=MIN_BID,
    bid_step=BID_STEP,
)

if optimizer_result is not None:
    optimizer_eval_df, optimizer_summary = optimizer_result
else:
    optimizer_eval_df, optimizer_summary = None, {}


# =============================================================================
# Save report
# =============================================================================

report_dir = REPORT_DIR
Path(report_dir).mkdir(exist_ok=True)

save_feature_summary_files(
    report_dir=report_dir,
    feature_summary_df=feature_summary_df,
    feature_counts_df=feature_counts_df,
)

save_performance_plots(
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

save_evaluation_summary(
    report_dir=report_dir,
    evaluation_summary=evaluation_summary,
    optimizer_eval_df=optimizer_eval_df,
)

print_saved_report_files(
    report_dir=report_dir,
    optimizer_eval_df=optimizer_eval_df,
)


# =============================================================================
# Save model locally
# =============================================================================

Path(MODEL_DIR).mkdir(exist_ok=True)
joblib.dump(model, LOCAL_MODEL_PATH)
print(f"Saved model to {LOCAL_MODEL_PATH}")


# =============================================================================
# MLflow
# =============================================================================

log_training_run(
    model=model,
    feature_cols=FEATURE_COLS,
    metrics=model_metrics,
    report_dir=report_dir,
    experiment_name=MLFLOW_EXPERIMENT_NAME,
    run_name=MLFLOW_RUN_NAME,
    registered_model_name=MLFLOW_REGISTERED_MODEL_NAME,
)

print("Done.")
