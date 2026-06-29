"""
Evaluation metrics for Anton model training.
"""

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_model(model, X_test, y_test):
    """Evaluate a trained model and return predictions and metrics."""
    pred = model.predict_proba(X_test)[:, 1]
    pred_class = (pred >= 0.5).astype(int)

    metrics = {
        "roc_auc": float(roc_auc_score(y_test, pred)),
        "pr_auc": float(average_precision_score(y_test, pred)),
        "log_loss": float(log_loss(y_test, pred)),
        "brier_score": float(brier_score_loss(y_test, pred)),
        "accuracy": float(accuracy_score(y_test, pred_class)),
        "precision": float(precision_score(y_test, pred_class, zero_division=0)),
        "recall": float(recall_score(y_test, pred_class, zero_division=0)),
        "f1": float(f1_score(y_test, pred_class, zero_division=0)),
        "observed_win_rate": float(y_test.mean()),
        "predicted_current_bid_win_rate": float(pred.mean()),
    }

    metrics["calibration_error"] = abs(
        metrics["observed_win_rate"] - metrics["predicted_current_bid_win_rate"]
    )

    return pred, pred_class, metrics


def print_model_evaluation(metrics, rows_trained, train_rows, test_rows):
    """Print model metrics to the console."""
    print()
    print("=" * 80)
    print("Anton Model Evaluation")
    print("=" * 80)
    print(f"Rows Trained:                       {rows_trained:,}")
    print(f"Train Rows:                         {train_rows:,}")
    print(f"Test Rows:                          {test_rows:,}")
    print()
    print(f"ROC AUC:                            {metrics['roc_auc']:.4f}")
    print(f"PR AUC:                             {metrics['pr_auc']:.4f}")
    print(f"Log Loss:                           {metrics['log_loss']:.4f}")
    print(f"Brier Score:                        {metrics['brier_score']:.4f}")
    print()
    print(f"Accuracy (TP+TN)/N:                 {metrics['accuracy']:.4f}")
    print(f"Precision TP/(TP+FP):               {metrics['precision']:.4f}")
    print(f"Recall TP/(TP+FN):                  {metrics['recall']:.4f}")
    print(f"F1 Score (2*Prec*Recl)/(Prec+Recl): {metrics['f1']:.4f}")
    print(f"Calibration Error:                  {metrics['calibration_error']:.4f}")
    print()
    print("=" * 80)
    print("Business Metrics")
    print("=" * 80)
    print(f"Observed Win Rate:                  " f"{metrics['observed_win_rate']:.4f}")
    print(
        f"Predicted Win Rate (Current Bids):  "
        f"{metrics['predicted_current_bid_win_rate']:.4f}"
    )
    print(
        f"Predicted Win Rate (Recommended):   "
        f"{metrics['predicted_recommended_bid_win_rate']:.4f}"
    )
