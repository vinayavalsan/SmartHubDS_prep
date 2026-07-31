"""Model evaluation metrics for SmartHub training.

This module computes, stores, and logs classifier and calibration metrics.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    fbeta_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

from smarthub.core.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class ModelMetrics:
    """Store classifier, calibration, and predicted win-rate metrics."""

    roc_auc: float
    pr_auc: float
    log_loss: float
    brier_score: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    f2: float
    observed_win_rate: float
    predicted_current_bid_win_rate: float
    calibration_error: float
    predicted_recommended_bid_win_rate: float = float("nan")

    def to_dict(self) -> dict:
        """Execute to dict.

        Returns
        -------
        dict
            Serializable dictionary representation.
        """
        return asdict(self)

    def with_recommended_win_rate(self, value: float) -> "ModelMetrics":
        """Return a copy with the optimizer win-rate estimate.

        Inputs
        ------
        value : Any
            Value to process.

        Returns
        -------
        ModelMetrics
            New metrics object with the supplied win rate.
        """
        values = self.to_dict()
        values["predicted_recommended_bid_win_rate"] = float(value)
        return ModelMetrics(**values)


def evaluate_model(model, X_test, y_test):
    """Evaluate a fitted classifier on held-out data.

    Inputs
    ------
    model : Any
        Fitted model or model pipeline.
    X_test : pandas.DataFrame
        Held-out feature matrix.
    y_test : pandas.Series | numpy.ndarray
        Held-out target values.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray, ModelMetrics]
        Predicted probabilities, predicted classes, and metrics, in that order.
    """
    pred = model.predict_proba(X_test)[:, 1]
    pred_class = (pred >= 0.5).astype(int)
    observed = float(y_test.mean())
    predicted = float(pred.mean())
    result = ModelMetrics(
        roc_auc=float(roc_auc_score(y_test, pred)),
        pr_auc=float(average_precision_score(y_test, pred)),
        log_loss=float(log_loss(y_test, pred)),
        brier_score=float(brier_score_loss(y_test, pred)),
        accuracy=float(accuracy_score(y_test, pred_class)),
        precision=float(precision_score(y_test, pred_class, zero_division=0)),
        recall=float(recall_score(y_test, pred_class, zero_division=0)),
        f1=float(f1_score(y_test, pred_class, zero_division=0)),
        f2=float(fbeta_score(y_test, pred_class, beta=2, zero_division=0)),
        observed_win_rate=observed,
        predicted_current_bid_win_rate=predicted,
        calibration_error=abs(observed - predicted),
    )
    return pred, pred_class, result


def log_model_evaluation(
    model_metrics: ModelMetrics,
    rows_trained: int,
    train_rows: int,
    test_rows: int,
):
    """Log classifier and business-facing evaluation metrics.

    Inputs
    ------
    model_metrics : ModelMetrics
        Computed model metrics.
    rows_trained : int
        Total prepared training rows.
    train_rows : int
        Number of model-fitting rows.
    test_rows : int
        Number of held-out rows.
    """
    logger.info("Model Evaluation")
    logger.info("  Rows trained                       : %s", f"{rows_trained:,}")
    logger.info(
        "  Train / test rows                  : %s / %s",
        f"{train_rows:,}",
        f"{test_rows:,}",
    )
    logger.info("  ROC AUC                            : %.4f", model_metrics.roc_auc)
    logger.info("  PR AUC                             : %.4f", model_metrics.pr_auc)
    logger.info("  Log loss                           : %.4f", model_metrics.log_loss)
    logger.info(
        "  Brier score                        : %.4f",
        model_metrics.brier_score,
    )
    logger.info(
        "  Accuracy / precision / recall / F1 / F2: %.4f / %.4f / %.4f / %.4f / %.4f",
        model_metrics.accuracy,
        model_metrics.precision,
        model_metrics.recall,
        model_metrics.f1,
        model_metrics.f2,
    )
    logger.info(
        "  Calibration error                  : %.4f",
        model_metrics.calibration_error,
    )
    logger.info("Business Metrics")
    logger.info(
        "  Observed win rate                  : %.4f",
        model_metrics.observed_win_rate,
    )
    logger.info(
        "  Predicted win rate, current bids   : %.4f",
        model_metrics.predicted_current_bid_win_rate,
    )
    logger.info(
        "  Predicted win rate, recommended    : %.4f",
        model_metrics.predicted_recommended_bid_win_rate,
    )
