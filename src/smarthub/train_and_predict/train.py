"""Train Anton's win-probability model: ``P(won | bid, lead features)``.

STEP 3 of the pipeline. Consumes the leakage-safe training table from
``smarthub.feature_engineering`` (STEP 2), trains, evaluates, runs the offline
bid-optimizer evaluation, saves the model + reports, and (optionally) logs the
run to MLflow.

Run directly:

    python -m smarthub.train_and_predict.train --lead-type-id 6   # auto
    python -m smarthub.train_and_predict.train --lead-type-id 1   # home

or call ``run_training(...)`` (used by the Prefect ``train_flow``).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

from . import (
    config,
    metrics,
    models,
    plots_and_reports,
    predict,
    preprocessing,
    registry,
)

logger = logging.getLogger("smarthub.train_and_predict.train")


def _split_train_test(frame, test_size):
    """Time-ordered split: the most recent ``test_size`` fraction is the test set.

    ``prepare_training_data`` sorts by ``created_at`` when available, so holding
    out the tail avoids training on the future (no look-ahead leakage).
    """
    n = len(frame)
    n_test = max(1, int(round(n * test_size)))
    train_df = frame.iloc[: n - n_test]
    test_df = frame.iloc[n - n_test :]
    return train_df, test_df


def run_training(
    lead_type_id=6,
    lead_type_name=None,
    version=None,
    register_mlflow=True,
    log=None,
):
    """Train + evaluate + save a model for one lead type. Returns a result dict.

    ``log`` is an optional logger (e.g. Prefect's ``get_run_logger()``); step
    messages are sent there so the whole run is visible in the caller's logs.
    """
    log = log or logger
    lead_type_name = lead_type_name or config.lead_type_name(lead_type_id)
    np.random.seed(config.RANDOM_SEED)

    log.info(
        "Loading training table for lead_type=%s (%s)", lead_type_name, lead_type_id
    )
    frame, numeric, categorical, prep_summary = preprocessing.prepare_training_data(
        lead_type_id, lead_type_name, version
    )
    feature_cols = numeric + categorical
    log.info(
        "Prepared %s rows (%s dropped); win_rate=%s; %s numeric + %s categorical "
        "features; missing columns=%s",
        prep_summary["training_rows"],
        prep_summary["dropped_rows"],
        prep_summary["win_rate"],
        len(numeric),
        len(categorical),
        prep_summary["missing_feature_columns"] or "none",
    )
    if prep_summary["training_rows"] < 50:
        raise ValueError(
            f"Only {prep_summary['training_rows']} training rows for "
            f"{lead_type_name}; need more data before training."
        )
    # Both wins and losses must be present (raises a clear message otherwise).
    preprocessing.assert_trainable(frame, lead_type_name)

    train_df, test_df = _split_train_test(frame, config.TEST_SIZE)

    # Drop no-signal (constant) feature columns from the model input.
    if config.DROP_ZERO_VARIANCE:
        numeric, categorical, dropped = preprocessing.drop_zero_variance(
            train_df, numeric, categorical
        )
        if dropped:
            log.info("Dropped %s zero-variance features: %s", len(dropped), dropped)
        feature_cols = numeric + categorical

    X_train, y_train = train_df[feature_cols], train_df[config.TARGET_COL]
    X_test, y_test = test_df[feature_cols], test_df[config.TARGET_COL]
    log.info("Split: %s train / %s test rows (time-ordered)", len(X_train), len(X_test))

    model_type = config.model_type()
    model_params = (
        config.LIGHTGBM_PARAMS
        if model_type == "lightgbm"
        else config.LOGISTIC_REGRESSION_PARAMS
    )
    calibrate = config.CALIBRATE
    log.info("Fitting %s (calibrate=%s)", model_type, calibrate)
    try:
        model = models.build_model(
            model_type, numeric, categorical, model_params, calibrate=calibrate
        )
        model.fit(X_train, y_train)
    except Exception as exc:  # noqa: BLE001 - fall back if calibration errors
        if not calibrate:
            raise
        log.warning("Calibration failed (%s); refitting without calibration", exc)
        calibrate = False
        model = models.build_model(
            model_type, numeric, categorical, model_params, calibrate=False
        )
        model.fit(X_train, y_train)

    log.info("Evaluating on held-out test set")
    pred, pred_class, model_metrics = metrics.evaluate_model(model, X_test, y_test)
    log.info(
        "ROC AUC=%.4f  PR AUC=%.4f  logloss=%.4f  brier=%.4f",
        model_metrics["roc_auc"],
        model_metrics["pr_auc"],
        model_metrics["log_loss"],
        model_metrics["brier_score"],
    )

    # Fixed once so the challenger and the currently-serving model (re-evaluated
    # below) are scored under the identical objective — a fair comparison.
    target_cm = config.target_cm_value()
    min_bid = config.min_bid_value()

    log.info("Running offline bid-optimizer evaluation")
    optimizer_result = predict.run_bid_optimizer_evaluation(
        test_eval_df=test_df,
        model=model,
        feature_cols=feature_cols,
        target_cm=target_cm,
        min_bid=min_bid,
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
        rows_trained=len(frame),
        train_rows=len(X_train),
        test_rows=len(X_test),
    )

    # --- Reports -------------------------------------------------------------
    report_dir = config.report_dir(lead_type_name)
    Path(report_dir).mkdir(parents=True, exist_ok=True)
    feature_summary_df, feature_counts_df = (
        plots_and_reports.print_training_data_summary(
            df=frame,
            continuous_features=[c for c in numeric if c in ("bid", "age")],
            discrete_features=[c for c in numeric if c not in ("bid", "age")],
            categorical_features=categorical,
            target_col=config.TARGET_COL,
        )
    )
    plots_and_reports.save_feature_summary_files(
        report_dir, feature_summary_df, feature_counts_df
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
    # Lineage: exactly which data this model was trained on.
    lineage = {
        "model_type": model_type,
        "calibrated": bool(calibrate),
        "training_table_version": prep_summary.get("training_table_version"),
        "data_min_created_at": prep_summary.get("data_min_created_at"),
        "data_max_created_at": prep_summary.get("data_max_created_at"),
        "source_row_count": prep_summary.get("source_row_count"),
    }
    log.info(
        "Lineage: model=%s trained on table version %s "
        "(data %s -> %s, %s source rows)",
        model_type,
        lineage["training_table_version"],
        lineage["data_min_created_at"],
        lineage["data_max_created_at"],
        lineage["source_row_count"],
    )
    evaluation_summary = {
        "lead_type_id": lead_type_id,
        "lead_type_name": lead_type_name,
        "rows_trained": int(len(frame)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        **lineage,
        **model_metrics,
        "bid_optimizer": optimizer_summary,
    }
    plots_and_reports.save_evaluation_summary(
        report_dir, evaluation_summary, optimizer_eval_df
    )

    # --- Promotion gate --------------------------------------------------------
    # Re-score the model CURRENTLY SERVING traffic on this run's exact
    # held-out test set, so the comparison is apples-to-apples instead of
    # diffing metrics computed on two different data snapshots. Any failure
    # here (e.g. it predates a feature-schema change) is logged and treated as
    # "nothing comparable currently serving" rather than blocking training.
    serving_metrics, serving_optimizer_summary = _evaluate_currently_serving_model(
        lead_type_name, test_df, y_test, target_cm, min_bid, log
    )

    decision = registry.decide_promotion(
        challenger_metrics=model_metrics,
        challenger_optimizer=optimizer_summary,
        currently_serving_metrics=serving_metrics,
        currently_serving_optimizer=serving_optimizer_summary,
        min_roc_auc_regression=config.PROMOTION_MIN_ROC_AUC_REGRESSION,
        min_profit_ratio=config.PROMOTION_MIN_PROFIT_RATIO,
    )
    log.info(
        "Promotion decision: %s — %s",
        "PROMOTE" if decision.promote else "HOLD (keep serving current model)",
        decision.reason,
    )

    # --- Save model (always versioned; never overwrites a prior version) -----
    manifest = registry.save_version(
        model,
        lead_type_name,
        feature_cols=feature_cols,
        metrics=model_metrics,
        optimizer_summary=optimizer_summary,
        lineage=lineage,
        model_params=model_params,
    )
    model_path = manifest["model_path"]
    log.info("Saved model version %s -> %s", manifest["version"], model_path)

    if decision.promote:
        registry.promote(lead_type_name, manifest["version"], reason=decision.reason)
        log.info(
            "Promoted %s to currently-serving for '%s'.",
            manifest["version"],
            lead_type_name,
        )
    else:
        log.warning(
            "Challenger %s NOT promoted for '%s'; currently-serving model "
            "unchanged.",
            manifest["version"],
            lead_type_name,
        )

    # --- MLflow --------------------------------------------------------------
    if register_mlflow:
        try:
            from . import mlflow_utils

            mlflow_utils.log_training_run(
                model=model,
                model_params=model_params,
                feature_cols=feature_cols,
                metrics=model_metrics,
                report_dir=report_dir,
                experiment_name=config.MLFLOW_EXPERIMENT_NAME,
                run_name=f"{config.MLFLOW_RUN_NAME}_{lead_type_name}",
                registered_model_name=config.MLFLOW_REGISTERED_MODEL_NAME,
                extra_params={
                    "lead_type_name": lead_type_name,
                    "model_version": manifest["version"],
                    "promoted": decision.promote,
                    "promotion_reason": decision.reason,
                    **lineage,
                },
            )
            log.info(
                "Logged run to MLflow experiment '%s'",
                config.MLFLOW_EXPERIMENT_NAME,
            )
        except Exception as exc:  # noqa: BLE001 - MLflow must not fail the run
            log.warning("MLflow logging skipped: %s", exc)

    return {
        "lead_type_id": lead_type_id,
        "lead_type_name": lead_type_name,
        "model_path": model_path,
        "model_version": manifest["version"],
        "promoted": decision.promote,
        "promotion_reason": decision.reason,
        "promotion_comparison": decision.comparison,
        "report_dir": report_dir,
        "metrics": model_metrics,
        "optimizer_summary": optimizer_summary,
        "prep_summary": prep_summary,
        "lineage": lineage,
        "feature_cols": list(feature_cols),
    }


def _evaluate_currently_serving_model(
    lead_type_name, test_df, y_test, target_cm, min_bid, log
):
    """Score the model currently serving traffic on this run's held-out test set.

    Returns ``(metrics, optimizer_summary)``, or ``(None, None)`` if nothing
    is serving yet, or if it can't be loaded/scored (e.g. its feature schema
    has since changed, or its registry pointer is corrupt) — in which case
    ``decide_promotion`` treats this as the bootstrap case and promotes the
    challenger. The ``registry.load_currently_serving_model`` call is
    deliberately INSIDE the try/except below (not before it): a corrupt
    ``current.json`` (e.g. left truncated by a process killed mid-write) once
    raised a raw ``JSONDecodeError`` from there that was outside the
    try/except, crashing the entire training run instead of just skipping
    this comparison — this is a "nice to have" re-score, not something that
    should ever be able to block training a new model.
    """
    try:
        serving_model, serving_manifest = registry.load_currently_serving_model(
            lead_type_name
        )
        if serving_model is None:
            log.info(
                "Nothing currently serving for '%s' (first model).", lead_type_name
            )
            return None, None

        serving_feature_cols = serving_manifest.get("feature_cols") or []
        X_serving = test_df[serving_feature_cols]
        _, _, serving_metrics = metrics.evaluate_model(serving_model, X_serving, y_test)
        serving_optimizer_result = predict.run_bid_optimizer_evaluation(
            test_eval_df=test_df,
            model=serving_model,
            feature_cols=serving_feature_cols,
            target_cm=target_cm,
            min_bid=min_bid,
            bid_step=config.BID_STEP,
        )
        serving_optimizer_summary = (
            serving_optimizer_result[1] if serving_optimizer_result else {}
        )
        log.info(
            "Currently-serving model (%s) re-scored on this test set: ROC AUC=%.4f",
            serving_manifest.get("version"),
            serving_metrics["roc_auc"],
        )
        return serving_metrics, serving_optimizer_summary
    except Exception as exc:  # noqa: BLE001 - corrupt pointer, bad schema, etc.
        log.warning(
            "Could not load/score the currently-serving model for '%s' on "
            "the current test set (%s); treating as nothing comparable "
            "currently serving.",
            lead_type_name,
            exc,
        )
        return None, None


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Train the Anton win-probability model."
    )
    parser.add_argument(
        "--lead-type-id", type=int, default=6, help="6=auto (default), 1=home"
    )
    parser.add_argument(
        "--version", default=None, help="Training-table version (default: latest)"
    )
    parser.add_argument(
        "--no-mlflow", action="store_true", help="Skip MLflow logging/registration"
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    run_training(
        lead_type_id=args.lead_type_id,
        register_mlflow=not args.no_mlflow,
    )
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
