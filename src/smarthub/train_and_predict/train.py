"""Training workflow for Anton's win-probability model.

This module trains, evaluates, versions, and optionally registers models.
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from smarthub.core.logging_utils import get_logger

from . import (
    config,
    metrics,
    models,
    optimizer_evaluation,
    plots_and_reports,
    preprocessing,
    registry,
    training_artifacts,
)

logger = get_logger(__name__)


def split_training_data(
    frame: pd.DataFrame,
    target_column: str,
    split_settings: dict[str, Any],
    random_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split prepared data using the configured strategy.

    Inputs
    ------
    frame : pandas.DataFrame
        Input dataframe.
    target_column : str
        Target column used for optional stratification.
    split_settings : dict[str, Any]
        Selected split strategy and options.
    random_seed : int
        Seed for reproducible random splitting.

    Returns
    -------
    tuple[pandas.DataFrame, pandas.DataFrame]
        Training dataframe followed by test dataframe.

    Raises
    ------
    ValueError
        If split settings are invalid or produce an empty dataset.
    """
    strategy = str(split_settings["strategy"]).strip().lower()
    test_size = float(split_settings["test_size"])

    if not 0.0 < test_size < 1.0:
        raise ValueError("Split test_size must be between 0 and 1.")

    if strategy == "time":
        n_test = max(1, int(round(len(frame) * test_size)))
        split_index = len(frame) - n_test
        train_df = frame.iloc[:split_index].copy()
        test_df = frame.iloc[split_index:].copy()
    elif strategy == "random":
        use_stratification = split_settings["stratify"]
        stratify = None

        if use_stratification:
            if target_column not in frame.columns:
                raise ValueError(
                    f"Cannot stratify because target column "
                    f"{target_column!r} is missing."
                )
            stratify = frame[target_column]

        train_df, test_df = train_test_split(
            frame,
            test_size=test_size,
            random_state=random_seed,
            shuffle=True,
            stratify=stratify,
        )
        train_df = train_df.copy()
        test_df = test_df.copy()
    else:
        raise ValueError(f"Unsupported split strategy: {strategy!r}.")

    if train_df.empty or test_df.empty:
        raise ValueError(
            "Configured train/test split produced an empty dataset. "
            "Adjust split.test_size."
        )

    return train_df, test_df


def _optimizer_metrics_for_mlflow(
    optimizer_summary: dict | None,
) -> dict[str, float | int]:
    """Select the retained numeric optimizer metrics for MLflow.

    Inputs
    ------
    optimizer_summary : dict | None
        Offline optimizer summary metrics.

    Returns
    -------
    dict[str, float | int]
        Flat numeric optimizer metrics.
    """
    if not optimizer_summary:
        return {}

    selected_keys = {
        "optimizer_rows",
        "target_cm",
        "current_bid_total_expected_profit",
        "recommended_bid_total_expected_profit",
        "expected_profit_lift_total",
        "expected_profit_lift_pct",
        "avg_current_bid_predicted_win_rate",
        "avg_recommended_bid_predicted_win_rate",
        "avg_bid_change",
        "median_bid_change",
        "bid_increase_pct",
        "bid_decrease_pct",
        "bid_unchanged_pct",
        "avg_recommended_bid_cm_if_won",
    }
    return {
        f"optimizer_{key}": value
        for key, value in optimizer_summary.items()
        if key in selected_keys and isinstance(value, (int, float))
    }


def run_training(
    lead_type_id: int,
    version: str | None = None,
    register_mlflow: bool = True,
) -> dict[str, Any]:
    """Train, evaluate, version, and optionally register one model.

    Inputs
    ------
    lead_type_id : int
        SmartHub lead type identifier.
    version : str | None
        Optional training-table or model version identifier.
    register_mlflow : bool
        Whether to log and register the run in MLflow.

    Returns
    -------
    dict[str, Any]
        Training outputs, metrics, lineage, and promotion information.

    Raises
    ------
    ValueError
        If the prepared data cannot support model training.
    """
    training_config = config.load_training_config()
    lead_type_name = config.lead_type_name(lead_type_id)
    np.random.seed(training_config.random_seed)

    logger.info(
        "Loading training table for lead_type=%s (%s)",
        lead_type_name,
        lead_type_id,
    )
    frame, numeric, categorical, prep_summary = preprocessing.prepare_training_data(
        lead_type_id,
        lead_type_name,
        version,
    )
    feature_cols = numeric + categorical

    logger.info("Dataset Prepared")
    logger.info(
        "  Training rows                         : %s",
        f"{prep_summary['training_rows']:,}",
    )
    logger.info(
        "  Dropped rows                          : %s",
        f"{prep_summary['dropped_rows']:,}",
    )
    logger.info(
        "  Win rate                              : %.4f",
        prep_summary["win_rate"] or 0.0,
    )
    logger.info(
        "  Feature count                         : %s numeric, %s categorical",
        len(numeric),
        len(categorical),
    )
    logger.info(
        "  Missing feature columns               : %s",
        prep_summary["missing_feature_columns"] or "none",
    )

    if prep_summary["training_rows"] < 50:
        raise ValueError(
            f"Only {prep_summary['training_rows']} training rows for "
            f"{lead_type_name}; need more data before training."
        )

    preprocessing.assert_trainable(frame, lead_type_name)

    split_settings = training_config.split
    train_df, test_df = split_training_data(
        frame=frame,
        target_column=config.TARGET_COL,
        split_settings=split_settings,
        random_seed=training_config.random_seed,
    )

    if training_config.drop_zero_variance:
        numeric, categorical, dropped = preprocessing.drop_zero_variance(
            train_df,
            numeric,
            categorical,
        )
        if dropped:
            logger.info("Cleaning Data")
            logger.info(
                "  Dropped zero-variance features        : %s",
                f"{len(dropped):,}",
            )
            logger.info(
                "  Zero-variance feature names           : %s",
                ", ".join(dropped),
            )
        feature_cols = numeric + categorical

    logger.info("Feature columns: %s", feature_cols)

    feature_summary_df, feature_counts_df = (
        plots_and_reports.build_training_data_summary(
            df=frame,
            continuous_features=[
                column for column in numeric if column in ("bid", "age")
            ],
            discrete_features=[
                column for column in numeric if column not in ("bid", "age")
            ],
            categorical_features=categorical,
        )
    )
    plots_and_reports.log_training_data_summary(
        df=frame,
        feature_summary_df=feature_summary_df,
        feature_counts_df=feature_counts_df,
        target_col=config.TARGET_COL,
    )

    X_train = train_df[feature_cols]
    y_train = train_df[config.TARGET_COL]
    X_test = test_df[feature_cols]
    y_test = test_df[config.TARGET_COL]

    logger.info("Train/Test Split")
    logger.info(
        "  Train rows                            : %s",
        f"{len(X_train):,}",
    )
    logger.info(
        "  Test rows                             : %s",
        f"{len(X_test):,}",
    )
    logger.info(
        "  Split method                          : %s",
        split_settings["strategy"],
    )
    logger.info(
        "  Test fraction                         : %.2f",
        split_settings["test_size"],
    )
    if split_settings["strategy"] == "random":
        logger.info(
            "  Stratified                            : %s",
            split_settings["stratify"],
        )
    logger.info(
        "  Train date range                      : %s → %s",
        train_df["created_at"].min(),
        train_df["created_at"].max(),
    )
    logger.info(
        "  Test date range                       : %s → %s",
        test_df["created_at"].min(),
        test_df["created_at"].max(),
    )
    logger.info("  Test rows by weekday")
    for day, count in (
        test_df["created_at"].dt.day_name().value_counts().sort_index().items()
    ):
        logger.info("    %-10s : %s", day, f"{count:,}")

    model_type = training_config.model_type
    model_params = training_config.model_parameters
    calibration_enabled = training_config.calibration_enabled

    logger.info("Training Model")
    logger.info("  Model type                            : %s", model_type)
    logger.info(
        "  Calibration enabled                   : %s",
        calibration_enabled,
    )

    model = models.build_model(
        model_type,
        numeric,
        categorical,
        model_params,
        calibration_enabled=calibration_enabled,
        calibration_method=training_config.calibration_method,
        calibration_cv=training_config.calibration_cv,
    )
    model.fit(X_train, y_train)

    logger.info("Evaluating Model")
    pred, pred_class, model_metrics = metrics.evaluate_model(
        model,
        X_test,
        y_test,
    )

    evaluation_df = test_df.copy()
    evaluation_df["predicted_win_probability"] = pred
    evaluation_df["predicted_class"] = pred_class

    optimizer_config = training_config.optimizer
    target_cm = optimizer_config.target_cm
    min_bid = optimizer_config.minimum_bid
    bid_step = optimizer_config.bid_step
    optimizer_chunk_size = optimizer_config.chunk_size

    optimizer_result = optimizer_evaluation.run_bid_optimizer_evaluation(
        test_eval_df=test_df,
        model=model,
        feature_cols=feature_cols,
        target_cm=target_cm,
        min_bid=min_bid,
        bid_step=bid_step,
        chunk_size=optimizer_chunk_size,
    )

    if optimizer_result is not None:
        optimizer_eval_df, optimizer_summary = optimizer_result
        model_metrics = model_metrics.with_recommended_win_rate(
            optimizer_summary.avg_recommended_bid_predicted_win_rate
        )
    else:
        optimizer_eval_df = None
        optimizer_summary = None

    optimizer_summary_dict = (
        optimizer_summary.to_dict() if optimizer_summary is not None else {}
    )
    optimizer_mlflow_metrics = _optimizer_metrics_for_mlflow(optimizer_summary_dict)

    metrics.log_model_evaluation(
        model_metrics=model_metrics,
        rows_trained=len(frame),
        train_rows=len(X_train),
        test_rows=len(X_test),
    )

    logger.info("Saving Reports")
    report_dir = training_config.report_dir(lead_type_name)
    Path(report_dir).mkdir(parents=True, exist_ok=True)

    plots_and_reports.save_feature_summary_files(
        report_dir,
        feature_summary_df,
        feature_counts_df,
    )
    plots_and_reports.save_performance_plots(
        report_dir=report_dir,
        y_test=y_test,
        pred=pred,
        pred_class=pred_class,
        roc_auc=model_metrics.roc_auc,
        pr_auc=model_metrics.pr_auc,
        accuracy=model_metrics.accuracy,
        precision=model_metrics.precision,
        recall=model_metrics.recall,
        f1=model_metrics.f1,
        f2=model_metrics.f2,
        optimizer_eval_df=optimizer_eval_df,
    )

    lineage = {
        "model_type": model_type,
        "calibrated": bool(calibration_enabled),
        "split_strategy": split_settings["strategy"],
        "split_test_size": split_settings["test_size"],
        "training_table_version": prep_summary["training_table_version"],
        "data_min_created_at": prep_summary["data_min_created_at"],
        "data_max_created_at": prep_summary["data_max_created_at"],
        "source_row_count": prep_summary["source_row_count"],
    }

    logger.info(
        "Lineage: model=%s trained on table version %s "
        "(data %s → %s, %s source rows)",
        model_type,
        lineage["training_table_version"],
        lineage["data_min_created_at"],
        lineage["data_max_created_at"],
        lineage["source_row_count"],
    )

    test_set_id = training_artifacts.build_test_set_id(
        test_df,
        training_table_version=prep_summary["training_table_version"],
        split_settings=split_settings,
    )
    lineage["test_set_id"] = test_set_id

    evaluation_summary = {
        "lead_type_id": lead_type_id,
        "lead_type_name": lead_type_name,
        "rows_trained": int(len(frame)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        **lineage,
        **model_metrics.to_dict(),
        "bid_optimizer": optimizer_summary_dict,
    }

    plots_and_reports.save_evaluation_summary(
        report_dir,
        evaluation_summary,
        optimizer_eval_df,
    )
    plots_and_reports.log_saved_report_files(
        report_dir,
        optimizer_eval_df,
    )

    promotion_mode = training_config.promotion_mode
    if promotion_mode == "disabled":
        decision = None
        eligibility_status = "not_evaluated"
        promotion_status = "not_evaluated"
        promotion_reason = "Promotion evaluation disabled."
        promotion_comparison = {}
        logger.info("Promotion Evaluation")
        logger.info("  Mode                                  : %s", promotion_mode)
        logger.info("  Status                                : SKIPPED")
        logger.info("  Reason                                : %s", promotion_reason)
    else:
        serving_metrics, serving_optimizer_summary = _evaluate_currently_serving_model(
            lead_type_name,
            test_df,
            y_test,
            target_cm,
            min_bid,
            bid_step,
            optimizer_chunk_size,
        )
        decision = registry.decide_promotion(
            challenger_metrics=model_metrics.to_dict(),
            challenger_optimizer=optimizer_summary_dict,
            currently_serving_metrics=serving_metrics,
            currently_serving_optimizer=serving_optimizer_summary,
            max_log_loss_regression=(training_config.promotion_max_log_loss_regression),
            min_profit_ratio=training_config.promotion_min_profit_ratio,
            max_absolute_profit_loss_tolerance=(
                training_config.promotion_max_absolute_profit_loss_tolerance
            ),
            target_cm=target_cm,
            max_log_loss=training_config.promotion_max_log_loss,
            min_expected_profit=training_config.promotion_min_expected_profit,
        )
        eligibility_status = "eligible" if decision.promote else "not_eligible"
        if promotion_mode == "automatic":
            promotion_status = "promoted" if decision.promote else "rejected"
        else:
            promotion_status = (
                "awaiting_manual_promotion" if decision.promote else "rejected"
            )
        promotion_reason = decision.reason
        promotion_comparison = decision.comparison
        logger.info("Promotion Evaluation")
        logger.info("  Mode                                  : %s", promotion_mode)
        logger.info(
            "  Policy recommendation                 : %s",
            "ELIGIBLE" if decision.promote else "NOT ELIGIBLE",
        )
        logger.info("  Reason                                : %s", decision.reason)

    manifest = registry.save_version(
        model,
        lead_type_name,
        feature_cols=feature_cols,
        metrics=model_metrics.to_dict(),
        optimizer_summary=optimizer_summary_dict,
        lineage=lineage,
        model_params=model_params,
        training_config=training_config.as_dict(),
        promotion_mode=promotion_mode,
        eligibility_status=eligibility_status,
        promotion_status=promotion_status,
        promotion_decision_reason=promotion_reason,
        promotion_comparison=promotion_comparison,
    )

    model_path = manifest["model_path"]

    artifact_metadata = {
        "training_run_id": manifest["training_run_id"],
        "lead_type_id": lead_type_id,
        "lead_type_name": lead_type_name,
        "test_set_id": test_set_id,
        "training_table_version": prep_summary["training_table_version"],
        "split_settings": split_settings,
        "random_seed": training_config.random_seed,
        "model_type": model_type,
        "model_parameters": model_params,
        "calibration": {
            "enabled": calibration_enabled,
            "method": training_config.calibration_method,
            "cv": training_config.calibration_cv,
        },
        "feature_cols": list(feature_cols),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "optimizer": optimizer_config.as_dict(),
        "lineage": lineage,
    }

    manifest = registry.update_manifest(
        lead_type_name,
        manifest["training_run_id"],
        test_set_id=test_set_id,
    )

    comparison_temp_dir = None
    comparison_artifacts = {}
    if training_config.comparison_artifacts:
        if not register_mlflow:
            logger.warning(
                "Model-comparison artifacts were requested, but MLflow logging "
                "is disabled. The comparison artifacts will not be retained."
            )
        elif optimizer_eval_df is None:
            logger.warning(
                "Model-comparison artifacts were requested, but optimizer "
                "results are unavailable. No comparison artifacts were created."
            )
        else:
            comparison_temp_dir = tempfile.TemporaryDirectory(
                prefix="smarthub_model_comparison_"
            )
            comparison_artifacts = training_artifacts.save_comparison_artifacts(
                output_dir=comparison_temp_dir.name,
                evaluation_df=evaluation_df,
                optimizer_df=optimizer_eval_df,
                metadata=artifact_metadata,
            )

    logger.info(
        "Saved training run %s → %s",
        manifest["training_run_id"],
        model_path,
    )

    promoted = False
    if promotion_mode == "automatic" and decision is not None and decision.promote:
        registry.promote(
            lead_type_name,
            manifest["training_run_id"],
            reason=decision.reason,
        )
        promoted = True
        manifest = registry.load_manifest(lead_type_name, manifest["training_run_id"])
        promotion_status = "promoted"
        logger.info(
            "Automatically promoted %s to currently-serving for '%s'.",
            manifest["production_model_version"],
            lead_type_name,
        )
    elif promotion_mode == "manual":
        logger.info(
            "Training run %s was saved with status %s. "
            "Currently-serving model remains unchanged.",
            manifest["training_run_id"],
            promotion_status,
        )
    elif promotion_mode == "disabled":
        logger.info(
            "Training run %s was saved; promotion evaluation and execution "
            "were skipped.",
            manifest["training_run_id"],
        )
    else:
        logger.warning(
            "Training run %s is not eligible for automatic promotion; "
            "currently-serving model remains unchanged.",
            manifest["training_run_id"],
        )

    if register_mlflow:
        try:
            from . import mlflow_utils

            experiment_name = (
                f"{training_config.mlflow_experiment_name}_{lead_type_name}"
            )

            mlflow_metadata = mlflow_utils.log_training_run(
                model=model,
                model_params=model_params,
                feature_cols=feature_cols,
                metrics=model_metrics.to_dict(),
                optimizer_metrics=optimizer_mlflow_metrics,
                report_dir=report_dir,
                comparison_artifact_dir=comparison_artifacts.get("artifact_dir"),
                tracking_db_path=training_config.mlflow_tracking_db_path,
                artifact_root=training_config.mlflow_artifact_root,
                experiment_name=experiment_name,
                run_name=manifest["training_run_id"],
                training_config_path=Path(
                    training_config.raw["resolved"]["config_path"]
                ),
                extra_params={
                    "lead_type_name": lead_type_name,
                    "training_run_id": manifest["training_run_id"],
                    "promotion_mode": promotion_mode,
                    **lineage,
                },
                extra_tags={
                    "eligibility_status": eligibility_status,
                    "promotion_status": promotion_status,
                    "promoted": promoted,
                    "production_model_version": manifest.get(
                        "production_model_version"
                    ),
                    "model_version": manifest.get("production_model_version"),
                    "promotion_reason": promotion_reason,
                },
            )
            manifest = registry.update_manifest(
                lead_type_name,
                manifest["training_run_id"],
                **mlflow_metadata,
            )
            if comparison_artifacts:
                manifest = registry.update_manifest(
                    lead_type_name,
                    manifest["training_run_id"],
                    comparison_artifact_path="comparison",
                )
            if promoted:
                promotion_mlflow_metadata = mlflow_utils.promote_training_run(
                    training_run_id=manifest["training_run_id"],
                    lead_type_name=lead_type_name,
                    production_model_version=manifest["production_model_version"],
                    reason=promotion_reason,
                    tracking_db_path=training_config.mlflow_tracking_db_path,
                    artifact_root=training_config.mlflow_artifact_root,
                    experiment_name=experiment_name,
                    registered_model_name=(
                        training_config.mlflow_registered_model_name
                    ),
                    mlflow_run_id=manifest.get("mlflow_run_id"),
                )
                manifest = registry.update_manifest(
                    lead_type_name,
                    manifest["training_run_id"],
                    **promotion_mlflow_metadata,
                )
            logger.info(
                "Logged run '%s' to MLflow experiment '%s'",
                manifest["training_run_id"],
                experiment_name,
            )
            logger.info(
                "Logged run to MLflow experiment '%s'",
                training_config.mlflow_experiment_name,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("MLflow logging skipped: %s", exc)
        finally:
            if comparison_temp_dir is not None:
                comparison_temp_dir.cleanup()

    return {
        "lead_type_id": lead_type_id,
        "lead_type_name": lead_type_name,
        "model_path": model_path,
        "training_run_id": manifest["training_run_id"],
        "model_version": manifest.get("production_model_version"),
        "production_model_version": manifest.get("production_model_version"),
        "promotion_mode": promotion_mode,
        "eligibility_status": eligibility_status,
        "promotion_status": promotion_status,
        "promotion_eligible": (
            True
            if eligibility_status == "eligible"
            else False if eligibility_status == "not_eligible" else None
        ),
        "promoted": promoted,
        "promotion_reason": promotion_reason,
        "promotion_comparison": promotion_comparison,
        "report_dir": report_dir,
        "metrics": model_metrics.to_dict(),
        "optimizer_summary": optimizer_summary_dict,
        "prep_summary": prep_summary,
        "lineage": lineage,
        "feature_cols": list(feature_cols),
        "split_settings": split_settings,
        "test_set_id": test_set_id,
        "comparison_artifact_path": manifest.get("comparison_artifact_path"),
    }


def _evaluate_currently_serving_model(
    lead_type_name: str,
    test_df: pd.DataFrame,
    y_test: pd.Series,
    target_cm: float,
    min_bid: float,
    bid_step: float,
    chunk_size: int,
) -> tuple[dict | None, dict | None]:
    """Evaluate the serving model on the challenger test rows.

    Inputs
    ------
    lead_type_name : str
        Human-readable lead type name.
    test_df : Any
        Input value.
    y_test : pandas.Series | numpy.ndarray
        Held-out target values.
    target_cm : float
        Target contribution margin as a decimal.
    min_bid : float
        Minimum candidate bid.
    bid_step : float
        Increment between candidate bids.
    chunk_size : int
        Maximum rows processed per optimizer scoring chunk.

    Returns
    -------
    tuple[dict | None, dict | None]
        Serving model metrics followed by optimizer metrics.
    """
    serving_version = registry.currently_serving_version(lead_type_name)
    if serving_version is None:
        logger.info(
            "Nothing currently serving for '%s' (first model).",
            lead_type_name,
        )
        return None, None

    serving_model, serving_manifest = registry.load_currently_serving_model(
        lead_type_name
    )
    if serving_model is None or serving_manifest is None:
        raise RuntimeError(
            "A currently-serving model is configured for "
            f"'{lead_type_name}' as version '{serving_version}', but its "
            "manifest or model artifact could not be loaded."
        )

    serving_feature_cols = serving_manifest.get("feature_cols") or []
    if not serving_feature_cols:
        raise ValueError(
            "Currently-serving model manifest "
            f"'{serving_version}' has no feature_cols."
        )

    # Feature-schema drift: the serving model may have been trained by an
    # older feature pipeline and expect columns the current training data no
    # longer produces. In that case, it cannot be re-scored like-for-like.
    missing = [
        column for column in serving_feature_cols if column not in test_df.columns
    ]
    if missing:
        logger.warning(
            "Currently-serving model '%s' expects %d feature column(s) not "
            "present in the current training data (e.g. %s); its feature "
            "schema predates the current pipeline. Skipping the head-to-head "
            "comparison and evaluating the challenger on the absolute "
            "promotion gates only.",
            serving_version,
            len(missing),
            missing[:6],
        )
        return None, None

    try:
        X_serving = test_df[serving_feature_cols]
        _, _, serving_metrics = metrics.evaluate_model(
            serving_model,
            X_serving,
            y_test,
        )
        serving_optimizer_result = optimizer_evaluation.run_bid_optimizer_evaluation(
            test_eval_df=test_df,
            model=serving_model,
            feature_cols=serving_feature_cols,
            target_cm=target_cm,
            min_bid=min_bid,
            bid_step=bid_step,
            chunk_size=chunk_size,
            log_summary_result=False,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Could not score the currently-serving model '%s' on the current "
            "test set (%s); treating it as not comparable and evaluating the "
            "challenger on the absolute promotion gates only.",
            serving_version,
            exc,
        )
        return None, None

    serving_optimizer_summary = (
        serving_optimizer_result[1].to_dict() if serving_optimizer_result else {}
    )
    logger.info(
        "Currently-serving model (%s) re-scored on this test set: "
        "log loss=%.4f, PR AUC=%.4f",
        serving_manifest.get("version"),
        serving_metrics.log_loss,
        serving_metrics.pr_auc,
    )
    return serving_metrics.to_dict(), serving_optimizer_summary


def main(argv: list[str] | None = None) -> int:
    """Run model training from command-line arguments.

    Inputs
    ------
    argv : list[str] | None
        Optional command-line argument sequence.

    Returns
    -------
    int
        Process exit status.
    """
    parser = argparse.ArgumentParser(
        description="Train the Anton win-probability model."
    )
    parser.add_argument(
        "--lead-type-id",
        type=int,
        required=True,
        help="Lead type ID: 6=auto, 1=home, 5=commercial",
    )
    parser.add_argument(
        "--version",
        default=None,
        help="Training-table version (default: latest)",
    )
    parser.add_argument(
        "--no-mlflow",
        action="store_true",
        help="Skip MLflow logging and registration",
    )
    args = parser.parse_args(argv)

    run_training(
        lead_type_id=args.lead_type_id,
        version=args.version,
        register_mlflow=not args.no_mlflow,
    )
    logger.info("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
