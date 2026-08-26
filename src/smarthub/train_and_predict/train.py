"""Training workflow for Anton's win-probability model.

This module trains, evaluates, versions, and optionally registers models.

The end-to-end run is expressed as ordered *stages* that share a single
:class:`TrainingContext`. ``run_training`` executes them sequentially (used by
the CLI), while ``train_and_predict.flow`` wraps each stage as its own Prefect
task for per-step observability and isolated retries. Keeping the logic here —
not in the Prefect entrypoint — preserves the canonical ``smarthub`` module path
for pickled model objects (see the note in ``flow.py``).
"""

from __future__ import annotations

import argparse
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from smarthub.core import notifications
from smarthub.core.lead_types import lead_type_name as resolve_lead_type_name
from smarthub.core.logging_utils import get_logger

from . import (
    config,
    feature_diagnostics,
    metrics,
    models,
    optimizer_evaluation,
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
        "observed_policy_wins",
        "observed_policy_win_rate",
        "observed_policy_total_expected_revenue",
        "observed_policy_total_bid_cost",
        "observed_policy_total_expected_profit",
        "observed_policy_expected_cm",
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


def _monotonicity_metrics_for_mlflow(
    monotonicity_summary: dict | None,
) -> dict[str, float | int]:
    """Select numeric monotonicity metrics for MLflow."""
    if not monotonicity_summary:
        return {}

    result = {}
    for key, value in monotonicity_summary.items():
        if key == "enabled" or value is None:
            continue
        if isinstance(value, bool):
            result[f"monotonicity_{key}"] = int(value)
        elif isinstance(value, (int, float)):
            result[f"monotonicity_{key}"] = value
    return result


@dataclass
class TrainingContext:
    """Mutable state shared across the ordered training stages.

    A single instance flows through :func:`stage_prepare_data` ...
    :func:`stage_mlflow`; each stage reads what earlier stages produced and
    writes its own outputs back. This is passed by reference between Prefect
    tasks (results are never persisted), so large objects such as the training
    frame and the fitted model are kept in memory rather than serialized.
    """

    # --- inputs ---
    lead_type_id: int
    version: str | None = None
    register_mlflow: bool = True

    # --- resolved config ---
    training_config: Any = None
    lead_type_name: str | None = None

    # --- stage: prepare data ---
    frame: Any = None
    numeric: Any = None
    categorical: Any = None
    prep_summary: Any = None
    feature_cols: Any = None

    # --- stage: split + diagnostics ---
    split_settings: Any = None
    train_df: Any = None
    test_df: Any = None
    X_train: Any = None
    y_train: Any = None
    X_test: Any = None
    y_test: Any = None
    feature_summary_df: Any = None
    feature_counts_df: Any = None
    feature_split_diagnostics: Any = None
    full_zero_variance_features: Any = None
    zero_variance_features: Any = None

    # --- stage: fit ---
    model_type: Any = None
    model_params: Any = None
    calibration_enabled: Any = None
    model: Any = None

    # --- stage: evaluate + optimizer ---
    pred: Any = None
    pred_class: Any = None
    model_metrics: Any = None
    evaluation_df: Any = None
    optimizer_config: Any = None
    target_cm: Any = None
    min_bid: Any = None
    bid_step: Any = None
    optimizer_chunk_size: Any = None
    monotonicity_config: Any = None
    optimizer_eval_df: Any = None
    optimizer_summary_dict: Any = None
    monotonicity_summary_dict: Any = None
    optimizer_mlflow_metrics: Any = None

    # --- stage: save reports ---
    report_dir: Any = None
    lineage: Any = None
    test_set_id: Any = None

    # --- stage: promotion decision ---
    promotion_mode: Any = None
    decision: Any = None
    eligibility_status: Any = None
    promotion_status: Any = None
    promotion_reason: Any = None
    promotion_comparison: Any = None

    # --- stage: save version + promote ---
    manifest: Any = None
    model_path: Any = None
    comparison_temp_dir: Any = None
    comparison_artifacts: Any = None
    promoted: bool = False


def stage_prepare_data(ctx: TrainingContext) -> TrainingContext:
    """Load config and the training table; validate it is trainable.

    Populates ``training_config``, ``lead_type_name``, ``frame``, ``numeric``,
    ``categorical``, ``prep_summary`` and ``feature_cols`` on ``ctx``.

    Raises
    ------
    ValueError
        If there are too few training rows to proceed.
    """
    ctx.training_config = config.load_training_config(ctx.lead_type_id)
    ctx.lead_type_name = resolve_lead_type_name(ctx.lead_type_id)
    np.random.seed(ctx.training_config.random_seed)

    logger.info(
        "Loading training table for lead_type=%s (%s)",
        ctx.lead_type_name,
        ctx.lead_type_id,
    )
    frame, numeric, categorical, prep_summary = preprocessing.prepare_training_data(
        ctx.lead_type_id,
        ctx.lead_type_name,
        ctx.version,
    )
    ctx.frame = frame
    ctx.numeric = numeric
    ctx.categorical = categorical
    ctx.prep_summary = prep_summary
    ctx.feature_cols = numeric + categorical

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
            f"{ctx.lead_type_name}; need more data before training."
        )

    preprocessing.assert_trainable(frame, ctx.lead_type_name)
    return ctx


def stage_split_and_diagnostics(ctx: TrainingContext) -> TrainingContext:
    """Split into train/test and compute feature-coverage diagnostics.

    Populates the split partitions, feature-summary artifacts, coverage
    diagnostics and the zero-variance feature list on ``ctx``. Emits a warning
    notification if any binary feature loses variance in the training split.
    """
    training_config = ctx.training_config
    frame = ctx.frame
    numeric = ctx.numeric
    categorical = ctx.categorical
    feature_cols = ctx.feature_cols
    lead_type_name = ctx.lead_type_name
    lead_type_id = ctx.lead_type_id

    full_zero_variance_features = feature_diagnostics.find_zero_variance_features(
        frame,
        numeric,
        categorical,
    )
    ctx.full_zero_variance_features = full_zero_variance_features
    feature_diagnostics.log_zero_variance_features(
        full_zero_variance_features,
        "Full Dataset",
    )

    split_settings = training_config.split
    train_df, test_df = split_training_data(
        frame=frame,
        target_column=config.TARGET_COL,
        split_settings=split_settings,
        random_seed=training_config.random_seed,
    )
    preprocessing.assert_partition_has_both_classes(
        train_df,
        lead_type_name,
        "Training partition",
    )
    preprocessing.assert_partition_has_both_classes(
        test_df,
        lead_type_name,
        "Test partition",
    )
    ctx.split_settings = split_settings
    ctx.train_df = train_df
    ctx.test_df = test_df

    logger.info("Feature columns: %s", feature_cols)

    feature_summary_df, feature_counts_df = (
        feature_diagnostics.build_training_data_summary(
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
    feature_diagnostics.log_training_data_summary(
        df=frame,
        feature_summary_df=feature_summary_df,
        feature_counts_df=feature_counts_df,
        target_col=config.TARGET_COL,
    )
    ctx.feature_summary_df = feature_summary_df
    ctx.feature_counts_df = feature_counts_df

    X_train = train_df[feature_cols]
    y_train = train_df[config.TARGET_COL]
    X_test = test_df[feature_cols]
    y_test = test_df[config.TARGET_COL]
    ctx.X_train = X_train
    ctx.y_train = y_train
    ctx.X_test = X_test
    ctx.y_test = y_test

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

    diagnostic_features = feature_diagnostics.coverage_features(
        frame,
        numeric,
        categorical,
    )
    feature_coverage_diagnostics = feature_diagnostics.feature_coverage_rows(
        train_df=train_df,
        eval_df=test_df,
        features=diagnostic_features,
        partition="test",
    )
    differing_coverage_df = feature_diagnostics.log_feature_coverage_diagnostics(
        feature_coverage_diagnostics,
        evaluation_label="test",
        include_partition=False,
    )
    ctx.feature_split_diagnostics = differing_coverage_df.to_dict(orient="records")

    binary_variance_loss = feature_diagnostics.find_binary_variance_loss(
        frame,
        feature_coverage_diagnostics,
    )

    if binary_variance_loss:
        affected = "\n".join(
            f"{row['feature']}: train={row['train_unique']}, "
            f"test={row['eval_unique']}"
            for row in binary_variance_loss
        )
        notifications.notify_warning(
            "train-model",
            {
                "Lead type": f"{lead_type_name} ({lead_type_id})",
                "Issue": "Binary feature(s) lost variance in training split",
                "Affected features": affected,
                "Action": "Training continues normally",
            },
        )

    ctx.zero_variance_features = feature_diagnostics.find_zero_variance_features(
        train_df,
        numeric,
        categorical,
    )
    feature_diagnostics.log_zero_variance_features(
        ctx.zero_variance_features,
        "Train Split",
    )
    return ctx


def stage_fit_model(ctx: TrainingContext) -> TrainingContext:
    """Build and fit the model on the training partition."""
    training_config = ctx.training_config

    model_type = training_config.model_type
    model_params = training_config.model_parameters
    calibration_enabled = training_config.calibration_enabled
    ctx.model_type = model_type
    ctx.model_params = model_params
    ctx.calibration_enabled = calibration_enabled

    logger.info("Training Model")
    logger.info("  Model type                            : %s", model_type)
    logger.info(
        "  Calibration enabled                   : %s",
        calibration_enabled,
    )

    model = models.build_model(
        model_type,
        ctx.numeric,
        ctx.categorical,
        model_params,
        calibration_enabled=calibration_enabled,
        calibration_method=training_config.calibration_method,
        calibration_cv=training_config.calibration_cv,
    )
    model.fit(ctx.X_train, ctx.y_train)
    ctx.model = model
    return ctx


def stage_evaluate(ctx: TrainingContext) -> TrainingContext:
    """Evaluate the fitted model and run the offline bid optimizer."""
    training_config = ctx.training_config
    model = ctx.model
    X_test = ctx.X_test
    y_test = ctx.y_test
    test_df = ctx.test_df
    feature_cols = ctx.feature_cols
    frame = ctx.frame

    logger.info("Evaluating Model")
    pred, pred_class, model_metrics = metrics.evaluate_model(
        model,
        X_test,
        y_test,
    )
    ctx.pred = pred
    ctx.pred_class = pred_class

    evaluation_df = test_df.copy()
    evaluation_df["predicted_win_probability"] = pred
    evaluation_df["predicted_class"] = pred_class
    ctx.evaluation_df = evaluation_df

    optimizer_config = training_config.optimizer
    target_cm = optimizer_config.target_cm
    min_bid = optimizer_config.minimum_bid
    bid_step = optimizer_config.bid_step
    optimizer_chunk_size = optimizer_config.chunk_size
    monotonicity_config = training_config.promotion_monotonicity
    ctx.optimizer_config = optimizer_config
    ctx.target_cm = target_cm
    ctx.min_bid = min_bid
    ctx.bid_step = bid_step
    ctx.optimizer_chunk_size = optimizer_chunk_size
    ctx.monotonicity_config = monotonicity_config

    optimizer_result = optimizer_evaluation.run_bid_optimizer_evaluation(
        test_eval_df=test_df,
        model=model,
        feature_cols=feature_cols,
        target_cm=target_cm,
        min_bid=min_bid,
        bid_step=bid_step,
        chunk_size=optimizer_chunk_size,
        monotonicity_enabled=True,
        monotonicity_tolerance=monotonicity_config.tolerance,
        monotonicity_max_violation_rate=(monotonicity_config.max_violation_rate),
    )

    if optimizer_result is not None:
        optimizer_eval_df, optimizer_summary = optimizer_result
        model_metrics = model_metrics.with_recommended_win_rate(
            optimizer_summary.avg_recommended_bid_predicted_win_rate
        )
        monotonicity_summary_dict = optimizer_eval_df.attrs.get(
            "monotonicity_summary",
            {},
        )
    else:
        optimizer_eval_df = None
        optimizer_summary = None
        monotonicity_summary_dict = {}
    ctx.optimizer_eval_df = optimizer_eval_df
    ctx.model_metrics = model_metrics
    ctx.monotonicity_summary_dict = monotonicity_summary_dict

    optimizer_summary_dict = (
        optimizer_summary.to_dict() if optimizer_summary is not None else {}
    )
    ctx.optimizer_summary_dict = optimizer_summary_dict
    optimizer_mlflow_metrics = _optimizer_metrics_for_mlflow(optimizer_summary_dict)
    optimizer_mlflow_metrics.update(
        _monotonicity_metrics_for_mlflow(monotonicity_summary_dict)
    )
    ctx.optimizer_mlflow_metrics = optimizer_mlflow_metrics

    metrics.log_model_evaluation(
        model_metrics=model_metrics,
        rows_trained=len(frame),
        train_rows=len(ctx.X_train),
        test_rows=len(X_test),
    )
    return ctx


def stage_save_reports(ctx: TrainingContext) -> TrainingContext:
    """Persist report files and build the run lineage."""
    training_config = ctx.training_config
    lead_type_name = ctx.lead_type_name
    lead_type_id = ctx.lead_type_id
    model_metrics = ctx.model_metrics
    prep_summary = ctx.prep_summary
    optimizer_eval_df = ctx.optimizer_eval_df
    split_settings = ctx.split_settings

    logger.info("Saving Reports")
    report_dir = training_config.report_dir(lead_type_name)
    Path(report_dir).mkdir(parents=True, exist_ok=True)
    ctx.report_dir = report_dir

    training_artifacts.save_feature_summary_files(
        report_dir,
        ctx.feature_summary_df,
        ctx.feature_counts_df,
    )
    training_artifacts.save_performance_plots(
        report_dir=report_dir,
        y_test=ctx.y_test,
        pred=ctx.pred,
        pred_class=ctx.pred_class,
        roc_auc=model_metrics.roc_auc,
        pr_auc=model_metrics.pr_auc,
        accuracy=model_metrics.accuracy,
        precision=model_metrics.precision,
        recall=model_metrics.recall,
        f1=model_metrics.f1,
        f2=model_metrics.f2,
        optimizer_eval_df=optimizer_eval_df,
        target_cm=ctx.target_cm,
        min_bid=ctx.min_bid,
        bid_step=ctx.bid_step,
    )

    lineage = {
        "model_type": ctx.model_type,
        "calibrated": bool(ctx.calibration_enabled),
        "split_strategy": split_settings["strategy"],
        "split_test_size": split_settings["test_size"],
        "training_table_version": prep_summary["training_table_version"],
        "data_min_created_at": prep_summary["data_min_created_at"],
        "data_max_created_at": prep_summary["data_max_created_at"],
        "source_row_count": prep_summary["source_row_count"],
        "full_zero_variance_features": list(ctx.full_zero_variance_features),
        "zero_variance_features": list(ctx.zero_variance_features),
        "feature_split_diagnostics": ctx.feature_split_diagnostics,
    }

    logger.info(
        "Lineage: model=%s trained on table version %s "
        "(data %s → %s, %s source rows)",
        ctx.model_type,
        lineage["training_table_version"],
        lineage["data_min_created_at"],
        lineage["data_max_created_at"],
        lineage["source_row_count"],
    )

    test_set_id = training_artifacts.build_test_set_id(
        ctx.test_df,
        training_table_version=prep_summary["training_table_version"],
        split_settings=split_settings,
    )
    lineage["test_set_id"] = test_set_id
    ctx.lineage = lineage
    ctx.test_set_id = test_set_id

    evaluation_summary = {
        "lead_type_id": lead_type_id,
        "lead_type_name": lead_type_name,
        "rows_trained": int(len(ctx.frame)),
        "train_rows": int(len(ctx.X_train)),
        "test_rows": int(len(ctx.X_test)),
        **lineage,
        **model_metrics.to_dict(),
        "bid_monotonicity": ctx.monotonicity_summary_dict,
        "bid_optimizer": ctx.optimizer_summary_dict,
    }

    training_artifacts.save_evaluation_summary(
        report_dir,
        evaluation_summary,
        optimizer_eval_df,
    )
    training_artifacts.log_saved_report_files(
        report_dir,
        optimizer_eval_df,
    )
    return ctx


def stage_promotion_decision(ctx: TrainingContext) -> TrainingContext:
    """Decide whether the challenger is eligible for promotion."""
    training_config = ctx.training_config
    lead_type_name = ctx.lead_type_name
    model_metrics = ctx.model_metrics
    optimizer_summary_dict = ctx.optimizer_summary_dict
    monotonicity_summary_dict = ctx.monotonicity_summary_dict
    monotonicity_config = ctx.monotonicity_config
    test_df = ctx.test_df
    y_test = ctx.y_test

    promotion_mode = training_config.promotion_mode
    ctx.promotion_mode = promotion_mode
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
        monotonicity_required = monotonicity_config.enabled
        monotonicity_passed = monotonicity_summary_dict.get("passed")
        monotonicity_comparison = {
            "monotonicity_required": monotonicity_required,
            "challenger_monotonicity_passed": monotonicity_passed,
            "challenger_monotonicity_violation_rate": (
                monotonicity_summary_dict.get("violation_rate")
            ),
            "maximum_monotonicity_violation_rate": (
                monotonicity_config.max_violation_rate
            ),
        }

        if monotonicity_required and monotonicity_passed is not True:
            decision = registry.PromotionDecision(
                promote=False,
                reason=(
                    "Challenger failed the required bid-monotonicity "
                    "promotion criterion."
                ),
                comparison=monotonicity_comparison,
            )
        else:
            serving_metrics, serving_optimizer_summary = (
                _evaluate_currently_serving_model(
                    lead_type_name,
                    test_df,
                    y_test,
                    ctx.target_cm,
                    ctx.min_bid,
                    ctx.bid_step,
                    ctx.optimizer_chunk_size,
                    monotonicity_config,
                )
            )
            decision = registry.decide_promotion(
                challenger_metrics=model_metrics.to_dict(),
                challenger_optimizer=optimizer_summary_dict,
                currently_serving_metrics=serving_metrics,
                currently_serving_optimizer=serving_optimizer_summary,
                max_log_loss_regression=(
                    training_config.promotion_max_log_loss_regression
                ),
                min_profit_ratio=training_config.promotion_min_profit_ratio,
                max_absolute_profit_loss_tolerance=(
                    training_config.promotion_max_absolute_profit_loss_tolerance
                ),
                max_log_loss=training_config.promotion_max_log_loss,
                min_expected_profit=(training_config.promotion_min_expected_profit),
            )
            decision.comparison.update(monotonicity_comparison)

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
            "  Monotonicity criterion                : %s",
            "REQUIRED" if monotonicity_required else "NOT REQUIRED",
        )
        if monotonicity_config.enabled:
            logger.info(
                "  Monotonicity result                   : %s",
                "PASS" if monotonicity_passed else "FAIL",
            )

        challenger_profit = promotion_comparison.get("challenger_profit")
        serving_profit = promotion_comparison.get("currently_serving_profit")
        if challenger_profit is not None:
            logger.info(
                "  Challenger probability-weighted expected profit: %.4f",
                challenger_profit,
            )
        if serving_profit is not None:
            logger.info(
                "  Serving probability-weighted expected profit   : %.4f",
                serving_profit,
            )
        if challenger_profit is not None and serving_profit is not None:
            logger.info(
                "  Profit difference (challenger-serving): %.4f",
                challenger_profit - serving_profit,
            )

        profit_ratio = promotion_comparison.get("profit_ratio")
        if profit_ratio is not None:
            logger.info(
                "  Challenger / serving profit ratio     : %.2f%%",
                profit_ratio * 100.0,
            )
            logger.info(
                "  Minimum required profit ratio         : %.2f%%",
                training_config.promotion_min_profit_ratio * 100.0,
            )

        absolute_profit_loss = promotion_comparison.get("absolute_profit_loss")
        if absolute_profit_loss is not None:
            logger.info(
                "  Absolute profit loss                   : %.4f",
                absolute_profit_loss,
            )
            logger.info(
                "  Maximum allowed absolute profit loss   : %.4f",
                training_config.promotion_max_absolute_profit_loss_tolerance,
            )

        challenger_log_loss = promotion_comparison.get("challenger_log_loss")
        serving_log_loss = promotion_comparison.get("currently_serving_log_loss")
        if challenger_log_loss is not None:
            logger.info(
                "  Challenger log loss                   : %.4f",
                challenger_log_loss,
            )
        if serving_log_loss is not None:
            logger.info(
                "  Serving log loss                      : %.4f",
                serving_log_loss,
            )

        log_loss_regression = promotion_comparison.get("log_loss_regression")
        if log_loss_regression is not None:
            logger.info(
                "  Log-loss regression                   : %.4f",
                log_loss_regression,
            )
            logger.info(
                "  Maximum allowed log-loss regression   : %.4f",
                training_config.promotion_max_log_loss_regression,
            )

        logger.info(
            "  Policy recommendation                 : %s",
            "ELIGIBLE" if decision.promote else "NOT ELIGIBLE",
        )
        logger.info("  Reason                                : %s", decision.reason)

    ctx.decision = decision
    ctx.eligibility_status = eligibility_status
    ctx.promotion_status = promotion_status
    ctx.promotion_reason = promotion_reason
    ctx.promotion_comparison = promotion_comparison
    return ctx


def stage_save_and_promote(ctx: TrainingContext) -> TrainingContext:
    """Persist the versioned model and, if eligible, promote it.

    Save-version and promote are intentionally in one stage: passing promotion
    eligibility is not the same as serving, and splitting them across retryable
    tasks risks a "saved but half-promoted" state. The atomic
    publish->flip->mark handling in ``registry.promote`` is preserved.
    """
    training_config = ctx.training_config
    lead_type_id = ctx.lead_type_id
    lead_type_name = ctx.lead_type_name
    model = ctx.model
    feature_cols = ctx.feature_cols
    model_metrics = ctx.model_metrics
    optimizer_summary_dict = ctx.optimizer_summary_dict
    lineage = ctx.lineage
    model_params = ctx.model_params
    promotion_mode = ctx.promotion_mode
    decision = ctx.decision
    optimizer_eval_df = ctx.optimizer_eval_df
    prep_summary = ctx.prep_summary
    test_set_id = ctx.test_set_id
    split_settings = ctx.split_settings
    monotonicity_config = ctx.monotonicity_config
    monotonicity_summary_dict = ctx.monotonicity_summary_dict

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
        eligibility_status=ctx.eligibility_status,
        promotion_status=ctx.promotion_status,
        promotion_decision_reason=ctx.promotion_reason,
        promotion_comparison=ctx.promotion_comparison,
    )

    model_path = manifest["model_path"]
    ctx.model_path = model_path

    artifact_metadata = {
        "training_run_id": manifest["training_run_id"],
        "lead_type_id": lead_type_id,
        "lead_type_name": lead_type_name,
        "test_set_id": test_set_id,
        "training_table_version": prep_summary["training_table_version"],
        "split_settings": split_settings,
        "random_seed": training_config.random_seed,
        "model_type": ctx.model_type,
        "model_parameters": model_params,
        "calibration": {
            "enabled": ctx.calibration_enabled,
            "method": training_config.calibration_method,
            "cv": training_config.calibration_cv,
        },
        "feature_cols": list(feature_cols),
        "full_zero_variance_features": list(ctx.full_zero_variance_features),
        "zero_variance_features": list(ctx.zero_variance_features),
        "feature_split_diagnostics": ctx.feature_split_diagnostics,
        "train_rows": int(len(ctx.X_train)),
        "test_rows": int(len(ctx.X_test)),
        "optimizer": ctx.optimizer_config.as_dict(),
        "observed_production_policy": {
            key: value
            for key, value in optimizer_summary_dict.items()
            if key.startswith("observed_policy_")
        },
        "promotion_monotonicity": monotonicity_config.as_dict(),
        "bid_monotonicity": monotonicity_summary_dict,
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
        if not ctx.register_mlflow:
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
                evaluation_df=ctx.evaluation_df,
                optimizer_df=optimizer_eval_df,
                metadata=artifact_metadata,
            )
    ctx.comparison_temp_dir = comparison_temp_dir
    ctx.comparison_artifacts = comparison_artifacts

    logger.info(
        "Saved training run %s → %s",
        manifest["training_run_id"],
        model_path,
    )

    promoted = False
    if promotion_mode == "automatic" and decision is not None and decision.promote:
        try:
            registry.promote(
                lead_type_name,
                manifest["training_run_id"],
                reason=decision.reason,
            )
        except Exception:
            # Passing eligibility is NOT the same as being promoted. If
            # publishing to production or switching the serving pointer failed,
            # the model is not serving — do not record a successful promotion.
            # Mark the run and fail loudly so it's visible and can be re-promoted.
            logger.error(
                "Automatic promotion FAILED for %s ('%s'): the model was "
                "trained but is NOT serving. Re-promote once production storage "
                "is healthy.",
                manifest["training_run_id"],
                lead_type_name,
                exc_info=True,
            )
            try:
                registry.update_manifest(
                    lead_type_name,
                    manifest["training_run_id"],
                    promotion_status="promotion_failed",
                )
            except Exception:  # noqa: BLE001 -- best-effort status write
                logger.warning(
                    "Could not record promotion_failed status.", exc_info=True
                )
            raise
        promoted = True
        manifest = registry.load_manifest(lead_type_name, manifest["training_run_id"])
        ctx.promotion_status = "promoted"
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
            ctx.promotion_status,
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

    ctx.manifest = manifest
    ctx.promoted = promoted
    return ctx


def stage_mlflow(ctx: TrainingContext) -> TrainingContext:
    """Log (and optionally register/promote) the run in MLflow.

    Best-effort: any MLflow error is logged and swallowed so it never fails the
    training run. Skipped entirely when ``register_mlflow`` is False.
    """
    training_config = ctx.training_config
    lead_type_name = ctx.lead_type_name
    model = ctx.model
    model_params = ctx.model_params
    feature_cols = ctx.feature_cols
    model_metrics = ctx.model_metrics
    report_dir = ctx.report_dir
    lineage = ctx.lineage
    promotion_mode = ctx.promotion_mode
    promotion_reason = ctx.promotion_reason
    comparison_artifacts = ctx.comparison_artifacts
    comparison_temp_dir = ctx.comparison_temp_dir
    manifest = ctx.manifest
    promoted = ctx.promoted

    try:
        from . import mlflow_utils

        experiment_name = f"{training_config.mlflow_experiment_name}_{lead_type_name}"

        mlflow_metadata = mlflow_utils.log_training_run(
            model=model,
            model_params=model_params,
            feature_cols=feature_cols,
            metrics=model_metrics.to_dict(),
            optimizer_metrics=ctx.optimizer_mlflow_metrics,
            report_dir=report_dir,
            comparison_artifact_dir=comparison_artifacts.get("artifact_dir"),
            tracking_db_path=training_config.mlflow_tracking_db_path,
            artifact_root=training_config.mlflow_artifact_root,
            experiment_name=experiment_name,
            run_name=manifest["training_run_id"],
            training_config_path=Path(training_config.raw["resolved"]["config_path"]),
            extra_params={
                "lead_type_name": lead_type_name,
                "training_run_id": manifest["training_run_id"],
                "promotion_mode": promotion_mode,
                **lineage,
            },
            extra_tags={
                "eligibility_status": ctx.eligibility_status,
                "promotion_status": ctx.promotion_status,
                "promoted": promoted,
                "production_model_version": manifest.get("production_model_version"),
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
                registered_model_name=(training_config.mlflow_registered_model_name),
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

    ctx.manifest = manifest
    return ctx


def build_result(ctx: TrainingContext) -> dict[str, Any]:
    """Assemble the training-run result dictionary from the context."""
    manifest = ctx.manifest
    eligibility_status = ctx.eligibility_status
    return {
        "lead_type_id": ctx.lead_type_id,
        "lead_type_name": ctx.lead_type_name,
        "model_path": ctx.model_path,
        "training_run_id": manifest["training_run_id"],
        "model_version": manifest.get("production_model_version"),
        "production_model_version": manifest.get("production_model_version"),
        "promotion_mode": ctx.promotion_mode,
        "eligibility_status": eligibility_status,
        "promotion_status": ctx.promotion_status,
        "promotion_eligible": (
            True
            if eligibility_status == "eligible"
            else False if eligibility_status == "not_eligible" else None
        ),
        "promoted": ctx.promoted,
        "promotion_reason": ctx.promotion_reason,
        "promotion_comparison": ctx.promotion_comparison,
        "report_dir": ctx.report_dir,
        "metrics": ctx.model_metrics.to_dict(),
        "optimizer_summary": ctx.optimizer_summary_dict,
        "monotonicity_summary": ctx.monotonicity_summary_dict,
        "prep_summary": ctx.prep_summary,
        "lineage": ctx.lineage,
        "feature_cols": list(ctx.feature_cols),
        "split_settings": ctx.split_settings,
        "test_set_id": ctx.test_set_id,
        "comparison_artifact_path": manifest.get("comparison_artifact_path"),
    }


# Ordered training stages. ``run_training`` (CLI) walks these in-process; the
# Prefect flow wraps each as its own task. Keep this list and the flow in sync.
TRAINING_STAGES = (
    stage_prepare_data,
    stage_split_and_diagnostics,
    stage_fit_model,
    stage_evaluate,
    stage_save_reports,
    stage_promotion_decision,
    stage_save_and_promote,
)


def run_training(
    lead_type_id: int,
    version: str | None = None,
    register_mlflow: bool = True,
) -> dict[str, Any]:
    """Train, evaluate, version, and optionally register one model.

    Runs the ordered training stages sequentially in-process and returns the
    assembled result. The Prefect flow runs the same stages as separate tasks.

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
    ctx = TrainingContext(
        lead_type_id=lead_type_id,
        version=version,
        register_mlflow=register_mlflow,
    )
    for stage in TRAINING_STAGES:
        stage(ctx)
    if register_mlflow:
        stage_mlflow(ctx)
    return build_result(ctx)


def _evaluate_currently_serving_model(
    lead_type_name: str,
    test_df: pd.DataFrame,
    y_test: pd.Series,
    target_cm: float,
    min_bid: float,
    bid_step: float,
    chunk_size: int,
    monotonicity_config,
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
    monotonicity_config : MonotonicityConfig
        Resolved bid-response monotonicity settings.

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
        logger.warning(
            "Currently-serving model '%s' for '%s' is configured but could not "
            "be loaded (corrupt/unloadable); skipping challenger comparison and "
            "proceeding to train/promote a replacement.",
            serving_version,
            lead_type_name,
        )
        return None, None

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
            monotonicity_enabled=True,
            monotonicity_tolerance=monotonicity_config.tolerance,
            monotonicity_max_violation_rate=(monotonicity_config.max_violation_rate),
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
