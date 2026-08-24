"""Prefect orchestration for SmartHub model training.

This module runs training, publishes run summaries, and sends notifications.
"""

from __future__ import annotations

import argparse

from prefect import flow, get_run_logger, task
from prefect.artifacts import create_markdown_artifact

from smarthub.core import notifications
from smarthub.core.lead_types import lead_type_name as resolve_lead_type_name
from smarthub.feature_engineering import features as fe

# NOTE: absolute (not `from . import config, train`). Prefect loads this
# entrypoint as a bare top-level module named ``flow`` (``__package__='flow'``),
# so a relative import would pull the whole training subtree in as ``flow.models``,
# ``flow.train``, ... and stamp pickled objects (e.g. the model's
# FunctionTransformer) with ``__module__='flow.models'`` -- unloadable anywhere
# else ("No module named 'flow'"). Importing by canonical path keeps trained
# models loadable by serve, the shap-worker, and the training eval step.
from smarthub.train_and_predict import train


def _feature_breakdown(lead_type_id, feature_cols):
    """Summarize how the trained columns relate to the feature registry.

    The registry (``feature_engineering.FEATURES``) is the source of truth for a
    lead type's applicable features; ``feature_cols`` are the columns the model
    was actually trained on. This reports how many trained columns come from the
    registry and which registry features went unused.

    Inputs
    ------
    lead_type_id : int
        SmartHub lead type identifier.
    feature_cols : list[str]
        Ordered model feature names.

    Returns
    -------
    dict
        Feature counts and the list of unused registry features.
    """
    feature_cols = list(feature_cols or [])
    numeric, categorical = fe.model_feature_columns(lead_type_id)
    registered = set(numeric) | set(categorical)
    used = set(feature_cols)
    return {
        "total": len(feature_cols),
        "n_registered": len(registered),
        "n_registered_used": len(registered & used),
        "unused": sorted(registered - used),
    }


@task(name="train-anton-model")
def _train_task(lead_type_id, version, register_mlflow):
    """Run model training and offline optimizer evaluation for one lead type.

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
    dict
        Training workflow result.
    """
    return train.run_training(
        lead_type_id=lead_type_id,
        version=version,
        register_mlflow=register_mlflow,
    )


@flow(
    name="smarthub-train-model",
    log_prints=True,
    on_failure=[notifications.flow_failure_hook],
)
def train_flow(
    lead_type_id: int,
    version: str | None = None,
    register_mlflow: bool = True,
) -> dict:
    """Run the Prefect model-training flow for one lead type.

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
    dict
        Completed flow summary.
    """
    logger = get_run_logger()
    lead_type_name = resolve_lead_type_name(lead_type_id)
    logger.info(
        "STEP 3 train-model starting: lead_type=%s (%s), version=%s, mlflow=%s",
        lead_type_name,
        lead_type_id,
        version or "latest",
        register_mlflow,
    )

    result = _train_task(lead_type_id, version, register_mlflow)

    m = result["metrics"]
    opt = result["optimizer_summary"] or {}
    logger.info(
        "Trained %s model %s: log loss=%.4f, rows=%s -> %s (promoted=%s)",
        lead_type_name,
        result.get("production_model_version") or result.get("training_run_id"),
        m.get("log_loss", float("nan")),
        result["prep_summary"]["training_rows"],
        result["model_path"],
        result.get("promoted"),
    )

    _report(lead_type_name, lead_type_id, result, m, opt)
    _notify_success(lead_type_name, lead_type_id, result, m, opt)
    return {
        "lead_type": lead_type_name,
        "log_loss": m.get("log_loss"),
        "rows_trained": result["prep_summary"]["training_rows"],
        "model_path": result["model_path"],
    }


def _report(lead_type_name, lead_type_id, result, m, opt) -> None:
    """Publish a Prefect markdown artifact for a training run.

    Inputs
    ------
    lead_type_name : str
        Human-readable lead type name.
    lead_type_id : int
        SmartHub lead type identifier.
    result : dict
        Training-run result dictionary.
    m : dict
        Model evaluation metrics.
    opt : dict
        Optimizer summary metrics.
    """
    prep = result["prep_summary"]

    def _f(value, fmt="{:.4f}"):
        """Execute  f.

        Inputs
        ------
        value : Any
            Value to process.
        fmt : str
            Format string used for display.
        """
        try:
            return fmt.format(float(value))
        except (TypeError, ValueError):
            return "—"

    lineage = result.get("lineage", {})
    fb = _feature_breakdown(lead_type_id, result.get("feature_cols"))
    promoted = result.get("promoted")
    eligibility_status = result.get("eligibility_status")
    promotion_status = result.get("promotion_status")
    promotion_mode = result.get("promotion_mode")
    if promotion_mode == "disabled":
        status_label = "⏭ PROMOTION EVALUATION DISABLED"
    elif promoted:
        status_label = "✅ PROMOTED to currently-serving"
    elif promotion_status == "awaiting_manual_promotion":
        status_label = "🟡 ELIGIBLE — awaiting manual promotion"
    elif eligibility_status == "eligible":
        status_label = "⏸ ELIGIBLE — promotion execution skipped"
    else:
        status_label = "⏸ NOT ELIGIBLE — currently-serving model unchanged"
    observed_policy_revenue = _f(opt.get("observed_policy_total_expected_revenue"))
    observed_policy_profit = _f(opt.get("observed_policy_total_expected_profit"))
    observed_policy_cm = _f(
        opt.get("observed_policy_expected_cm"),
    )
    total_profit = (
        f"{_f(opt.get('current_bid_total_expected_profit'))} → "
        f"{_f(opt.get('recommended_bid_total_expected_profit'))}"
    )
    avg_win_rate = (
        f"{_f(opt.get('avg_current_bid_predicted_win_rate'))} → "
        f"{_f(opt.get('avg_recommended_bid_predicted_win_rate'))}"
    )
    bid_direction = (
        f"{_f(opt.get('bid_increase_pct'), '{:.2f}%')} / "
        f"{_f(opt.get('bid_decrease_pct'), '{:.2f}%')} / "
        f"{_f(opt.get('bid_unchanged_pct'), '{:.2f}%')}"
    )
    bid_change = (
        f"{_f(opt.get('avg_bid_change'))} / " f"{_f(opt.get('median_bid_change'))}"
    )
    probability_weighted_profit_lift = _f(opt.get("expected_profit_lift_total"))
    probability_weighted_profit_lift_pct = _f(
        opt.get("expected_profit_lift_pct"),
        "{:.2%}",
    )
    md = f"""# Train model — {lead_type_name}

## Promotion decision: {status_label}
{result.get('promotion_reason', '—')}

| field | value |
| --- | --- |
| lead_type_id | {lead_type_id} |
| training run ID | `{result.get('training_run_id')}` |
| production model version | `{result.get('production_model_version') or '—'}` |
| promotion mode | {promotion_mode} |
| eligibility status | {eligibility_status} |
| promotion status | {promotion_status} |
| model | {lineage.get('model_type')} (calibrated={lineage.get('calibrated')}) |
| rows trained | {prep.get('training_rows')} |
| observed win rate | {_f(prep.get('win_rate'))} |
| training table version | `{lineage.get('training_table_version')}` |
| trained on data | `{lineage.get('data_min_created_at')}` → \
`{lineage.get('data_max_created_at')}` |
| model path | `{result['model_path']}` |
| report dir | `{result['report_dir']}` |
| features | {fb['total']} ({fb['n_registered_used']}/{fb['n_registered']} \
from registry) |
| registry features unused | {', '.join(fb['unused']) or '—'} |

## Model quality (held-out test)
| metric | value |
| --- | --- |
| ROC AUC | {_f(m.get('roc_auc'))} |
| PR AUC | {_f(m.get('pr_auc'))} |
| Log loss | {_f(m.get('log_loss'))} |
| F2 | {_f(m.get('f2'))} |
| Brier score | {_f(m.get('brier_score'))} |
| Calibration error | {_f(m.get('calibration_error'))} |

## Observed production policy (held-out optimizer rows)
| metric | value |
| --- | --- |
| rows evaluated | {opt.get('optimizer_rows', '—')} |
| wins | {opt.get('observed_policy_wins', '—')} |
| observed win rate | {_f(opt.get('observed_policy_win_rate'), '{:.2%}')} |
| expected revenue on observed wins | {observed_policy_revenue} |
| bid cost on observed wins | {_f(opt.get('observed_policy_total_bid_cost'))} |
| observed profit on historical wins | {observed_policy_profit} |
| expected CM on observed wins | {observed_policy_cm} |

## Offline bid optimizer (predicted, not measured)
| metric | value |
| --- | --- |
| rows evaluated | {opt.get('optimizer_rows', '—')} |
| probability-weighted expected-profit lift | {probability_weighted_profit_lift} |
| probability-weighted expected-profit lift % | {probability_weighted_profit_lift_pct} |
| total probability-weighted expected profit: current → recommended | {total_profit} |
| avg predicted win rate: current → recommended | {avg_win_rate} |
| avg / median bid change | {bid_change} |
| bid up / down / same | {bid_direction} |
| avg recommended CM if won | {_f(opt.get('avg_recommended_bid_cm_if_won'))} |
"""
    create_markdown_artifact(
        key=f"train-model-{lead_type_name.strip().lower()}",
        markdown=md,
        description=f"Latest {lead_type_name} model training run",
    )


def _notify_success(lead_type_name, lead_type_id, result, m, opt) -> None:
    """Send the successful training-run notification.

    Inputs
    ------
    lead_type_name : str
        Human-readable lead type name.
    lead_type_id : int
        SmartHub lead type identifier.
    result : dict
        Training-run result dictionary.
    m : dict
        Model evaluation metrics.
    opt : dict
        Optimizer summary metrics.
    """

    def _f(value, fmt="{:.4f}"):
        """Execute  f.

        Inputs
        ------
        value : Any
            Value to process.
        fmt : str
            Format string used for display.
        """
        try:
            return fmt.format(float(value))
        except (TypeError, ValueError):
            return "—"

    lineage = result.get("lineage", {})
    fb = _feature_breakdown(lead_type_id, result.get("feature_cols"))
    promoted = result.get("promoted")
    eligibility_status = result.get("eligibility_status")
    promotion_status = result.get("promotion_status")
    promotion_mode = result.get("promotion_mode")
    if promotion_mode == "disabled":
        icon = ":fast_forward:"
        state = "Promotion evaluation disabled"
    elif promoted:
        icon = ":white_check_mark:"
        state = "Promoted to serving"
    elif promotion_status == "awaiting_manual_promotion":
        icon = ":large_yellow_circle:"
        state = "Eligible — awaiting manual promotion"
    elif eligibility_status == "eligible":
        icon = ":pause_button:"
        state = "Eligible — promotion execution skipped"
    else:
        icon = ":pause_button:"
        state = "Not eligible — serving model unchanged"
    headline = (
        f"{icon} *{state}* · "
        f"`{result.get('production_model_version') or result.get('training_run_id')}`\n"
        f"{result.get('promotion_reason', '—')}"
    )

    reg_used = f"{fb['n_registered_used']}/{fb['n_registered']}"
    feature_title = f"Features · {fb['total']} ({reg_used} from registry)"
    groups = [
        (
            "Model",
            {
                "Model": f"{lineage.get('model_type')} "
                f"(cal={lineage.get('calibrated')})",
                "Rows trained": result["prep_summary"].get("training_rows"),
                "Training run ID": result.get("training_run_id"),
                "Production version": (result.get("production_model_version") or "—"),
                "Promotion mode": promotion_mode,
                "Eligibility status": eligibility_status,
                "Promotion status": promotion_status,
                "Data range": (
                    f"{lineage.get('data_min_created_at')} → "
                    f"{lineage.get('data_max_created_at')}"
                ),
                "Trained on table": f"`{lineage.get('training_table_version')}`",
            },
        ),
        (
            "Performance (held-out)",
            {
                "ROC AUC": _f(m.get("roc_auc")),
                "PR AUC": _f(m.get("pr_auc")),
                "Log loss": _f(m.get("log_loss")),
                "F2": _f(m.get("f2")),
                "Calibration error": _f(m.get("calibration_error")),
            },
        ),
        (
            "Observed production policy (held-out)",
            {
                "Rows": opt.get("optimizer_rows", "—"),
                "Wins": opt.get("observed_policy_wins", "—"),
                "Win rate": _f(
                    opt.get("observed_policy_win_rate"),
                    "{:.2%}",
                ),
                "Expected revenue": _f(
                    opt.get("observed_policy_total_expected_revenue")
                ),
                "Bid cost": _f(opt.get("observed_policy_total_bid_cost")),
                "Observed profit on historical wins": _f(
                    opt.get("observed_policy_total_expected_profit")
                ),
                "Expected CM": _f(
                    opt.get("observed_policy_expected_cm"),
                ),
            },
        ),
        (
            "Bid optimizer (predicted)",
            {
                "Probability-weighted expected-profit lift": _f(
                    opt.get("expected_profit_lift_total")
                ),
                "Probability-weighted expected-profit lift %": _f(
                    opt.get("expected_profit_lift_pct"),
                    "{:.2%}",
                ),
                "Probability-weighted expected profit": (
                    f"{_f(opt.get('current_bid_total_expected_profit'))} → "
                    f"{_f(opt.get('recommended_bid_total_expected_profit'))}"
                ),
                "Avg win rate": (
                    f"{_f(opt.get('avg_current_bid_predicted_win_rate'))} → "
                    f"{_f(opt.get('avg_recommended_bid_predicted_win_rate'))}"
                ),
                "Avg / median bid change": (
                    f"{_f(opt.get('avg_bid_change'))} / "
                    f"{_f(opt.get('median_bid_change'))}"
                ),
                "Bid up/down/same": (
                    f"{_f(opt.get('bid_increase_pct'), '{:.2f}%')} / "
                    f"{_f(opt.get('bid_decrease_pct'), '{:.2f}%')} / "
                    f"{_f(opt.get('bid_unchanged_pct'), '{:.2f}%')}"
                ),
                "Avg recommended CM": _f(opt.get("avg_recommended_bid_cm_if_won")),
            },
        ),
        (
            feature_title,
            {
                "Registry features used": reg_used,
                "Registry features unused": ", ".join(fb["unused"]) or "none",
            },
        ),
    ]
    notifications.notify_success_grouped(
        "train-model",
        subject=f"{lead_type_name} ({lead_type_id})",
        headline=headline,
        groups=groups,
        footer_extra=f"model `{result['model_path']}`",
    )


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments for local flow execution.

    Returns
    -------
    argparse.Namespace
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Run the SmartHub Prefect model-training flow."
    )
    parser.add_argument(
        "--lead-type-id",
        type=int,
        required=True,
        help="SmartHub lead type identifier to train.",
    )
    parser.add_argument(
        "--version",
        default=None,
        help="Optional training-table or model version identifier.",
    )
    parser.add_argument(
        "--no-register-mlflow",
        action="store_true",
        help="Disable MLflow logging and registration for this run.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train_flow(
        lead_type_id=args.lead_type_id,
        version=args.version,
        register_mlflow=not args.no_register_mlflow,
    )
