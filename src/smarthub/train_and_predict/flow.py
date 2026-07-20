"""Prefect flow that trains the Anton model and runs the offline bid optimizer.

STEP 3 of the pipeline (after data-pull → build-features). One deployment, two
schedules (auto / home). ``log_prints=True`` so every print() from the training
pipeline (metrics tables, optimizer summary) lands in the Prefect run logs — you
get maximum visibility. Failures are alerted to Slack via the shared
``flow_failure_hook``; on success a rich Slack notification + a Prefect markdown
artifact summarise the run.

Requires the ``ml`` extra (scikit-learn, mlflow, matplotlib, joblib).
"""

from __future__ import annotations

from prefect import flow, get_run_logger, task
from prefect.artifacts import create_markdown_artifact

from smarthub.core import notifications
from smarthub.feature_engineering import features as fe

from . import train


def _feature_breakdown(lead_type_id, feature_cols):
    """Split the trained features into mandatory / optional-on / optional-off."""
    feature_cols = list(feature_cols or [])
    mandatory = fe.mandatory_features(lead_type_id)
    optional_all = fe.optional_features(lead_type_id)
    used = set(feature_cols)
    return {
        "total": len(feature_cols),
        "n_mandatory": len(mandatory & used),
        "optional_on": sorted(optional_all & used),
        "optional_off": sorted(optional_all - used),
    }


@task(name="train-anton-model")
def _train_task(lead_type_id, lead_type_name, version, register_mlflow):
    """Run the full training + offline optimizer evaluation for one lead type."""
    logger = get_run_logger()
    return train.run_training(
        lead_type_id=lead_type_id,
        lead_type_name=lead_type_name,
        version=version,
        register_mlflow=register_mlflow,
        log=logger,
    )


@flow(
    name="smarthub-train-model",
    log_prints=True,
    on_failure=[notifications.flow_failure_hook],
)
def train_flow(
    lead_type_id: int = 6,
    lead_type_name: str = "auto",
    version: str | None = None,
    register_mlflow: bool = True,
) -> dict:
    """Train + evaluate + score (offline optimizer) one lead type's model.

    Depends on STEP 2 (build-features): if no training table exists yet, the
    load fails and ``flow_failure_hook`` sends a "run build-features first" alert.
    """
    logger = get_run_logger()
    logger.info(
        "STEP 3 train-model starting: lead_type=%s (%s), version=%s, mlflow=%s",
        lead_type_name,
        lead_type_id,
        version or "latest",
        register_mlflow,
    )

    result = _train_task(lead_type_id, lead_type_name, version, register_mlflow)

    m = result["metrics"]
    opt = result["optimizer_summary"] or {}
    logger.info(
        "Trained %s model %s: ROC AUC=%.4f, rows=%s -> %s (promoted=%s)",
        lead_type_name,
        result.get("model_version"),
        m.get("roc_auc", float("nan")),
        result["prep_summary"]["training_rows"],
        result["model_path"],
        result.get("promoted"),
    )

    _report(lead_type_name, lead_type_id, result, m, opt)
    _notify_success(lead_type_name, lead_type_id, result, m, opt)
    return {
        "lead_type": lead_type_name,
        "roc_auc": m.get("roc_auc"),
        "rows_trained": result["prep_summary"]["training_rows"],
        "model_path": result["model_path"],
    }


def _report(lead_type_name, lead_type_id, result, m, opt) -> None:
    """Publish a Prefect markdown artifact summarising the training run."""
    prep = result["prep_summary"]

    def _f(value, fmt="{:.4f}"):
        try:
            return fmt.format(float(value))
        except (TypeError, ValueError):
            return "—"

    lineage = result.get("lineage", {})
    fb = _feature_breakdown(lead_type_id, result.get("feature_cols"))
    promoted = result.get("promoted")
    status_label = (
        "✅ PROMOTED to currently-serving"
        if promoted
        else "⏸ HELD (currently-serving model unchanged)"
    )
    md = f"""# Train model — {lead_type_name}

## Promotion decision: {status_label}
{result.get('promotion_reason', '—')}

| field | value |
| --- | --- |
| lead_type_id | {lead_type_id} |
| model version | `{result.get('model_version')}` |
| model | {lineage.get('model_type')} (calibrated={lineage.get('calibrated')}) |
| rows trained | {prep.get('training_rows')} |
| observed win rate | {_f(prep.get('win_rate'))} |
| training table version | `{lineage.get('training_table_version')}` |
| trained on data | `{lineage.get('data_min_created_at')}` → \
`{lineage.get('data_max_created_at')}` |
| model path | `{result['model_path']}` |
| report dir | `{result['report_dir']}` |
| features | {fb['total']} ({fb['n_mandatory']} mandatory + \
{len(fb['optional_on'])} optional) |
| optional included | {', '.join(fb['optional_on']) or '—'} |
| optional excluded | {', '.join(fb['optional_off']) or '—'} |

## Model quality (held-out test)
| metric | value |
| --- | --- |
| ROC AUC | {_f(m.get('roc_auc'))} |
| PR AUC | {_f(m.get('pr_auc'))} |
| Log loss | {_f(m.get('log_loss'))} |
| Brier score | {_f(m.get('brier_score'))} |
| Calibration error | {_f(m.get('calibration_error'))} |

## Offline bid optimizer (predicted, not measured)
| metric | value |
| --- | --- |
| rows evaluated | {opt.get('optimizer_rows', '—')} |
| expected-profit lift % | {_f(opt.get('expected_profit_lift_pct'), '{:.2%}')} |
| avg recommended CM if won | {_f(opt.get('avg_recommended_bid_cm_if_won'))} |
| bid up / down / same | {opt.get('bid_increase_count', '—')} / \
{opt.get('bid_decrease_count', '—')} / {opt.get('bid_unchanged_count', '—')} |
"""
    create_markdown_artifact(
        key=f"train-model-{lead_type_name.strip().lower()}",
        markdown=md,
        description=f"Latest {lead_type_name} model training run",
    )


def _notify_success(lead_type_name, lead_type_id, result, m, opt) -> None:
    """Send the Slack 'training completed' notification, grouped for readability.

    Leads with the promotion decision, then groups the rest under titled
    sections (Model / Performance / Bid optimizer / Features) instead of one
    long flat field list.
    """

    def _f(value, fmt="{:.4f}"):
        try:
            return fmt.format(float(value))
        except (TypeError, ValueError):
            return "—"

    lineage = result.get("lineage", {})
    fb = _feature_breakdown(lead_type_id, result.get("feature_cols"))
    promoted = result.get("promoted")
    icon = ":white_check_mark:" if promoted else ":pause_button:"
    state = (
        "Promoted to serving"
        if promoted
        else "Held — currently-serving model unchanged"
    )
    headline = (
        f"{icon} *{state}* · `{result.get('model_version')}`\n"
        f"{result.get('promotion_reason', '—')}"
    )

    feature_title = (
        f"Features · {fb['total']} "
        f"({fb['n_mandatory']} mandatory + {len(fb['optional_on'])} optional)"
    )
    model_label = f"{lineage.get('model_type')} (cal={lineage.get('calibrated')})"
    groups = [
        (
            "Model",
            {
                "Model": model_label,
                "Rows trained": result["prep_summary"].get("training_rows"),
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
                "Calibration error": _f(m.get("calibration_error")),
            },
        ),
        (
            "Bid optimizer (predicted)",
            {
                "Profit lift %": _f(opt.get("expected_profit_lift_pct"), "{:.2%}"),
                "Avg recommended CM": _f(opt.get("avg_recommended_bid_cm_if_won")),
            },
        ),
        (
            feature_title,
            {
                "Optional included": ", ".join(fb["optional_on"]) or "none",
                "Optional excluded": ", ".join(fb["optional_off"]) or "none",
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


if __name__ == "__main__":
    train_flow(lead_type_id=6, lead_type_name="auto")
