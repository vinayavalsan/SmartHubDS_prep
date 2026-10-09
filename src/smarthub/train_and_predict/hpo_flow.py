"""Prefect HPO flows, fixed-cadence scheduling, and daily candidate retries."""

from __future__ import annotations

import time
from datetime import date, datetime, timedelta, timezone
from functools import wraps
from zoneinfo import ZoneInfo

import yaml
from prefect import flow, task

from smarthub.core import notifications, paths
from smarthub.core.lead_types import lead_type_name as resolve_lead_type_name
from smarthub.core.logging_utils import get_logger
from smarthub.train_and_predict import config, model_parameters
from smarthub.train_and_predict.flow import train_flow
from smarthub.train_and_predict.hyperparameter_search import run_hyperparameter_search

logger = get_logger(__name__)


def _notification_fields(lead_type_id):
    try:
        label = f"{resolve_lead_type_name(lead_type_id)} ({lead_type_id})"
    except (ValueError, KeyError, TypeError):
        label = str(lead_type_id)
    return {"Lead type": label}


def _send_notification(status, fields, error=None):
    try:
        fields = dict(fields)
        subject = fields.pop("Lead type", None)
        headline = fields.pop("Status", None)
        severity = "success" if status == "started" else status
        display_status = {
            "started": "started",
            "success": "completed",
            "warning": "WARNING",
            "failure": "FAILED",
        }.get(status, status)
        if headline == "HPO candidate promoted":
            display_status = "promoted"
        elif headline == "HPO candidate not promoted":
            display_status = "completed (not promoted)"
        if error:
            fields["Error"] = str(error).strip()[:1500]
        delivered = notifications.notify_grouped(
            severity,
            "hpo",
            subject=subject,
            status=display_status,
            headline=headline,
            groups=[("HPO", fields)],
        )
        if not delivered:
            logger.info("HPO Slack alert was not delivered or Slack is disabled.")
    except Exception:
        logger.warning("HPO Slack notification failed; run continues.", exc_info=True)


def _report_search(function):
    """Report the Prefect search task; standalone searches remain notification-free."""

    @wraps(function)
    def wrapped(lead_type_id, version=None, config_path=None):
        started = time.perf_counter()
        fields = {
            **_notification_fields(lead_type_id),
            "Status": "HPO started",
            "Requested dataset": version or "latest",
        }
        _send_notification("started", fields)
        result = function(lead_type_id, version, config_path)
        _send_notification(
            "success",
            {
                **fields,
                "Status": "HPO completed; parameters saved",
                "HPO run": result.get("hpo_run_id"),
                "Parameter version": result.get("parameter_version"),
                "HPO MLflow run": result.get("hpo_mlflow_run_id"),
                "Model": result.get("model_type"),
                "Selected trial": result.get("selected_trial"),
                "Calibration": result.get("selected_calibration_method"),
                "Holdout log loss": (
                    result.get("holdout_probability_metrics") or {}
                ).get("log_loss"),
                "Parameter file": result.get("parameters_path"),
                "Duration (seconds)": round(time.perf_counter() - started, 1),
            },
        )
        return result

    return wrapped


def notify_candidate_result(lead_type_id, hpo_result, training_result, state=None):
    """Keep successful search separate from acceptance of its trained candidate."""
    promoted = training_result.get("promoted") is True
    state = state or {}
    _send_notification(
        "success",
        {
            **_notification_fields(lead_type_id),
            "Status": (
                "HPO candidate promoted" if promoted else "HPO candidate not promoted"
            ),
            "HPO run": hpo_result.get("hpo_run_id"),
            "Parameter version": hpo_result.get("parameter_version"),
            "Training run": training_result.get("training_run_id"),
            "Promotion status": training_result.get("promotion_status"),
            "Reason": training_result.get("promotion_reason"),
            "Current parameters": "Updated" if promoted else "Unchanged",
            "Next HPO retry": state.get("retry_date") if not promoted else None,
            "Next scheduled HPO": state.get("next_scheduled_date"),
            "Timezone": (state.get("schedule") or {}).get("timezone"),
        },
    )


def notify_candidate_error(lead_type_id, hpo_result, state, error):
    """Report execution errors and the persistent next-day retry decision."""
    error._smarthub_hpo_reported = True
    phase = state["status"]
    _send_notification(
        "failure",
        {
            **_notification_fields(lead_type_id),
            "Status": (
                "HPO candidate training failed"
                if phase == "training_failed"
                else (
                    "HPO failed; retry scheduled"
                    if state.get("retry_date")
                    else "HPO failed"
                )
            ),
            "HPO run": (hpo_result or {}).get("hpo_run_id"),
            "Promotion": (
                "Outcome not confirmed"
                if phase == "training_failed"
                else "Candidate training not reached"
            ),
            "Next HPO retry": state.get("retry_date"),
            "Next scheduled HPO": state.get("next_scheduled_date"),
            "Timezone": (state.get("schedule") or {}).get("timezone"),
        },
        error=f"{type(error).__name__}: {error}",
    )


def _report_flow_failure(function):
    """Report preflight errors; candidate errors already carry retry details."""

    @wraps(function)
    def wrapped(lead_type_id, *args, **kwargs):
        try:
            return function(lead_type_id, *args, **kwargs)
        except Exception as exc:
            if not getattr(exc, "_smarthub_hpo_reported", False):
                _send_notification(
                    "failure",
                    {**_notification_fields(lead_type_id), "Status": "HPO flow failed"},
                    error=f"{type(exc).__name__}: {exc}",
                )
            raise

    return wrapped


@task(name="hyperparameter-search", persist_result=False, cache_policy=None)
@_report_search
def _hpo_task(lead_type_id, version, config_path):
    return run_hyperparameter_search(lead_type_id, version, config_path)


@flow(name="smarthub-hpo", log_prints=True)
@_report_flow_failure
def hpo_flow(
    lead_type_id: int,
    version: str | None = None,
    config_path: str | None = None,
    train_candidate: bool = False,
) -> dict:
    """Save HPO results and optionally train the exact candidate as a subflow.

    Ordinary scheduled training uses train_flow without parameter_file.
    A future HPO deployment can enable train_candidate to perform the handoff.
    """
    result = None
    phase = "hpo_failed"
    try:
        result = _hpo_task(lead_type_id, version, config_path)
        if train_candidate:
            phase = "training_failed"
            result["candidate_training"] = train_flow.with_options(on_failure=[])(
                lead_type_id=lead_type_id,
                parameter_file=result["parameters_path"],
                register_mlflow=True,
            )
            notify_candidate_result(lead_type_id, result, result["candidate_training"])
        return result
    except Exception as exc:
        notify_candidate_error(lead_type_id, result, {"status": phase}, exc)
        raise


def run_daily_cycle(lead_type_id, *, now=None, hpo_runner, training_runner):
    """Run HPO when scheduled/pending, otherwise run ordinary daily training.

    Deploy one serialized daily cycle per lead type on a shared persistent
    filesystem. A rejected candidate is a completed run with retry_pending;
    exceptions are persisted and re-raised so Prefect also reports failure.
    """
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        raise ValueError("Scheduling requires a timezone-aware timestamp.")
    cfg = config.load_training_config(lead_type_id)
    policy = cfg.raw.get("hpo_schedule") or {}
    interval = policy.get("interval_days")
    retry = policy.get("retry_days")
    if any(type(value) is not int or value < 1 for value in (interval, retry)):
        raise ValueError("HPO interval_days and retry_days must be positive integers.")
    anchor = date.fromisoformat(policy["anchor_date"])
    today = now.astimezone(ZoneInfo(policy["timezone"])).date()
    path = paths.resolve(policy["state_file"])
    if path.exists():
        with path.open(encoding="utf-8") as stream:
            state = yaml.safe_load(stream)
        if (
            not isinstance(state, dict)
            or state.get("schema_version") != 1
            or state.get("lead_type_id") != lead_type_id
            or state.get("schedule") != policy
        ):
            raise ValueError(
                "Invalid/changed HPO schedule state; reconcile it explicitly."
            )
    else:
        state = {
            "schema_version": 1,
            "lead_type_id": lead_type_id,
            "schedule": policy,
            "next_scheduled_date": anchor.isoformat(),
            "retry_pending": False,
            "retry_date": None,
        }
    current = model_parameters.resolve_training_parameters(cfg, lead_type_id)
    # Recover a worker crash after model promotion, before the success journal.
    if (
        state["retry_pending"]
        and state.get("hpo_run_id")
        and current.get("hpo_run_id") == state["hpo_run_id"]
        and current.get("approved_training_run_id")
    ):
        state.update(
            status="promoted",
            retry_pending=False,
            retry_date=None,
            training_run_id=current["approved_training_run_id"],
        )
        model_parameters.atomic_write_yaml(path, state)
        notify_candidate_result(
            lead_type_id,
            current,
            {
                "promoted": True,
                "training_run_id": current["approved_training_run_id"],
                "promotion_reason": "Recovered already-serving HPO candidate",
            },
            state,
        )
    scheduled_due = today >= date.fromisoformat(state["next_scheduled_date"])
    retry_due = state["retry_pending"] and (
        state["retry_date"] is None or today >= date.fromisoformat(state["retry_date"])
    )
    if not scheduled_due and not retry_due:
        return {
            "action": "daily_training",
            "hpo_state": state,
            "training": training_runner(lead_type_id=lead_type_id),
        }
    if scheduled_due:
        # Advance from the calendar anchor, never from a retry/promotion date.
        boundary = ((today - anchor).days // interval + 1) * interval
        state["next_scheduled_date"] = (anchor + timedelta(days=boundary)).isoformat()
    state.update(
        status="running_hpo",
        retry_pending=True,
        retry_date=(today + timedelta(days=retry)).isoformat(),
        started_at=now.isoformat(),
        hpo_run_id=None,
        training_run_id=None,
        error=None,
    )
    model_parameters.atomic_write_yaml(path, state)
    phase = "hpo_failed"
    hpo_result = None
    try:
        hpo_result = hpo_runner(lead_type_id, None, None)
        state.update(status="running_training", hpo_run_id=hpo_result["hpo_run_id"])
        model_parameters.atomic_write_yaml(path, state)
        phase = "training_failed"
        training_result = training_runner(
            lead_type_id=lead_type_id,
            parameter_file=hpo_result["parameters_path"],
            register_mlflow=True,
        )
        promoted = training_result["promoted"] is True
        state.update(
            status="promoted" if promoted else "not_promoted",
            retry_pending=not promoted,
            retry_date=(
                None if promoted else (today + timedelta(days=retry)).isoformat()
            ),
            training_run_id=training_result.get("training_run_id"),
            promotion_reason=training_result.get("promotion_reason"),
        )
        model_parameters.atomic_write_yaml(path, state)
        notify_candidate_result(lead_type_id, hpo_result, training_result, state)
        return {
            "action": "hpo_candidate",
            "hpo_state": state,
            "hpo": hpo_result,
            "training": training_result,
        }
    except Exception as exc:
        state.update(status=phase, error=f"{type(exc).__name__}: {exc}")
        model_parameters.atomic_write_yaml(path, state)
        notify_candidate_error(lead_type_id, hpo_result, state, exc)
        raise


@flow(
    name="smarthub-daily-model-cycle",
    log_prints=True,
)
@_report_flow_failure
def daily_model_cycle(lead_type_id: int, register_mlflow: bool = True) -> dict:
    """Deploy daily per lead type with concurrency one and persistent shared data."""

    def training_runner(**kwargs):
        kwargs.setdefault("register_mlflow", register_mlflow)
        # The parent reports candidate failures with the retry schedule.
        runner = (
            train_flow.with_options(on_failure=[])
            if kwargs.get("parameter_file")
            else train_flow
        )
        try:
            return runner(**kwargs)
        except Exception as exc:
            if not kwargs.get("parameter_file"):
                # Ordinary daily training keeps its existing failure hook.
                exc._smarthub_hpo_reported = True
            raise

    return run_daily_cycle(
        lead_type_id,
        hpo_runner=_hpo_task,
        training_runner=training_runner,
    )
