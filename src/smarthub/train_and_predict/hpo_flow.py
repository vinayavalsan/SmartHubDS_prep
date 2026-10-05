"""Prefect HPO flows, fixed-cadence scheduling, and daily candidate retries."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import yaml
from prefect import flow, task

from smarthub.core import notifications, paths
from smarthub.train_and_predict import config, model_parameters
from smarthub.train_and_predict.flow import train_flow
from smarthub.train_and_predict.hyperparameter_search import (
    notify_candidate_error,
    notify_candidate_result,
    run_hyperparameter_search,
)


@task(name="hyperparameter-search", persist_result=False, cache_policy=None)
def _hpo_task(lead_type_id, version, config_path):
    return run_hyperparameter_search(lead_type_id, version, config_path)


@flow(
    name="smarthub-hpo", log_prints=True, on_failure=[notifications.flow_failure_hook]
)
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
    result = _hpo_task(lead_type_id, version, config_path)
    if train_candidate:
        try:
            result["candidate_training"] = train_flow(
                lead_type_id=lead_type_id,
                parameter_file=result["parameters_path"],
                register_mlflow=True,
            )
        except Exception as exc:
            notify_candidate_error(
                lead_type_id, result, {"status": "training_failed"}, exc
            )
            raise
        notify_candidate_result(lead_type_id, result, result["candidate_training"])
    return result


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
    on_failure=[notifications.flow_failure_hook],
)
def daily_model_cycle(lead_type_id: int, register_mlflow: bool = True) -> dict:
    """Deploy daily per lead type with concurrency one and persistent shared data."""

    def training_runner(**kwargs):
        kwargs.setdefault("register_mlflow", register_mlflow)
        return train_flow(**kwargs)

    return run_daily_cycle(
        lead_type_id,
        hpo_runner=_hpo_task,
        training_runner=training_runner,
    )
