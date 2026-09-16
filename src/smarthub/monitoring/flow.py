"""Prefect orchestration for hourly SmartHub model-degradation monitoring."""

from __future__ import annotations

import argparse

import pandas as pd
from prefect import flow, get_run_logger, task

from smarthub.core import notifications
from smarthub.core.lead_types import lead_type_name as resolve_lead_type_name
from smarthub.monitoring.model_degradation import check_once


@task(name="check-model-degradation", retries=2, retry_delay_seconds=60)
def check_model_degradation(lead_type_id: int) -> pd.DataFrame:
    """Run one degradation check for a registered lead type."""
    return check_once(lead_type_id=lead_type_id)


@flow(
    name="smarthub-model-degradation",
    log_prints=True,
    on_failure=[notifications.flow_failure_hook],
)
def model_degradation_flow(lead_type_id: int) -> dict[str, object]:
    """Evaluate degradation and return an hourly run summary."""
    logger = get_run_logger()
    lead_type = resolve_lead_type_name(lead_type_id)
    logger.info(
        "Model degradation monitoring starting: lead_type=%s (%d).",
        lead_type,
        lead_type_id,
    )

    degraded = check_model_degradation(lead_type_id)
    warning_count = (
        int(degraded["severity"].eq("warning").sum())
        if "severity" in degraded.columns
        else 0
    )
    critical_count = (
        int(degraded["severity"].eq("critical").sum())
        if "severity" in degraded.columns
        else 0
    )
    result = {
        "lead_type": lead_type,
        "lead_type_id": lead_type_id,
        "degraded_cohorts": int(len(degraded)),
        "warning_cohorts": warning_count,
        "critical_cohorts": critical_count,
    }
    logger.info("Model degradation monitoring completed: %s.", result)
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the SmartHub Prefect model-degradation flow."
    )
    parser.add_argument(
        "--lead-type-id",
        type=int,
        required=True,
        help="SmartHub lead type identifier to monitor.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    model_degradation_flow(lead_type_id=args.lead_type_id)
