"""Scheduled SLO alert check for the bid API -> Slack.

Computes the same SLIs as the Health page over a recent window, evaluates them
against the SLO thresholds, and posts a Slack alert (via the existing
``core.notifications`` webhook) when any threshold is breached. Reuses existing
infrastructure only -- no Prometheus/Alertmanager.

Run once (cron / Prefect deployment friendly):
    python -m smarthub.monitoring.slo_alerts

Or loop in-process (e.g. a lightweight sidecar):
    python -m smarthub.monitoring.slo_alerts --loop --interval 60

Knobs (env):
    SMARTHUB_SLO_WINDOW_MINUTES   window to evaluate (default 15)
    SLACK_WEBHOOK_URL             where alerts go (unset -> logs only, no-op)
"""

from __future__ import annotations

import argparse
import logging
import os
import time

from smarthub.core import notifications
from smarthub.monitoring import slo

logger = logging.getLogger("smarthub.monitoring.slo_alerts")


def _store():
    from smarthub.train_and_predict.prediction_log_schema import PredictionLogStore

    return PredictionLogStore()


def check_once(window_minutes: int | None = None) -> list[dict]:
    """Evaluate SLOs once; Slack-alert on any breach. Returns the breaches."""
    window_minutes = window_minutes or int(
        os.getenv("SMARTHUB_SLO_WINDOW_MINUTES", "15")
    )
    slis = slo.compute_slis(_store(), window_minutes=window_minutes)
    breaches = slo.evaluate_alerts(slis)

    if not breaches:
        logger.info(
            "SLO OK (window=%dm): %d req, p99=%s, err=%.2f%%, backlog=%d",
            window_minutes,
            slis["requests"],
            slis["tat_p99"],
            slis["error_rate_pct"],
            slis["shap_backlog"],
        )
        return []

    fields = {
        "Window": f"last {window_minutes} min",
        "Requests": slis["requests"],
        "Rate/min": slis["rate_per_min"],
        "TAT p99 (s)": slis["tat_p99"],
        "Within 1s %": slis["within_1s_pct"],
        "Error rate %": slis["error_rate_pct"],
        "SHAP backlog": slis["shap_backlog"],
        "Breaches": ", ".join(b["metric"] for b in breaches),
    }
    error_text = "\n".join(b["message"] for b in breaches)
    logger.warning("SLO breach: %s", error_text)
    notifications.notify_failure("bid-api-slo", fields, error=error_text)
    return breaches


def main() -> None:
    logging.basicConfig(
        level=os.getenv("SMARTHUB_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    ap = argparse.ArgumentParser()
    ap.add_argument("--loop", action="store_true", help="check repeatedly")
    ap.add_argument("--interval", type=float, default=60.0, help="loop seconds")
    ap.add_argument("--window", type=int, default=None, help="window minutes")
    args = ap.parse_args()

    if not notifications.slack_enabled():
        logger.warning(
            "SLACK_WEBHOOK_URL not set -- alerts will be logged only, not sent."
        )

    if not args.loop:
        check_once(args.window)
        return
    while True:
        try:
            check_once(args.window)
        except Exception:  # noqa: BLE001 -- never let the alerter die
            logger.warning("SLO check failed", exc_info=True)
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
