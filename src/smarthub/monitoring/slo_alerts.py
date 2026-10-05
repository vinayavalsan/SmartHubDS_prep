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
from datetime import datetime, timezone

from smarthub.core import notifications
from smarthub.monitoring import slo

logger = logging.getLogger("smarthub.monitoring.slo_alerts")

_AREA = "bid-api"

# Short label shown in the alert title's subject, per breach metric.
_METRIC_LABEL = {
    "tat_p99_seconds": "p99",
    "error_rate_pct": "error-rate",
    "no_requests_minutes": "no-requests",
    "shap_backlog": "SHAP backlog",
    "predictions_over_1s": "over-1s TAT",
}
# Value label shown inside the code-block for non-"over-1s" breaches.
_VALUE_LABEL = {
    "tat_p99_seconds": "TAT p99 (s)",
    "error_rate_pct": "Error rate %",
    "no_requests_minutes": "No requests (min)",
    "shap_backlog": "SHAP backlog",
}

# In-memory alert state so we notify on START / severity-change (not every loop),
# remind only for unresolved CRITICAL conditions, and send a RECOVERED when a
# breach clears. The slo-alerts container runs a continuous loop, so process-local
# state is sufficient (a restart simply re-sends "new" for anything still breaching).
_ALERT_STATE: dict[str, dict] = {}


def _store():
    from smarthub.train_and_predict.prediction_log_schema import PredictionLogStore

    return PredictionLogStore()


def _reminder_minutes() -> float:
    """Minutes between reminders for an unresolved CRITICAL breach (default 60)."""
    try:
        return float(os.getenv("SMARTHUB_SLO_REMINDER_MINUTES", "60"))
    except ValueError:
        return 60.0


def _fmt_time(value) -> str:
    """Format a row timestamp as HH:MM:SSZ (best-effort)."""
    if isinstance(value, datetime):
        return value.strftime("%H:%M:%SZ")
    return str(value) if value else "?"


def _breach_fields(breach: dict, window_minutes: int) -> dict:
    """Build the code-block fields for one breach."""
    fields = {"Window": f"last {window_minutes} min"}
    metric = breach["metric"]
    if metric == "predictions_over_1s":
        reqs = breach.get("requests")
        fields["Over 1s"] = (
            f"{breach['value']} of {reqs} requests" if reqs else str(breach["value"])
        )
        examples = breach.get("examples") or []
        if examples:
            pid, tat, ts = examples[0]
            fields["Slowest"] = f"{pid}  {tat:.2f}s  {_fmt_time(ts)}"
            if len(examples) > 1:
                fields["Also"] = "; ".join(
                    f"{pid} {tat:.2f}s" for pid, tat, _ in examples[1:]
                )
    else:
        fields[_VALUE_LABEL.get(metric, metric)] = breach["value"]
    return fields


def _send_breach(breach: dict, window_minutes: int, tag: str) -> None:
    """Route one breach to its severity's channel with a state tag in the title."""
    label = _METRIC_LABEL.get(breach["metric"], breach["metric"])
    footer = (
        "look up prediction_id in smarthub_prediction_log"
        if breach["metric"] == "predictions_over_1s"
        else None
    )
    notifications.notify_grouped(
        breach["severity"],
        _AREA,
        subject=f"{label} ({tag})",
        headline=breach["message"],
        groups=[("", _breach_fields(breach, window_minutes))],
        footer_extra=footer,
    )


def _send_recovered(metric: str, _prev: dict) -> None:
    """Send a RECOVERED (success) notification when a breach clears."""
    label = _METRIC_LABEL.get(metric, metric)
    notifications.notify_grouped(
        "success",
        _AREA,
        subject=f"{label} (recovered)",
        headline=f"{label} back within threshold",
    )


def _dispatch(breaches: list[dict], window_minutes: int) -> None:
    """Notify on new/severity-change, remind unresolved criticals, and recover."""
    now = datetime.now(timezone.utc)
    reminder = _reminder_minutes()
    active = {b["metric"]: b for b in breaches}

    # Recovered: previously firing, no longer breaching.
    for metric in list(_ALERT_STATE):
        if metric not in active:
            prev = _ALERT_STATE.pop(metric)
            _send_recovered(metric, prev)

    # New / severity-change / due reminder.
    for metric, breach in active.items():
        prev = _ALERT_STATE.get(metric)
        if prev is None or prev["severity"] != breach["severity"]:
            tag, since = "new", now
        elif (
            breach["severity"] == "critical"
            and (now - prev["last_notified"]).total_seconds() >= reminder * 60
        ):
            hours = max(1, round((now - prev["since"]).total_seconds() / 3600))
            tag, since = f"reminder · {hours}h", prev["since"]
        else:
            continue  # already notified and not yet due for a reminder
        _send_breach(breach, window_minutes, tag)
        _ALERT_STATE[metric] = {
            "severity": breach["severity"],
            "since": since,
            "last_notified": now,
        }


def check_once(window_minutes: int | None = None) -> list[dict]:
    """Evaluate SLOs once; route per-breach Slack alerts. Returns the breaches."""
    window_minutes = window_minutes or int(
        os.getenv("SMARTHUB_SLO_WINDOW_MINUTES", "15")
    )
    slis = slo.compute_slis(_store(), window_minutes=window_minutes)
    breaches = slo.evaluate_alerts(slis)

    if breaches:
        logger.warning("SLO breach: %s", "; ".join(b["message"] for b in breaches))
    else:
        logger.info(
            "SLO OK (window=%dm): %d req, p99=%s, err=%.2f%%, backlog=%d",
            window_minutes,
            slis["requests"],
            slis["tat_p99"],
            slis["error_rate_pct"],
            slis["shap_backlog"],
        )

    _dispatch(breaches, window_minutes)
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
