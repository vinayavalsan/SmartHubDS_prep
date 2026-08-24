"""Service-level indicators (SLIs) + alert evaluation for the bid API.

Computes live SLIs from the prediction log (the same Postgres table serving
writes to) over a recent window, and evaluates them against SLO thresholds.
Shared by the Streamlit **Health** page (``health_app``) and the scheduled
Slack alert check (``slo_alerts``), so both report identical numbers.

No new infrastructure: everything is derived from rows already logged, plus the
shap-worker backlog gauge. Thresholds come from ``config/smarthub.yaml`` under
the ``slo`` section (env/file), with sensible code defaults.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from smarthub.core import task_config


def thresholds() -> dict[str, float]:
    """Alert thresholds (config-overridable). Breaching any one fires an alert."""
    return {
        # p99 turnaround time must stay under the 1s TAT. Alert a bit early.
        "tat_p99_seconds": task_config.get_float("slo", "tat_p99_seconds", 0.8),
        # fraction of requests that errored, as a percentage.
        "error_rate_pct": task_config.get_float("slo", "error_rate_pct", 1.0),
        # SHAP enrichment backlog (rows awaiting explanation).
        "shap_backlog": task_config.get_int("slo", "shap_backlog", 1000),
        # No successful request for this long => possible outage / no traffic.
        "no_requests_minutes": task_config.get_float(
            "slo", "no_requests_minutes", 10.0
        ),
    }


def _pct(sorted_vals: list[float], p: float) -> float | None:
    if not sorted_vals:
        return None
    idx = min(int(p / 100.0 * len(sorted_vals)), len(sorted_vals) - 1)
    return sorted_vals[idx]


def compute_slis(store, window_minutes: int = 15) -> dict[str, Any]:
    """Compute SLIs from the prediction log over the last ``window_minutes``.

    Returns a flat dict of indicators (counts, TAT percentiles, error rate,
    request rate, SHAP backlog, freshness, decision-path mix).
    """
    rows = store.window_rows(minutes=window_minutes)
    n = len(rows)

    tats = sorted(
        float(r["tat_seconds"]) for r in rows if r.get("tat_seconds") is not None
    )
    errors = sum(1 for r in rows if r.get("status") == "error")
    within_1s = sum(1 for t in tats if t <= 1.0)

    # Freshness: seconds since the most recent successful request.
    last_ok_age = None
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    ok_times = [
        r["created_at"]
        for r in rows
        if r.get("status") in ("success", "ok") and r.get("created_at")
    ]
    if ok_times:
        newest = max(ok_times)
        if isinstance(newest, datetime):
            last_ok_age = max(0.0, (now - newest.replace(tzinfo=None)).total_seconds())

    paths: dict[str, int] = {}
    for r in rows:
        p = r.get("decision_path") or "unknown"
        paths[p] = paths.get(p, 0) + 1

    return {
        "window_minutes": window_minutes,
        "requests": n,
        "rate_per_min": round(n / window_minutes, 1) if window_minutes else None,
        "tat_p50": _pct(tats, 50),
        "tat_p95": _pct(tats, 95),
        "tat_p99": _pct(tats, 99),
        "tat_max": tats[-1] if tats else None,
        "within_1s_pct": round(100 * within_1s / len(tats), 2) if tats else None,
        "errors": errors,
        "error_rate_pct": round(100 * errors / n, 2) if n else 0.0,
        "shap_backlog": store.pending_shap_count(),
        "last_ok_age_seconds": last_ok_age,
        "decision_paths": paths,
    }


def evaluate_alerts(slis: dict, thr: dict | None = None) -> list[dict]:
    """Return a list of breached SLOs: ``[{metric, value, threshold, message}]``.

    An empty list means all green. Freshness only alerts when there has been at
    least one request in the window (so a genuinely idle service that has never
    served isn't a false alarm -- but a service that *was* serving and went
    silent is).
    """
    thr = thr or thresholds()
    breaches: list[dict] = []

    p99 = slis.get("tat_p99")
    if p99 is not None and p99 > thr["tat_p99_seconds"]:
        breaches.append(
            {
                "metric": "tat_p99_seconds",
                "value": round(p99, 3),
                "threshold": thr["tat_p99_seconds"],
                "message": f"TAT p99 {p99:.3f}s > {thr['tat_p99_seconds']}s "
                "(approaching/over the 1s SLA).",
            }
        )

    err = slis.get("error_rate_pct") or 0.0
    if slis.get("requests") and err > thr["error_rate_pct"]:
        breaches.append(
            {
                "metric": "error_rate_pct",
                "value": err,
                "threshold": thr["error_rate_pct"],
                "message": f"Error rate {err:.2f}% > {thr['error_rate_pct']}%.",
            }
        )

    backlog = slis.get("shap_backlog") or 0
    if backlog > thr["shap_backlog"]:
        breaches.append(
            {
                "metric": "shap_backlog",
                "value": backlog,
                "threshold": thr["shap_backlog"],
                "message": f"SHAP backlog {backlog} rows > {thr['shap_backlog']} "
                "(shap-worker falling behind; bids unaffected).",
            }
        )

    age = slis.get("last_ok_age_seconds")
    if age is not None and age > thr["no_requests_minutes"] * 60:
        breaches.append(
            {
                "metric": "no_requests_minutes",
                "value": round(age / 60, 1),
                "threshold": thr["no_requests_minutes"],
                "message": f"No successful request for {age/60:.1f} min "
                f"> {thr['no_requests_minutes']} min (possible outage).",
            }
        )

    return breaches
