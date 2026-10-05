"""Tests for the SLO over-1s breach and the alert dispatch state machine."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from smarthub.monitoring import slo, slo_alerts


class _Store:
    def __init__(self, rows, backlog=0):
        self._rows = rows
        self._backlog = backlog

    def window_rows(self, minutes=15, limit=50000):
        return self._rows

    def pending_shap_count(self):
        return self._backlog


def _row(tat, status="success", pid=None, created=None):
    return {
        "tat_seconds": tat,
        "status": status,
        "prediction_id": pid,
        "created_at": created or datetime.now(timezone.utc).replace(tzinfo=None),
        "decision_path": "model",
    }


def test_over_1s_breach_detected_and_sorted():
    rows = [_row(0.2), _row(1.5, pid="p-slow"), _row(1.1, pid="p-mid")]
    slis = slo.compute_slis(_Store(rows), window_minutes=15)
    assert slis["over_1s_count"] == 2
    breaches = {b["metric"]: b for b in slo.evaluate_alerts(slis)}
    assert "predictions_over_1s" in breaches
    b = breaches["predictions_over_1s"]
    assert b["severity"] == "failure" and b["value"] == 2
    assert b["examples"][0][0] == "p-slow"  # slowest first


def test_breach_severities():
    """Each breach carries the severity used to route its channel."""
    rows = [_row(2.0, pid="x")] + [_row(0.1, status="error") for _ in range(10)]
    slis = slo.compute_slis(_Store(rows, backlog=5000), window_minutes=15)
    sev = {b["metric"]: b["severity"] for b in slo.evaluate_alerts(slis)}
    assert sev["tat_p99_seconds"] == "critical"
    assert sev["error_rate_pct"] == "critical"
    assert sev["shap_backlog"] == "warning"
    assert sev["predictions_over_1s"] == "failure"


def _capture(monkeypatch):
    sent = []
    monkeypatch.setattr(
        slo_alerts.notifications,
        "notify_grouped",
        lambda severity, area, **k: sent.append((severity, k.get("subject"))) or True,
    )
    slo_alerts._ALERT_STATE.clear()
    return sent


def test_dispatch_new_then_quiet_then_recovered(monkeypatch):
    sent = _capture(monkeypatch)
    b = [
        {
            "metric": "tat_p99_seconds",
            "severity": "critical",
            "value": 1.8,
            "message": "p99 high",
        }
    ]
    slo_alerts._dispatch(b, 15)  # new
    assert sent[-1][0] == "critical" and "(new)" in sent[-1][1]
    slo_alerts._dispatch(b, 15)  # still firing, not due -> quiet
    assert len(sent) == 1
    slo_alerts._dispatch([], 15)  # cleared -> recovered
    assert sent[-1][0] == "success" and "recovered" in sent[-1][1]


def test_dispatch_reminder_after_interval(monkeypatch):
    sent = _capture(monkeypatch)
    monkeypatch.setenv("SMARTHUB_SLO_REMINDER_MINUTES", "60")
    b = [
        {
            "metric": "tat_p99_seconds",
            "severity": "critical",
            "value": 1.8,
            "message": "p99 high",
        }
    ]
    slo_alerts._dispatch(b, 15)  # new
    st = slo_alerts._ALERT_STATE["tat_p99_seconds"]
    st["last_notified"] = st["last_notified"] - timedelta(hours=2)
    st["since"] = st["since"] - timedelta(hours=2)
    slo_alerts._dispatch(b, 15)  # reminder now due
    assert "reminder" in sent[-1][1]
