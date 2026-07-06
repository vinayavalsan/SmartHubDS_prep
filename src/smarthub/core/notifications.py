"""Slack notifications for the SmartHub pipelines (success + failure).

Sends to a Slack **Incoming Webhook** whose URL lives in ``SLACK_WEBHOOK_URL``
(a Tier-1 secret in ``.env``). Design goals:

- **Best-effort:** every send is wrapped so a notification problem (bad URL,
  Slack down, network blip) is logged and swallowed — it must never break or
  fail a data pull / feature build.
- **Cleanly disabled:** with no webhook configured, calls are no-ops.
- **Dependency-free:** uses only the standard library (``urllib``), so it works
  everywhere the package runs (worker, CLI, tests) without extra installs.

Env vars:
- ``SLACK_WEBHOOK_URL``        — Incoming Webhook URL (required to enable).
- ``SLACK_ENV_LABEL``         — label shown on every message (e.g. ``prod``,
                                 ``staging``, ``local``); defaults to hostname.
- ``SLACK_MENTION_ON_FAILURE``— optional Slack id to @-mention on failures,
                                 e.g. ``<@U123ABC>`` or ``<!subteam^S123>``.
"""

from __future__ import annotations

import json
import logging
import os
import socket
import urllib.error
import urllib.request
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

WEBHOOK_ENV = "SLACK_WEBHOOK_URL"
ENV_LABEL_ENV = "SLACK_ENV_LABEL"
MENTION_ENV = "SLACK_MENTION_ON_FAILURE"

_SUCCESS = "success"
_FAILURE = "failure"
_EMOJI = {_SUCCESS: ":white_check_mark:", _FAILURE: ":red_circle:"}
_TIMEOUT_SECONDS = 10


def slack_enabled() -> bool:
    """True when a webhook URL is configured (otherwise sends are no-ops)."""
    return bool(_webhook_url())


def _webhook_url() -> str:
    return os.environ.get(WEBHOOK_ENV, "").strip()


def _env_label() -> str:
    return os.environ.get(ENV_LABEL_ENV, "").strip() or socket.gethostname()


def _utc_now_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _post(payload: dict) -> bool:
    """POST a Slack payload. Returns True on success; never raises."""
    url = _webhook_url()
    if not url:
        logger.info("Slack webhook not configured; skipping notification.")
        return False
    try:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url, data=data, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=_TIMEOUT_SECONDS) as resp:
            resp.read()
        return True
    except urllib.error.URLError as exc:  # network / DNS / HTTP error
        logger.warning("Slack notification failed (network): %s", exc)
    except Exception as exc:  # noqa: BLE001 - notifications must never break flows
        logger.warning("Slack notification failed: %s", exc)
    return False


def _build_payload(status: str, pipeline: str, fields: dict, error: str | None) -> dict:
    """Build a Block Kit message with a text fallback."""
    emoji = _EMOJI.get(status, "")
    verb = "completed" if status == _SUCCESS else "FAILED"
    header = f"SmartHub · {pipeline} · {verb}"

    blocks: list[dict] = [
        {
            "type": "header",
            "text": {"type": "plain_text", "text": f"{emoji} {header}"[:150]},
        }
    ]

    # Optional @-mention on failure.
    if status == _FAILURE:
        mention = os.environ.get(MENTION_ENV, "").strip()
        if mention:
            blocks.append(
                {
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": f"{mention} attention needed"},
                }
            )

    # Field grid (2 columns) — skip empty values.
    field_blocks = [
        {"type": "mrkdwn", "text": f"*{k}:*\n{v}"}
        for k, v in fields.items()
        if v not in (None, "", [])
    ]
    for i in range(0, len(field_blocks), 10):  # Slack caps 10 fields/section
        blocks.append({"type": "section", "fields": field_blocks[i:i + 10]})

    if error:
        text = str(error)
        if len(text) > 2800:
            text = text[:2800] + "\n… (truncated)"
        blocks.append(
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": f"*Error:*\n```{text}```"},
            }
        )

    blocks.append(
        {
            "type": "context",
            "elements": [
                {"type": "mrkdwn", "text": f"env: `{_env_label()}` · {_utc_now_str()}"}
            ],
        }
    )

    # Fallback text (notifications, screen readers, clients without blocks).
    lines = [f"{emoji} {header}"]
    lines += [f"{k}: {v}" for k, v in fields.items() if v not in (None, "", [])]
    if error:
        lines.append(f"Error: {error}")
    return {"text": "\n".join(lines), "blocks": blocks}


def notify(status: str, pipeline: str, fields: dict, error: str | None = None) -> bool:
    """Send a Slack notification. Best-effort; returns True if delivered."""
    return _post(_build_payload(status, pipeline, fields, error))


def notify_success(pipeline: str, fields: dict) -> bool:
    """Notify that a pipeline run completed successfully."""
    return notify(_SUCCESS, pipeline, fields)


def notify_failure(pipeline: str, fields: dict, error: str | None = None) -> bool:
    """Notify that a pipeline run failed."""
    return notify(_FAILURE, pipeline, fields, error=error)


def _run_url(flow_run) -> str:
    """Best-effort Prefect UI URL for a flow run (blank if unknown)."""
    base = (
        os.environ.get("PREFECT_UI_URL")
        or os.environ.get("PREFECT_API_URL", "").replace("/api", "")
    ).strip().rstrip("/")
    run_id = getattr(flow_run, "id", None)
    if base and run_id:
        return f"{base}/runs/flow-run/{run_id}"
    return ""


def flow_failure_hook(flow, flow_run, state) -> None:
    """Prefect ``on_failure`` hook — notify Slack when a flow run fails.

    Attach with ``@flow(..., on_failure=[flow_failure_hook])``. Pulls the lead
    type from the run's parameters so alerts are self-identifying. Never raises.
    """
    try:
        params = dict(getattr(flow_run, "parameters", {}) or {})
        pipeline = getattr(flow, "name", None) or getattr(
            flow_run, "flow_name", "smarthub-flow"
        )
        fields = {
            "Lead type": _lead_type_label(params),
            "Run": getattr(flow_run, "name", None),
            "Deployment": getattr(flow_run, "deployment_id", None),
            "Run URL": _run_url(flow_run) or None,
        }
        message = getattr(state, "message", None) or "Flow run entered a FAILED state."
        notify_failure(pipeline, fields, error=message)
    except Exception as exc:  # noqa: BLE001 - a failing hook must not mask the error
        logger.warning("flow_failure_hook could not send Slack alert: %s", exc)


def _lead_type_label(params: dict) -> str | None:
    """Human label like ``auto (6)`` from flow parameters, if present."""
    name = params.get("lead_type_name")
    lead_id = params.get("lead_type_id")
    if name and lead_id is not None:
        return f"{name} ({lead_id})"
    if name:
        return str(name)
    if lead_id is not None:
        return str(lead_id)
    return None
