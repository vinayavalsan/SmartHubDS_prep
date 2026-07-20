"""Slack notifications for the SmartHub pipelines (success and failure).

Posts to a Slack Incoming Webhook whose URL lives in ``SLACK_WEBHOOK_URL``.
Sends are best-effort (any failure is logged and swallowed so a notification
problem never breaks a pipeline) and cleanly disabled (no-ops) when no webhook
is configured. Uses only the standard library so it works everywhere the
package runs. ``SLACK_ENV_LABEL`` sets the message label (defaults to the
hostname) and ``SLACK_MENTION_ON_FAILURE`` an optional @-mention on failures.
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
    """Return the configured Slack webhook URL (stripped, may be empty)."""
    return os.environ.get(WEBHOOK_ENV, "").strip()


def _env_label() -> str:
    """Return the environment label, defaulting to the hostname."""
    return os.environ.get(ENV_LABEL_ENV, "").strip() or socket.gethostname()


def _utc_now_str() -> str:
    """Return the current UTC time as a display string."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _post(payload: dict) -> bool:
    """POST a Slack payload to the configured webhook.

    Best-effort: logs and swallows any error, never raising.

    Inputs
    ------
    payload : dict
        The Slack message payload to send as JSON.

    Returns
    -------
    bool
        True when delivered; False when disabled or the send failed.
    """
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
    """Build a Block Kit Slack message with a plain-text fallback.

    Inputs
    ------
    status : str
        ``success`` or ``failure``; selects the emoji and verb.
    pipeline : str
        Pipeline name shown in the header.
    fields : dict
        Label/value pairs rendered as a field grid (empty values skipped).
    error : str | None
        Error text shown in a code block (truncated if very long).

    Returns
    -------
    dict
        A payload with ``text`` and ``blocks`` keys.
    """
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
        blocks.append({"type": "section", "fields": field_blocks[i : i + 10]})

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
    """Send a Slack notification (best-effort).

    Inputs
    ------
    status : str
        ``success`` or ``failure``.
    pipeline : str
        Pipeline name shown in the header.
    fields : dict
        Label/value pairs to display.
    error : str | None
        Optional error text to include.

    Returns
    -------
    bool
        True when delivered.
    """
    return _post(_build_payload(status, pipeline, fields, error))


def notify_success(pipeline: str, fields: dict) -> bool:
    """Notify that a pipeline run completed successfully.

    Inputs
    ------
    pipeline : str
        Pipeline name shown in the header.
    fields : dict
        Label/value pairs to display.

    Returns
    -------
    bool
        True when delivered.
    """
    return notify(_SUCCESS, pipeline, fields)


def notify_failure(pipeline: str, fields: dict, error: str | None = None) -> bool:
    """Notify that a pipeline run failed.

    Inputs
    ------
    pipeline : str
        Pipeline name shown in the header.
    fields : dict
        Label/value pairs to display.
    error : str | None
        Optional error text to include.

    Returns
    -------
    bool
        True when delivered.
    """
    return notify(_FAILURE, pipeline, fields, error=error)


def _field_blocks(fields: dict) -> list[dict]:
    """2-column Block Kit field blocks from a dict; empty values are skipped."""
    return [
        {"type": "mrkdwn", "text": f"*{k}:*\n{v}"}
        for k, v in fields.items()
        if v not in (None, "", [])
    ]


def _build_grouped_payload(
    status: str,
    pipeline: str,
    subject: str | None,
    headline: str | None,
    groups: list,
    footer_extra: str | None,
) -> dict:
    """Build a Block Kit message grouped into titled sections with dividers.

    Inputs
    ------
    status : str
        ``success`` or ``failure``.
    pipeline : str
        Pipeline name shown in the header.
    subject : str | None
        Optional subject appended to the header.
    headline : str | None
        Prominent mrkdwn line under the header (e.g. the decision).
    groups : list
        Ordered ``(title, fields_dict)`` pairs; each renders as a divider
        plus a titled 2-column field grid.
    footer_extra : str | None
        Extra text appended to the context footer.

    Returns
    -------
    dict
        A payload with ``text`` and ``blocks`` keys.
    """
    emoji = _EMOJI.get(status, "")
    verb = "completed" if status == _SUCCESS else "FAILED"
    subj = f" · {subject}" if subject else ""
    header = f"SmartHub · {pipeline} · {verb}{subj}"

    blocks: list[dict] = [
        {
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": f"{emoji} {header}"[:150],
                "emoji": True,
            },
        }
    ]
    if headline:
        blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": headline}})

    for title, fields in groups:
        fb = _field_blocks(fields)
        if not fb:
            continue
        blocks.append({"type": "divider"})
        first = {"type": "section", "fields": fb[:10]}
        if title:
            first["text"] = {"type": "mrkdwn", "text": f"*{title}*"}
        blocks.append(first)
        for i in range(10, len(fb), 10):
            blocks.append({"type": "section", "fields": fb[i : i + 10]})

    ctx = f"env: `{_env_label()}` · {_utc_now_str()}"
    if footer_extra:
        ctx += f" · {footer_extra}"
    blocks.append({"type": "context", "elements": [{"type": "mrkdwn", "text": ctx}]})

    # Fallback text (notifications, screen readers, no-block clients).
    lines = [f"{emoji} {header}"]
    if headline:
        lines.append(headline)
    for title, fields in groups:
        rendered = [f"{k}: {v}" for k, v in fields.items() if v not in (None, "", [])]
        if not rendered:
            continue
        if title:
            lines.append(f"— {title} —")
        lines += rendered
    lines.append(ctx)
    return {"text": "\n".join(lines), "blocks": blocks}


def notify_success_grouped(
    pipeline: str,
    *,
    subject: str | None = None,
    headline: str | None = None,
    groups: list | None = None,
    footer_extra: str | None = None,
) -> bool:
    """Notify success with a grouped, sectioned layout (best-effort).

    Inputs
    ------
    pipeline : str
        Pipeline name shown in the header.
    subject : str | None
        Optional subject appended to the header.
    headline : str | None
        Prominent mrkdwn line under the header.
    groups : list | None
        Ordered ``(title, fields_dict)`` pairs to render.
    footer_extra : str | None
        Extra text appended to the context footer.

    Returns
    -------
    bool
        True when delivered.
    """
    return _post(
        _build_grouped_payload(
            _SUCCESS, pipeline, subject, headline, groups or [], footer_extra
        )
    )


def _run_url(flow_run) -> str:
    """Best-effort Prefect UI URL for a flow run (blank if unknown)."""
    base = (
        (
            os.environ.get("PREFECT_UI_URL")
            or os.environ.get("PREFECT_API_URL", "").replace("/api", "")
        )
        .strip()
        .rstrip("/")
    )
    run_id = getattr(flow_run, "id", None)
    if base and run_id:
        return f"{base}/runs/flow-run/{run_id}"
    return ""


def flow_failure_hook(flow, flow_run, state) -> None:
    """Prefect ``on_failure`` hook that notifies Slack when a flow fails.

    Attach with ``@flow(..., on_failure=[flow_failure_hook])``. Pulls the
    lead type from the run's parameters so alerts are self-identifying.
    Never raises.

    Inputs
    ------
    flow : Flow
        The Prefect flow whose name is used as the pipeline label.
    flow_run : FlowRun
        The failed run; supplies parameters, name and deployment id.
    state : State
        The terminal state; its message is used as the error text.
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
