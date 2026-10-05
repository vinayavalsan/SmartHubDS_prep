"""Slack notifications for the SmartHub pipelines and services.

Severity-routed to four production channels, each with its own Slack Incoming
Webhook:

    success  -> updates   (SLACK_WEBHOOK_UPDATES_URL)
    warning  -> warnings  (SLACK_WEBHOOK_WARNINGS_URL)
    critical -> critical  (SLACK_WEBHOOK_CRITICAL_URL)   [@here]
    failure  -> failures  (SLACK_WEBHOOK_FAILURES_URL)   [@here]

If a category's webhook is not configured, the sender falls back to the single
legacy ``SLACK_WEBHOOK_URL`` so nothing breaks during rollout. Sends are
best-effort (any failure is logged and swallowed so a notification problem never
breaks a pipeline) and cleanly disabled (no-ops) when no webhook is configured.
Standard library only, so it works everywhere the package runs.

``SLACK_ENV_LABEL`` sets the environment label shown in the title (e.g. ``PROD``)
and the footer (defaults to the hostname in the footer only).
``SLACK_MENTION_ON_FAILURE`` overrides the default ``@here`` ping used on
critical/failure alerts (e.g. a specific ``<!subteam^ID>`` or ``<@USER>``).
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

# Legacy single webhook — used as the fallback for every category.
WEBHOOK_ENV = "SLACK_WEBHOOK_URL"
ENV_LABEL_ENV = "SLACK_ENV_LABEL"
MENTION_ENV = "SLACK_MENTION_ON_FAILURE"

# Severities (internal status values) and the channel category each maps to.
_SUCCESS = "success"
_WARNING = "warning"
_FAILURE = "failure"
_CRITICAL = "critical"

# severity -> category (channel) -> per-category webhook env var.
_SEVERITY_CATEGORY = {
    _SUCCESS: "updates",
    _WARNING: "warnings",
    _CRITICAL: "critical",
    _FAILURE: "failures",
}
_CATEGORY_ENV = {
    "updates": "SLACK_WEBHOOK_UPDATES_URL",
    "warnings": "SLACK_WEBHOOK_WARNINGS_URL",
    "critical": "SLACK_WEBHOOK_CRITICAL_URL",
    "failures": "SLACK_WEBHOOK_FAILURES_URL",
}
# Severities that @here the channel so they are not missed.
_PING_SEVERITIES = {_CRITICAL, _FAILURE}

_EMOJI = {
    _SUCCESS: ":white_check_mark:",
    _WARNING: ":warning:",
    _CRITICAL: ":red_circle:",
    _FAILURE: ":x:",
}
_VERB = {
    _SUCCESS: "completed",
    _WARNING: "WARNING",
    _CRITICAL: "CRITICAL",
    _FAILURE: "FAILED",
}
_TIMEOUT_SECONDS = 10


def _single_webhook_url() -> str:
    """Return the legacy single webhook URL (stripped, may be empty)."""
    return os.environ.get(WEBHOOK_ENV, "").strip()


def _category_webhook_url(category: str | None) -> str:
    """Resolve the webhook for a channel category, falling back to the single one.

    Inputs
    ------
    category : str | None
        One of ``updates``/``warnings``/``critical``/``failures``; ``None`` or an
        unknown value uses the legacy single webhook.

    Returns
    -------
    str
        The webhook URL to post to (empty when nothing is configured).
    """
    env = _CATEGORY_ENV.get(category or "")
    if env:
        url = os.environ.get(env, "").strip()
        if url:
            return url
    return _single_webhook_url()


def slack_enabled() -> bool:
    """True when any webhook (single or per-category) is configured."""
    if _single_webhook_url():
        return True
    return any(os.environ.get(env, "").strip() for env in _CATEGORY_ENV.values())


def _env_label() -> str:
    """Environment label for the footer, defaulting to the hostname."""
    return os.environ.get(ENV_LABEL_ENV, "").strip() or socket.gethostname()


def _env_tag() -> str:
    """Uppercased env tag for the title, only when explicitly set (else blank)."""
    return os.environ.get(ENV_LABEL_ENV, "").strip().upper()


def _mention(severity: str) -> str:
    """The @-mention prefix for a severity (``@here`` on critical/failure)."""
    if severity not in _PING_SEVERITIES:
        return ""
    return os.environ.get(MENTION_ENV, "").strip() or "<!here>"


def _utc_now_str() -> str:
    """Return the current UTC time as a display string."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _post(payload: dict, category: str | None = None) -> bool:
    """POST a Slack payload to the webhook for ``category`` (best-effort).

    Logs and swallows any error, never raising.

    Inputs
    ------
    payload : dict
        The Slack message payload to send as JSON.
    category : str | None
        Channel category used to pick the webhook; falls back to the single one.

    Returns
    -------
    bool
        True when delivered; False when disabled or the send failed.
    """
    url = _category_webhook_url(category)
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


def _rows(fields: dict) -> list[tuple[str, str]]:
    """(key, value) pairs, skipping empties and stripping backticks.

    Backticks are removed because values render inside a ``` code block ```,
    where backticks would show literally rather than as inline code.
    """
    out: list[tuple[str, str]] = []
    for k, v in (fields or {}).items():
        if v in (None, "", []):
            continue
        out.append((str(k), str(v).replace("`", "")))
    return out


def _table(rows: list[tuple[str, str]]) -> str:
    """Left-align keys into a monospace column: ``key   value``."""
    if not rows:
        return ""
    width = max(len(k) for k, _ in rows)
    return "\n".join(f"{k.ljust(width)}   {v}" for k, v in rows)


def _title(severity: str, pipeline: str, subject: str | None) -> str:
    """Build the bold title line (optional @mention + emoji + env + area + verb)."""
    emoji = _EMOJI.get(severity, "")
    verb = _VERB.get(severity, severity.upper())
    env = f" {_env_tag()}" if _env_tag() else ""
    subj = f" · {subject}" if subject else ""
    mention = _mention(severity)
    prefix = f"{mention} " if mention else ""
    return f"{prefix}{emoji} *SmartHub{env} · {pipeline} · {verb}{subj}*"


def _assemble(head: list[str], rows: list[tuple[str, str]], footer_extra) -> dict:
    """Assemble the common layout: rendered head, a code-block table, a footer.

    ``head`` lines render as normal mrkdwn (bold title, emoji, headline, links);
    ``rows`` go inside a ``` code block ``` as a column-aligned key/value table
    so each alert reads as one compact, low-clutter monospace box. The env label
    and timestamp (plus any ``footer_extra``) go in a small context footer.
    """
    body = "\n".join(head)
    if rows:
        body += "\n```\n" + _table(rows) + "\n```"
    ctx = f"env: `{_env_label()}` · {_utc_now_str()}"
    if footer_extra:
        ctx += f" · {footer_extra}"
    blocks = [
        {"type": "section", "text": {"type": "mrkdwn", "text": body}},
        {"type": "context", "elements": [{"type": "mrkdwn", "text": ctx}]},
    ]
    # Fallback text (notifications, screen readers, clients without blocks).
    lines = list(head) + [f"{k}: {v}" for k, v in rows] + [ctx]
    return {"text": "\n".join(lines), "blocks": blocks}


def _build_payload(
    severity: str,
    pipeline: str,
    fields: dict,
    error: str | None,
    subject: str | None = None,
) -> dict:
    """Build a code-block Slack message (title outside, key/values in a box).

    Inputs
    ------
    severity : str
        ``success``/``warning``/``critical``/``failure``; selects emoji, verb,
        channel, and whether to @here.
    pipeline : str
        Pipeline/area name shown in the title.
    fields : dict
        Label/value pairs rendered as a monospace table (empty values skipped).
    error : str | None
        Error text added as a final ``Error`` row (truncated if very long).
    subject : str | None
        Optional subject appended to the title (e.g. the lead type).

    Returns
    -------
    dict
        A payload with ``text`` and ``blocks`` keys.
    """
    head = [_title(severity, pipeline, subject)]
    rows = _rows(fields)
    if error:
        text = str(error).strip()
        if len(text) > 1500:
            text = text[:1500] + " … (truncated)"
        rows.append(("Error", text))
    return _assemble(head, rows, footer_extra=None)


def _build_grouped_payload(
    severity: str,
    pipeline: str,
    subject: str | None,
    headline: str | None,
    groups: list,
    footer_extra: str | None,
) -> dict:
    """Build a code-block Slack message from grouped fields.

    Title (mention + emoji + pipeline + verb + subject) and the headline render
    as normal mrkdwn above the box; the groups are flattened into one
    column-aligned key/value table inside a ``` code block ``` (group titles are
    dropped — the trimmed field keys are self-describing). Empty values/groups
    are skipped.
    """
    head = [_title(severity, pipeline, subject)]
    if headline:
        head.append(headline)
    rows: list[tuple[str, str]] = []
    for _group_title, fields in groups or []:
        rows.extend(_rows(fields))
    return _assemble(head, rows, footer_extra)


def _category(severity: str) -> str:
    """Channel category for a severity (defaults to ``failures`` if unknown)."""
    return _SEVERITY_CATEGORY.get(severity, "failures")


def notify_raw(payload: dict, category: str | None = None) -> bool:
    """Send a fully custom Slack Block Kit payload (best-effort).

    Escape hatch for callers that need a layout the structured helpers don't
    produce. Routes to ``category``'s webhook (or the single fallback) and goes
    through the same config check, timeout, and swallow-errors behavior.
    """
    return _post(payload, category)


def notify(
    severity: str,
    pipeline: str,
    fields: dict,
    error: str | None = None,
    subject: str | None = None,
) -> bool:
    """Send a severity-routed Slack notification (best-effort)."""
    return _post(
        _build_payload(severity, pipeline, fields, error, subject),
        _category(severity),
    )


def notify_grouped(
    severity: str,
    pipeline: str,
    *,
    subject: str | None = None,
    headline: str | None = None,
    groups: list | None = None,
    footer_extra: str | None = None,
) -> bool:
    """Send a severity-routed grouped notification (best-effort)."""
    return _post(
        _build_grouped_payload(
            severity, pipeline, subject, headline, groups or [], footer_extra
        ),
        _category(severity),
    )


def notify_success(pipeline: str, fields: dict, subject: str | None = None) -> bool:
    """Notify a successful operation -> #updates."""
    return notify(_SUCCESS, pipeline, fields, subject=subject)


def notify_warning(pipeline: str, fields: dict, subject: str | None = None) -> bool:
    """Notify a non-fatal warning -> #warnings."""
    return notify(_WARNING, pipeline, fields, subject=subject)


def notify_failure(
    pipeline: str, fields: dict, error: str | None = None, subject: str | None = None
) -> bool:
    """Notify a failed operation/workflow -> #failures (@here)."""
    return notify(_FAILURE, pipeline, fields, error=error, subject=subject)


def notify_critical(
    pipeline: str, fields: dict, error: str | None = None, subject: str | None = None
) -> bool:
    """Notify a serious API-health/model condition -> #critical (@here)."""
    return notify(_CRITICAL, pipeline, fields, error=error, subject=subject)


def notify_success_grouped(
    pipeline: str,
    *,
    subject: str | None = None,
    headline: str | None = None,
    groups: list | None = None,
    footer_extra: str | None = None,
) -> bool:
    """Notify success with a grouped, sectioned layout -> #updates."""
    return notify_grouped(
        _SUCCESS,
        pipeline,
        subject=subject,
        headline=headline,
        groups=groups,
        footer_extra=footer_extra,
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
    """Prefect ``on_failure`` hook that notifies #failures when a flow fails.

    Attach with ``@flow(..., on_failure=[flow_failure_hook])``. Pulls the lead
    type from the run's parameters so alerts are self-identifying. Never raises.
    """
    try:
        params = dict(getattr(flow_run, "parameters", {}) or {})
        pipeline = getattr(flow, "name", None) or getattr(
            flow_run, "flow_name", "smarthub-flow"
        )
        # Normalise the area to match the success alerts (e.g. the flow
        # "smarthub-data-pull" -> "data-pull"), so a pipeline reads the same
        # whether it succeeded or failed.
        if pipeline.startswith("smarthub-"):
            pipeline = pipeline[len("smarthub-") :]
        fields = {
            "Run": getattr(flow_run, "name", None),
            "Run URL": _run_url(flow_run) or None,
        }
        message = getattr(state, "message", None) or "Flow run entered a FAILED state."
        # Lead type goes in the title subject (like the success alerts).
        notify_failure(
            pipeline, fields, error=message, subject=_lead_type_label(params)
        )
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
