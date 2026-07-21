"""Send a Slack notification summarizing this CI/CD run (success or failure).

Called from the `notify` job in .github/workflows/ci_cd.yml, which only runs
on a push to the deploy branch (smarthub.etl.pipeline) -- the one case where
Docker images actually get built, so a success notification's "pull this
image" is meaningful.

Layout is a specific, custom one (bulleted metadata, emoji section headers,
plain -- not code-fenced -- pull commands), so this builds its own Block Kit
blocks rather than using core.notifications's structured templates, but
still sends through `notify_raw` -- same webhook, same best-effort/
disabled-when-unconfigured behavior as every other Slack alert in this repo.

All inputs come from environment variables set by the `notify` job -- see
that job's `env:` block for exactly what each one is.
"""

from __future__ import annotations

import os
import socket
from datetime import datetime, timezone

from smarthub.core.notifications import notify_raw

# Job name (as used in RESULT_*/MSG_* env var suffixes) -> human label.
STAGES = {
    "isort": "isort",
    "black": "black",
    "flake8": "flake8",
    "pytest": "pytest",
    "build_worker": "build-worker",
    "build_dashboard": "build-dashboard",
}

_DIAMOND = ":small_blue_diamond:"


def _env(name: str) -> str:
    return os.environ.get(name, "") or ""


def _env_label() -> str:
    """SLACK_ENV_LABEL if set, else hostname (mirrors core.notifications)."""
    return os.environ.get("SLACK_ENV_LABEL", "").strip() or socket.gethostname()


def _time_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _metadata_bullets(extra: list[str] | None = None) -> str:
    sha = _env("GITHUB_SHA_FULL")
    lines = [
        f"{_DIAMOND} Branch: {_env('GITHUB_REF_NAME_VALUE') or '?'}",
        f"{_DIAMOND} Commit: {sha[:7] if sha else '?'}",
        f"{_DIAMOND} Triggered by: {_env('GITHUB_ACTOR_NAME') or '?'}",
        f"{_DIAMOND} Environment: {_env_label()}",
        f"{_DIAMOND} Time: {_time_str()}",
    ]
    if extra:
        lines.extend(f"{_DIAMOND} {line}" for line in extra)
    return "\n".join(lines)


def _mention_block() -> dict | None:
    mention = os.environ.get("SLACK_MENTION_ON_FAILURE", "").strip()
    if not mention:
        return None
    return {
        "type": "section",
        "text": {"type": "mrkdwn", "text": f"{mention} attention needed"},
    }


def _section(text: str) -> dict:
    return {"type": "section", "text": {"type": "mrkdwn", "text": text}}


def _header(text: str) -> dict:
    return {
        "type": "header",
        "text": {"type": "plain_text", "text": text[:150], "emoji": True},
    }


def _send(
    header_text: str, blocks_middle: list[dict], fallback_lines: list[str]
) -> None:
    run_url = _env("RUN_URL")
    blocks = [_header(header_text)]
    blocks.extend(blocks_middle)
    blocks.append(_section(f":link: *Workflow:*\n{run_url}"))
    fallback_lines = fallback_lines + [f"Workflow: {run_url}"]
    notify_raw({"text": "\n".join(fallback_lines), "blocks": blocks})


def _notify_success() -> None:
    dockerhub_user = _env("DOCKERHUB_USERNAME")
    repo = f"{dockerhub_user}/smarthub" if dockerhub_user else "smarthub"
    sha = _env("GITHUB_SHA_FULL")
    tag = sha[:7] if sha else "latest"
    pulls = "\n".join(
        [
            f"docker pull {repo}:worker-{tag}",
            f"docker pull {repo}:dashboard-{tag}",
        ]
    )
    metadata = _metadata_bullets()
    blocks_middle = [
        _section(metadata),
        _section(":whale: *Pull the latest images:*"),
        _section(pulls),
    ]
    _send(
        "✅ SmartHub CI/CD Pipeline Succeeded",
        blocks_middle,
        [
            "✅ SmartHub CI/CD Pipeline Succeeded",
            metadata,
            "Pull the latest images:",
            pulls,
        ],
    )


def _notify_failure(failed: list[str], not_run: list[str], messages: dict) -> None:
    stage_label = ", ".join(STAGES[s] for s in failed) or "unknown"

    error_text = "\n\n".join(
        f"[{STAGES[s]}]\n{messages[s]}" for s in failed if messages.get(s)
    )
    if not error_text:
        error_text = (
            "(no captured output for this stage -- see the workflow link below)"
        )

    extra = [f"Failed stage: {stage_label}"]
    if not_run:
        extra.append(f"Skipped as a result: {', '.join(STAGES[s] for s in not_run)}")
    metadata = _metadata_bullets(extra)

    blocks_middle = [_section(metadata)]
    mention = _mention_block()
    if mention:
        blocks_middle.append(mention)
    blocks_middle.append(_section(f":rotating_light: *Error:*\n{error_text}"))

    _send(
        "\U0001f534 SmartHub CI/CD Pipeline Failed",
        blocks_middle,
        [
            "\U0001f534 SmartHub CI/CD Pipeline Failed",
            metadata,
            f"Error:\n{error_text}",
        ],
    )


def main() -> None:
    results = {stage: _env(f"RESULT_{stage.upper()}") for stage in STAGES}
    messages = {stage: _env(f"MSG_{stage.upper()}") for stage in STAGES}

    failed = [s for s in STAGES if results[s] == "failure"]
    not_run = [s for s in STAGES if results[s] not in ("success", "failure")]

    if not failed and not not_run:
        _notify_success()
        return

    # Prefer the stages that actually reported failure; if none did (e.g.
    # everything upstream was cancelled rather than failing outright), fall
    # back to whatever didn't complete so the alert isn't silently dropped.
    if not failed:
        failed = not_run
        not_run = []

    _notify_failure(failed, not_run, messages)


if __name__ == "__main__":
    main()
