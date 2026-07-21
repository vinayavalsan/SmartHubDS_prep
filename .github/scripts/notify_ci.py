"""Send a Slack notification summarizing this CI/CD run (success or failure).

Called from the `notify` job in .github/workflows/ci_cd.yml, which only runs
on a push to the deploy branch (smarthub.etl.pipeline) -- the one case where
Docker images actually get built, so a success notification's "pull this
image" is meaningful. Reuses smarthub.core.notifications so this looks and
behaves exactly like the existing Prefect flow-failure Slack alerts (same
Block Kit formatting, same best-effort/no-op-when-unconfigured behavior),
rather than being a second, differently-styled notification path.

All inputs come from environment variables set by the `notify` job -- see
that job's `env:` block for exactly what each one is.
"""

from __future__ import annotations

import os

from smarthub.core.notifications import notify_failure, notify_success

# Job name (as used in RESULT_*/MSG_* env var suffixes) -> human label.
STAGES = {
    "isort": "isort",
    "black": "black",
    "flake8": "flake8",
    "pytest": "pytest",
    "build_worker": "build-worker",
    "build_dashboard": "build-dashboard",
}


def _env(name: str) -> str:
    return os.environ.get(name, "") or ""


def _common_fields() -> dict:
    sha = _env("GITHUB_SHA_FULL")
    return {
        "Branch": _env("GITHUB_REF_NAME_VALUE"),
        "Commit": sha[:7] if sha else None,
        "Triggered by": _env("GITHUB_ACTOR_NAME"),
        "Run": _env("RUN_URL"),
    }


def _notify_success() -> None:
    sha = _env("GITHUB_SHA_FULL")[:7]
    dockerhub_user = _env("DOCKERHUB_USERNAME")
    repo = f"{dockerhub_user}/smarthub" if dockerhub_user else "smarthub"
    pulls = "\n".join(
        [
            f"docker pull {repo}:worker-latest",
            f"docker pull {repo}:worker-{sha}",
            f"docker pull {repo}:dashboard-latest",
            f"docker pull {repo}:dashboard-{sha}",
        ]
    )
    fields = _common_fields()
    fields["Pull the new images"] = f"```{pulls}```"
    notify_success("CI/CD", fields)


def _notify_failure(failed: list[str], not_run: list[str], messages: dict) -> None:
    fields = _common_fields()
    fields["Failed stage"] = ", ".join(STAGES[s] for s in failed) or "unknown"
    if not_run:
        fields["Skipped as a result"] = ", ".join(STAGES[s] for s in not_run)

    error_text = "\n\n".join(
        f"[{STAGES[s]}]\n{messages[s]}" for s in failed if messages.get(s)
    )
    if not error_text:
        error_text = "(no captured output for this stage -- see the run link above)"

    notify_failure("CI/CD", fields, error=error_text)


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
