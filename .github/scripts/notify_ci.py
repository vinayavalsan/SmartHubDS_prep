"""Send a Slack notification summarizing this CI/CD run (success or failure).

Called from the `notify` job in .github/workflows/ci_cd.yml, which only runs
on a push to the deploy branch (smarthub.etl.pipeline) -- the one case where
Docker images actually get built, so a success notification's "pull this
image" is meaningful. Reuses smarthub.core.notifications's grouped layout
(notify_success_grouped / notify_failure_grouped) so this looks and behaves
like the existing Prefect flow alerts, not a second, differently-styled
notification path.

Compact layout: branch/commit/actor collapse into one headline under the
header, the run link lives in the small context footer, and the group
section holds only the part that actually matters -- the pull commands on
success, the error excerpt on failure.

Only the `-latest` tag is shown for each image (not `-<sha>`) -- that's the
tag Watchtower actually pulls on the server (see README's CI/CD section), so
it's the command someone would realistically run right after this alert.

All inputs come from environment variables set by the `notify` job -- see
that job's `env:` block for exactly what each one is.
"""

from __future__ import annotations

import os

from smarthub.core.notifications import notify_failure_grouped, notify_success_grouped

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


def _headline() -> str:
    sha = _env("GITHUB_SHA_FULL")
    branch = _env("GITHUB_REF_NAME_VALUE") or "?"
    commit = sha[:7] if sha else "?"
    actor = _env("GITHUB_ACTOR_NAME") or "?"
    return f"`{branch}` @ `{commit}` · triggered by {actor}"


def _footer() -> str | None:
    run_url = _env("RUN_URL")
    return f"Run: {run_url}" if run_url else None


def _notify_success() -> None:
    dockerhub_user = _env("DOCKERHUB_USERNAME")
    repo = f"{dockerhub_user}/smarthub" if dockerhub_user else "smarthub"
    pulls = "\n".join(
        [
            f"docker pull {repo}:worker-latest",
            f"docker pull {repo}:dashboard-latest",
        ]
    )
    notify_success_grouped(
        "CI/CD",
        headline=_headline(),
        groups=[(None, {"Pull the new images": f"```{pulls}```"})],
        footer_extra=_footer(),
    )


def _notify_failure(failed: list[str], not_run: list[str], messages: dict) -> None:
    stage_label = ", ".join(STAGES[s] for s in failed) or "unknown"

    error_text = "\n\n".join(
        f"[{STAGES[s]}]\n{messages[s]}" for s in failed if messages.get(s)
    )
    if not error_text:
        error_text = "(no captured output for this stage -- see the run link below)"

    fields = {"Error": f"```{error_text}```"}
    if not_run:
        fields["Skipped as a result"] = ", ".join(STAGES[s] for s in not_run)

    notify_failure_grouped(
        "CI/CD",
        subject=f"stage: {stage_label}",
        headline=_headline(),
        groups=[(None, fields)],
        footer_extra=_footer(),
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
