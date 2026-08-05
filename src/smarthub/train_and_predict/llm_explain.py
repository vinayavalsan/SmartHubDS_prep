"""LLM (Ollama) formatting of a SHAP factor breakdown into plain English.

Split out of ``explain.py`` (2026-07-24) — this module owns the LLM prompt
template, the Ollama HTTP calls, and the model-pull/dedup infrastructure
around them. ``explain.py`` remains the thin orchestrator that calls into
``shap_explain.py`` (for the numeric breakdown) and this module (to turn that
breakdown into plain English); no logic changed in this split, only which
file each piece lives in.

The LLM never sees model internals and never computes anything — it only
formats facts it's handed, with an explicit "don't invent numbers"
instruction, since this is a formatting task, not a reasoning task, and small
models hallucinate numbers readily if given room to reason freely.

Heavy/optional deps (requests) are imported lazily so the rest of
`train_and_predict` keeps working without the `explain` extra installed —
same pattern as `predict.py`'s lazy joblib/mlflow/fastapi imports.
"""

from __future__ import annotations

import fcntl
import logging
import os
import threading

from smarthub.core import task_config

logger = logging.getLogger(__name__)

DEFAULT_OLLAMA_HOST = "http://localhost:11434"

# Task config: smarthub.yaml `explain` section — used only by this module, not live
# bidding path, so it's kept local rather than in train_and_predict/config.py.
# $SMARTHUB_LLM_MODEL wins over the file config (same env-over-file convention as
# OLLAMA_HOST below), so compose can keep the serve container and the
# `ollama-init` pull step pointed at exactly the same model with one env var.
LLM_MODEL = os.getenv("SMARTHUB_LLM_MODEL") or task_config.get(
    "explain", "llm_model", "qwen2.5:1.5b-instruct"
)
# $SMARTHUB_OLLAMA_HOST wins over the file config when set -- same
# env-wins-over-file convention as SMARTHUB_PREDICTION_LOG_DB_URL/
# SMARTHUB_CONFIG_DB_URL. Lets docker-compose point this at the `ollama`
# service's Docker-network hostname (see docker-compose.prefect.yml) without
# changing config/smarthub.yaml's own default, which stays `localhost` for
# non-Docker/local-dev use (a natively-running Ollama).
OLLAMA_HOST = os.getenv("SMARTHUB_OLLAMA_HOST") or task_config.get(
    "explain", "ollama_host", DEFAULT_OLLAMA_HOST
)
LLM_TIMEOUT_SECONDS = task_config.get_float("explain", "timeout_seconds", 30.0)

# Dedupes the startup model-pull check across uvicorn's SERVE_WORKERS worker
# PROCESSES (`--workers N` forks child processes within the same container --
# see docker/Dockerfile.serve -- not separate containers, so they all share
# this path). Deliberately under /tmp, not the `data` volume: container-
# local and ephemeral is exactly right for a lock whose only job is "don't
# let two workers of the same container race the same pull at once".
_OLLAMA_PULL_LOCK_PATH = "/tmp/smarthub_ollama_pull.lock"


def format_llm_prompt(facts: dict) -> str:
    """Render structured facts into a tightly-templated LLM prompt.

    Deliberately rigid (not "be creative") and explicit about not inventing
    numbers — this is a formatting task, not a reasoning task, so the prompt
    is written to minimize the model's room to hallucinate. An optional
    ``decision_note`` (from ``predict.decide_bid`` — e.g. "this was a
    scheduled exploration probe") is included as another fact the LLM may
    use, not something it has to guess at from the numbers alone.

    Two guardrails added after observing a small model's actual output on a
    real lead:
    - An explicit statement of the model's monotonic bid constraint, because
      without it the LLM sometimes reasoned backwards about the `bid`
      factor's SHAP sign (e.g. implying a *lower* bid would have won more
      often, which the model's design rules out).
    - An optional ``bid_curve`` (from ``predict.bid_curve_around``) — actual
      win-rate/profit numbers at nearby bids — so claims about "what a
      different bid would do" are grounded in real numbers instead of the
      LLM guessing from the single chosen bid alone.
    """
    lines = [
        "You explain a pricing model's decision in plain English for a "
        "business user.",
        "Use ONLY the facts below. Do not invent numbers. 2-3 sentences, no " "jargon.",
        "Model rule: predicted win rate never decreases as the bid rises "
        "(built into the model by design) -- never claim a lower bid would "
        "win more often than a higher one.",
        "",
        f"Recommended bid: ${facts['recommended_bid']:.2f}",
        f"Predicted win rate at this bid: {facts['predicted_win_rate']:.0%} "
        f"(vs. average {facts['base_win_rate']:.0%})",
        f"Expected profit: ${facts['expected_profit']:.2f}",
    ]
    if facts.get("decision_note"):
        lines += ["", f"Note: {facts['decision_note']}"]
    if facts.get("bid_curve"):
        lines += ["", "Nearby bids explored (bid -> predicted win rate):"]
        for point in facts["bid_curve"]:
            lines.append(f"- ${point['bid']:.2f} -> {point['predicted_win_rate']:.0%}")
    lines += ["", "Top factors:"]
    for f in facts["top_factors"]:
        lines.append(f"- {f['feature']}={f['value']}: {f['direction']} win likelihood")
    lines += ["", "Explanation:"]
    return "\n".join(lines)


def call_ollama(prompt, model=None, host=None, timeout=None):
    """Call a local Ollama model; returns the generated text.

    Best-effort: if Ollama isn't reachable (not installed / not running),
    this logs nothing scary and returns a clear fallback message instead of
    raising — an explanation feature must never break its caller, and it's
    not on the live bidding path anyway, so a degraded (but honest) response
    is the right failure mode.
    """
    import requests

    model = model or LLM_MODEL
    host = host or OLLAMA_HOST
    timeout = timeout or LLM_TIMEOUT_SECONDS
    try:
        resp = requests.post(
            f"{host}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=timeout,
        )
        resp.raise_for_status()
        return resp.json()["response"].strip()
    except requests.HTTPError as exc:
        # Ollama reachable but the request failed (commonly a 404/500 because the
        # model isn't pulled yet). Surface Ollama's own error text, and if the
        # model is missing, kick off a background pull so it self-heals for the
        # next call instead of erroring forever.
        body = ""
        try:
            body = (exc.response.text or "")[:300]
        except Exception:  # noqa: BLE001
            pass
        logger.warning(
            "Ollama /api/generate failed for model %r at %s: %s %s",
            model,
            host,
            exc,
            body,
        )
        if not is_model_pulled(model, host):
            ensure_model_pulled_async(model, host)
            return (
                f"(Explanation temporarily unavailable: the local model "
                f"{model!r} isn't ready yet — a download was just started. "
                "The numeric factors above are accurate; try again in a minute.)"
            )
        return (
            f"(Explanation unavailable: local LLM error at {host} — {exc}. "
            f"{body} The numeric factors above are still accurate.)"
        )
    except Exception as exc:  # noqa: BLE001 - best-effort, never break the caller
        return (
            "(Explanation unavailable: could not reach the local LLM at "
            f"{host} — {exc}. The numeric factors above are still accurate.)"
        )


def is_model_pulled(model=None, host=None, timeout=None) -> bool:
    """Return whether ``model`` is already pulled (available) in the local
    Ollama instance's model list.

    Best-effort: any failure reaching Ollama (not running yet, network blip)
    is treated as "not pulled" rather than raised -- the only caller
    (``_ensure_model_pulled_sync``) reacts to either case by attempting a
    pull, which is the safe default either way, so there's no need to
    distinguish "definitely absent" from "couldn't check" here.
    """
    import requests

    model = model or LLM_MODEL
    host = host or OLLAMA_HOST
    timeout = timeout or LLM_TIMEOUT_SECONDS
    try:
        resp = requests.get(f"{host}/api/tags", timeout=timeout)
        resp.raise_for_status()
        names = {m.get("name") for m in resp.json().get("models", [])}
    except Exception:  # noqa: BLE001 -- best-effort, see docstring
        return False
    return model in names


def pull_model(model=None, host=None) -> bool:
    """Pull ``model`` into the local Ollama instance.

    Blocking -- a real pull can take minutes for a multi-GB model, with no
    fixed time budget, so this deliberately sets no request timeout. Callers
    that must not block (e.g. FastAPI startup) should run this off the main
    thread -- see ``ensure_model_pulled_async``.

    Returns
    -------
    bool
        Whether the pull request completed successfully.
    """
    import requests

    model = model or LLM_MODEL
    host = host or OLLAMA_HOST
    try:
        resp = requests.post(
            f"{host}/api/pull",
            json={"name": model, "stream": False},
            timeout=None,
        )
        resp.raise_for_status()
        return True
    except Exception:  # noqa: BLE001 -- best-effort; caller only logs a warning
        logger.warning(
            "Failed to pull Ollama model %r from %s", model, host, exc_info=True
        )
        return False


def _ensure_model_pulled_sync(
    model=None,
    host=None,
    wait_for_host_seconds=60,
    poll_interval_seconds=2,
) -> None:
    """Check-then-pull ``model``, after waiting briefly for Ollama itself to
    become reachable (it may still be starting -- e.g. right after `docker
    compose up`, `serve` and `ollama` start around the same time).

    Synchronous/blocking by design -- kept separate from
    ``ensure_model_pulled_async`` purely so the actual logic is directly
    unit-testable without touching real threads; the async wrapper is the
    only thing that needs a background thread.
    """
    import time

    import requests

    model = model or LLM_MODEL
    host = host or OLLAMA_HOST

    deadline = time.monotonic() + wait_for_host_seconds
    reachable = False
    while time.monotonic() < deadline:
        try:
            requests.get(f"{host}/api/tags", timeout=5).raise_for_status()
            reachable = True
            break
        except Exception:  # noqa: BLE001 -- still starting up, keep polling
            time.sleep(poll_interval_seconds)

    if not reachable:
        logger.warning(
            "Ollama at %s was not reachable within %ss -- skipping the "
            "model-pull check for now (retried automatically on next "
            "startup).",
            host,
            wait_for_host_seconds,
        )
        return

    if is_model_pulled(model, host):
        logger.info("Ollama model %r already pulled at %s.", model, host)
        return

    logger.info("Ollama model %r not found at %s -- pulling now...", model, host)
    if pull_model(model, host):
        logger.info("Finished pulling Ollama model %r.", model)


def _ensure_model_pulled_locked(model=None, host=None) -> None:
    """Run ``_ensure_model_pulled_sync`` behind a non-blocking file lock, so
    only one of uvicorn's `SERVE_WORKERS` worker PROCESSES (they fork inside
    one container -- see docker/Dockerfile.serve -- and so share one
    filesystem) actually performs the check/pull; the rest see the lock
    already held and return immediately.

    Without this, every worker independently calls this at its own startup
    and all of them fire a simultaneous `POST /api/pull` for the identical
    model -- quadrupling (at `SERVE_WORKERS=4`) bandwidth and disk I/O for
    the exact same bytes, and contending for CPU badly enough to make the
    API itself sluggish. Observed live: with 4 workers, `/docs` and
    `/health` started timing out under nginx's 5s `proxy_read_timeout` while
    all 4 workers pulled at once (see docs/CHANGELOG.md).

    A plain advisory `flock` (non-blocking) is enough -- no need to wait or
    retry if another worker already holds it, since whichever worker wins
    pulls the model into the shared `ollama` service, which benefits every
    worker's future `/explain_bid` (or /recommend_bid background SHAP) call
    either way.
    """
    try:
        lock_fd = os.open(_OLLAMA_PULL_LOCK_PATH, os.O_CREAT | os.O_RDWR, 0o644)
    except OSError:
        # Can't even open the lock file -- better to just run the check than
        # silently never attempt a pull at all.
        _ensure_model_pulled_sync(model, host)
        return

    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        # Another worker already holds the lock -- it's handling this.
        os.close(lock_fd)
        return

    try:
        _ensure_model_pulled_sync(model, host)
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)


def ensure_model_pulled_async(model=None, host=None) -> None:
    """Kick off the (lock-deduped) check/pull in a background daemon thread.

    Returns immediately -- never blocks the caller (FastAPI startup, in this
    codebase's only caller today, see ``predict._lifespan``). A real model
    pull can take minutes for a multi-GB model, and neither startup nor any
    in-flight `/recommend_bid` / `/explain_bid` request should ever wait on
    it -- same "never hold up live serving" principle as SHAP/logging
    elsewhere in this codebase. Daemon so it can't prevent process exit.
    """
    threading.Thread(
        target=_ensure_model_pulled_locked,
        args=(model, host),
        daemon=True,
        name="ollama-model-pull",
    ).start()
