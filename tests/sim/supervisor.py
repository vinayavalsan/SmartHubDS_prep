#!/usr/bin/env python3
"""
Piece 7 — Supervisor: keep the replay running all week, unattended.

Runs the replay one day-segment at a time, so a crash restarts only the current
day (checkpointed) rather than the whole week. After each day it runs the
analyzer for a rollup and posts a heartbeat to Slack. Put THIS under a process
manager (systemd `Restart=always`, or `nohup`) so a machine reboot brings it
back; the checkpoint makes it resume the right day.

    nohup python supervisor.py --data data/sim/snapshot.parquet \
        --url http://staging:8000 --days 4 --burst-prob 0.6 \
        --ledger-dir data/sim/ledgers > data/sim/supervisor.log 2>&1 &

Slack: set SLACK_WEBHOOK in the environment to receive heartbeats (else printed).
Quick test: add `--segment-minutes 6 --speed 60` to run a "day" in ~6 seconds.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))


# level -> (Slack attachment colour, leading emoji)
_LEVELS = {
    "info": ("#2eb67d", ":large_green_circle:"),
    "warn": ("#ecb22e", ":large_yellow_circle:"),
    "error": ("#e01e5a", ":red_circle:"),
}


def notify(
    title: str,
    detail: str = "",
    level: str = "info",
    code: str = "",
    suggestion: str = "",
) -> None:
    """Post a colour-coded Slack alert if SLACK_WEBHOOK is set; always echo.

    ``level`` is one of info/warn/error. On an errors-only channel, set
    SIM_ALERTS_ERRORS_ONLY=1 to suppress routine info-level lifecycle posts
    (start / day-complete / finished) while still delivering warn + error.
    """
    import socket

    line = f"[{level}] {title}" + (f" - {detail}" if detail else "")
    print(f"[alert] {line}", flush=True)

    hook = os.environ.get("SLACK_WEBHOOK")
    if not hook:
        return
    if level == "info" and os.environ.get(
        "SIM_ALERTS_ERRORS_ONLY", ""
    ).strip().lower() in {"1", "true", "yes"}:
        return

    color, emoji = _LEVELS.get(level, _LEVELS["info"])
    when = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%SZ")
    blocks = [
        {
            "type": "section",
            "text": {"type": "mrkdwn", "text": f"{emoji} *{title}*"},
        }
    ]
    if detail:
        blocks.append(
            {"type": "section", "text": {"type": "mrkdwn", "text": detail}}
        )
    if code:
        blocks.append(
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": "*Traceback (tail):*\n```" + code[-2500:] + "```",
                },
            }
        )
    if suggestion:
        blocks.append(
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": ":bulb: *Suggested fix (LLM):*\n" + suggestion[:2500],
                },
            }
        )
    blocks.append(
        {
            "type": "section",
            "fields": [
                {"type": "mrkdwn", "text": f"*Host:*\n{socket.gethostname()}"},
                {
                    "type": "mrkdwn",
                    "text": f"*Target:*\n{os.environ.get('SIM_URL', 'n/a')}",
                },
                {"type": "mrkdwn", "text": f"*When (UTC):*\n{when}"},
                {"type": "mrkdwn", "text": "*Component:*\nreplay-supervisor"},
            ],
        }
    )
    payload = {"attachments": [{"color": color, "blocks": blocks}]}
    try:
        req = urllib.request.Request(
            hook,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        urllib.request.urlopen(req, timeout=10)
    except Exception as exc:  # noqa: BLE001 - never let alerting kill the run
        print(f"[alert] slack post failed: {exc}", flush=True)


def heartbeat(text: str) -> None:  # back-compat shim
    notify(text, level="info")


# Last captured child output of a failed day, so the FAILED alert in main() can
# attach the traceback + an LLM fix suggestion.
_LAST_FAIL_OUTPUT = ""


def _tail_lines(text: str, n: int) -> str:
    return "\n".join(text.rstrip("\n").splitlines()[-n:])


def _run_capture(cmd, tail_lines: int = 120):
    """Run cmd, stream its output live to the console, and return
    (returncode, tail) where tail is the last N lines (incl. any traceback)."""
    from collections import deque

    buf: deque = deque(maxlen=tail_lines)
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,  # fold stderr in so tracebacks are captured
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    for ln in proc.stdout:
        print(ln, end="", flush=True)
        buf.append(ln)
    proc.wait()
    return proc.returncode, "".join(buf)


def _llm_fix_suggestion(tb_text: str) -> str:
    """Ask the SAME local LLM prod uses (llm_explain.call_ollama) for a terse
    root-cause + fix. Returns a fallback string if the LLM is unreachable."""
    if not tb_text.strip():
        return ""
    try:
        from smarthub.train_and_predict.llm_explain import call_ollama
    except Exception as exc:  # noqa: BLE001
        return f"(LLM unavailable: {exc})"
    prompt = (
        "You are an SRE assistant for the SmartHub bid-recommendation service. "
        "A replay/load-test day just failed. From the traceback below, reply "
        "with (1) the single most likely root cause in one sentence, then "
        "(2) 1-3 concrete fix steps as short bullet lines. Be terse and "
        "technical; do not repeat the traceback.\n\nTRACEBACK:\n"
        + tb_text[-4000:]
    )
    try:
        return call_ollama(prompt)
    except Exception as exc:  # noqa: BLE001
        return f"(LLM suggestion failed: {exc})"


def load_ckpt(path: str) -> int:
    try:
        with open(path) as f:
            return int(json.load(f).get("last_completed_day", 0))
    except Exception:
        return 0


def save_ckpt(path: str, day: int) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(
            {
                "last_completed_day": day,
                "updated": datetime.now().isoformat(timespec="seconds"),
            },
            f,
        )
    os.replace(tmp, path)


def _wait_healthy(url: str, api_key: str | None, timeout_s: float = 180.0) -> bool:
    """Poll <url>/health until the model is loaded, so a day never runs against
    a cold or just-rebooted serve. Returns True once healthy, else False."""
    import urllib.request

    deadline = time.time() + timeout_s
    probe = url.rstrip("/") + "/health?lead_type_id=6"
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(probe, timeout=8) as resp:
                if json.load(resp).get("model_loaded"):
                    return True
        except Exception:  # noqa: BLE001 - serve still starting; keep polling
            pass
        time.sleep(5)
    return False


def _finish(args) -> int:
    """Exit 0, or -- for a restart:unless-stopped container -- hold idle so a
    completed run is not restart-looped by the container manager."""
    if getattr(args, "hold_when_done", False):
        print(
            "[supervisor] all days complete -- holding idle "
            "(use 'sim_test.sh down' to remove).",
            flush=True,
        )
        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            pass
    return 0


def run_day(args, day: int, ledger: str) -> bool:
    """Run one day-segment; restart on crash up to --max-restarts. True if ok."""
    cmd = [
        args.python,
        os.path.join(HERE, "run.py"),
        "--data",
        args.data,
        "--url",
        args.url,
        "--minutes",
        str(args.segment_minutes),
        "--out",
        ledger,
        "--dispatch",
        "--seed",
        str(args.seed + day),
        "--rate-min",
        str(args.rate_min),
        "--rate-max",
        str(args.rate_max),
        "--burst-prob",
        str(args.burst_prob),
        "--burst-frac-min",
        str(args.burst_frac_min),
        "--burst-frac-max",
        str(args.burst_frac_max),
        "--max-bursts",
        str(args.max_bursts),
        "--speed",
        str(args.speed),
    ]
    if args.api_key:
        cmd += ["--api-key", args.api_key]
    for attempt in range(1, args.max_restarts + 1):
        if not _wait_healthy(args.url, args.api_key):
            notify(
                f"Serve not healthy before day {day}",
                f"attempt {attempt} - retrying",
                level="warn",
            )
            time.sleep(min(30, 5 * attempt))
            continue
        print(f"\n=== day {day} attempt {attempt}: {' '.join(cmd)}", flush=True)
        rc, tail = _run_capture(cmd)
        if rc == 0:
            return True
        global _LAST_FAIL_OUTPUT
        _LAST_FAIL_OUTPUT = tail
        notify(
            f"Replay day {day} restarting",
            f"exited rc={rc} (attempt {attempt}/{args.max_restarts})",
            level="warn",
            code=_tail_lines(tail, 25),
        )
        time.sleep(min(30, 5 * attempt))
    return False


def rollup(args, day: int, ledger: str) -> str:
    """Run the analyzer on the day's ledger; return a short summary line."""
    if not args.analyze or not os.path.exists(ledger):
        return "(no rollup)"
    out = subprocess.run(
        [args.python, os.path.join(HERE, "analyze.py"), ledger, "--data", args.data],
        capture_output=True,
        text=True,
    ).stdout
    with open(os.path.join(args.ledger_dir, f"rollup_day{day}.txt"), "w") as f:
        f.write(out)
    # pull the two lines humans care about
    picks = [
        ln.strip()
        for ln in out.splitlines()
        if ln.strip().startswith(("requests", "latency ms")) or "[FAIL]" in ln
    ]
    return " | ".join(picks) if picks else "rollup written"


def main() -> int:
    ap = argparse.ArgumentParser(description="Supervise the week-long replay.")
    ap.add_argument("--data", required=True)
    ap.add_argument("--url", required=True)
    ap.add_argument("--days", type=int, default=4, help="historical days to run")
    ap.add_argument("--segment-minutes", type=int, default=1440, help="minutes per day")
    ap.add_argument("--ledger-dir", default="data/sim/ledgers")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--rate-min", type=int, default=100)
    ap.add_argument("--rate-max", type=int, default=150)
    ap.add_argument("--burst-prob", type=float, default=0.5)
    ap.add_argument("--burst-frac-min", type=float, default=0.3)
    ap.add_argument("--burst-frac-max", type=float, default=0.6)
    ap.add_argument("--max-bursts", type=int, default=1)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--speed", type=float, default=1.0, help="test only: compress time")
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--max-restarts", type=int, default=5)
    ap.add_argument("--no-analyze", dest="analyze", action="store_false")
    ap.add_argument(
        "--hold-when-done",
        action="store_true",
        help="after all days complete, sleep instead of exiting (keeps a "
        "restart:unless-stopped container Up without a restart loop)",
    )
    ap.add_argument("--python", default=sys.executable)
    args = ap.parse_args()

    os.makedirs(args.ledger_dir, exist_ok=True)
    ckpt = args.checkpoint or os.path.join(args.ledger_dir, "supervisor.ckpt.json")
    done = load_ckpt(ckpt)
    if done >= args.days:
        print(f"All {args.days} days already complete (checkpoint={ckpt}).")
        return _finish(args)

    notify(
        "Replay supervisor started",
        f"Days {done+1}-{args.days}  |  target `{args.url}`",
        level="info",
    )
    for day in range(done + 1, args.days + 1):
        ledger = os.path.join(args.ledger_dir, f"day{day}.jsonl")
        ok = run_day(args, day, ledger)
        if not ok:
            suggestion = _llm_fix_suggestion(_LAST_FAIL_OUTPUT)
            notify(
                f"Replay day {day} FAILED",
                f"after {args.max_restarts} restarts - supervisor stopping",
                level="error",
                code=_tail_lines(_LAST_FAIL_OUTPUT, 40),
                suggestion=suggestion,
            )
            return 1
        summary = rollup(args, day, ledger)
        save_ckpt(ckpt, day)
        notify(
            f"Replay day {day}/{args.days} complete",
            summary,
            level="info",
        )

    notify(
        "Replay finished all historical days",
        f"{args.days} days complete",
        level="info",
    )
    return _finish(args)


if __name__ == "__main__":
    sys.exit(main())
