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


def heartbeat(text: str) -> None:
    """Post to Slack if SLACK_WEBHOOK is set; always echo locally."""
    print(f"[heartbeat] {text}", flush=True)
    hook = os.environ.get("SLACK_WEBHOOK")
    if not hook:
        return
    try:
        req = urllib.request.Request(
            hook,
            data=json.dumps({"text": text}).encode(),
            headers={"Content-Type": "application/json"},
        )
        urllib.request.urlopen(req, timeout=10)
    except Exception as exc:  # noqa: BLE001 - never let alerting kill the run
        print(f"[heartbeat] slack post failed: {exc}", flush=True)


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
        print(f"\n=== day {day} attempt {attempt}: {' '.join(cmd)}", flush=True)
        rc = subprocess.run(cmd).returncode
        if rc == 0:
            return True
        heartbeat(
            f":warning: replay day {day} exited rc={rc} "
            f"(attempt {attempt}/{args.max_restarts}) — restarting"
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
    ap.add_argument("--python", default=sys.executable)
    args = ap.parse_args()

    os.makedirs(args.ledger_dir, exist_ok=True)
    ckpt = args.checkpoint or os.path.join(args.ledger_dir, "supervisor.ckpt.json")
    done = load_ckpt(ckpt)
    if done >= args.days:
        print(f"All {args.days} days already complete (checkpoint={ckpt}).")
        return 0

    heartbeat(
        f":rocket: replay supervisor starting — days {done+1}..{args.days}, "
        f"target {args.url}"
    )
    for day in range(done + 1, args.days + 1):
        ledger = os.path.join(args.ledger_dir, f"day{day}.jsonl")
        ok = run_day(args, day, ledger)
        if not ok:
            heartbeat(
                f":rotating_light: replay day {day} FAILED after "
                f"{args.max_restarts} restarts — supervisor stopping."
            )
            return 1
        summary = rollup(args, day, ledger)
        save_ckpt(ckpt, day)
        heartbeat(
            f":white_check_mark: replay day {day}/{args.days} complete — {summary}"
        )

    heartbeat(f":checkered_flag: replay finished all {args.days} historical days.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
