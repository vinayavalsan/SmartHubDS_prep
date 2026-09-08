#!/usr/bin/env python3
"""Closed-loop CONCURRENCY sweep for the SmartHub bid API (/recommend_bid).

Unlike loadtest.py / bidload.py (open-loop: fire at a target rate), this holds a
FIXED number of concurrent clients, each firing back-to-back (send -> wait ->
send again) for a fixed duration, then steps the concurrency up. That directly
measures how much concurrency FastAPI/uvicorn can absorb before latency breaks,
which is the number to use for uvicorn `--limit-concurrency`.

Read the knee off the table: the concurrency level where throughput stops rising
AND p99 climbs past the SLO (1s). Size --limit-concurrency at (or just below)
that level, PER worker process. Run serve with SERVE_WORKERS=1 for a clean
per-worker number; global cap = per-worker N x workers.

Bypass nginx: point --url at the internal uvicorn (http://serve:8000) and run
this from a sibling container on the stack network (NOT the serve container, so
the generator doesn't steal serve's CPU). See the runbook printed by --help-run.

Writes junk prediction-log rows under a single sentinel lead_ping_id so they are
trivially deletable afterwards (see --sentinel and the printed cleanup SQL).

Only dependency: requests.

Example:
    python3 concurrency_sweep.py \
        --url http://serve:8000 \
        --concurrency 1,2,4,8,16,32,64,128 \
        --duration 30 --warmup-secs 5
"""

from __future__ import annotations

import argparse
import statistics
import threading
import time
from collections import Counter
from datetime import datetime, timezone

import requests

# Fixed sentinel so every load row is identifiable and deletable in one query.
DEFAULT_SENTINEL_LEAD_PING_ID = 2_000_000_000
_CREATED_AT = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def _payload(sentinel_id: int) -> dict:
    """Valid /recommend_bid body (current contract) with a sentinel lead id."""
    return {
        "expected_revenue": 25.00,
        "target_cm": 0.25,
        "min_bid": 0.00,
        "bid_step": 0.25,
        "lead_type_id": 6,
        "campaign_id": 40088,
        "source_type_id": 574,
        "traffic_tier": "tier_2",
        "state": "TX",
        "created_at": _CREATED_AT,
        "lead_ping_id": sentinel_id,
        "insured": "true",
        "home_owner": "false",
        "dui": "false",
        "num_vehicles": 2,
        "num_auto_accidents": 0,
        "age": 34,
    }


def _pct(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    v = sorted(values)
    k = max(0, min(len(v) - 1, int(round((p / 100.0) * (len(v) - 1)))))
    return v[k]


def _run_level(
    endpoint: str,
    concurrency: int,
    duration: float,
    payload: dict,
    headers: dict,
    timeout: float,
) -> dict:
    """Hold `concurrency` back-to-back clients for `duration`s; collect stats.

    Each worker thread loops fire->record->fire until the shared deadline, so
    there are always exactly `concurrency` requests in flight (closed loop).
    """
    latencies: list[float] = []  # ms, successes only
    errors: Counter = Counter()
    lock = threading.Lock()
    stop = threading.Event()

    # One reusable session per thread (keep-alive) — matches how a real client
    # would call, and keeps the generator from paying TCP setup every request.
    def worker() -> None:
        sess = requests.Session()
        local_lat: list[float] = []
        local_err: Counter = Counter()
        while not stop.is_set():
            t0 = time.perf_counter()
            try:
                r = sess.post(endpoint, json=payload, headers=headers, timeout=timeout)
                dt = (time.perf_counter() - t0) * 1000.0
                if r.status_code == 200:
                    local_lat.append(dt)
                else:
                    local_err[f"HTTP {r.status_code}"] += 1
            except Exception as exc:  # noqa: BLE001
                local_err[type(exc).__name__] += 1
        with lock:
            latencies.extend(local_lat)
            for k, n in local_err.items():
                errors[k] += n

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(concurrency)]
    started = time.perf_counter()
    for t in threads:
        t.start()
    time.sleep(duration)
    stop.set()
    for t in threads:
        t.join(timeout=timeout + 5)
    wall = time.perf_counter() - started

    ok = len(latencies)
    return {
        "concurrency": concurrency,
        "ok": ok,
        "errors": dict(errors),
        "err_total": sum(errors.values()),
        "wall": wall,
        "throughput": ok / wall if wall else 0.0,
        "p50": _pct(latencies, 50),
        "p95": _pct(latencies, 95),
        "p99": _pct(latencies, 99),
        "max": max(latencies) if latencies else float("nan"),
        "within_1s": (sum(1 for x in latencies if x <= 1000) / ok * 100) if ok else 0.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Closed-loop concurrency sweep")
    ap.add_argument("--url", required=True, help="base URL, e.g. http://serve:8000")
    ap.add_argument("--path", default="/recommend_bid", help="endpoint path")
    ap.add_argument(
        "--concurrency",
        default="1,2,4,8,16,32,64,128",
        help="comma-separated concurrency levels to sweep",
    )
    ap.add_argument(
        "--duration", type=float, default=30.0, help="measured seconds per level"
    )
    ap.add_argument(
        "--warmup-secs",
        type=float,
        default=5.0,
        help="warm-up seconds at concurrency=4 (primes model cache; not counted)",
    )
    ap.add_argument(
        "--cooldown", type=float, default=2.0, help="idle seconds between levels"
    )
    ap.add_argument("--timeout", type=float, default=15.0, help="per-request timeout(s)")
    ap.add_argument(
        "--key",
        default=None,
        help="API key (sent as Authorization: Bearer). Omit if auth is disabled.",
    )
    ap.add_argument(
        "--sentinel",
        type=int,
        default=DEFAULT_SENTINEL_LEAD_PING_ID,
        help="lead_ping_id used for all load rows (deletable sentinel)",
    )
    ap.add_argument("--slo", type=float, default=1.0, help="p99 SLO seconds")
    args = ap.parse_args()

    endpoint = args.url.rstrip("/") + args.path
    levels = [int(x) for x in args.concurrency.split(",") if x.strip()]
    payload = _payload(args.sentinel)
    headers = {"Authorization": f"Bearer {args.key}"} if args.key else {}

    print(f">> endpoint:   {endpoint}")
    print(f">> auth:       {'Bearer key' if args.key else 'NONE (auth disabled?)'}")
    print(f">> sentinel lead_ping_id: {args.sentinel}")
    print(f">> levels:     {levels}")
    print(f">> per level:  {args.duration:.0f}s measured, {args.warmup_secs:.0f}s warmup\n")

    # Warm up once so the first level doesn't eat cold-start model download.
    if args.warmup_secs > 0:
        print(f">> warming up ({args.warmup_secs:.0f}s @ concurrency 4) ...")
        _run_level(endpoint, 4, args.warmup_secs, payload, headers, args.timeout)
        print(">> warm-up done\n")

    header = (
        f"{'C':>5}  {'thru/s':>8}  {'ok':>7}  {'p50':>6}  {'p95':>7}  "
        f"{'p99':>7}  {'max':>7}  {'<1s%':>6}  errors"
    )
    print(header)
    print("-" * len(header))

    rows = []
    prev_thru = 0.0
    for c in levels:
        res = _run_level(endpoint, c, args.duration, payload, headers, args.timeout)
        rows.append(res)
        knee = ""
        # Flag the likely knee: throughput barely rose but p99 blew the SLO.
        if res["p99"] > args.slo * 1000 and res["throughput"] <= prev_thru * 1.1:
            knee = "  <-- knee?"
        print(
            f"{c:>5}  {res['throughput']:>8.1f}  {res['ok']:>7}  "
            f"{res['p50']:>6.0f}  {res['p95']:>7.0f}  {res['p99']:>7.0f}  "
            f"{res['max']:>7.0f}  {res['within_1s']:>6.1f}  "
            f"{res['errors'] if res['errors'] else '-'}{knee}"
        )
        prev_thru = max(prev_thru, res["throughput"])
        time.sleep(args.cooldown)

    print("\n(latencies in ms. Pick --limit-concurrency at the highest C where p99 "
          f"<= {args.slo*1000:.0f}ms and throughput is still rising — PER worker.)")
    print("\nCleanup the sentinel load rows afterwards, e.g.:")
    print(
        "  DELETE FROM smarthub_prediction_log WHERE lead_ping_id = "
        f"{args.sentinel};"
    )
    print("  (adjust table name/schema to your prediction-log store)")


if __name__ == "__main__":
    main()
