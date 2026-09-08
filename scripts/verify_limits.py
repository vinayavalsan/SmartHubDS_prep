#!/usr/bin/env python3
"""Verify the SmartHub bid-path rate/concurrency limits actually reject overload.

Fires a hard burst at /recommend_bid and tallies the status mix. What you expect
depends on WHERE you point it:

  * serve directly (http://serve:8000, bypassing nginx) with concurrency above
    the aggregate cap (4 workers x --limit-concurrency 8 = 32) -> you should see
    HTTP 503 (uvicorn shedding load) once you exceed 32 in flight, and p99 on the
    200s should stay bounded (~550ms) instead of climbing to the 1s cliff.

  * through nginx (http://nginx) at a rate above the limit_req zones (per-IP
    50 r/s, per-key 100 r/s, plus their bursts) -> you should see HTTP 429
    (nginx rate limiting) once the burst tokens are spent.

A healthy result is a MIX: plenty of 200s (real capacity served) plus 503/429
(overflow rejected fast). Zero rejections under a big burst means the limit isn't
taking effect; all rejections means it's far too tight.

Uses the same deletable sentinel lead_ping_id as concurrency_sweep.py.
Only dependency: requests.

Examples:
    # Trip uvicorn --limit-concurrency (hit serve directly, >32 concurrent):
    python3 verify_limits.py --url http://serve:8000 --key shk_... \
        --concurrency 64 --total 4000

    # Trip nginx limit_req (hit the proxy, high rate from one IP/key):
    python3 verify_limits.py --url http://nginx --key shk_... \
        --concurrency 64 --total 4000
"""

from __future__ import annotations

import argparse
import statistics
import threading
import time
from collections import Counter
from datetime import datetime, timezone

import requests

DEFAULT_SENTINEL_LEAD_PING_ID = 2_000_000_000
_CREATED_AT = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def _payload(sentinel_id: int) -> dict:
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


def _run_soak(endpoint: str, payload: dict, headers: dict, args) -> None:
    """Open-loop SOAK: dispatch at a steady --rpm for --duration, then report.

    Unlike the burst (which hammers flat-out), this fires at a fixed rate so it
    models real production traffic. A healthy soak is ~all 200s with p99 < SLO
    and no 429/503 -- i.e. the service holds up at that rate without tripping
    any limit.
    """
    from concurrent.futures import ThreadPoolExecutor

    interval = 60.0 / args.rpm
    statuses: Counter = Counter()
    ok_lat: list[float] = []
    lock = threading.Lock()
    session_local = threading.local()

    def fire() -> None:
        sess = getattr(session_local, "s", None)
        if sess is None:
            sess = requests.Session()
            session_local.s = sess
        t0 = time.perf_counter()
        try:
            r = sess.post(endpoint, json=payload, headers=headers, timeout=args.timeout)
            dt = (time.perf_counter() - t0) * 1000.0
            with lock:
                statuses[str(r.status_code)] += 1
                if r.status_code == 200:
                    ok_lat.append(dt)
        except Exception as exc:  # noqa: BLE001
            with lock:
                statuses[type(exc).__name__] += 1

    print(f">> endpoint:   {endpoint}")
    print(f">> auth:       {'Bearer key' if args.key else 'NONE'}")
    print(f">> SOAK:       {args.rpm}/min ({args.rpm/60:.1f}/s) for "
          f"{args.duration:.0f}s\n")

    started = time.perf_counter()
    sent = 0
    # Pool big enough to never be the bottleneck at this rate + latency headroom.
    pool_size = max(64, int(args.rpm / 60 * args.timeout) + 16)
    with ThreadPoolExecutor(max_workers=pool_size) as pool:
        next_at = started
        while time.perf_counter() - started < args.duration:
            now = time.perf_counter()
            if now >= next_at:
                pool.submit(fire)
                sent += 1
                next_at += interval
            else:
                time.sleep(min(interval, next_at - now))
    wall = time.perf_counter() - started

    total = sum(statuses.values())
    ok = statuses.get("200", 0)
    r429 = statuses.get("429", 0)
    r503 = statuses.get("503", 0)

    print("---- status mix ----")
    for code, n in sorted(statuses.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"  {code:<16} {n:>6}  ({n/total*100:.1f}%)" if total else code)
    print(f"\nsent:        {sent}   completed: {total}")
    print(f"wall:        {wall:.1f}s   effective rate: {total/wall:.1f}/s "
          f"(target {args.rpm/60:.1f}/s)")
    if ok_lat:
        print(
            f"200 latency: p50={_pct(ok_lat,50):.0f}  p95={_pct(ok_lat,95):.0f}  "
            f"p99={_pct(ok_lat,99):.0f}  max={max(ok_lat):.0f} ms  "
            f"(mean={statistics.mean(ok_lat):.0f})"
        )
        within = sum(1 for x in ok_lat if x <= 1000) / len(ok_lat) * 100
        print(f"within 1s:   {within:.2f}%")

    print("\n---- verdict ----")
    p99 = _pct(ok_lat, 99) if ok_lat else float("nan")
    if ok and (r429 + r503) == 0 and p99 <= 1000:
        print(f"  => PASS: sustained {args.rpm/60:.0f}/s cleanly — p99 {p99:.0f}ms "
              "< 1s, no 429/503.")
    elif r429 or r503:
        print(f"  => Limits tripped at this rate: {r429} x429, {r503} x503. "
              "You're at/above the per-IP (50 r/s) cap — lower --rpm or raise the "
              "nginx rate for a pure service soak.")
    elif ok and p99 > 1000:
        print(f"  => WARN: p99 {p99:.0f}ms exceeds the 1s SLO at {args.rpm/60:.0f}/s "
              "even without rejections — service is the bottleneck, not the limit.")
    else:
        print("  => FAIL: little/nothing served. Check auth (401?) / target URL.")

    print(f"\nCleanup sentinel rows:  DELETE FROM prediction_log WHERE "
          f"lead_ping_id = {args.sentinel};")


def main() -> None:
    ap = argparse.ArgumentParser(description="Verify bid-path rate/concurrency limits")
    ap.add_argument("--url", required=True, help="base URL (serve:8000 or nginx)")
    ap.add_argument("--path", default="/recommend_bid")
    ap.add_argument(
        "--concurrency", type=int, default=64, help="in-flight workers (>32 to trip)"
    )
    ap.add_argument(
        "--total", type=int, default=4000, help="total requests to fire in the burst"
    )
    ap.add_argument("--timeout", type=float, default=15.0)
    ap.add_argument("--key", default=None, help="API key (Bearer). Omit if auth off.")
    ap.add_argument("--sentinel", type=int, default=DEFAULT_SENTINEL_LEAD_PING_ID)
    ap.add_argument(
        "--rpm",
        type=int,
        default=None,
        help="SOAK mode: dispatch at this steady requests/min for --duration "
        "instead of a flat-out burst. Keep under the nginx per-IP cap (50 r/s = "
        "3000 rpm) so a soak measures the service, not the rate limiter.",
    )
    ap.add_argument(
        "--duration",
        type=float,
        default=300.0,
        help="SOAK mode: seconds to sustain --rpm (default 300 = 5 min).",
    )
    args = ap.parse_args()

    endpoint = args.url.rstrip("/") + args.path
    payload = _payload(args.sentinel)
    headers = {"Authorization": f"Bearer {args.key}"} if args.key else {}

    if args.rpm:
        _run_soak(endpoint, payload, headers, args)
        return

    statuses: Counter = Counter()
    ok_lat: list[float] = []  # ms, 200s only
    lock = threading.Lock()
    remaining = args.total
    rem_lock = threading.Lock()

    def take() -> bool:
        nonlocal remaining
        with rem_lock:
            if remaining <= 0:
                return False
            remaining -= 1
            return True

    def worker() -> None:
        sess = requests.Session()
        loc: Counter = Counter()
        loc_lat: list[float] = []
        while take():
            t0 = time.perf_counter()
            try:
                r = sess.post(endpoint, json=payload, headers=headers, timeout=args.timeout)
                dt = (time.perf_counter() - t0) * 1000.0
                loc[str(r.status_code)] += 1
                if r.status_code == 200:
                    loc_lat.append(dt)
            except Exception as exc:  # noqa: BLE001
                loc[type(exc).__name__] += 1
        with lock:
            for k, n in loc.items():
                statuses[k] += n
            ok_lat.extend(loc_lat)

    print(f">> endpoint:    {endpoint}")
    print(f">> auth:        {'Bearer key' if args.key else 'NONE'}")
    print(f">> burst:       {args.total} requests @ concurrency {args.concurrency}\n")

    started = time.perf_counter()
    threads = [
        threading.Thread(target=worker, daemon=True) for _ in range(args.concurrency)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.perf_counter() - started

    total = sum(statuses.values())
    ok = statuses.get("200", 0)
    r429 = statuses.get("429", 0)
    r503 = statuses.get("503", 0)

    print("---- status mix ----")
    for code, n in sorted(statuses.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"  {code:<16} {n:>6}  ({n/total*100:.1f}%)")
    print(f"\nwall:        {wall:.1f}s   effective rate: {total/wall:.0f}/s")
    if ok_lat:
        print(
            f"200 latency: p50={_pct(ok_lat,50):.0f}  p95={_pct(ok_lat,95):.0f}  "
            f"p99={_pct(ok_lat,99):.0f}  max={max(ok_lat):.0f} ms  "
            f"(mean={statistics.mean(ok_lat):.0f})"
        )

    print("\n---- verdict ----")
    print(f"  served (200):        {ok}")
    print(f"  nginx rate-limited:  {r429}  (429)")
    print(f"  uvicorn shed load:   {r503}  (503)")
    if ok and (r429 or r503):
        print("  => PASS: real capacity served AND overflow rejected fast.")
    elif ok and not (r429 or r503):
        print("  => WARN: no 429/503 under burst — limit not tripping. Increase "
              "--concurrency/--total, or confirm you hit the right target "
              "(serve:8000 for 503, nginx for 429).")
    elif not ok:
        print("  => FAIL: nothing served. Check auth (401?), target URL, or a "
              "far-too-tight limit.")

    print(f"\nCleanup sentinel rows:  DELETE FROM prediction_log WHERE "
          f"lead_ping_id = {args.sentinel};")


if __name__ == "__main__":
    main()
