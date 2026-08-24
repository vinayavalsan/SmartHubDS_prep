#!/usr/bin/env python3
"""Load test for the SmartHub bid API (/recommend_bid).

Fires requests at a target rate for a fixed duration and reports throughput,
latency percentiles, and whether p99 stayed under the 1s TAT SLO.

Examples:
    # SLO baseline: 200 requests/min for 60s
    python loadtest.py --url http://54.161.161.100:8000 --rpm 200 --duration 60

    # Stress: 3000 requests/min for 60s
    python loadtest.py --url http://54.161.161.100:8000 --rpm 3000 --duration 60

    # Local
    python loadtest.py --url http://localhost:8000 --rpm 200 --duration 30

Only dependency is `requests` (already used elsewhere in the repo).
"""

from __future__ import annotations

import argparse
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

import requests

PAYLOAD = {
    "expected_revenue": 25.00,
    "target_cm": 0.25,
    "min_bid": 0.00,
    "bid_step": 0.25,
    "campaign_id": 12345,
    "lead_type_id": 6,
    "created_hour": 14,
    "created_dayofweek": 2,
    "state": "TX",
    "insured": "true",
    "home_owner": "false",
    "dui": "false",
    "num_vehicles": 2,
    "num_drivers": 2,
    "num_auto_violations": 0,
    "num_auto_accidents": 0,
    "continuous_coverage_months": 24,
    "military_affiliation": "false",
    "gender": "Female",
    "marital_status": "Single",
    "age": 34,
}

_lock = Lock()
_latencies: list[float] = []
_errors: dict[str, int] = {}


def _fire(url: str, timeout: float) -> None:
    start = time.perf_counter()
    try:
        r = requests.post(url, json=PAYLOAD, timeout=timeout)
        elapsed = time.perf_counter() - start
        with _lock:
            if r.status_code == 200:
                _latencies.append(elapsed)
            else:
                _errors[f"HTTP {r.status_code}"] = (
                    _errors.get(f"HTTP {r.status_code}", 0) + 1
                )
    except Exception as exc:  # noqa: BLE001
        key = type(exc).__name__
        with _lock:
            _errors[key] = _errors.get(key, 0) + 1


def _pct(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    values = sorted(values)
    k = max(0, min(len(values) - 1, int(round((p / 100.0) * (len(values) - 1)))))
    return values[k]


def main() -> None:
    ap = argparse.ArgumentParser(description="Load test /recommend_bid")
    ap.add_argument(
        "--url", required=True, help="base URL, e.g. http://54.161.161.100:8000"
    )
    ap.add_argument("--rpm", type=int, default=200, help="target requests per minute")
    ap.add_argument("--duration", type=int, default=60, help="test duration in seconds")
    ap.add_argument(
        "--workers", type=int, default=64, help="max concurrent in-flight requests"
    )
    ap.add_argument(
        "--timeout", type=float, default=10.0, help="per-request timeout (s)"
    )
    ap.add_argument(
        "--slo", type=float, default=1.0, help="TAT SLO in seconds (p99 target)"
    )
    ap.add_argument(
        "--warmup",
        type=int,
        default=0,
        help="fire N warm-up requests (not timed) to prime model caches first",
    )
    args = ap.parse_args()

    endpoint = args.url.rstrip("/") + "/recommend_bid"

    if args.warmup:
        print(f">> warming up ({args.warmup} requests, not counted) ...")
        for _ in range(args.warmup):
            try:
                requests.post(endpoint, json=PAYLOAD, timeout=args.timeout)
            except Exception:  # noqa: BLE001
                pass
        print(">> warm-up done\n")
    interval = 60.0 / args.rpm  # seconds between dispatches
    total = int(args.rpm * args.duration / 60)

    print(f">> {endpoint}")
    print(
        f">> rate={args.rpm}/min  duration={args.duration}s  ~{total} requests  "
        f"workers={args.workers}\n"
    )

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        next_at = started
        sent = 0
        while time.perf_counter() - started < args.duration:
            now = time.perf_counter()
            if now >= next_at:
                pool.submit(_fire, endpoint, args.timeout)
                sent += 1
                next_at += interval
            else:
                time.sleep(min(interval, next_at - now))
        # pool context exit waits for in-flight requests to finish
    wall = time.perf_counter() - started

    ok = len(_latencies)
    err = sum(_errors.values())
    print("---- results ----")
    print(f"sent:        {sent}")
    print(f"succeeded:   {ok}")
    print(f"failed:      {err}  {dict(_errors) if _errors else ''}")
    print(f"wall time:   {wall:.1f}s")
    print(f"throughput:  {ok / wall * 60:.0f} ok/min ({ok / wall:.1f}/s)")
    if _latencies:
        print(
            f"latency ms:  p50={_pct(_latencies, 50)*1000:.0f}  "
            f"p95={_pct(_latencies, 95)*1000:.0f}  "
            f"p99={_pct(_latencies, 99)*1000:.0f}  "
            f"max={max(_latencies)*1000:.0f}  "
            f"mean={statistics.mean(_latencies)*1000:.0f}"
        )
        p99 = _pct(_latencies, 99)
        verdict = "PASS" if p99 <= args.slo and err == 0 else "FAIL"
        print(
            f"\nSLO (p99 <= {args.slo:.0f}s, 0 errors): {verdict}  "
            f"(p99={p99*1000:.0f}ms)"
        )


if __name__ == "__main__":
    main()
