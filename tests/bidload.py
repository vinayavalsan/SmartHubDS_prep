#!/usr/bin/env python3
"""Load-test the SmartHub bid API, capture every response to a file, analyze it.

Two modes:

  # 1) RUN the test -- saves one JSON line per request to --out
  python3 bidload.py run --url http://localhost:8000 --rpm 2000 --duration 60 \
      --out results.jsonl --warmup 5

  # 2) ANALYZE a saved results file (latency + status + response-field stats)
  python3 bidload.py analyze results.jsonl

Each line written by `run` looks like:
  {"ts": 1.23, "latency_ms": 31.4, "status": 200, "ok": true, "response": {...}}
  {"ts": 4.56, "latency_ms": 0.0,  "status": 0,   "ok": false, "error": "ReadTimeout"}

Only dependency: requests.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import requests

# Sample /recommend_bid request (matches server/manual_api_check.py).
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


def gen_payload() -> dict:
    """Randomized but valid /recommend_bid request (auto lead type) so a run
    sweeps the model across scenarios instead of repeating one input."""
    return {
        "expected_revenue": round(random.uniform(5, 60), 2),
        "target_cm": round(random.uniform(0.15, 0.40), 3),
        "min_bid": 0.00,
        "bid_step": 0.25,
        "campaign_id": random.choice([12345, 22221, 33310, 40088]),
        "lead_type_id": 6,
        "created_hour": random.randint(0, 23),
        "created_dayofweek": random.randint(0, 6),
        "state": random.choice(["TX", "CA", "FL", "NY", "OH", "GA", "PA", "IL"]),
        "insured": random.choice(["true", "false"]),
        "home_owner": random.choice(["true", "false"]),
        "dui": random.choice(["true", "false"]),
        "num_vehicles": random.randint(1, 4),
        "num_drivers": random.randint(1, 4),
        "num_auto_violations": random.randint(0, 3),
        "num_auto_accidents": random.randint(0, 2),
        "continuous_coverage_months": random.choice([0, 6, 12, 24, 36]),
        "military_affiliation": random.choice(["true", "false"]),
        "gender": random.choice(["Male", "Female"]),
        "marital_status": random.choice(["Single", "Married"]),
        "age": random.randint(18, 75),
    }


def _pct(values, p):
    if not values:
        return float("nan")
    v = sorted(values)
    k = max(0, min(len(v) - 1, int(round((p / 100.0) * (len(v) - 1)))))
    return v[k]


# --------------------------------------------------------------------------- #
# run                                                                         #
# --------------------------------------------------------------------------- #
def cmd_run(args) -> None:
    endpoint = args.url.rstrip("/") + args.path
    interval = 60.0 / args.rpm
    results: list[dict] = []  # list.append is GIL-atomic; no lock needed

    def fire(t0: float) -> None:
        start = time.perf_counter()
        payload = gen_payload() if args.vary else PAYLOAD
        rec: dict = {"ts": round(start - t0, 4)}
        if args.vary:
            rec["request"] = payload   # keep input to correlate with the bid later
        try:
            r = requests.post(endpoint, json=payload, timeout=args.timeout)
            rec["latency_ms"] = round((time.perf_counter() - start) * 1000, 2)
            rec["status"] = r.status_code
            rec["ok"] = r.status_code == 200
            try:
                rec["response"] = r.json()
            except Exception:  # noqa: BLE001 -- non-JSON body
                rec["response"] = r.text[:500]
        except Exception as exc:  # noqa: BLE001
            rec["latency_ms"] = round((time.perf_counter() - start) * 1000, 2)
            rec["status"] = 0
            rec["ok"] = False
            rec["error"] = type(exc).__name__
        results.append(rec)

    if args.warmup:
        print(f">> warming up ({args.warmup} requests, not saved) ...")
        for _ in range(args.warmup):
            try:
                requests.post(endpoint, json=PAYLOAD, timeout=args.timeout)
            except Exception:  # noqa: BLE001
                pass

    print(f">> {endpoint}")
    print(f">> rate={args.rpm}/min  duration={args.duration}s  "
          f"workers={args.workers}  out={args.out}\n")

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        next_at = started
        sent = 0
        while time.perf_counter() - started < args.duration:
            now = time.perf_counter()
            if now >= next_at:
                pool.submit(fire, started)
                sent += 1
                next_at += interval
            else:
                time.sleep(min(interval, next_at - now))
    wall = time.perf_counter() - started

    results.sort(key=lambda r: r["ts"])
    with open(args.out, "w") as fh:
        for rec in results:
            fh.write(json.dumps(rec) + "\n")

    ok = sum(1 for r in results if r["ok"])
    print(f">> dispatched {sent}, captured {len(results)} "
          f"({ok} ok / {len(results)-ok} failed) in {wall:.1f}s")
    print(f">> saved to {args.out}")
    print(f">> analyze with:  python3 bidload.py analyze {args.out}")


# --------------------------------------------------------------------------- #
# analyze                                                                      #
# --------------------------------------------------------------------------- #
def _flatten(d, prefix=""):
    """Flatten a nested dict into dotted leaf keys."""
    out = {}
    if isinstance(d, dict):
        for k, v in d.items():
            out.update(_flatten(v, f"{prefix}{k}."))
    else:
        out[prefix[:-1]] = d
    return out


def cmd_analyze(args) -> None:
    rows = []
    with open(args.file) as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    if not rows:
        print("no rows in file")
        return

    ok = [r for r in rows if r.get("ok")]
    bad = [r for r in rows if not r.get("ok")]
    lat = [r["latency_ms"] for r in ok if "latency_ms" in r]
    ts = [r["ts"] for r in rows if "ts" in r]
    span = (max(ts) - min(ts)) if len(ts) > 1 else 0.0

    print("==================== SUMMARY ====================")
    print(f"file:        {args.file}")
    print(f"total:       {len(rows)}")
    print(f"ok:          {len(ok)}")
    print(f"failed:      {len(bad)}")
    if bad:
        errs = Counter(r.get("error") or f"HTTP {r.get('status')}" for r in bad)
        print(f"  errors:    {dict(errs)}")
    if span > 0:
        print(f"span:        {span:.1f}s")
        print(f"throughput:  {len(ok)/span*60:.0f} ok/min ({len(ok)/span:.1f}/s)")

    print("\n==================== LATENCY (ms) ====================")
    if lat:
        print(f"n={len(lat)}  min={min(lat):.0f}  p50={_pct(lat,50):.0f}  "
              f"p90={_pct(lat,90):.0f}  p95={_pct(lat,95):.0f}  "
              f"p99={_pct(lat,99):.0f}  max={max(lat):.0f}  "
              f"mean={statistics.mean(lat):.0f}")
        within1s = sum(1 for x in lat if x <= 1000) / len(lat) * 100
        print(f"within 1s:   {within1s:.1f}%   "
              f"SLO(p99<=1s): {'PASS' if _pct(lat,99) <= 1000 and not bad else 'FAIL'}")
    else:
        print("no successful requests to measure")

    # ---- response-field breakdown ---------------------------------------- #
    fields: dict[str, list] = {}
    for r in ok:
        resp = r.get("response")
        if isinstance(resp, dict):
            for k, v in _flatten(resp).items():
                fields.setdefault(k, []).append(v)

    if fields:
        print("\n==================== RESPONSE FIELDS ====================")
        for key in sorted(fields):
            vals = fields[key]
            nums = [v for v in vals if isinstance(v, (int, float))
                    and not isinstance(v, bool)]
            if nums and len(nums) == len(vals):
                print(f"[num] {key:42s} "
                      f"min={min(nums):.4g} p50={_pct(nums,50):.4g} "
                      f"p95={_pct(nums,95):.4g} max={max(nums):.4g} "
                      f"mean={statistics.mean(nums):.4g}")
            else:
                top = Counter(str(v) for v in vals).most_common(5)
                shown = ", ".join(f"{k}={n}" for k, n in top)
                extra = "" if len(set(map(str, vals))) <= 5 else " ..."
                print(f"[cat] {key:42s} {shown}{extra}")

        # Highlight the money fields if present.
        print("\n-------------------- KEY BID METRICS --------------------")
        for key in ("recommended_bid",
                    "recommended_bid_predicted_win_rate",
                    "recommended_bid_predicted_profit",
                    "recommended_bid_predicted_cm",
                    "decision_path",
                    "decision_reason",
                    "model_version"):
            match = next((k for k in fields if k == key or k.endswith("." + key)), None)
            if not match:
                continue
            vals = fields[match]
            nums = [v for v in vals if isinstance(v, (int, float))
                    and not isinstance(v, bool)]
            if nums and len(nums) == len(vals):
                print(f"{key:38s} p50={_pct(nums,50):.4g}  "
                      f"p95={_pct(nums,95):.4g}  mean={statistics.mean(nums):.4g}")
            else:
                top = Counter(str(v) for v in vals).most_common(5)
                print(f"{key:38s} " + ", ".join(f"{k}({n})" for k, n in top))
    else:
        print("\n(no JSON response bodies captured to analyze)")


def main() -> None:
    ap = argparse.ArgumentParser(description="Bid API load test + response capture + analysis")
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run the load test and save responses")
    r.add_argument("--url", required=True, help="base URL, e.g. http://localhost:8000")
    r.add_argument("--path", default="/recommend_bid", help="endpoint path")
    r.add_argument("--rpm", type=int, default=1000, help="requests per minute")
    r.add_argument("--duration", type=int, default=60, help="seconds")
    r.add_argument("--workers", type=int, default=128, help="max concurrent in-flight")
    r.add_argument("--timeout", type=float, default=10.0, help="per-request timeout (s)")
    r.add_argument("--warmup", type=int, default=0, help="warm-up requests (not saved)")
    r.add_argument("--vary", action="store_true",
                   help="randomize each request so responses span real scenarios")
    r.add_argument("--out", default="results.jsonl", help="output JSONL file")
    r.set_defaults(func=cmd_run)

    a = sub.add_parser("analyze", help="analyze a saved results file")
    a.add_argument("file", help="results JSONL file")
    a.set_defaults(func=cmd_analyze)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

