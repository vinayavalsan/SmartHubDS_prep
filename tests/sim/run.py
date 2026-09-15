#!/usr/bin/env python3
"""
Orchestrator — ties pieces 2 (payload builder) + 3 (arrivals) + 4 (sender).

Reads the historical snapshot, builds a bursty arrival schedule, turns real
leads into real `/recommend_bid` requests, and fires them open-loop at `--url`
(each request submitted to a thread pool at its scheduled instant, so bursts
run concurrently), writing a capture ledger (one JSON line per request).

Uses only pandas + requests — runs in the worker container with no extra installs.

    # preview only (instant, sends nothing): payload sample + schedule stats
    python run.py --data snapshot.parquet --minutes 6 --burst-prob 0.6

    # actually fire at the stub (or later, staging):
    python stub_server.py --port 8080 &
    python run.py --data snapshot.parquet --url http://127.0.0.1:8080 \
        --minutes 6 --burst-prob 0.6 --dispatch --out data/sim/ledger.jsonl

Analyse the ledger afterwards with analyze.py.
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from collections import Counter

import pandas as pd

import arrivals
import payload_builder as pb
from sender import Ledger, Sender, Stats


def _load_payloads(data: str, n_needed: int, rng: random.Random,
                   lead_type_ids: list[int] | None) -> tuple[list[dict], int]:
    """Sample rows from the snapshot and build valid request bodies.

    Returns (payloads, skipped) where skipped = rows missing a required field.
    Samples ~15% extra to cover skips, without replacement when possible.
    """
    import pyarrow.parquet as pq
    avail = set(pq.ParquetFile(data).schema.names)
    cols = [c for c in pb.required_columns() if c in avail]
    df = pd.read_parquet(data, columns=cols)
    if lead_type_ids:
        df = df[df["lead_type_id"].isin(lead_type_ids)]
    if df.empty:
        raise SystemExit(f"No rows in {data} for lead_type_ids={lead_type_ids}")

    want = int(n_needed * 1.15) + 10
    replace = want > len(df)
    sample = df.sample(n=want, replace=replace, random_state=rng.randint(0, 2**31))
    payloads: list[dict] = []
    skipped = 0
    for row in sample.to_dict("records"):
        try:
            payloads.append(pb.build_payload(row))
        except pb.PayloadError:
            skipped += 1
        if len(payloads) >= n_needed:
            break
    if not payloads:
        raise SystemExit("Every sampled row was missing a required field.")
    return payloads, skipped


def dispatch(schedule: list[float], payloads: list[dict], sender: Sender,
             minutes: int, speed: float) -> None:
    """Open-loop: sleep until each arrival, then submit (don't wait)."""
    start = time.perf_counter()
    last_min = -1
    for i, offset in enumerate(schedule):
        cur_min = int(offset // 60)
        if cur_min != last_min:
            print(f"  ...minute {cur_min + 1}/{minutes} — {i} submitted", flush=True)
            last_min = cur_min
        wait = offset / speed - (time.perf_counter() - start)
        if wait > 0:
            time.sleep(wait)
        sender.submit(payloads[i % len(payloads)], offset)
    sender.close()  # waits for all in-flight requests to finish


def _print_summary(stats: Stats, minutes: int, skipped: int, wall: float) -> None:
    lat = sorted(stats.latencies_ms)

    def pct(p):
        return lat[min(len(lat) - 1, int(p / 100 * len(lat)))] if lat else 0.0

    print("\n================  RUN SUMMARY  ================")
    print(f"sent           : {stats.sent}  (ok {stats.ok}, "
          f"skipped-invalid {skipped}, wall {wall:.1f}s)")
    print(f"status codes   : {dict(stats.by_status)}")
    if lat:
        print(f"latency ms     : p50 {pct(50):.1f}  p95 {pct(95):.1f}  "
              f"p99 {pct(99):.1f}  max {max(lat):.1f}")
        print(f"within 1s      : {sum(1 for x in lat if x < 1000)}/{len(lat)}")
    print(f"decision_path  : {dict(stats.decision_path)}   null_bids {stats.null_bids}")
    print(f"max concurrent : {stats.max_inflight} in-flight")
    if stats.errors:
        print(f"errors         : {dict(stats.errors)}")
    print("===============================================\n")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Replay real leads at /recommend_bid.")
    ap.add_argument("--data", required=True, help="snapshot .parquet")
    ap.add_argument("--url", default="http://127.0.0.1:8080",
                    help="API base URL (stub or staging) — /recommend_bid is appended")
    ap.add_argument("--out", default="data/sim/ledger.jsonl", help="capture ledger (JSONL)")
    ap.add_argument("--minutes", type=int, default=6)
    ap.add_argument("--rate-min", type=int, default=100)
    ap.add_argument("--rate-max", type=int, default=150)
    ap.add_argument("--burst-prob", type=float, default=0.5)
    ap.add_argument("--burst-frac-min", type=float, default=0.3)
    ap.add_argument("--burst-frac-max", type=float, default=0.6)
    ap.add_argument("--burst-jitter", type=float, default=0.08)
    ap.add_argument("--max-bursts", type=int, default=1)
    ap.add_argument("--lead-type-ids", type=int, nargs="*", default=None,
                    help="restrict replay to these lead types (default: all in file)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dispatch", action="store_true",
                    help="actually FIRE requests. Default: preview only, sends nothing.")
    ap.add_argument("--speed", type=float, default=1.0, help="with --dispatch: compress time")
    ap.add_argument("--api-key", default=None, help="bearer key (when auth is on)")
    ap.add_argument("--timeout", type=float, default=5.0)
    ap.add_argument("--workers", type=int, default=512, help="max concurrent requests")
    args = ap.parse_args(argv)

    rng = random.Random(args.seed)
    cfg = arrivals.BurstConfig(
        rate_min=args.rate_min, rate_max=args.rate_max, burst_prob=args.burst_prob,
        burst_frac_min=args.burst_frac_min, burst_frac_max=args.burst_frac_max,
        burst_jitter_s=args.burst_jitter, max_bursts=args.max_bursts)
    schedule, _infos = arrivals.build_schedule(rng, args.minutes, cfg)
    payloads, skipped = _load_payloads(args.data, len(schedule), rng, args.lead_type_ids)

    print(f"Snapshot: {args.data} | schedule: {len(schedule)} requests over "
          f"{args.minutes} min | valid payloads: {len(payloads)} "
          f"(skipped-invalid {skipped})")
    print("Sample payload:", payloads[0])
    counts = Counter(int(o) for o in schedule)
    print(f"Arrival shape: peak {max(counts.values())}/s, "
          f"{args.minutes*60 - len(counts)} idle seconds (bursty={args.burst_prob>0})")

    if not args.dispatch:
        print("\nPreview only — nothing sent. Add --dispatch to fire at --url.")
        return 0

    ledger = Ledger(args.out)
    stats = Stats()
    url = args.url.rstrip("/") + "/recommend_bid"
    sender = Sender(url, ledger, stats, api_key=args.api_key,
                    timeout=args.timeout, workers=args.workers)
    est = (args.minutes * 60.0) / args.speed
    print(f"\nDispatching to {url}  (~{est:.0f}s wall at speed={args.speed}x, "
          f"Ctrl-C to stop)", flush=True)
    t0 = time.time()
    try:
        dispatch(schedule, payloads, sender, args.minutes, args.speed)
    finally:
        ledger.close()
    _print_summary(stats, args.minutes, skipped, time.time() - t0)
    print(f"ledger written: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
