#!/usr/bin/env python3
"""
SmartHub bid-API replayer — PHASE 1 (arrival-distribution harness, dry-run).

Goal of this phase
------------------
Pull historical lead rows and "make requests" at a realistic, *non-uniform*
rate of 100–150 per minute — WITHOUT hitting the real API yet. We only prove
the arrival distribution is right (bursty, not evenly spaced) and that the
open-loop scheduler can actually fire clustered requests concurrently.

Two ideas do all the work
-------------------------
1. ARRIVAL PROCESS (when each request fires):
   * "scatter" — a Poisson process. A Poisson process conditioned on N events
     in [0, 60s] places those N events UNIFORMLY at random in the window, so
     scattering = drawing uniform timestamps. Natural bunching, no even spacing.
   * "burst"   — with probability `burst_prob`, a fraction `burst_frac` of the
     minute's requests are dumped at ONE random instant (jittered by a few tens
     of ms), i.e. your "50% at a single point of time" case.
   Burstiness is fully tunable; set burst_prob=0 for pure Poisson scatter.

2. OPEN-LOOP DISPATCH (how they fire):
   Timestamps are scheduled up front; each request is launched with
   asyncio.create_task() and NOT awaited. That is what lets a 50-request burst
   actually run concurrently. (A closed-loop send->await->send loop can never
   produce a real spike, and would hide server slowdowns behind the client.)

Phase 2 will replace `send_stub` with a real httpx POST to /recommend_bid and
`load_rows` with the real lead_pings pull + full BidRequest builder. Those
seams are marked with TODO(phase2).

Run
---
    python phase1_replayer.py --minutes 5 --seed 42
    python phase1_replayer.py --minutes 10 --burst-prob 0.6 --burst-frac-min 0.4 \
        --burst-frac-max 0.6 --data snapshot.parquet
"""
from __future__ import annotations

import argparse
import asyncio
import random
import time
from collections import Counter
from dataclasses import dataclass, field


# --------------------------------------------------------------------------- #
# 1. DATA SOURCE  (TODO(phase2): swap synthetic for the real lead_pings pull)  #
# --------------------------------------------------------------------------- #
def load_rows(path: str | None, n_needed: int, rng: random.Random) -> list[dict]:
    """Return a pool of lead rows to replay.

    Phase 1: if `path` is given, read a parquet/CSV snapshot; otherwise generate
    synthetic rows so the distribution can be exercised with no DB access.
    Phase 2: this becomes the field-registry-driven lead_pings extract.
    """
    if path:
        import pandas as pd  # lazy: only needed when a real snapshot is used

        df = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)
        rows = df.to_dict("records")
        if not rows:
            raise SystemExit(f"No rows in {path}")
        return rows

    # Synthetic fallback — enough variety to look like real payloads.
    states = ["CA", "TX", "FL", "NY", "OH", "GA", "PA", "IL"]
    tiers = ["tier_1", "tier_2", "tier_3"]
    rows = []
    for i in range(max(n_needed, 500)):
        lead_type = rng.choice([6, 6, 6, 1])  # mostly auto (6), some home (1)
        rows.append(
            {
                "id": 900_000_000 + i,             # -> lead_ping_id
                "lead_type_id": lead_type,
                "campaign_id": rng.choice([40088, 40122, 51007, 60231]),
                "source_type_id": rng.choice([574, 588, 601, 620]),
                "traffic_tier": rng.choice(tiers),
                "state": rng.choice(states),
                "exp_rev": round(rng.uniform(20, 400), 2),  # -> expected_revenue
                "age": rng.choice([None, 24, 31, 38, 45, 52, 67]),
                "insured": rng.choice([None, "true", "false"]),
            }
        )
    return rows


def build_payload(row: dict) -> dict:
    """Light Phase-1 mapping lead_pings row -> BidRequest-shaped dict.

    TODO(phase2): replace with the full field_registry-driven builder
    (per-lead-type required subset, created_at->Pacific, point-in-time filter).
    """
    return {
        "expected_revenue": row.get("exp_rev"),
        "lead_type_id": row.get("lead_type_id"),
        "campaign_id": row.get("campaign_id"),
        "source_type_id": row.get("source_type_id"),
        "traffic_tier": row.get("traffic_tier"),
        "state": row.get("state"),
        "lead_ping_id": row.get("id"),
        # attributes sent only when present
        **({"age": row["age"]} if row.get("age") is not None else {}),
        **({"insured": row["insured"]} if row.get("insured") is not None else {}),
    }


# --------------------------------------------------------------------------- #
# 2. ARRIVAL SCHEDULE  (the part you asked about)                              #
# --------------------------------------------------------------------------- #
@dataclass
class BurstConfig:
    rate_min: int = 100          # requests per minute, lower bound
    rate_max: int = 150          # requests per minute, upper bound
    burst_prob: float = 0.5      # chance a given minute contains a burst
    burst_frac_min: float = 0.3  # burst = this..that fraction of the minute's load
    burst_frac_max: float = 0.6
    burst_jitter_s: float = 0.08 # spread of a burst around its instant (std, seconds)
    max_bursts: int = 1          # bursts per minute (set >1 for multi-spike minutes)


def build_minute_offsets(rng: random.Random, cfg: BurstConfig) -> tuple[list[float], dict]:
    """Return (sorted arrival offsets in [0,60), info) for ONE minute.

    total N ~ Uniform(rate_min, rate_max).
    With prob burst_prob: carve out burst(s) of frac*N at random instants
    (gaussian-jittered), then scatter the remainder uniformly (= Poisson).
    `info` records the composition so --verify can prove what happened.
    """
    n = rng.randint(cfg.rate_min, cfg.rate_max)
    offsets: list[float] = []
    remaining = n
    bursts: list[dict] = []

    n_bursts = 0
    while n_bursts < cfg.max_bursts and remaining > 0 and rng.random() < cfg.burst_prob:
        frac = rng.uniform(cfg.burst_frac_min, cfg.burst_frac_max)
        burst_n = min(remaining, max(1, round(frac * n)))
        t0 = rng.uniform(0.0, 60.0)                     # the "single point of time"
        for _ in range(burst_n):
            t = t0 + rng.gauss(0.0, cfg.burst_jitter_s)  # near-simultaneous
            offsets.append(min(59.999, max(0.0, t)))
        bursts.append({"size": burst_n, "pct": 100.0 * burst_n / n, "at_s": t0})
        remaining -= burst_n
        n_bursts += 1

    # Scatter the rest uniformly == a Poisson process conditioned on `remaining`.
    offsets.extend(rng.uniform(0.0, 60.0) for _ in range(remaining))
    offsets.sort()
    info = {"n": n, "bursts": bursts, "scattered": remaining}
    return offsets, info


def build_schedule(rng: random.Random, minutes: int,
                   cfg: BurstConfig) -> tuple[list[float], list[dict]]:
    """Absolute arrival offsets (seconds from start) + per-minute composition."""
    schedule: list[float] = []
    infos: list[dict] = []
    for m in range(minutes):
        base = m * 60.0
        offs, info = build_minute_offsets(rng, cfg)
        info["minute"] = m
        infos.append(info)
        schedule.extend(base + off for off in offs)
    schedule.sort()
    return schedule, infos


def print_verify(infos: list[dict]) -> None:
    """Prove the intended composition per minute: burst %-of-minute + scatter."""
    print("\n----------------  VERIFY: per-minute composition  ----------------")
    print("(what the scheduler INTENDED before firing — cross-check vs histogram)\n")
    burst_minutes = 0
    burst_pcts: list[float] = []
    for info in infos:
        m, n, sc = info["minute"], info["n"], info["scattered"]
        if info["bursts"]:
            burst_minutes += 1
            parts = ", ".join(
                f"{b['size']} reqs ({b['pct']:.0f}% of minute) @ t={b['at_s']:.1f}s"
                for b in info["bursts"]
            )
            burst_pcts.extend(b["pct"] for b in info["bursts"])
            print(f"  min {m:>2} | N={n:>3} | BURST → {parts}  + {sc} scattered")
        else:
            print(f"  min {m:>2} | N={n:>3} | no burst — all {sc} scattered "
                  f"(pure Poisson)")
    avg = (sum(burst_pcts) / len(burst_pcts)) if burst_pcts else 0.0
    print(f"\n  summary: {burst_minutes}/{len(infos)} minutes had a burst; "
          f"avg burst share {avg:.0f}% of that minute.")
    print("------------------------------------------------------------------")


# --------------------------------------------------------------------------- #
# 3. OPEN-LOOP DISPATCH                                                        #
# --------------------------------------------------------------------------- #
@dataclass
class Stats:
    sent: int = 0
    per_second: Counter = field(default_factory=Counter)   # arrivals per wall-second
    drift_ms: list[float] = field(default_factory=list)    # scheduled vs actual dispatch
    inflight: int = 0
    max_inflight: int = 0


async def send_stub(stats: Stats, speed: float = 1.0, sim_latency_range=(0.01, 0.05)):
    """Phase-1 stub: pretend to call the API. TODO(phase2): real httpx POST.

    Latency is scaled by `speed` so that under demo time-compression the
    observed in-flight concurrency still reflects real overlap.
    """
    stats.inflight += 1
    stats.max_inflight = max(stats.max_inflight, stats.inflight)
    try:
        await asyncio.sleep(random.uniform(*sim_latency_range) / speed)
    finally:
        stats.inflight -= 1


async def run(schedule: list[float], rows: list[dict], stats: Stats,
              sender=send_stub, speed: float = 1.0) -> None:
    """Fire each request at scheduled_offset/speed. Reporting buckets use the
    real (un-compressed) scheduled second, so the histogram is meaningful at
    any --speed."""
    loop = asyncio.get_running_loop()
    start = loop.time()
    tasks: list[asyncio.Task] = []
    last_min = -1
    for i, offset in enumerate(schedule):
        cur_min = int(offset // 60)            # progress heartbeat each scheduled minute
        if cur_min != last_min:
            print(f"  ...running: minute {cur_min + 1} — {i} requests fired so far",
                  flush=True)
            last_min = cur_min
        target = offset / speed
        wait = target - (loop.time() - start)
        if wait > 0:
            await asyncio.sleep(wait)          # sleep until this arrival's instant
        actual = loop.time() - start
        stats.drift_ms.append((actual - target) * 1000.0)
        stats.per_second[int(offset)] += 1     # bucket by real scheduled second
        stats.sent += 1
        row = rows[i % len(rows)]
        _payload = build_payload(row)          # built now (used for real by phase 2)
        tasks.append(asyncio.create_task(sender(stats, speed)))  # fire, don't await
    await asyncio.gather(*tasks)               # let in-flight requests finish


# --------------------------------------------------------------------------- #
# 4. REPORT — show that the distribution is bursty, not even                   #
# --------------------------------------------------------------------------- #
def report(counts_by_sec: dict, minutes: int, stats: Stats | None = None) -> None:
    """Render the arrival distribution. Works from the planned schedule alone
    (instant); if `stats` is given (a --dispatch run), also show measured
    scheduler drift and peak concurrency."""
    seconds = minutes * 60
    counts = [counts_by_sec.get(s, 0) for s in range(seconds)]
    total = sum(counts)
    mx = max(counts) if counts else 0
    nonzero = [c for c in counts if c]
    label = "MEASURED" if stats is not None else "PLANNED"
    print(f"\n================  ARRIVAL DISTRIBUTION ({label})  ================")
    print(f"total requests : {total}  over {minutes} min "
          f"({total / minutes:.1f}/min avg)")
    print(f"per-second     : max {mx}  |  mean(active s) "
          f"{(sum(nonzero)/len(nonzero) if nonzero else 0):.1f}  |  idle seconds "
          f"{seconds - len(nonzero)}/{seconds}")
    if stats is not None and stats.drift_ms:
        drift = sorted(stats.drift_ms)
        p99 = drift[min(len(drift) - 1, int(0.99 * len(drift)))]
        print(f"schedule drift : p50 {drift[len(drift)//2]:.1f}ms  "
              f"p99 {p99:.1f}ms   (how late the scheduler fired vs plan)")
        print(f"max concurrent : {stats.max_inflight}  in-flight at once "
              f"(the spikes)")
    else:
        print("               : add --dispatch to actually fire the schedule and "
              "measure drift + peak concurrency")

    # compact per-second sparkline histogram, 60s per row
    blocks = " ▁▂▃▄▅▆▇█"
    print("\nper-second arrivals (each char = 1s; height ∝ count, '#' = peak second):")
    for m in range(minutes):
        row = counts[m * 60:(m + 1) * 60]
        line = "".join(
            " " if c == 0 else "#" if c == mx and mx > 0
            else blocks[max(1, min(8, round(8 * c / mx)))] if mx else " "
            for c in row
        )
        print(f"  min {m:>2} |{line}| {sum(row):>3}")
    print("=======================================================\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase-1 bursty arrival replayer (dry-run)")
    ap.add_argument("--data", default=None, help="parquet/CSV snapshot; omit for synthetic")
    ap.add_argument("--minutes", type=int, default=5)
    ap.add_argument("--rate-min", type=int, default=100)
    ap.add_argument("--rate-max", type=int, default=150)
    ap.add_argument("--burst-prob", type=float, default=0.5)
    ap.add_argument("--burst-frac-min", type=float, default=0.3)
    ap.add_argument("--burst-frac-max", type=float, default=0.6)
    ap.add_argument("--burst-jitter", type=float, default=0.08, help="burst spread std (s)")
    ap.add_argument("--max-bursts", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dispatch", action="store_true",
                    help="actually FIRE the schedule (real-time, open-loop). "
                         "Default: off — just analyse the distribution and exit instantly.")
    ap.add_argument("--speed", type=float, default=1.0,
                    help="with --dispatch: time compression (e.g. 30 = 30x faster)")
    ap.add_argument("--verify", action="store_true",
                    help="print the intended per-minute burst/scatter composition")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    cfg = BurstConfig(
        rate_min=args.rate_min, rate_max=args.rate_max,
        burst_prob=args.burst_prob, burst_frac_min=args.burst_frac_min,
        burst_frac_max=args.burst_frac_max, burst_jitter_s=args.burst_jitter,
        max_bursts=args.max_bursts,
    )

    schedule, infos = build_schedule(rng, args.minutes, cfg)  # real offsets
    rows = load_rows(args.data, n_needed=len(schedule), rng=rng)

    print(f"Scheduled {len(schedule)} requests over {args.minutes} min "
          f"(seed={args.seed}, burst_prob={cfg.burst_prob}, DRY-RUN: no real HTTP).")
    if args.verify:
        print_verify(infos)

    counts = Counter(int(o) for o in schedule)   # planned per-second distribution

    if not args.dispatch:
        # Default: analyse the schedule and exit immediately — no real-time wait.
        report(counts, args.minutes, stats=None)
        return

    # --dispatch: actually fire the schedule open-loop (real-time or --speed).
    est_wall = (args.minutes * 60.0) / args.speed
    print(f"\nDispatching in REAL TIME: ~{est_wall:.0f}s wall-clock "
          f"({est_wall/60:.1f} min at speed={args.speed}x). Ctrl-C to stop.")
    stats = Stats()
    t0 = time.time()
    asyncio.run(run(schedule, rows, stats, speed=args.speed))
    print(f"Wall time: {time.time() - t0:.1f}s")
    report(counts, args.minutes, stats=stats)


if __name__ == "__main__":
    main()
