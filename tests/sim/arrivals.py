"""
Piece 3 — Arrival schedule: WHEN each request fires (bursty, not even).

Two building blocks (see phase1_replayer for the standalone proof):
  * scatter -- a Poisson process. Conditioned on N events in [0,60s], the events
    are UNIFORM in the window, so scattering = uniform random timestamps.
  * burst   -- with prob `burst_prob`, dump a fraction of the minute's requests
    at one random instant (gaussian-jittered) = "50% at a single point".

Total per minute ~ Uniform(rate_min, rate_max). Set burst_prob=0 for pure scatter.
"""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass
class BurstConfig:
    rate_min: int = 100
    rate_max: int = 150
    burst_prob: float = 0.5
    burst_frac_min: float = 0.3
    burst_frac_max: float = 0.6
    burst_jitter_s: float = 0.08
    max_bursts: int = 1


def build_minute_offsets(
    rng: random.Random, cfg: BurstConfig
) -> tuple[list[float], dict]:
    """Return (sorted offsets in [0,60), composition-info) for one minute."""
    n = rng.randint(cfg.rate_min, cfg.rate_max)
    offsets: list[float] = []
    remaining = n
    bursts: list[dict] = []
    nb = 0
    while nb < cfg.max_bursts and remaining > 0 and rng.random() < cfg.burst_prob:
        frac = rng.uniform(cfg.burst_frac_min, cfg.burst_frac_max)
        burst_n = min(remaining, max(1, round(frac * n)))
        t0 = rng.uniform(0.0, 60.0)
        for _ in range(burst_n):
            t = t0 + rng.gauss(0.0, cfg.burst_jitter_s)
            offsets.append(min(59.999, max(0.0, t)))
        bursts.append({"size": burst_n, "pct": 100.0 * burst_n / n, "at_s": t0})
        remaining -= burst_n
        nb += 1
    offsets.extend(rng.uniform(0.0, 60.0) for _ in range(remaining))
    offsets.sort()
    return offsets, {"n": n, "bursts": bursts, "scattered": remaining}


def build_schedule(
    rng: random.Random, minutes: int, cfg: BurstConfig
) -> tuple[list[float], list[dict]]:
    """Absolute arrival offsets (seconds from start) + per-minute composition."""
    schedule: list[float] = []
    infos: list[dict] = []
    for m in range(minutes):
        offs, info = build_minute_offsets(rng, cfg)
        info["minute"] = m
        infos.append(info)
        schedule.extend(m * 60.0 + o for o in offs)
    schedule.sort()
    return schedule, infos
