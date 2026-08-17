#!/usr/bin/env python3
"""Endurance / soak test for the SmartHub bid API.

Drives a steady, moderate request rate for hours and logs a one-row-per-minute
time-series of BOTH client-side latency AND server-side health, so you can spot
slow creep that a 60s test can't:

  * latency  -- p50 / p95 / p99 per window (should stay FLAT)
  * errors   -- per window (should stay 0)
  * serve    -- CPU % and memory (should stay FLAT; rising memory = leak)
  * pred_log -- prediction_log row count (grows linearly -- watch the rate)
  * shap_lag -- rows still awaiting SHAP (should stay bounded, not grow forever)
  * pg_conns -- Postgres connections (should stay flat, not climb)

Run it ON the EC2 host (needs docker access for the server-side samples).

Example -- 3000 req/min for 2 hours, sampling every 60s:
  python3 tests/soak.py --url http://localhost:8000 --rpm 3000 --hours 2 \
      --vary --out soak_metrics.csv

Only client dependency: requests. Server samples use `docker` (best-effort;
pass --no-server-metrics to skip them).
"""

from __future__ import annotations

import argparse
import csv
import random
import statistics
import subprocess
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import requests

# ---- request payload ------------------------------------------------------- #
_STATES = ["TX", "CA", "FL", "NY", "OH", "GA", "PA", "IL"]


def gen_payload(vary: bool) -> dict:
    if not vary:
        base_rev, base_age = 25.00, 34
    return {
        "expected_revenue": round(random.uniform(5, 60), 2) if vary else 25.00,
        "target_cm": round(random.uniform(0.15, 0.40), 3) if vary else 0.25,
        "min_bid": 0.00,
        "bid_step": 0.25,
        "campaign_id": random.choice([12345, 22221, 33310]) if vary else 12345,
        "lead_type_id": 6,
        "created_hour": random.randint(0, 23) if vary else 14,
        "created_dayofweek": random.randint(0, 6) if vary else 2,
        "state": random.choice(_STATES) if vary else "TX",
        "insured": random.choice(["true", "false"]) if vary else "true",
        "home_owner": random.choice(["true", "false"]) if vary else "false",
        "dui": random.choice(["true", "false"]) if vary else "false",
        "num_vehicles": random.randint(1, 4) if vary else 2,
        "num_drivers": random.randint(1, 4) if vary else 2,
        "num_auto_violations": random.randint(0, 3) if vary else 0,
        "num_auto_accidents": random.randint(0, 2) if vary else 0,
        "continuous_coverage_months": random.choice([0, 6, 12, 24, 36]) if vary else 24,
        "military_affiliation": random.choice(["true", "false"]) if vary else "false",
        "gender": random.choice(["Male", "Female"]) if vary else "Female",
        "marital_status": random.choice(["Single", "Married"]) if vary else "Single",
        "age": random.randint(18, 75) if vary else 34,
    }


def _pct(v, p):
    if not v:
        return float("nan")
    v = sorted(v)
    k = max(0, min(len(v) - 1, int(round((p / 100.0) * (len(v) - 1)))))
    return v[k]


# ---- server-side samplers (best-effort; blank on any failure) -------------- #
def _sh(cmd: list[str], timeout: float = 8.0) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True,
                              timeout=timeout).stdout.strip()
    except Exception:  # noqa: BLE001
        return ""


def serve_cpu_mem(container: str):
    out = _sh(["docker", "stats", "--no-stream", "--format",
               "{{.CPUPerc}}|{{.MemUsage}}", container])
    cpu = mem = ""
    if "|" in out:
        c, m = out.split("|", 1)
        cpu = c.replace("%", "").strip()
        mem = m.split("/")[0].strip()  # e.g. "512MiB"
    return cpu, mem


def pg_scalar(pg_container: str, sql: str) -> str:
    return _sh(["docker", "exec", pg_container, "psql", "-U", "prefect",
                "-d", "prefect", "-tAc", sql])


# ---- shared window state --------------------------------------------------- #
lock = threading.Lock()
win_lat: list[float] = []
win_err: Counter = Counter()
tot_ok = 0
tot_err = 0
stop = threading.Event()


def main() -> None:
    ap = argparse.ArgumentParser(description="Bid API soak / endurance test")
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--path", default="/recommend_bid")
    ap.add_argument("--rpm", type=int, default=3000)
    ap.add_argument("--hours", type=float, default=1.0)
    ap.add_argument("--workers", type=int, default=128)
    ap.add_argument("--timeout", type=float, default=10.0)
    ap.add_argument("--sample-secs", type=int, default=60)
    ap.add_argument("--vary", action="store_true")
    ap.add_argument("--out", default="soak_metrics.csv")
    ap.add_argument("--serve-container", default="smarthub-serve")
    ap.add_argument("--pg-container", default="prefect-postgres")
    ap.add_argument("--no-server-metrics", action="store_true")
    args = ap.parse_args()

    endpoint = args.url.rstrip("/") + args.path
    interval = 60.0 / args.rpm
    total_secs = args.hours * 3600

    def fire() -> None:
        global tot_ok, tot_err
        t = time.perf_counter()
        try:
            r = requests.post(endpoint, json=gen_payload(args.vary), timeout=args.timeout)
            ms = (time.perf_counter() - t) * 1000
            with lock:
                if r.status_code == 200:
                    win_lat.append(ms); tot_ok += 1
                else:
                    win_err[f"HTTP {r.status_code}"] += 1; tot_err += 1
        except Exception as exc:  # noqa: BLE001
            with lock:
                win_err[type(exc).__name__] += 1; tot_err += 1

    pool = ThreadPoolExecutor(max_workers=args.workers)

    def dispatcher() -> None:
        next_at = time.perf_counter()
        while not stop.is_set():
            now = time.perf_counter()
            if now >= next_at:
                pool.submit(fire)
                next_at += interval
            else:
                time.sleep(min(interval, next_at - now))

    cols = ["elapsed_min", "ok", "errors", "err_detail", "req_per_min",
            "p50_ms", "p95_ms", "p99_ms", "max_ms",
            "serve_cpu_pct", "serve_mem", "pred_log_rows", "shap_lag", "pg_conns"]
    fh = open(args.out, "w", newline="")
    writer = csv.writer(fh)
    writer.writerow(cols)
    fh.flush()

    print(f">> soak: {endpoint}  rate={args.rpm}/min  duration={args.hours}h  "
          f"sample={args.sample_secs}s  out={args.out}")
    print(">> " + "  ".join(f"{c}" for c in
          ["min", "ok", "err", "req/min", "p50", "p95", "p99", "cpu%", "mem",
           "predlog", "shaplag", "pgconn"]))

    started = time.perf_counter()
    dt = threading.Thread(target=dispatcher, daemon=True)
    dt.start()

    try:
        while (elapsed := time.perf_counter() - started) < total_secs:
            time.sleep(max(0, args.sample_secs - ((time.perf_counter() - started) %
                                                  args.sample_secs)))
            with lock:
                lat = win_lat[:]; errs = dict(win_err)
                win_lat.clear(); win_err.clear()
            e = sum(errs.values())
            rpm = len(lat) / args.sample_secs * 60
            p50 = _pct(lat, 50); p95 = _pct(lat, 95); p99 = _pct(lat, 99)
            mx = max(lat) if lat else float("nan")
            cpu = mem = predlog = shaplag = pgconn = ""
            if not args.no_server_metrics:
                cpu, mem = serve_cpu_mem(args.serve_container)
                predlog = pg_scalar(args.pg_container,
                                    "SELECT count(*) FROM smarthub_prediction_log")
                shaplag = pg_scalar(args.pg_container,
                                    "SELECT count(*) FROM smarthub_prediction_log "
                                    "WHERE shap_explanation IS NULL")
                pgconn = pg_scalar(args.pg_container,
                                   "SELECT count(*) FROM pg_stat_activity")
            mins = (time.perf_counter() - started) / 60
            writer.writerow([f"{mins:.1f}", len(lat), e, errs or "", f"{rpm:.0f}",
                             f"{p50:.0f}", f"{p95:.0f}", f"{p99:.0f}", f"{mx:.0f}",
                             cpu, mem, predlog, shaplag, pgconn])
            fh.flush()
            print(f">> {mins:5.1f}  {len(lat):5d}  {e:3d}  {rpm:6.0f}  "
                  f"{p50:4.0f} {p95:4.0f} {p99:4.0f}  {cpu:>6} {mem:>9}  "
                  f"{predlog:>8} {shaplag:>7} {pgconn:>6}")
    except KeyboardInterrupt:
        print("\n>> interrupted -- stopping")
    finally:
        stop.set()
        pool.shutdown(wait=False, cancel_futures=True)
        fh.close()

    print(f"\n>> done. total ok={tot_ok}  errors={tot_err}  metrics -> {args.out}")
    print(">> check: is p99 flat across rows? did serve_mem creep? did shap_lag "
          "stay bounded? did pg_conns stay flat?")


if __name__ == "__main__":
    main()
