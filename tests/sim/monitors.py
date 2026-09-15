#!/usr/bin/env python3
"""
Piece 6 — Monitors: server-side health while the replay runs.

Samples the machine and (optionally) the Docker containers + Postgres every
`--interval` seconds and appends a CSV row, so week-long creep (memory leak,
climbing DB connections, disk filling) is visible that a short run would miss.
Pure stdlib + `docker`/`psql` via subprocess — no Python deps.

    python monitors.py --interval 30 --out data/sim/health.csv \
        --serve-container prefect-serve --pg-container prefect-postgres \
        --disk-path /app/data
"""
from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
import time


def _host_cpu_idle_total():
    with open("/proc/stat") as f:
        p = f.readline().split()[1:]
    v = list(map(int, p))
    idle = v[3] + (v[4] if len(v) > 4 else 0)
    return idle, sum(v)


def _host_mem_mb():
    d = {}
    with open("/proc/meminfo") as f:
        for line in f:
            k, _, rest = line.partition(":")
            d[k] = int(rest.split()[0]) if rest.split() else 0
    total = d.get("MemTotal", 0) / 1024
    avail = d.get("MemAvailable", 0) / 1024
    return round(total - avail, 1), round(total, 1)


def _docker_stats(container: str):
    """(cpu_pct, mem_mb) for a container, or (None, None) if unavailable."""
    try:
        out = subprocess.run(
            [
                "docker",
                "stats",
                "--no-stream",
                "--format",
                "{{.CPUPerc}};{{.MemUsage}}",
                container,
            ],
            capture_output=True,
            text=True,
            timeout=15,
        ).stdout.strip()
        if not out:
            return None, None
        cpu, mem = out.split(";")
        cpu_v = float(cpu.strip().rstrip("%"))
        mem_v = mem.split("/")[0].strip()  # e.g. "1.2GiB"
        num = float("".join(c for c in mem_v if (c.isdigit() or c == ".")))
        mult = 1024 if "GiB" in mem_v else (1.0 / 1024 if "KiB" in mem_v else 1.0)
        return cpu_v, round(num * mult, 1)
    except Exception:
        return None, None


def _pg_scalar(pg_container: str, db: str, user: str, sql: str):
    try:
        out = subprocess.run(
            ["docker", "exec", pg_container, "psql", "-U", user, "-d", db, "-tAc", sql],
            capture_output=True,
            text=True,
            timeout=15,
        ).stdout.strip()
        return int(out) if out.lstrip("-").isdigit() else out or None
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description="Sample host/container/DB health to CSV.")
    ap.add_argument("--interval", type=float, default=30)
    ap.add_argument(
        "--duration", type=float, default=0, help="seconds; 0 = until Ctrl-C"
    )
    ap.add_argument("--out", default="data/sim/health.csv")
    ap.add_argument("--serve-container", default=None)
    ap.add_argument("--pg-container", default=None)
    ap.add_argument("--pg-db", default="prefect")
    ap.add_argument("--pg-user", default="prefect")
    ap.add_argument("--disk-path", default="/app/data")
    args = ap.parse_args()

    fields = [
        "ts",
        "host_cpu_pct",
        "host_mem_used_mb",
        "host_mem_total_mb",
        "disk_used_pct",
        "serve_cpu_pct",
        "serve_mem_mb",
        "pg_conns",
        "pred_log_rows",
    ]
    fh = open(args.out, "a", newline="", buffering=1)
    w = csv.DictWriter(fh, fieldnames=fields)
    if fh.tell() == 0:
        w.writeheader()

    idle0, tot0 = _host_cpu_idle_total()
    t_start = time.time()
    print(
        f"monitoring every {args.interval}s -> {args.out} (Ctrl-C to stop)", flush=True
    )
    try:
        while True:
            time.sleep(args.interval)
            idle1, tot1 = _host_cpu_idle_total()
            dtot = (tot1 - tot0) or 1
            cpu = round(100.0 * (1 - (idle1 - idle0) / dtot), 1)
            idle0, tot0 = idle1, tot1
            mem_used, mem_total = _host_mem_mb()
            try:
                du = shutil.disk_usage(args.disk_path)
                disk_pct = round(100.0 * du.used / du.total, 1)
            except Exception:
                disk_pct = None
            s_cpu, s_mem = (
                _docker_stats(args.serve_container)
                if args.serve_container
                else (None, None)
            )
            pg_conns = pred_rows = None
            if args.pg_container:
                pg_conns = _pg_scalar(
                    args.pg_container,
                    args.pg_db,
                    args.pg_user,
                    "SELECT count(*) FROM pg_stat_activity;",
                )
                pred_rows = _pg_scalar(
                    args.pg_container,
                    args.pg_db,
                    args.pg_user,
                    "SELECT count(*) FROM smarthub_prediction_log;",
                )
            row = {
                "ts": round(time.time() - t_start, 1),
                "host_cpu_pct": cpu,
                "host_mem_used_mb": mem_used,
                "host_mem_total_mb": mem_total,
                "disk_used_pct": disk_pct,
                "serve_cpu_pct": s_cpu,
                "serve_mem_mb": s_mem,
                "pg_conns": pg_conns,
                "pred_log_rows": pred_rows,
            }
            w.writerow(row)
            print(
                f"  t={row['ts']:>6.0f}s cpu={cpu:>5}% mem={mem_used:.0f}/"
                f"{mem_total:.0f}MB disk={disk_pct}% serve_mem={s_mem} "
                f"pg_conns={pg_conns} pred_rows={pred_rows}",
                flush=True,
            )
            if args.duration and (time.time() - t_start) >= args.duration:
                break
    except KeyboardInterrupt:
        pass
    finally:
        fh.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
