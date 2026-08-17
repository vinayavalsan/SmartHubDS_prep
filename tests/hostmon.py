#!/usr/bin/env python3
"""Whole-machine resource monitor -- run it during a load/soak test."""
from __future__ import annotations
import argparse, csv, os, time


def read_cpu():
    out = {}
    with open("/proc/stat") as f:
        for line in f:
            if not line.startswith("cpu"):
                break
            p = line.split()
            v = list(map(int, p[1:]))
            idle = v[3] + (v[4] if len(v) > 4 else 0)
            out[p[0]] = (sum(v), idle)
    return out


def read_mem():
    d = {}
    with open("/proc/meminfo") as f:
        for line in f:
            k, _, rest = line.partition(":")
            d[k] = int(rest.strip().split()[0])
    total = d["MemTotal"]
    avail = d.get("MemAvailable", d.get("MemFree", 0))
    return (total - avail) / 1024.0, total / 1024.0


def pct(prev, cur):
    dt = cur[0] - prev[0]; di = cur[1] - prev[1]
    return 0.0 if dt <= 0 else max(0.0, min(100.0, (dt - di) / dt * 100.0))


def main() -> None:
    ap = argparse.ArgumentParser(description="Whole-machine CPU/RAM monitor")
    ap.add_argument("--interval", type=float, default=2.0)
    ap.add_argument("--duration", type=float, default=0)
    ap.add_argument("--out", default="")
    ap.add_argument("--busy-threshold", type=float, default=80.0)
    args = ap.parse_args()

    ncores = os.cpu_count() or 1
    core_names = [f"cpu{i}" for i in range(ncores)]
    print(f">> host monitor: {ncores} cores, interval={args.interval}s"
          + (f", duration={args.duration}s" if args.duration else ", Ctrl-C to stop"), flush=True)
    print(f">> time      cpu_total%  (of {ncores*100})  cores_busy  ram_used_mb  ram%", flush=True)

    writer = None; fh = None
    if args.out:
        fh = open(args.out, "w", newline="")
        writer = csv.writer(fh)
        writer.writerow(["time", "cpu_total_pct", "cores_of_work", "cores_busy",
                         "ram_used_mb", "ram_pct"] + core_names)

    peak_cpu = (-1.0, "", 0); peak_ram = (-1.0, "", 0.0)
    sum_cpu = 0.0; n = 0
    core_peak = {c: 0.0 for c in core_names}; core_sum = {c: 0.0 for c in core_names}

    prev = read_cpu(); started = time.time()
    try:
        while True:
            time.sleep(args.interval)
            cur = read_cpu()
            per_core = [pct(prev[c], cur[c]) for c in core_names if c in prev and c in cur]
            prev = cur
            cpu_total = sum(per_core); cores_work = cpu_total / 100.0
            cores_busy = sum(1 for x in per_core if x >= args.busy_threshold)
            ram_used, ram_total = read_mem(); ram_pct = ram_used / ram_total * 100.0
            ts = time.strftime("%H:%M:%S")
            n += 1; sum_cpu += cpu_total
            for c, x in zip(core_names, per_core):
                core_peak[c] = max(core_peak[c], x); core_sum[c] += x
            if cpu_total > peak_cpu[0]: peak_cpu = (cpu_total, ts, cores_busy)
            if ram_used > peak_ram[0]: peak_ram = (ram_used, ts, ram_pct)
            print(f">> {ts}   {cpu_total:6.0f}   ({cores_work:4.1f} cores)   "
                  f"{cores_busy:2d}/{ncores}      {ram_used:7.0f}    {ram_pct:4.0f}%", flush=True)
            if writer:
                writer.writerow([ts, f"{cpu_total:.0f}", f"{cores_work:.2f}", cores_busy,
                                 f"{ram_used:.0f}", f"{ram_pct:.0f}"] + [f"{x:.0f}" for x in per_core])
                fh.flush()
            if args.duration and (time.time() - started) >= args.duration:
                break
    except KeyboardInterrupt:
        print("\n>> stopped", flush=True)
    finally:
        if fh: fh.close()

    if n == 0: return
    print("\n==================== HOST SUMMARY ====================")
    print(f"cores:            {ncores}")
    print(f"samples:          {n}  over {(time.time()-started)/60:.1f} min")
    print(f"CPU avg:          {sum_cpu/n:.0f}% of {ncores*100}%  (~{sum_cpu/n/100:.1f} cores' worth, sustained)")
    print(f"CPU peak:         {peak_cpu[0]:.0f}% of {ncores*100}%  (~{peak_cpu[0]/100:.1f} cores' worth) at {peak_cpu[1]}  [{peak_cpu[2]}/{ncores} cores >= {args.busy_threshold:.0f}%]")
    print(f"RAM peak:         {peak_ram[0]:.0f} MB  ({peak_ram[2]:.0f}%) at {peak_ram[1]}")
    print("\nper-core utilisation (avg / peak):")
    for c in core_names:
        print(f"  {c:6s}  avg {core_sum[c]/n:5.0f}%   peak {core_peak[c]:5.0f}%")


if __name__ == "__main__":
    main()
