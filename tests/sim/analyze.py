#!/usr/bin/env python3
"""
Piece 8 — Analyzer: turn a capture ledger into a verdict.

Reads the JSONL ledger written by run.py and reports latency, errors,
throughput and the decision-path mix. If the snapshot is given, it also joins
each response back to that lead's historical outcome (bid / won / rev) for a
limited parity + economic backtest.

    python analyze.py data/sim/ledger.jsonl
    python analyze.py data/sim/ledger.jsonl --data data/sim/snapshot.parquet
"""
from __future__ import annotations

import argparse
import sys

import pandas as pd


def _pct(s: pd.Series, p: float) -> float:
    return float(s.quantile(p / 100.0)) if len(s) else 0.0


def analyze(ledger_path: str, snapshot: str | None) -> int:
    df = pd.read_json(ledger_path, lines=True)
    if df.empty:
        print("empty ledger")
        return 1
    n = len(df)
    ok = df["ok"].fillna(False)
    lat = df.loc[ok, "latency_ms"].dropna()
    span = max(float(df["ts"].max()), 1e-9)
    thru = n / span * 60.0

    n_5xx = int(((df["status"] >= 500) | (df["status"] == 0)).sum())
    n_422 = int((df["status"] == 422).sum())
    within1s = int((lat < 1000).sum())

    print("\n================  LEDGER ANALYSIS  ================")
    print(f"requests       : {n}  (ok {int(ok.sum())}, over {span:.0f}s "
          f"= {thru:.0f}/min)")
    print(f"status codes   : {df['status'].value_counts().sort_index().to_dict()}")
    if len(lat):
        print(f"latency ms     : p50 {_pct(lat,50):.1f}  p90 {_pct(lat,90):.1f}  "
              f"p95 {_pct(lat,95):.1f}  p99 {_pct(lat,99):.1f}  max {lat.max():.1f}")
        print(f"within 1s      : {within1s}/{len(lat)} "
              f"({100*within1s/len(lat):.1f}%)")
    if "decision_path" in df:
        print(f"decision_path  : "
              f"{df['decision_path'].value_counts(dropna=True).to_dict()}")
    if "recommended_bid" in df:
        null_bids = int(df["recommended_bid"].isna().sum() - (n - int(ok.sum())))
        served = df.loc[ok, "recommended_bid"].dropna()
        if len(served):
            print(f"recommended_bid: n={len(served)}  min {served.min():.2f}  "
                  f"p50 {_pct(served,50):.2f}  p95 {_pct(served,95):.2f}  "
                  f"max {served.max():.2f}  | null(no-bid) {max(null_bids,0)}")

    # ---- parity / backtest against historical outcomes ----
    if snapshot:
        snap = pd.read_parquet(snapshot, columns=["id", "bid", "won", "rev",
                                                  "expected_revenue"])
        snap = snap.rename(columns={"id": "lead_ping_id", "bid": "hist_bid",
                                    "won": "hist_won", "rev": "hist_rev"})
        j = df.merge(snap, on="lead_ping_id", how="left")
        served = j[j["ok"] & j["recommended_bid"].notna()]
        if len(served):
            print("\n--- parity vs history (served bids only) ---")
            rb, hb = served["recommended_bid"], pd.to_numeric(served["hist_bid"],
                                                              errors="coerce")
            both = served[hb.notna()]
            if len(both):
                d = both["recommended_bid"] - pd.to_numeric(both["hist_bid"])
                print(f"recommended vs historical bid: n={len(both)}  "
                      f"mean Δ {d.mean():+.2f}  p50 Δ {_pct(d,50):+.2f}  "
                      f"(recommended {'higher' if d.mean()>0 else 'lower'} on avg)")
            wr = pd.to_numeric(served["recommended_bid_predicted_win_rate"],
                               errors="coerce").dropna()
            won = pd.to_numeric(served["hist_won"], errors="coerce")
            if len(wr):
                print(f"predicted win-rate: mean {wr.mean():.3f}  "
                      f"| historical win rate {won.mean():.3f} "
                      f"(calibration sanity — not a match, different bids)")

    # ---- verdict ----
    print("\n--- verdict ---")
    v = []
    v.append(("no 5xx/timeouts", n_5xx == 0, f"{n_5xx} errors"))
    v.append(("p99 < 1000ms", (_pct(lat, 99) < 1000) if len(lat) else False,
              f"p99={_pct(lat,99):.0f}ms"))
    v.append(("no unexpected 422", n_422 == 0, f"{n_422} rejected"))
    for name, passed, detail in v:
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}  ({detail})")
    print("==================================================\n")
    return 0 if all(p for _, p, _ in v) else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Analyze a run.py capture ledger.")
    ap.add_argument("ledger", help="JSONL ledger from run.py")
    ap.add_argument("--data", default=None, help="snapshot .parquet for parity/backtest")
    args = ap.parse_args(argv)
    return analyze(args.ledger, args.data)


if __name__ == "__main__":
    sys.exit(main())
