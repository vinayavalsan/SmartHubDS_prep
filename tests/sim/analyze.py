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
import glob
import os
import sys

import pandas as pd


def _pct(s: pd.Series, p: float) -> float:
    return float(s.quantile(p / 100.0)) if len(s) else 0.0


def _to01(s: pd.Series) -> pd.Series:
    """Coerce a won/boolean-ish column to 0/1, handling numeric, bool and common
    string encodings (true/false, t/f, yes/no, won/lost)."""
    out = pd.to_numeric(s, errors="coerce")
    mask = out.isna() & s.notna()
    if mask.any():
        m = {
            "true": 1,
            "false": 0,
            "t": 1,
            "f": 0,
            "1": 1,
            "0": 0,
            "yes": 1,
            "no": 0,
            "won": 1,
            "lost": 0,
            "y": 1,
            "n": 0,
        }
        out.loc[mask] = s[mask].astype(str).str.strip().str.lower().map(m)
    return pd.to_numeric(out, errors="coerce")


def analyze(
    ledger_path: str, snapshot: str | None, report_csv: str | None = None
) -> int:
    files = sorted(glob.glob(ledger_path)) or [ledger_path]
    frames = [
        pd.read_json(f, lines=True)
        for f in files
        if os.path.exists(f) and os.path.getsize(f) > 0
    ]
    if not frames:
        print(f"no ledger data at {ledger_path}")
        return 1
    df = pd.concat(frames, ignore_index=True)
    if len(files) > 1:
        print(f"(analyzing {len(files)} ledger files)")
    if df.empty:
        print("empty ledger")
        return 1
    n = len(df)
    ok = df["ok"].fillna(False)
    lat = df.loc[ok, "latency_ms"].dropna()
    span = max(float(df["ts"].max()), 1e-9)
    thru = n / span * 60.0

    n_500 = int((df["status"] == 500).sum())
    n_503 = int((df["status"] == 503).sum())  # graceful shed above concurrency cap
    n_timeout = int((df["status"] == 0).sum())
    n_422 = int((df["status"] == 422).sum())
    within1s = int((lat < 1000).sum())

    print("\n================  LEDGER ANALYSIS  ================")
    print(
        f"requests       : {n}  (ok {int(ok.sum())}, over {span:.0f}s "
        f"= {thru:.0f}/min)"
    )
    print(f"status codes   : {df['status'].value_counts().sort_index().to_dict()}")
    if len(lat):
        print(
            f"latency ms     : p50 {_pct(lat, 50):.1f}  p90 {_pct(lat, 90):.1f}  "
            f"p95 {_pct(lat, 95):.1f}  p99 {_pct(lat, 99):.1f}  max {lat.max():.1f}"
        )
        print(
            f"within 1s      : {within1s}/{len(lat)} " f"({100*within1s/len(lat):.1f}%)"
        )
    if "decision_path" in df:
        print(
            f"decision_path  : "
            f"{df['decision_path'].value_counts(dropna=True).to_dict()}"
        )
    if n_503 or n_500 or n_timeout:
        print(
            f"shed/errors    : 503 (burst>concurrency) {n_503}  |  "
            f"500 (crash) {n_500}  |  timeouts {n_timeout}"
        )
    if "recommended_bid" in df:
        null_bids = int(df["recommended_bid"].isna().sum() - (n - int(ok.sum())))
        served = df.loc[ok, "recommended_bid"].dropna()
        if len(served):
            print(
                f"recommended_bid: n={len(served)}  min {served.min():.2f}  "
                f"p50 {_pct(served, 50):.2f}  p95 {_pct(served, 95):.2f}  "
                f"max {served.max():.2f}  | null(no-bid) {max(null_bids, 0)}"
            )

    # ---- parity / backtest against historical outcomes ----
    if snapshot and "recommended_bid" not in df.columns:
        print(
            "\n(no successful bids in this ledger yet — skipping parity/report; "
            "check the status codes above)"
        )
    elif snapshot:
        snap = pd.read_parquet(
            snapshot, columns=["id", "bid", "won", "rev", "expected_revenue"]
        )
        snap = snap.rename(
            columns={
                "id": "lead_ping_id",
                "bid": "hist_bid",
                "won": "hist_won",
                "rev": "hist_rev",
            }
        )
        j = df.merge(snap, on="lead_ping_id", how="left")
        served = j[j["ok"] & j["recommended_bid"].notna()].copy()
        if len(served):
            # Per-lead comparison table: model's bid vs the bid actually placed.
            comp = pd.DataFrame(
                {
                    "lead_ping_id": served["lead_ping_id"].astype("Int64"),
                    "lead_type": served["lead_type_id"].map({6: "auto", 1: "home"}),
                    "recommended_bid": served["recommended_bid"].round(2),
                    "actual_bid": pd.to_numeric(
                        served["hist_bid"], errors="coerce"
                    ).round(2),
                    "model_win_prob": pd.to_numeric(
                        served.get("recommended_bid_predicted_win_rate"),
                        errors="coerce",
                    ).round(3),
                    "historical_won": _to01(served["hist_won"]).astype("Int64"),
                    "decision_path": served["decision_path"],
                }
            )
            comp["diff"] = (comp["recommended_bid"] - comp["actual_bid"]).round(2)
            # "won/loss if the bid is chosen by the ML model" = the model's own
            # call at its recommended bid (win_prob >= 0.5). It is an ESTIMATE,
            # not observed truth (we never see the market's response to a new bid).
            comp["model_says"] = comp["model_win_prob"].apply(
                lambda p: (
                    "win"
                    if pd.notna(p) and p >= 0.5
                    else ("loss" if pd.notna(p) else "n/a")
                )
            )

            # Ground-truth-anchored win/loss for the MODEL's bid, decided by the
            # KNOWN historical outcome at the ACTUAL bid (not a model self-estimate):
            #   won at actual  & recommended >= actual -> WIN  (>= a winning bid wins)
            #   lost at actual & recommended <= actual -> LOSS (<= a losing bid loses)
            # anything else is UNCERTAIN: the counterfactual price region we never
            # placed a bid in, so the true outcome there is unobservable.
            def _ground_truth(row):
                won = row["historical_won"]
                rec, act = row["recommended_bid"], row["actual_bid"]
                if pd.isna(won) or pd.isna(rec) or pd.isna(act):
                    return "n/a"
                if won == 1:
                    return "win" if rec >= act else "uncertain"
                return "loss" if rec <= act else "uncertain"

            comp["ground_truth"] = comp.apply(_ground_truth, axis=1)
            comp = comp[
                [
                    "lead_ping_id",
                    "lead_type",
                    "recommended_bid",
                    "actual_bid",
                    "diff",
                    "model_win_prob",
                    "model_says",
                    "historical_won",
                    "ground_truth",
                    "decision_path",
                ]
            ]

            print(
                "\n--- per-lead: model bid vs actual bid (sample of "
                f"{min(12, len(comp))} of {len(comp)}) ---"
            )
            with pd.option_context("display.max_columns", None, "display.width", 200):
                print(comp.head(12).to_string(index=False))

            d = comp["diff"].dropna()
            mw = (comp["model_says"] == "win").sum()
            hw = comp["historical_won"].dropna()
            print("\n--- parity summary (served bids) ---")
            if len(d):
                print(
                    f"recommended vs actual bid: n={len(d)}  "
                    f"mean Δ {d.mean():+.2f}  p50 Δ {_pct(d, 50):+.2f}  "
                    f"(model bids {'higher' if d.mean() > 0 else 'lower'} on avg)"
                )
            print(
                f"model says WIN at its bid : {mw}/{len(comp)} "
                f"({100*mw/len(comp):.0f}%)  [model's own estimate]"
            )
            gt = comp["ground_truth"].value_counts().to_dict()
            print(
                f"ground-truth outcome      : "
                f"win {gt.get('win', 0)}  loss {gt.get('loss', 0)}  "
                f"uncertain {gt.get('uncertain', 0)}  "
                f"[settled by the actual bid's known result]"
            )
            if len(hw):
                print(
                    f"historical win rate       : {hw.mean():.3f} "
                    f"(at the bid actually placed — different bid, for context)"
                )

            if report_csv:
                comp.to_csv(report_csv, index=False)
                print(f"\nfull per-lead report ({len(comp)} rows) -> {report_csv}")

    # ---- verdict ----
    print("\n--- verdict ---")
    v = []
    v.append(
        (
            "no crashes/timeouts (500/conn)",
            n_500 == 0 and n_timeout == 0,
            f"{n_500} x500, {n_timeout} timeouts",
        )
    )
    if n_503:
        v.append(
            (
                "503 shed under burst (capacity note)",
                n_503 == 0,
                f"{n_503} shed — burst exceeded serve concurrency cap "
                f"(raise SERVE_LIMIT_CONCURRENCY/workers, or bursts are "
                f"unrealistically large)",
            )
        )
    v.append(
        (
            "p99 < 1000ms",
            (_pct(lat, 99) < 1000) if len(lat) else False,
            f"p99={_pct(lat, 99):.0f}ms",
        )
    )
    v.append(("no unexpected 422", n_422 == 0, f"{n_422} rejected"))
    for name, passed, detail in v:
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}  ({detail})")
    print("==================================================\n")
    return 0 if all(p for _, p, _ in v) else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Analyze a run.py capture ledger.")
    ap.add_argument("ledger", help="JSONL ledger from run.py (or a glob for many)")
    ap.add_argument(
        "--data", default=None, help="snapshot .parquet for parity/backtest"
    )
    ap.add_argument(
        "--report-csv",
        default=None,
        help="write the full per-lead bid-comparison table to this CSV",
    )
    args = ap.parse_args(argv)
    return analyze(args.ledger, args.data, args.report_csv)


if __name__ == "__main__":
    sys.exit(main())
