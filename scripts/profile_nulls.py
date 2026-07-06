"""Null / blank profiling for the pulled leads and the training features.

Reports, per column, the % that is missing (NULL or blank string) for:
  1. all raw pulled columns, and
  2. only the training features (leakage-safe set, per lead type),
plus row-level completeness of the training features.

Usage:
    python scripts/profile_nulls.py                 # data/leads, auto (6)
    python scripts/profile_nulls.py --lead-type-id 1
    python scripts/profile_nulls.py --parquet-dir data/leads
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import pandas as pd

# Allow running as `python3 scripts/profile_nulls.py` without installing the pkg.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from smarthub.data import features  # noqa: E402


def missing_mask(series: pd.Series) -> pd.Series:
    """True where the value is NULL or (for text) blank/whitespace."""
    mask = series.isna()
    if series.dtype == object or str(series.dtype).startswith("string"):
        mask = mask | series.astype("string").str.strip().eq("")
    return mask


def report(df: pd.DataFrame, cols: list[str], title: str) -> None:
    n = len(df)
    print(f"\n=== {title} — {n} rows ===")
    stats = []
    for col in cols:
        if col in df.columns:
            miss = int(missing_mask(df[col]).sum())
            stats.append((col, miss, round(100 * miss / n, 1) if n else 0.0))
    for col, miss, pct in sorted(stats, key=lambda r: -r[1]):
        print(f"  {col:30} {miss:>7} ({pct}%)")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Null/blank profiler.")
    parser.add_argument("--parquet-dir", default="data/leads")
    parser.add_argument("--lead-type-id", type=int, default=6)
    args = parser.parse_args(argv)

    paths = sorted(glob.glob(f"{args.parquet_dir}/**/*.parquet", recursive=True))
    if not paths:
        print(f"No parquet files under {args.parquet_dir}")
        return 1
    raw = pd.concat((pd.read_parquet(p) for p in paths), ignore_index=True)
    print(f"Loaded {len(raw)} rows, {raw.shape[1]} columns from {len(paths)} files")

    report(raw, list(raw.columns), "ALL COLUMNS (raw pulled)")

    train = features.build_training_table(raw, lead_type_id=args.lead_type_id)
    feats = [c for c in train.columns if c not in ("id", "created_at", "won_flag")]
    report(train, feats, f"TRAINING FEATURES (lead_type_id={args.lead_type_id})")

    if len(train):
        mm = pd.DataFrame({c: missing_mask(train[c]) for c in feats})
        per_row = mm.sum(axis=1)
        print(f"\n  rows with NO missing feature: {int((per_row == 0).sum())} "
              f"({round(100 * (per_row == 0).mean(), 1)}%)")
        print(f"  avg missing features / row  : {round(per_row.mean(), 2)} "
              f"of {len(feats)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
