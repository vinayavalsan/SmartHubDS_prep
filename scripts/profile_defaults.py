"""Suspected default-value profiler.

Kiran's point: providers often send default values regardless of the real
consumer (e.g. marital_status always "Single", num_vehicles always 1), so a
"populated" field can be meaningless. This surfaces that:

  1. Global: per candidate feature, the most common value + its share + #unique
     (a column dominated by one value is a suspected default / low-signal).
  2. Per source (account_id): the same field can be REAL from one source and a
     hardcoded default from another — flags sources whose value is ~constant.

Runs on the training universe (one lead type + real bidding decisions).

Usage:
    python3 scripts/profile_defaults.py                       # auto (6)
    python3 scripts/profile_defaults.py --lead-type-id 1
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from smarthub.feature_engineering import features  # noqa: E402


def norm(s: pd.Series) -> pd.Series:
    """Strip strings and treat blanks as missing.

    Inputs
    ------
    s : pd.Series
        Series to normalize.

    Returns
    -------
    pd.Series
        The series with text stripped and blanks set to NA.
    """
    if s.dtype == object or str(s.dtype).startswith("string"):
        s = s.astype("string").str.strip()
        s = s.mask(s.eq(""))
    return s


def main(argv=None) -> int:
    """Profile suspected default values across the training universe.

    Inputs
    ------
    argv : list[str], optional
        Command-line arguments; defaults to ``sys.argv``.

    Returns
    -------
    int
        Process exit code (0 on success, 1 if no files found).
    """
    parser = argparse.ArgumentParser(description="Suspected default-value profiler.")
    parser.add_argument("--parquet-dir", default="data/leads")
    parser.add_argument("--lead-type-id", type=int, default=6)
    parser.add_argument("--min-source-rows", type=int, default=200)
    args = parser.parse_args(argv)

    paths = sorted(glob.glob(f"{args.parquet_dir}/**/*.parquet", recursive=True))
    if not paths:
        print(f"No parquet files under {args.parquet_dir}")
        return 1
    raw = pd.concat((pd.read_parquet(p) for p in paths), ignore_index=True)
    won = raw["won"].astype("string").str.strip().str.lower()
    uni = raw[
        (raw["lead_type_id"] == args.lead_type_id) & won.isin(["true", "false"])
    ].copy()
    n = len(uni)
    print(f"Training universe (lead_type_id={args.lead_type_id} + real bids): "
          f"{n} rows\n")

    cand = [c for c in features.PRE_BID_FEATURES if c in uni.columns]
    key = "account_id"  # grouping key: constancy-within-source is degenerate
    thr = 98.0          # %-share within a source to call the field "defaulted"

    drop, mixed, source_linked, use = [], [], [], []
    for col in cand:
        pop_unique = int(norm(uni[col]).nunique())
        pop_missing = int(norm(uni[col]).isna().all())
        if pop_unique <= 1:
            const = norm(uni[col]).dropna()
            val = "ALL MISSING" if pop_missing or const.empty else const.iloc[0]
            drop.append(f"{col} (constant = {val})")
            continue
        # per-source dominant share
        stats = []
        for aid, sub in uni.groupby(key):
            if len(sub) < args.min_source_rows:
                continue
            s = norm(sub[col]).dropna()
            if s.empty:
                continue
            vc = s.value_counts()
            stats.append((int(aid), str(vc.index[0]), 100 * vc.iloc[0] / len(s)))
        defaulters = [(a, v) for a, v, sh in stats if sh >= thr]
        reals = [a for a, v, sh in stats if sh < thr]
        if col == key or (stats and not reals):
            source_linked.append(col)               # varies across, flat within
        elif defaulters:
            dv = defaulters[0][1]
            mixed.append(
                f"{col}: defaulted (~{dv!r}) by {[a for a, _ in defaulters]}; "
                f"real from {reals}"
            )
        else:
            use.append(col)

    print("=== VERDICT (per candidate feature) ===\n")
    print("DROP — no signal (globally constant / all-missing):")
    for x in drop:
        print(f"   - {x}")
    print("\nSOURCE-DEFAULTED — keep value + add a source-reliability flag:")
    for x in mixed:
        print(f"   - {x}")
    print("\nSOURCE-LINKED — constant within a source, varies across "
          "(usable; correlated with source id):")
    print(f"   {', '.join(source_linked) or '(none)'}")
    print("\nUSE — genuine within-source variation:")
    print(f"   {', '.join(use) or '(none)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
