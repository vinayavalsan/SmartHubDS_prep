#!/usr/bin/env python3
"""
Extract a historical ``lead_pings`` snapshot to a single Parquet file.

One-time bulk pull that feeds the replay test's historical days (Days 1-4).
It pulls N months of ``lead_pings`` from Redshift in TIME CHUNKS -- so the whole
window never loads into memory at once -- and STREAMS them into one Parquet file
(one row-group per chunk, constant memory).

It reuses the project's own pull code, so it connects and types data EXACTLY like
``smarthub-pull``:
  * ``PullSettings.from_env()``    -- SSH + Redshift creds from env / .env
  * ``fetch_leads(...)``           -- SSH tunnel (or direct-VPC) + Redshift query
  * ``coerce_leads_dtypes(...)``   -- stable ORM dtypes + ``expected_revenue``

Run on the EC2 host (same environment as the pipeline)::

    # last 4 months, weekly chunks, auto (6) + home (1)
    python tests/sim/extract_snapshot.py --months 4 --out data/sim/snapshot.parquet

    # explicit window instead of --months
    python tests/sim/extract_snapshot.py \
        --min-created-at "2026-05-01 00:00:00" \
        --max-created-at "2026-09-01 00:00:00" \
        --chunk-days 7 --lead-type-ids 6 1 --out data/sim/snapshot.parquet

    # just print the chunk plan, pull nothing
    python tests/sim/extract_snapshot.py --months 4 --plan-only

Notes
-----
* One SSH tunnel is opened and closed per chunk (``fetch_leads`` manages it),
  so a bigger ``--chunk-days`` means fewer tunnels but more memory per chunk;
  7 days is a good balance on a 32 GB box.
* ``lead_pings.id`` is the primary key, so every row is a DISTINCT
  ``lead_ping_id`` -- the row count is the number of unique leads you can replay
  without repeating an id.
* Writes to ``<out>.partial`` and atomically renames on success; also writes
  ``<out>.manifest.json`` (row count, window, columns) next to it.
* Output goes under ``data/`` by default, which is git-ignored -- the snapshot
  is data, not code.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from smarthub.core.config import PullSettings
from smarthub.core.logging_utils import configure_logging, get_logger
from smarthub.data_pull.models import coerce_leads_dtypes
from smarthub.data_pull.pull import fetch_leads

logger = get_logger(__name__)

_DT = "%Y-%m-%d %H:%M:%S"


def iter_chunks(min_ts: pd.Timestamp, max_ts: pd.Timestamp, chunk_days: int):
    """Yield (chunk_min, chunk_max) windows of `chunk_days` covering [min, max)."""
    step = pd.Timedelta(days=chunk_days)
    cur = min_ts
    while cur < max_ts:
        nxt = min(cur + step, max_ts)
        yield cur, nxt
        cur = nxt


def extract(
    min_ts: pd.Timestamp,
    max_ts: pd.Timestamp,
    out: str,
    chunk_days: int,
    lead_type_ids: list[int] | None,
    with_expected_revenue: bool = True,
) -> int:
    """Pull [min, max) in chunks and stream into a single Parquet file at `out`.

    Returns the total row count written.
    """
    settings = PullSettings.from_env()
    out_dir = os.path.dirname(out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    tmp = out + ".partial"

    writer: pq.ParquetWriter | None = None
    schema: pa.Schema | None = None
    columns: list[str] | None = None
    total = 0
    chunk_no = 0

    try:
        for c_min, c_max in iter_chunks(min_ts, max_ts, chunk_days):
            chunk_no += 1
            s_min, s_max = c_min.strftime(_DT), c_max.strftime(_DT)
            logger.info("chunk %d: %s -> %s ...", chunk_no, s_min, s_max)

            raw = fetch_leads(
                settings,
                s_min,
                s_max,
                with_expected_revenue=with_expected_revenue,
                lead_type_ids=lead_type_ids,
            )
            df = coerce_leads_dtypes(raw)
            if df.empty:
                logger.info("  chunk %d: 0 rows", chunk_no)
                continue

            # Freeze the column set/order on the first non-empty chunk so every
            # row-group shares one schema (coerce_leads_dtypes already makes the
            # dtypes stable regardless of all-null columns).
            if columns is None:
                columns = list(df.columns)
            df = df.reindex(columns=columns)

            table = pa.Table.from_pandas(df, preserve_index=False)
            if writer is None:
                schema = table.schema
                writer = pq.ParquetWriter(tmp, schema, compression="snappy")
            else:
                try:
                    table = table.cast(schema)
                except (pa.ArrowInvalid, pa.ArrowTypeError) as exc:
                    raise SystemExit(
                        f"Schema mismatch on chunk {chunk_no} ({s_min}..{s_max}): "
                        f"{exc}. Re-run just this window to inspect."
                    ) from exc

            writer.write_table(table)
            total += len(df)
            logger.info(
                "  chunk %d: +%s rows (cumulative %s)",
                chunk_no,
                f"{len(df):,}",
                f"{total:,}",
            )
            print(f"  ...chunk {chunk_no} done: {total:,} rows so far", flush=True)
    finally:
        if writer is not None:
            writer.close()

    if writer is None:
        # Nothing written -- remove any empty temp file, fail loudly.
        if os.path.exists(tmp):
            os.remove(tmp)
        raise SystemExit("No rows pulled for the given window; nothing written.")

    os.replace(tmp, out)  # atomic: only a complete file appears at `out`

    manifest = {
        "out": os.path.abspath(out),
        "rows": total,
        "min_created_at": min_ts.strftime(_DT),
        "max_created_at": max_ts.strftime(_DT),
        "chunk_days": chunk_days,
        "lead_type_ids": lead_type_ids if lead_type_ids is not None else "all",
        "columns": columns,
        "created_at": datetime.now().strftime(_DT),
        "note": "lead_pings.id is the PK; rows == distinct lead_ping_id count.",
    }
    with open(out + ".manifest.json", "w") as fh:
        json.dump(manifest, fh, indent=2, default=str)

    size_mb = os.path.getsize(out) / 1e6
    logger.info(
        "DONE: %s rows over %s chunk(s) -> %s (%.1f MB). Manifest: %s.manifest.json",
        f"{total:,}",
        chunk_no,
        out,
        size_mb,
        out,
    )
    print(
        f"\nDONE: {total:,} rows -> {out} ({size_mb:.1f} MB)\n"
        f"      each row is a distinct lead_ping_id; manifest at {out}.manifest.json",
        flush=True,
    )
    return total


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Bulk lead_pings -> single Parquet snapshot (chunked, low-memory)."
    )
    ap.add_argument(
        "--out",
        default="data/sim/snapshot.parquet",
        help="output Parquet path (default under git-ignored data/)",
    )
    ap.add_argument(
        "--months",
        type=int,
        default=4,
        help="pull the last N months ending at --max-created-at/now "
        "(ignored when both --min/--max are given)",
    )
    ap.add_argument("--min-created-at", default=None, help="YYYY-MM-DD HH:MM:SS")
    ap.add_argument(
        "--max-created-at",
        default=None,
        help="YYYY-MM-DD HH:MM:SS (default: today 00:00)",
    )
    ap.add_argument(
        "--chunk-days",
        type=int,
        default=7,
        help="pull window size per chunk (default 7)",
    )
    ap.add_argument(
        "--lead-type-ids",
        type=int,
        nargs="*",
        default=[6, 1],
        help="lead types to pull (default: 6=auto 1=home)",
    )
    ap.add_argument(
        "--all-lead-types",
        action="store_true",
        help="pull every lead type (overrides --lead-type-ids)",
    )
    ap.add_argument(
        "--plan-only",
        action="store_true",
        help="print the chunk windows and exit without pulling",
    )
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args(argv)
    configure_logging(args.log_level)

    max_ts = (
        pd.Timestamp(args.max_created_at)
        if args.max_created_at
        else pd.Timestamp.now().normalize()
    )
    min_ts = (
        pd.Timestamp(args.min_created_at)
        if args.min_created_at
        else max_ts - pd.DateOffset(months=args.months)
    )
    if min_ts >= max_ts:
        raise SystemExit(f"min ({min_ts}) must be before max ({max_ts}).")

    lead_type_ids = None if args.all_lead_types else (args.lead_type_ids or None)

    windows = list(iter_chunks(min_ts, max_ts, args.chunk_days))
    print(
        f"Window : {min_ts.strftime(_DT)} -> {max_ts.strftime(_DT)}  "
        f"({(max_ts - min_ts).days} days, {len(windows)} chunks of "
        f"{args.chunk_days}d)",
        flush=True,
    )
    print(
        f"Types  : {lead_type_ids if lead_type_ids is not None else 'all'}   "
        f"Out: {args.out}",
        flush=True,
    )

    if args.plan_only:
        for i, (a, b) in enumerate(windows, 1):
            print(f"  chunk {i:>2}: {a.strftime(_DT)} -> {b.strftime(_DT)}")
        print("(plan-only: nothing pulled)")
        return 0

    extract(min_ts, max_ts, args.out, args.chunk_days, lead_type_ids)
    return 0


if __name__ == "__main__":
    sys.exit(main())
