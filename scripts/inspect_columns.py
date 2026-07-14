"""Print column names (and types) from saved Parquet files — schema only, no rows.

Usage:
    python scripts/inspect_columns.py                 # defaults to data/leads
    python scripts/inspect_columns.py data/leads/2026/06/20-06-2026.parquet
    python scripts/inspect_columns.py data/leads --check   # verify all files match
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pyarrow.parquet as pq


def find_parquet(path: Path) -> list[Path]:
    """Return the Parquet files at a path (the file, or all under a dir).

    Inputs
    ------
    path : Path
        A Parquet file or a directory to search recursively.

    Returns
    -------
    list[Path]
        Matching Parquet file paths, sorted.
    """
    if path.is_file():
        return [path]
    return sorted(path.glob("**/*.parquet"))


def main(argv=None) -> int:
    """Print the columns of saved Parquet files; optionally check consistency.

    Inputs
    ------
    argv : list[str], optional
        Command-line arguments; defaults to ``sys.argv``.

    Returns
    -------
    int
        Process exit code (0 on success, 1 if no files found).
    """
    parser = argparse.ArgumentParser(
        description="List columns in saved Parquet files (schema only)."
    )
    parser.add_argument(
        "path",
        nargs="?",
        default="data/training/auto",
        help="Parquet file or directory (default: data/leads).",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify every file under the directory shares the same columns.",
    )
    args = parser.parse_args(argv)

    files = find_parquet(Path(args.path))
    if not files:
        print(f"No parquet files found under {args.path}", file=sys.stderr)
        return 1

    schema = pq.ParquetFile(files[0]).schema_arrow
    print(f"Source : {files[-1]}")
    print(f"Files  : {len(files)} found")
    print(f"Columns: {len(schema.names)}\n")
    for field in schema:
        print(f"  {field.name:<30} {field.type}")

    if args.check and len(files) > 1:
        base = set(schema.names)
        mismatches = [
            (f, sorted(base ^ set(pq.ParquetFile(f).schema_arrow.names)))
            for f in files[1:]
            if set(pq.ParquetFile(f).schema_arrow.names) != base
        ]
        if mismatches:
            print("\n[!] Column mismatches found:")
            for f, diff in mismatches:
                print(f"  {f}: differs by {diff}")
        else:
            print(f"\nAll {len(files)} files share the same {len(base)} columns.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
