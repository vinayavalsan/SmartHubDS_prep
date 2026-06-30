"""Command-line argument parsing for the package scripts."""

from __future__ import annotations

import argparse
from datetime import datetime

_DT_FORMAT = "%Y-%m-%d %H:%M:%S"


def _valid_datetime(value: str) -> str:
    """Validate that a CLI date string matches the expected format."""
    try:
        datetime.strptime(value, _DT_FORMAT)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"'{value}' is not in 'YYYY-MM-DD HH:MM:SS' format"
        ) from exc
    return value


def build_pull_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="smarthub.data_pull",
        description="Pull lead_pings from Redshift through an SSH tunnel.",
    )
    parser.add_argument(
        "--min-created-at",
        type=_valid_datetime,
        required=True,
        help="Inclusive lower bound for lp.created_at (YYYY-MM-DD HH:MM:SS).",
    )
    parser.add_argument(
        "--max-created-at",
        type=_valid_datetime,
        required=True,
        help="Exclusive upper bound for lp.created_at (YYYY-MM-DD HH:MM:SS).",
    )
    parser.add_argument(
        "--no-expected-revenue",
        action="store_true",
        help="Pull lead_pings only, skipping the lead_ping_listings join.",
    )
    parser.add_argument(
        "--all-listings",
        action="store_true",
        help="Aggregate expected revenue over ALL listings, not just selected.",
    )
    parser.add_argument(
        "--lead-type-id",
        type=int,
        default=None,
        help="Restrict to one lead type (e.g. 6=auto, 1=home). Default: all.",
    )
    parser.add_argument(
        "--log-level",
        default=None,
        help="Logging level (default: env LOG_LEVEL or INFO).",
    )
    return parser
