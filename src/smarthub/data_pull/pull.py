"""Pull lead_pings from Redshift through an SSH tunnel and write a parquet file.

This is STEP 1 of the pipeline (data-pull); build-features (STEP 2) reads the
data this produces.

Usage:
    smarthub-pull \\
        --min-created-at "2026-06-07 00:00:00" \\
        --max-created-at "2026-06-20 00:00:00"
    # or:  python -m smarthub.data_pull.pull --min-created-at ... --max-...

Configuration (SSH + Redshift credentials) comes from the environment / .env.
See .env.example and CONTEXT.md.
"""

from __future__ import annotations

import sys

import pandas as pd
import redshift_connector
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sshtunnel import SSHTunnelForwarder

from smarthub.core.config import (
    ConfigError,
    PullSettings,
    RedshiftSettings,
    StorageSettings,
)
from smarthub.core.logging_utils import configure_logging, get_logger
from smarthub.core import notifications
from smarthub.core import storage
from smarthub.data_pull.cli import build_pull_parser
from smarthub.data_pull.models import (
    coerce_leads_dtypes,
    leads_select,
    leads_with_expected_revenue_select,
)

logger = get_logger(__name__)


def _build_engine(rs: RedshiftSettings, local_port: int) -> Engine:
    """Create a SQLAlchemy engine that connects through the open SSH tunnel.

    A ``creator`` is used so the actual socket goes through ``localhost`` on the
    tunnel's local port while SQLAlchemy still uses the Redshift dialect to
    compile queries.
    """

    def _connect():
        return redshift_connector.connect(
            host="localhost",
            port=local_port,
            database=rs.database,
            user=rs.user,
            password=rs.password,
            timeout=rs.connect_timeout,
        )

    return create_engine("redshift+redshift_connector://", creator=_connect)


def fetch_leads(
    settings: PullSettings,
    min_created_at: str,
    max_created_at: str,
    with_expected_revenue: bool = True,
    selected_only: bool = True,
    lead_type_id: int | None = None,
) -> pd.DataFrame:
    """Open the tunnel, run the ORM query, and return the result as a dataframe.

    The SSH tunnel and the SQLAlchemy engine are both disposed deterministically
    even if the query raises. ``lead_type_id`` restricts to one lead type.
    """
    ssh, rs = settings.ssh, settings.redshift
    if with_expected_revenue:
        stmt = leads_with_expected_revenue_select(
            min_created_at,
            max_created_at,
            selected_only=selected_only,
            lead_type_id=lead_type_id,
        )
    else:
        stmt = leads_select(min_created_at, max_created_at, lead_type_id=lead_type_id)

    with SSHTunnelForwarder(
        (ssh.host, ssh.port),
        ssh_username=ssh.user,
        ssh_pkey=str(ssh.private_key_path),
        ssh_private_key_password=ssh.private_key_password,
        remote_bind_address=(rs.host, rs.port),
    ) as tunnel:
        logger.info("SSH tunnel established on localhost:%s", tunnel.local_bind_port)

        engine = _build_engine(rs, tunnel.local_bind_port)
        try:
            with engine.connect() as conn:
                leads_df = pd.read_sql(stmt, conn)
        finally:
            engine.dispose()

    leads_df = coerce_leads_dtypes(leads_df)
    logger.info("Fetched leads frame with shape %s", leads_df.shape)
    return leads_df


def run(
    min_created_at: str,
    max_created_at: str,
    with_expected_revenue: bool = True,
    selected_only: bool = True,
    lead_type_id: int | None = None,
) -> pd.DataFrame:
    """End-to-end pull: load config, fetch, persist. Returns the frame.

    Upserts (keyed on ``id``) into whichever backend(s) ``STORAGE_BACKEND``
    enables, so overlapping re-pulled windows update late-resolving outcomes in
    place rather than duplicating them.
    """
    leads_df = fetch_leads(
        PullSettings.from_env(),
        min_created_at,
        max_created_at,
        with_expected_revenue=with_expected_revenue,
        selected_only=selected_only,
        lead_type_id=lead_type_id,
    )
    results = storage.save_pull(leads_df, StorageSettings.from_env())
    logger.info("Persisted pull: %s", results)
    return leads_df


def main(argv: list[str] | None = None) -> int:
    args = build_pull_parser().parse_args(argv)
    configure_logging(args.log_level)
    try:
        run(
            args.min_created_at,
            args.max_created_at,
            with_expected_revenue=not args.no_expected_revenue,
            selected_only=not args.all_listings,
            lead_type_id=args.lead_type_id,
        )
    except ConfigError as exc:
        logger.error("Configuration error: %s", exc)
        _notify_cli_failure(args, exc)
        return 2
    except Exception as exc:  # noqa: BLE001 - top-level guard for a CLI entry point
        logger.exception("Data pull failed")
        _notify_cli_failure(args, exc)
        return 1
    return 0


def _notify_cli_failure(args, exc: Exception) -> None:
    """Best-effort Slack alert for a failed manual (CLI) pull."""
    notifications.notify_failure(
        "data-pull (manual/CLI)",
        {
            "Lead type": args.lead_type_id if args.lead_type_id is not None else "all",
            "Data window (created_at)": (
                f"`{args.min_created_at}` → `{args.max_created_at}`"
            ),
        },
        error=f"{type(exc).__name__}: {exc}",
    )


if __name__ == "__main__":
    sys.exit(main())
