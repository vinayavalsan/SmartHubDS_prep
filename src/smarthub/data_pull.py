"""Pull lead_pings from Redshift through an SSH tunnel and write a parquet file.

Usage:
    python -m smarthub.data_pull \\
        --min-created-at "2026-06-07 00:00:00" \\
        --max-created-at "2026-06-20 00:00:00"

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

from . import io
from .cli import build_pull_parser
from .config import ConfigError, PullSettings, RedshiftSettings
from .logging_utils import configure_logging, get_logger
from .models import leads_select

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
    settings: PullSettings, min_created_at: str, max_created_at: str
) -> pd.DataFrame:
    """Open the tunnel, run the ORM query, and return the result as a dataframe.

    The SSH tunnel and the SQLAlchemy engine are both disposed deterministically
    even if the query raises.
    """
    ssh, rs = settings.ssh, settings.redshift
    stmt = leads_select(min_created_at, max_created_at)

    with SSHTunnelForwarder(
        (ssh.host, ssh.port),
        ssh_username=ssh.user,
        ssh_pkey=str(ssh.private_key_path),
        remote_bind_address=(rs.host, rs.port),
    ) as tunnel:
        logger.info("SSH tunnel established on localhost:%s", tunnel.local_bind_port)

        engine = _build_engine(rs, tunnel.local_bind_port)
        try:
            with engine.connect() as conn:
                leads_df = pd.read_sql(stmt, conn)
        finally:
            engine.dispose()

    logger.info("Fetched leads frame with shape %s", leads_df.shape)
    return leads_df


def run(
    min_created_at: str,
    max_created_at: str,
    output: str | None = None,
) -> pd.DataFrame:
    """End-to-end pull: load config, fetch, persist. Returns the dataframe."""
    settings = PullSettings.from_env()
    leads_df = fetch_leads(settings, min_created_at, max_created_at)
    written = io.save_leads(leads_df, output)
    logger.info("Wrote %s rows to %s", len(leads_df), written)
    return leads_df


def main(argv: list[str] | None = None) -> int:
    args = build_pull_parser().parse_args(argv)
    configure_logging(args.log_level)
    try:
        run(args.min_created_at, args.max_created_at, args.output)
    except ConfigError as exc:
        logger.error("Configuration error: %s", exc)
        return 2
    except Exception:  # noqa: BLE001 - top-level guard for a CLI entry point
        logger.exception("Data pull failed")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
