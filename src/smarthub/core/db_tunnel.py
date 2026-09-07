"""Optional SSH tunnel to the SmartHub Postgres (prediction log + config store).

The stack's Postgres runs on the Docker network of the EC2 host and is **not**
published to the internet, so a local run can't reach it directly. Turn this on
to forward it over SSH: with ``SMARTHUB_DB_SSH_TUNNEL=true`` the SmartHub DB URLs
(``prediction_log_db_url`` / ``config_db_url``) are transparently rewritten to a
``localhost`` port tunneled to the remote Postgres — so nothing that builds a
store needs to change, and a local ``data-pull --include-prediction-logs`` or a
local ``monitoring_app`` just works.

Off by default (production and tests are unaffected). Mirrors the existing
Redshift SSH tunnel (``data_pull/pull.py`` + ``core/config.py``) and reuses the
same ``SSH_*`` vars by default, with ``SMARTHUB_DB_SSH_*`` overrides.

Env
---
SMARTHUB_DB_SSH_TUNNEL       enable (1/true/yes/on); default off
SMARTHUB_DB_SSH_HOST         SSH host (default: SSH_HOST) — e.g. the EC2 IP
SMARTHUB_DB_SSH_PORT         SSH port (default: 22)
SMARTHUB_DB_SSH_USER         SSH user (default: SSH_USER, else "ubuntu")
SMARTHUB_DB_SSH_KEY          private key path (default: SSH_PRIVATE_KEY_PATH)
SMARTHUB_DB_SSH_KEY_PASSWORD key passphrase (default: SSH_PRIVATE_KEY_PASSWORD)
SMARTHUB_DB_REMOTE_HOST      Postgres host as seen from the SSH server
                             (default: 127.0.0.1 — needs Postgres published on
                             the EC2 host loopback; see README)
SMARTHUB_DB_REMOTE_PORT      remote Postgres port (default: 5432)
"""

from __future__ import annotations

import logging
import os
import threading

logger = logging.getLogger(__name__)

TUNNEL_ENV = "SMARTHUB_DB_SSH_TUNNEL"

_lock = threading.Lock()
_forwarder = None
_local_bind: tuple[str, int] | None = None


def db_tunnel_enabled() -> bool:
    """Whether the DB SSH tunnel is enabled (``SMARTHUB_DB_SSH_TUNNEL``)."""
    return os.getenv(TUNNEL_ENV, "").strip().lower() in {"1", "true", "yes", "on"}


def _env(*names: str, default: str | None = None) -> str | None:
    """Return the first non-empty value among ``names`` (env lookup), else default."""
    for name in names:
        value = os.getenv(name)
        if value not in (None, ""):
            return value
    return default


def _ensure_tunnel() -> tuple[str, int]:
    """Open the SSH tunnel once (cached) and return its local ``(host, port)``."""
    global _forwarder, _local_bind
    if _local_bind is not None:
        return _local_bind
    with _lock:
        if _local_bind is not None:
            return _local_bind

        ssh_host = _env("SMARTHUB_DB_SSH_HOST", "SSH_HOST")
        ssh_port = int(_env("SMARTHUB_DB_SSH_PORT", default="22"))
        ssh_user = _env("SMARTHUB_DB_SSH_USER", "SSH_USER", default="ubuntu")
        key_path = _env("SMARTHUB_DB_SSH_KEY", "SSH_PRIVATE_KEY_PATH")
        key_pw = _env("SMARTHUB_DB_SSH_KEY_PASSWORD", "SSH_PRIVATE_KEY_PASSWORD")
        remote_host = _env("SMARTHUB_DB_REMOTE_HOST", default="127.0.0.1")
        remote_port = int(_env("SMARTHUB_DB_REMOTE_PORT", default="5432"))

        # Validate config before importing sshtunnel so the error is clear and
        # independent of whether the (base-dep) sshtunnel package is present.
        if not ssh_host or not key_path:
            raise RuntimeError(
                f"{TUNNEL_ENV} is enabled but SSH host/key are not configured. "
                "Set SMARTHUB_DB_SSH_HOST (or SSH_HOST) and SMARTHUB_DB_SSH_KEY "
                "(or SSH_PRIVATE_KEY_PATH)."
            )

        from sshtunnel import SSHTunnelForwarder

        forwarder = SSHTunnelForwarder(
            (ssh_host, ssh_port),
            ssh_username=ssh_user,
            ssh_pkey=os.path.expanduser(key_path),
            ssh_private_key_password=key_pw,
            remote_bind_address=(remote_host, remote_port),
            local_bind_address=("127.0.0.1", 0),  # 0 -> auto-pick a free port
        )
        forwarder.start()
        _forwarder = forwarder
        _local_bind = ("127.0.0.1", int(forwarder.local_bind_port))
        logger.info(
            "DB SSH tunnel up: 127.0.0.1:%s -> %s:%s (via %s@%s:%s)",
            _local_bind[1],
            remote_host,
            remote_port,
            ssh_user,
            ssh_host,
            ssh_port,
        )
        return _local_bind


def resolve_db_url(url: str) -> str:
    """Return ``url`` unchanged, or rewritten to the tunnel's localhost endpoint.

    A no-op unless ``SMARTHUB_DB_SSH_TUNNEL`` is enabled. When enabled, the
    tunnel is opened on first use and the URL's host/port are swapped for the
    local forwarded endpoint; username, password, and database name are kept.
    """
    if not db_tunnel_enabled():
        return url
    host, port = _ensure_tunnel()
    from sqlalchemy.engine import make_url

    rewritten = make_url(url).set(host=host, port=port)
    return rewritten.render_as_string(hide_password=False)


def close_tunnel() -> None:
    """Close the tunnel if open (best-effort; for tests / clean shutdown)."""
    global _forwarder, _local_bind
    with _lock:
        if _forwarder is not None:
            try:
                _forwarder.stop()
            except Exception:  # noqa: BLE001 - shutdown is best-effort
                logger.warning("Failed to stop DB SSH tunnel cleanly", exc_info=True)
        _forwarder = None
        _local_bind = None
