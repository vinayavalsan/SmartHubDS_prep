"""Tests for the optional DB SSH tunnel URL rewriting (core.db_tunnel).

Only the URL-resolution logic is exercised — the SSH forwarder itself is
monkeypatched, so no real network/SSH is needed.
"""

from __future__ import annotations

import pytest

from smarthub.core import db_tunnel


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    """Ensure each test starts with the tunnel disabled and no cached bind."""
    monkeypatch.delenv(db_tunnel.TUNNEL_ENV, raising=False)
    db_tunnel._local_bind = None
    db_tunnel._forwarder = None
    yield
    db_tunnel._local_bind = None
    db_tunnel._forwarder = None


def test_disabled_is_noop():
    url = "postgresql+psycopg2://prefect:prefect@postgres:5432/prefect"
    assert db_tunnel.db_tunnel_enabled() is False
    assert db_tunnel.resolve_db_url(url) == url


def test_enabled_env_parsing(monkeypatch):
    for truthy in ("1", "true", "YES", "on"):
        monkeypatch.setenv(db_tunnel.TUNNEL_ENV, truthy)
        assert db_tunnel.db_tunnel_enabled() is True
    monkeypatch.setenv(db_tunnel.TUNNEL_ENV, "false")
    assert db_tunnel.db_tunnel_enabled() is False


def test_enabled_rewrites_host_and_port(monkeypatch):
    # Pretend the tunnel is up on a local port; don't open a real SSH session.
    monkeypatch.setenv(db_tunnel.TUNNEL_ENV, "true")
    monkeypatch.setattr(db_tunnel, "_ensure_tunnel", lambda: ("127.0.0.1", 15432))

    out = db_tunnel.resolve_db_url(
        "postgresql+psycopg2://prefect:secret@postgres:5432/prefect"
    )
    # Host/port swapped for the tunnel; credentials + db name preserved.
    assert "@127.0.0.1:15432/prefect" in out
    assert out.startswith("postgresql+psycopg2://prefect:secret@")
    assert "postgres:5432" not in out


def test_missing_ssh_config_raises_when_enabled(monkeypatch):
    monkeypatch.setenv(db_tunnel.TUNNEL_ENV, "true")
    for var in (
        "SMARTHUB_DB_SSH_HOST",
        "SSH_HOST",
        "SMARTHUB_DB_SSH_KEY",
        "SSH_PRIVATE_KEY_PATH",
    ):
        monkeypatch.delenv(var, raising=False)
    with pytest.raises(RuntimeError):
        db_tunnel.resolve_db_url("postgresql+psycopg2://u:p@postgres:5432/db")
