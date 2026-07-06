"""Typed, validated configuration loaded from the environment.

Replaces ad-hoc ``os.getenv`` calls scattered through the scripts. Required
values are validated up front so a misconfigured environment fails fast with a
clear message instead of a deep ``TypeError``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Load a local .env if present. Safe to call repeatedly.
load_dotenv()


class ConfigError(RuntimeError):
    """Raised when required configuration is missing or invalid."""


def _require(name: str) -> str:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        raise ConfigError(
            f"Required environment variable '{name}' is not set. "
            f"Copy .env.example to .env and fill it in."
        )
    return value.strip()


@dataclass(frozen=True)
class SSHSettings:
    host: str
    user: str
    private_key_path: Path
    port: int = 22
    private_key_password: str | None = None

    @classmethod
    def from_env(cls) -> "SSHSettings":
        key_path = Path(os.path.expanduser(_require("SSH_PRIVATE_KEY_PATH")))
        if not key_path.exists():
            raise ConfigError(f"SSH private key not found at: {key_path}")
        passphrase = os.getenv("SSH_PRIVATE_KEY_PASSWORD") or None
        return cls(
            host=_require("SSH_HOST"),
            user=_require("SSH_USER"),
            private_key_path=key_path,
            port=int(os.getenv("SSH_PORT", "22")),
            private_key_password=passphrase,
        )


@dataclass(frozen=True)
class RedshiftSettings:
    host: str
    database: str
    user: str
    password: str
    port: int = 5439
    connect_timeout: int = 10

    @classmethod
    def from_env(cls) -> "RedshiftSettings":
        return cls(
            host=_require("REDSHIFT_HOST"),
            database=_require("REDSHIFT_DB"),
            user=_require("REDSHIFT_USER"),
            password=_require("REDSHIFT_PASSWORD"),
            port=int(os.getenv("REDSHIFT_PORT", "5439")),
            connect_timeout=int(os.getenv("REDSHIFT_CONNECT_TIMEOUT", "10")),
        )


@dataclass(frozen=True)
class PullSettings:
    """Settings for a single data-pull run."""

    ssh: SSHSettings
    redshift: RedshiftSettings

    @classmethod
    def from_env(cls) -> "PullSettings":
        return cls(
            ssh=SSHSettings.from_env(),
            redshift=RedshiftSettings.from_env(),
        )


_VALID_BACKENDS = {"duckdb", "parquet", "both"}


@dataclass(frozen=True)
class StorageSettings:
    """Where and how pulled data is persisted (env-controlled).

    ``backend`` selects DuckDB, partitioned Parquet, or both.
    """

    backend: str
    duckdb_path: Path
    parquet_dir: Path
    partition_date_col: str

    @classmethod
    def from_env(cls) -> "StorageSettings":
        backend = os.getenv("STORAGE_BACKEND", "both").strip().lower()
        if backend not in _VALID_BACKENDS:
            raise ConfigError(
                f"STORAGE_BACKEND must be one of {sorted(_VALID_BACKENDS)}, "
                f"got '{backend}'."
            )
        return cls(
            backend=backend,
            duckdb_path=Path(os.getenv("DUCKDB_PATH", "data/smarthub.duckdb")),
            parquet_dir=Path(os.getenv("PARQUET_DIR", "data/leads")),
            partition_date_col=os.getenv("PARTITION_DATE_COL", "created_at"),
        )

    @property
    def use_duckdb(self) -> bool:
        return self.backend in ("duckdb", "both")

    @property
    def use_parquet(self) -> bool:
        return self.backend in ("parquet", "both")


DEFAULT_TRAINING_WINDOW_DAYS = 21


def training_window_days() -> int:
    """Rolling training window in days (env ``TRAINING_WINDOW_DAYS``, default 21).

    The feature build trains on this many recent days, since the market is
    non-stationary (CONTEXT.md §7). Set ``0`` to use all accumulated data.
    """
    raw = os.getenv("TRAINING_WINDOW_DAYS")
    return int(raw) if raw not in (None, "") else DEFAULT_TRAINING_WINDOW_DAYS
