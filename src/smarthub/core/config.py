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
    """Return the trimmed value of env var ``name``; raise if unset/blank."""
    value = os.getenv(name)
    if value is None or value.strip() == "":
        raise ConfigError(
            f"Required environment variable '{name}' is not set. "
            f"Copy .env.example to .env and fill it in."
        )
    return value.strip()


@dataclass(frozen=True)
class SSHSettings:
    """SSH connection settings loaded from the environment."""

    host: str
    user: str
    private_key_path: Path
    port: int = 22
    private_key_password: str | None = None

    @classmethod
    def from_env(cls) -> "SSHSettings":
        """Build SSH settings from ``SSH_*`` environment variables.

        Returns
        -------
        SSHSettings
            Settings populated from the ``SSH_*`` environment variables.

        Raises
        ------
        ConfigError
            If a required variable is unset or the key file is missing.
        """
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
    """Redshift connection settings loaded from the environment."""

    host: str
    database: str
    user: str
    password: str
    port: int = 5439
    connect_timeout: int = 10

    @classmethod
    def from_env(cls) -> "RedshiftSettings":
        """Build Redshift settings from ``REDSHIFT_*`` env variables.

        Returns
        -------
        RedshiftSettings
            Settings populated from the ``REDSHIFT_*`` env variables.

        Raises
        ------
        ConfigError
            If a required variable is unset or blank.
        """
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
        """Build combined SSH + Redshift settings from the environment.

        Returns
        -------
        PullSettings
            The combined SSH and Redshift settings.

        Raises
        ------
        ConfigError
            If any required variable is unset or invalid.
        """
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
        """Build storage settings from the environment.

        Returns
        -------
        StorageSettings
            Settings from ``STORAGE_BACKEND``, ``DUCKDB_PATH``,
            ``PARQUET_DIR`` and ``PARTITION_DATE_COL``.

        Raises
        ------
        ConfigError
            If ``STORAGE_BACKEND`` is not duckdb, parquet or both.
        """
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
        """True when the DuckDB backend is enabled."""
        return self.backend in ("duckdb", "both")

    @property
    def use_parquet(self) -> bool:
        """True when the Parquet backend is enabled."""
        return self.backend in ("parquet", "both")


DEFAULT_TRAINING_WINDOW_DAYS = 21


def training_window_days() -> int:
    """Rolling training window in days for the feature build (0 = all data).

    Reads ``[feature_engineering] training_window_days`` from
    ``config/smarthub.ini`` (via ``task_config``); falls back to the default
    of 21 when unset. The market is non-stationary (CONTEXT.md §7).

    Returns
    -------
    int
        Number of days in the rolling window (0 means all data).
    """
    from smarthub.core import task_config

    return task_config.get_int(
        "feature_engineering", "training_window_days", DEFAULT_TRAINING_WINDOW_DAYS
    )
