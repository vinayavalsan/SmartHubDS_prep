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

    @classmethod
    def from_env(cls) -> "SSHSettings":
        key_path = Path(os.path.expanduser(_require("SSH_PRIVATE_KEY_PATH")))
        if not key_path.exists():
            raise ConfigError(f"SSH private key not found at: {key_path}")
        return cls(
            host=_require("SSH_HOST"),
            user=_require("SSH_USER"),
            private_key_path=key_path,
            port=int(os.getenv("SSH_PORT", "22")),
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
