"""Runtime (Tier-2) configuration store, backed by the shared Postgres.

Holds business/tuning knobs edited from the config UI (not secrets or the DB
connection, which stay in the environment). Values are typed and validated
against ``REGISTRY``, env-scoped (``staging``/``prod``) with a global fallback,
and versioned to a history table for audit and rollback. The connection URL
comes from ``SMARTHUB_CONFIG_DB_URL``; tests point it at SQLite.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import (
    Column,
    DateTime,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    UniqueConstraint,
    and_,
    create_engine,
    insert,
    select,
    update,
)

DEFAULT_CONFIG_DB_URL = "postgresql+psycopg2://prefect:prefect@postgres:5432/prefect"
ENVIRONMENTS = ("staging", "prod")


def config_db_url() -> str:
    """Return the config DB URL from the environment, or the default."""
    return os.getenv("SMARTHUB_CONFIG_DB_URL", DEFAULT_CONFIG_DB_URL)


class ConfigError(ValueError):
    """Raised on unknown keys or values that fail validation."""


@dataclass(frozen=True)
class ConfigParam:
    """A single tunable parameter: its type, default, docs and validation."""

    key: str
    type: str  # 'float' | 'int' | 'bool' | 'str'
    default: Any
    description: str
    minimum: float | None = None
    maximum: float | None = None
    choices: tuple | None = None

    def cast(self, raw: Any) -> Any:
        """Cast ``raw`` to this parameter's type and validate it.

        Inputs
        ------
        raw : Any
            The value to cast (e.g. a string from JSON or the UI).

        Returns
        -------
        Any
            The value cast to ``int``, ``float``, ``bool`` or ``str``.

        Raises
        ------
        ConfigError
            If ``raw`` cannot be cast, or falls outside min/max or choices.
        """
        try:
            if self.type == "bool":
                value = (
                    raw
                    if isinstance(raw, bool)
                    else str(raw).strip().lower() in ("1", "true", "yes", "y", "on")
                )
            elif self.type == "int":
                value = int(raw)
            elif self.type == "float":
                value = float(raw)
            else:
                value = str(raw)
        except (TypeError, ValueError) as exc:
            raise ConfigError(
                f"{self.key}: cannot cast {raw!r} to {self.type}"
            ) from exc

        if self.type in ("int", "float"):
            if self.minimum is not None and value < self.minimum:
                raise ConfigError(f"{self.key}: {value} < min {self.minimum}")
            if self.maximum is not None and value > self.maximum:
                raise ConfigError(f"{self.key}: {value} > max {self.maximum}")
        if self.choices is not None and value not in self.choices:
            raise ConfigError(f"{self.key}: {value!r} not in {self.choices}")
        return value


# Tier-2 BUSINESS settings — the ONLY things the UI exposes (team decision:
# "business settings in the UI and nothing else; secrets in env"). Task configs
# (model_type, training window, bid_step, …) live in config/smarthub.yaml via
# smarthub.core.task_config; secrets live in .env.
REGISTRY: list[ConfigParam] = [
    ConfigParam("target_cm", "float", 0.25,
                "Target contribution margin kept in the bid (0–1).", 0.0, 1.0),
    ConfigParam("bid_floor", "float", 0.0,
                "Minimum bid in dollars (0 = no floor).", 0.0),
    ConfigParam("bid_max_cap", "float", 100.0,
                "Hard maximum bid in dollars (safety cap).", 0.0),
    ConfigParam("min_source_quality", "float", 0.0,
                "Minimum source-quality score required to bid (0–1).", 0.0, 1.0),
]
REGISTRY_BY_KEY = {p.key: p for p in REGISTRY}

_metadata = MetaData()

config_table = Table(
    "smarthub_config", _metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("env", String(32), nullable=False),
    Column("scope", String(64), nullable=False, default="global"),
    Column("key", String(128), nullable=False),
    Column("value", Text, nullable=False),
    Column("updated_by", String(128)),
    Column("updated_at", DateTime, nullable=False),
    UniqueConstraint("env", "scope", "key", name="uq_smarthub_config"),
)

history_table = Table(
    "smarthub_config_history", _metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("env", String(32), nullable=False),
    Column("scope", String(64), nullable=False),
    Column("key", String(128), nullable=False),
    Column("value", Text, nullable=False),
    Column("updated_by", String(128)),
    Column("updated_at", DateTime, nullable=False),
)


def _require(key: str) -> ConfigParam:
    """Return the registered :class:`ConfigParam` for ``key`` or raise."""
    param = REGISTRY_BY_KEY.get(key)
    if param is None:
        raise ConfigError(f"Unknown config key: {key!r}")
    return param


class ConfigStore:
    """Read/write the Tier-2 config table (creates it if absent)."""

    def __init__(self, url: str | None = None):
        """Open the config DB, creating the tables if absent.

        Inputs
        ------
        url : str | None
            SQLAlchemy URL; defaults to ``config_db_url()`` when omitted.
        """
        self.engine = create_engine(url or config_db_url(), future=True)
        _metadata.create_all(self.engine)

    def get(self, key: str, env: str = "prod", scope: str = "global") -> Any:
        """Resolve a config value with scope fallback.

        Looks up ``(env, scope)``, then ``(env, 'global')``, then the
        registry default.

        Inputs
        ------
        key : str
            Registered config key to resolve.
        env : str
            Environment scope, e.g. ``prod`` or ``staging``.
        scope : str
            Sub-scope; falls back to ``global`` when no override exists.

        Returns
        -------
        Any
            The typed, validated value for the key.

        Raises
        ------
        ConfigError
            If ``key`` is not registered.
        """
        param = _require(key)
        with self.engine.begin() as conn:
            row = conn.execute(
                select(config_table.c.value).where(
                    and_(
                        config_table.c.env == env,
                        config_table.c.scope == scope,
                        config_table.c.key == key,
                    )
                )
            ).first()
            if row is None and scope != "global":
                row = conn.execute(
                    select(config_table.c.value).where(
                        and_(
                            config_table.c.env == env,
                            config_table.c.scope == "global",
                            config_table.c.key == key,
                        )
                    )
                ).first()
        if row is None:
            return param.default
        return param.cast(json.loads(row[0]))

    def set(
        self,
        key: str,
        value: Any,
        env: str = "prod",
        scope: str = "global",
        updated_by: str = "unknown",
    ) -> Any:
        """Validate and upsert a value, appending a history row.

        Inputs
        ------
        key : str
            Registered config key to write.
        value : Any
            New value; cast and validated against the registry.
        env : str
            Environment scope to write to.
        scope : str
            Sub-scope to write to.
        updated_by : str
            Identifier recorded in the audit history.

        Returns
        -------
        Any
            The typed, validated value that was stored.

        Raises
        ------
        ConfigError
            If ``key`` is not registered or ``value`` fails validation.
        """
        param = _require(key)
        typed = param.cast(value)
        payload = json.dumps(typed)
        now = datetime.now(timezone.utc)
        with self.engine.begin() as conn:
            existing = conn.execute(
                select(config_table.c.id).where(
                    and_(
                        config_table.c.env == env,
                        config_table.c.scope == scope,
                        config_table.c.key == key,
                    )
                )
            ).first()
            if existing is None:
                conn.execute(insert(config_table).values(
                    env=env, scope=scope, key=key, value=payload,
                    updated_by=updated_by, updated_at=now,
                ))
            else:
                conn.execute(update(config_table).where(
                    config_table.c.id == existing[0]
                ).values(value=payload, updated_by=updated_by, updated_at=now))
            conn.execute(insert(history_table).values(
                env=env, scope=scope, key=key, value=payload,
                updated_by=updated_by, updated_at=now,
            ))
        return typed

    def resolved(self, env: str = "prod", scope: str = "global") -> list[dict]:
        """List every registered param with its current value and metadata.

        Inputs
        ------
        env : str
            Environment scope to resolve values for.
        scope : str
            Sub-scope to resolve values for.

        Returns
        -------
        list[dict]
            One dict per registered param, with keys such as ``key``,
            ``type``, ``value``, ``default``, ``overridden`` and audit
            fields (for the UI).
        """
        with self.engine.begin() as conn:
            rows = conn.execute(
                select(
                    config_table.c.key,
                    config_table.c.value,
                    config_table.c.updated_by,
                    config_table.c.updated_at,
                ).where(
                    and_(config_table.c.env == env, config_table.c.scope == scope)
                )
            ).all()
        stored = {r[0]: r for r in rows}
        out = []
        for param in REGISTRY:
            row = stored.get(param.key)
            overridden = row is not None
            value = param.cast(json.loads(row[1])) if overridden else param.default
            out.append({
                "key": param.key,
                "type": param.type,
                "value": value,
                "default": param.default,
                "description": param.description,
                "choices": param.choices,
                "minimum": param.minimum,
                "maximum": param.maximum,
                "overridden": overridden,
                "updated_by": row[2] if overridden else None,
                "updated_at": row[3] if overridden else None,
            })
        return out
