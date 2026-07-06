"""Runtime (Tier-2) configuration store, backed by the shared Postgres.

Business/tuning knobs Kiran edits from the config UI live **here** — not secrets
or the DB connection itself (those stay in the environment; see CONTEXT.md §10 /
PLAN). Values are:

- **typed & validated** against the ``REGISTRY`` below,
- **env-scoped** (``staging`` / ``prod``) with per-scope overrides + global
  fallback,
- **versioned** — every write is appended to a history table for audit/rollback.

The connection URL comes from ``SMARTHUB_CONFIG_DB_URL`` (defaults to the shared
Postgres inside the Docker network). Tests point it at SQLite.
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
        """Cast ``raw`` to this param's type and validate; raise ConfigError."""
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


# Tier-2 tunable parameters (the knobs the UI exposes).
REGISTRY: list[ConfigParam] = [
    ConfigParam("target_cm", "float", 0.25,
                "Target contribution margin kept in the bid (0–1).", 0.0, 1.0),
    ConfigParam("bid_floor", "float", 0.0,
                "Minimum bid in dollars (0 = no floor).", 0.0),
    ConfigParam("bid_max_cap", "float", 100.0,
                "Hard maximum bid in dollars (safety cap).", 0.0),
    ConfigParam("recency_window_days", "int", 21,
                "Rolling training window in days.", 1, 365),
    ConfigParam("exploration_variance_pct", "float", 0.10,
                "Bid noise std as a fraction of the base bid (0–1).", 0.0, 1.0),
    ConfigParam("model_type", "str", "lightgbm",
                "Model family used for training.",
                choices=("logistic_regression", "lightgbm")),
    ConfigParam("active_model_version", "str", "none",
                "MLflow model version currently served."),
    ConfigParam("min_source_quality", "float", 0.0,
                "Minimum source-quality score required to bid (0–1).", 0.0, 1.0),
    ConfigParam("holiday_calendar", "str", "US",
                "Holiday calendar used for is_workday.",
                choices=("US", "NONE")),
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
    param = REGISTRY_BY_KEY.get(key)
    if param is None:
        raise ConfigError(f"Unknown config key: {key!r}")
    return param


class ConfigStore:
    """Read/write the Tier-2 config table (creates it if absent)."""

    def __init__(self, url: str | None = None):
        self.engine = create_engine(url or config_db_url(), future=True)
        _metadata.create_all(self.engine)

    def get(self, key: str, env: str = "prod", scope: str = "global") -> Any:
        """Resolve a value: (env,scope) → (env,'global') → registry default."""
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
        """Validate + upsert a value, appending to history. Returns typed value."""
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
        """Every registered param with its current value + metadata (for the UI)."""
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
