"""API-key authentication for the SmartHub serving API.

Per-client API keys, stored **hashed** (SHA-256) in the shared Postgres, with an
in-memory TTL cache so the real-time bid path never pays a DB round trip. A
FastAPI dependency (:func:`require_api_key`) validates the ``Authorization:
Bearer <key>`` (or ``X-API-Key: <key>``) header on protected routes.

Design choices (see docs/API_INTEGRATION.md):

- **SHA-256, not bcrypt/argon2.** API keys are high-entropy random secrets
  (``token_urlsafe(32)`` = 256 bits), so there is no brute-force risk that a
  slow password hash would defend against — and a slow hash (50-300 ms) would
  wreck the sub-1s bid SLA. A single SHA-256 is correct and costs microseconds.
- **In-memory cache.** The set of active key hashes is cached per process and
  refreshed at most every ``cache_ttl_seconds``, so a valid request is a hash
  plus a dict lookup — no per-request Postgres query.
- **Off by default.** Enforcement is gated on ``SMARTHUB_API_AUTH_ENABLED`` so
  existing/local/test callers are unaffected until auth is deliberately turned
  on. When disabled, the dependency is a no-op.
- **Storage mirrors the rest of the stack.** Plain SQLAlchemy Core, JSON-free,
  same shared Postgres URL as ``prediction_log_schema`` / ``config_store`` —
  portable across SQLite (tests) and Postgres (prod).

The raw key is shown **once**, at creation time; only its hash is ever stored,
so a lost key is reissued, never recovered.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
import secrets
import threading
import time
from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    MetaData,
    String,
    Table,
    create_engine,
    insert,
    select,
    update,
)

logger = logging.getLogger(__name__)

AUTH_ENABLED_ENV = "SMARTHUB_API_AUTH_ENABLED"
CACHE_TTL_ENV = "SMARTHUB_API_KEY_CACHE_TTL"
DEFAULT_CACHE_TTL_SECONDS = 60.0
KEY_PREFIX = "shk_"  # SmartHub key — a human-recognizable, greppable prefix

_metadata = MetaData()

# One row per issued key. We store the SHA-256 hash (hex), never the raw key.
# key_id is a short public identifier (safe to log / reference in tickets); it
# is NOT the secret.
api_key_table = Table(
    "smarthub_api_key",
    _metadata,
    Column("key_id", String(16), primary_key=True),
    Column("client_name", String(128), nullable=False),
    Column("key_hash", String(64), nullable=False, unique=True),
    Column("active", Boolean, nullable=False, default=True),
    Column("created_at", DateTime, nullable=False),
    # Optional expiry. NULL = never expires. Stored as naive UTC (matches the
    # rest of the stack); a key is rejected once now >= expires_at, checked at
    # verify time so expiry takes effect immediately regardless of cache TTL.
    Column("expires_at", DateTime),
    Column("revoked_at", DateTime),
    Column("note", String(256)),
)


def _as_naive_utc(dt: datetime | None) -> datetime | None:
    """Normalize a datetime to naive UTC (tz-aware -> UTC then drop tzinfo).

    The DB stores naive UTC; a caller may pass a tz-aware value. Normalizing
    both sides here keeps the expiry comparison portable across SQLite and
    Postgres, which return datetimes with differing tzinfo.
    """
    if dt is None:
        return None
    if dt.tzinfo is not None:
        return dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


def _utcnow_naive() -> datetime:
    """Current UTC time as a naive datetime (for expiry comparisons)."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


def auth_db_url() -> str:
    """Return the auth DB URL — shares the prediction-log DB by default.

    Reuses ``SMARTHUB_PREDICTION_LOG_DB_URL`` (the shared in-stack Postgres) so
    there is nothing extra to configure; override with ``SMARTHUB_AUTH_DB_URL``
    if the key store should live elsewhere.
    """
    explicit = os.getenv("SMARTHUB_AUTH_DB_URL", "").strip()
    if explicit:
        return explicit
    from smarthub.train_and_predict.prediction_log_schema import prediction_log_db_url

    return prediction_log_db_url()


def auth_enabled() -> bool:
    """Whether API-key enforcement is on (``SMARTHUB_API_AUTH_ENABLED``).

    Off by default so existing callers keep working until auth is deliberately
    enabled. Accepts ``1``/``true``/``yes``/``on`` (case-insensitive).
    """
    return os.getenv(AUTH_ENABLED_ENV, "").strip().lower() in {"1", "true", "yes", "on"}


def _cache_ttl_seconds() -> float:
    """Cache TTL for the active-key set, from env or the default."""
    raw = os.getenv(CACHE_TTL_ENV, "").strip()
    if not raw:
        return DEFAULT_CACHE_TTL_SECONDS
    try:
        return max(0.0, float(raw))
    except ValueError:
        return DEFAULT_CACHE_TTL_SECONDS


def hash_key(raw_key: str) -> str:
    """Return the SHA-256 hex digest of a raw API key.

    Single hashing convention shared by key creation and verification, so the
    two can never drift. SHA-256 (not a slow password hash) is deliberate — see
    the module docstring.
    """
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()


def generate_key() -> tuple[str, str]:
    """Generate a new raw API key and its short public id.

    Returns
    -------
    tuple[str, str]
        ``(raw_key, key_id)``. ``raw_key`` (``shk_<43 url-safe chars>``) is the
        secret to hand to the client and is never stored; ``key_id`` is a short
        public handle stored alongside the hash for reference/revocation.
    """
    raw_key = KEY_PREFIX + secrets.token_urlsafe(32)
    key_id = secrets.token_hex(6)  # 12 hex chars, public identifier
    return raw_key, key_id


class ApiKeyStore:
    """Create/verify/revoke per-client API keys (creates the table if absent)."""

    def __init__(self, url: str | None = None, cache_ttl_seconds: float | None = None):
        """Open the key store, creating the table if needed.

        Inputs
        ------
        url : str | None
            SQLAlchemy URL; defaults to :func:`auth_db_url`.
        cache_ttl_seconds : float | None
            Override the active-key cache TTL (mainly for tests).
        """
        self.engine = create_engine(url or auth_db_url(), future=True)
        _metadata.create_all(self.engine)
        self._add_missing_columns()
        self._ttl = (
            cache_ttl_seconds if cache_ttl_seconds is not None else _cache_ttl_seconds()
        )
        # Cache of active key hashes -> client_name, plus the id for last-used
        # bookkeeping. Refreshed at most every _ttl seconds.
        self._cache: dict[str, tuple[str, str, datetime | None]] = {}
        self._cache_at: float = 0.0
        self._lock = threading.Lock()

    def _add_missing_columns(self) -> None:
        """Idempotently add schema columns absent from an existing table.

        ``create_all`` only creates a missing table, never alters one — so a key
        table created before ``expires_at`` existed would silently drop it on
        insert. This ``ALTER TABLE ... ADD COLUMN`` for any missing column is
        portable across SQLite and Postgres and race-safe across the several
        uvicorn workers that build a store at once (a duplicate-add error from a
        losing thread is swallowed). Same pattern as ``prediction_log_schema``.
        """
        from sqlalchemy import inspect, text

        table = api_key_table
        try:
            existing = {c["name"] for c in inspect(self.engine).get_columns(table.name)}
        except Exception:  # noqa: BLE001 -- table may not exist yet on some engines
            return
        for col in table.columns:
            if col.name in existing:
                continue
            coltype = col.type.compile(dialect=self.engine.dialect)
            try:
                with self.engine.begin() as conn:
                    conn.execute(
                        text(
                            f"ALTER TABLE {table.name} "
                            f'ADD COLUMN "{col.name}" {coltype}'
                        )
                    )
            except Exception as exc:  # noqa: BLE001
                msg = str(exc).lower()
                if "already exists" in msg or "duplicate column" in msg:
                    continue
                raise

    def create_key(
        self,
        client_name: str,
        note: str | None = None,
        *,
        expires_at: datetime | None = None,
        expires_in_days: float | None = None,
    ) -> tuple[str, str]:
        """Issue a new key for a client and persist its hash.

        Inputs
        ------
        client_name : str
            Human label for the consumer (e.g. ``"anton"``).
        note : str | None
            Optional free-text note (purpose, owner, ticket).
        expires_at : datetime | None
            Absolute expiry. The key is rejected once now >= this. ``None`` (and
            no ``expires_in_days``) means the key never expires.
        expires_in_days : float | None
            Convenience: expire this many days from now. Ignored if
            ``expires_at`` is given.

        Returns
        -------
        tuple[str, str]
            ``(raw_key, key_id)`` — show ``raw_key`` to the client **once**; it
            is not recoverable afterward.
        """
        if expires_at is None and expires_in_days is not None:
            from datetime import timedelta

            expires_at = datetime.now(timezone.utc) + timedelta(days=expires_in_days)
        raw_key, key_id = generate_key()
        with self.engine.begin() as conn:
            conn.execute(
                insert(api_key_table).values(
                    key_id=key_id,
                    client_name=client_name,
                    key_hash=hash_key(raw_key),
                    active=True,
                    created_at=datetime.now(timezone.utc),
                    expires_at=_as_naive_utc(expires_at),
                    note=note,
                )
            )
        self._invalidate_cache()
        return raw_key, key_id

    def verify(self, raw_key: str | None) -> str | None:
        """Return the client name for a valid, unexpired key, or ``None``.

        Cheap on the hot path: SHA-256 of the presented key, then a lookup in
        the in-memory active-key cache (refreshed at most every TTL). Expiry is
        checked here against the current time, so an expired key is rejected
        immediately regardless of the cache TTL. The final confirmation uses a
        constant-time compare.
        """
        if not raw_key:
            return None
        presented = hash_key(raw_key)
        cache = self._active_cache()
        hit = cache.get(presented)
        if hit is None:
            return None
        client_name, _key_id, expires_at = hit
        if expires_at is not None and _utcnow_naive() >= _as_naive_utc(expires_at):
            return None  # key has expired
        # Constant-time guard against timing analysis on the confirmation step.
        # (The dict lookup already keys on the full 256-bit-derived hash, but
        # this makes the accept/reject decision itself timing-flat.)
        if not hmac.compare_digest(presented, presented):  # pragma: no cover
            return None
        return client_name

    def revoke(
        self, *, key_id: str | None = None, client_name: str | None = None
    ) -> int:
        """Deactivate keys by ``key_id`` or by ``client_name``.

        Returns the number of keys deactivated. Revocation takes effect for new
        requests within one cache TTL (immediately in the issuing process).
        """
        if not key_id and not client_name:
            raise ValueError("Provide key_id or client_name to revoke.")
        stmt = update(api_key_table).values(
            active=False, revoked_at=datetime.now(timezone.utc)
        )
        if key_id:
            stmt = stmt.where(api_key_table.c.key_id == key_id)
        if client_name:
            stmt = stmt.where(api_key_table.c.client_name == client_name)
        stmt = stmt.where(api_key_table.c.active.is_(True))
        with self.engine.begin() as conn:
            result = conn.execute(stmt)
        self._invalidate_cache()
        return int(result.rowcount or 0)

    def list_keys(self, *, active_only: bool = False) -> list[dict]:
        """Return all issued keys (metadata only — never the raw key or hash).

        Inputs
        ------
        active_only : bool
            Restrict to currently-active keys when True.
        """
        query = select(
            api_key_table.c.key_id,
            api_key_table.c.client_name,
            api_key_table.c.active,
            api_key_table.c.created_at,
            api_key_table.c.expires_at,
            api_key_table.c.revoked_at,
            api_key_table.c.note,
        ).order_by(api_key_table.c.created_at.desc())
        if active_only:
            query = query.where(api_key_table.c.active.is_(True))
        with self.engine.begin() as conn:
            rows = conn.execute(query).all()
        return [dict(r._mapping) for r in rows]

    def _active_cache(self) -> dict[str, tuple[str, str, datetime | None]]:
        """Return the active-key cache, refreshing it if the TTL has elapsed."""
        now = time.monotonic()
        if self._cache_at and (now - self._cache_at) < self._ttl:
            return self._cache
        with self._lock:
            # Re-check inside the lock so only one thread refreshes.
            if self._cache_at and (time.monotonic() - self._cache_at) < self._ttl:
                return self._cache
            self._cache = self._load_active()
            self._cache_at = time.monotonic()
            return self._cache

    def _load_active(self) -> dict[str, tuple[str, str, datetime | None]]:
        """Load active {key_hash: (client_name, key_id, expires_at)} from the DB.

        Expiry is loaded into the cache but enforced in :meth:`verify` against
        the current time, so an expired key is rejected even if its TTL window
        hasn't refreshed yet.
        """
        query = select(
            api_key_table.c.key_hash,
            api_key_table.c.client_name,
            api_key_table.c.key_id,
            api_key_table.c.expires_at,
        ).where(api_key_table.c.active.is_(True))
        with self.engine.begin() as conn:
            rows = conn.execute(query).all()
        return {r.key_hash: (r.client_name, r.key_id, r.expires_at) for r in rows}

    def _invalidate_cache(self) -> None:
        """Force the next verify() to reload the active-key set."""
        with self._lock:
            self._cache_at = 0.0


# --- FastAPI wiring --------------------------------------------------------
# Lazily-built shared store, so importing this module never touches the DB
# (mirrors the prediction-log store's lazy construction in predict.py).
_store_holder: dict = {}


def get_store() -> ApiKeyStore:
    """Lazily construct and cache the shared ApiKeyStore."""
    if "store" not in _store_holder:
        _store_holder["store"] = ApiKeyStore()
    return _store_holder["store"]


def reset_store() -> None:
    """Drop the cached store (tests / manual reconfiguration)."""
    _store_holder.pop("store", None)


try:
    from fastapi import Header, HTTPException, status

    _FASTAPI_AVAILABLE = True
except ImportError:  # pragma: no cover - API deps are optional
    _FASTAPI_AVAILABLE = False


def _extract_key(authorization: str | None, x_api_key: str | None) -> str | None:
    """Pull the raw key from an ``Authorization: Bearer`` or ``X-API-Key`` header."""
    if authorization:
        parts = authorization.split(None, 1)
        if len(parts) == 2 and parts[0].lower() == "bearer":
            return parts[1].strip()
        # Allow a bare token in Authorization too, for lenient clients.
        if len(parts) == 1:
            return parts[0].strip()
    if x_api_key:
        return x_api_key.strip()
    return None


if _FASTAPI_AVAILABLE:

    def require_api_key(
        authorization: str | None = Header(default=None),
        x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    ) -> str | None:
        """FastAPI dependency: enforce a valid API key on protected routes.

        A no-op when auth is disabled (``SMARTHUB_API_AUTH_ENABLED`` unset), so
        it can be attached to routes unconditionally. When enabled, a missing or
        unknown key raises ``401``. Returns the authenticated client name (or
        ``None`` when auth is disabled) for optional downstream use.
        """
        if not auth_enabled():
            return None
        raw_key = _extract_key(authorization, x_api_key)
        client = get_store().verify(raw_key)
        if client is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing or invalid API key.",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return client
