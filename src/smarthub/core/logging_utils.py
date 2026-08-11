"""Centralised logging configuration.

Two output formats, chosen by the ``LOG_FORMAT`` env var (or the ``fmt`` arg):

  * ``text`` (default) -- human-readable
    ``ts | LEVEL | file:func:line | message``.
  * ``json``           -- one JSON object per line (JSONL) with a stable schema,
                          built for machine consumption (grep/jq, log stores, and
                          the error-triage tool). Set ``LOG_FORMAT=json`` in the
                          container env to switch a service over.

JSON schema (per line)::

    ts          RFC3339 UTC with milliseconds
    level       INFO / WARNING / ERROR / ...
    service     value of SMARTHUB_SERVICE (serve / worker / shap-worker / ...)
    logger      logger name (usually __name__)
    msg         human-readable message
    <context>   any key bound via bind_context()/request_context() or passed as
                logging `extra=` (e.g. request_id, event, lead_type, latency_ms)
    error       ONLY on exceptions: {type, msg, fingerprint, stack}

``error.fingerprint`` is a short stable hash of the exception type + call frames
(file:func:line), so identical failures group together. The full traceback rides
in ``error.stack`` as one field, so nothing multi-line leaks across log lines.

Public API is unchanged for callers: ``get_logger(__name__)`` and
``configure_logging()``. The rest is additive: ``bind_context``,
``request_context``, ``new_request_id``, ``log_event``.
"""

from __future__ import annotations

import contextvars
import datetime as _dt
import hashlib
import json
import logging
import os
import traceback
import uuid
from contextlib import contextmanager

_CONFIGURED = False
_SERVICE: str | None = None

# Human-readable text format (kept from the shared logging setup).
_TEXT_FORMAT = (
    "%(asctime)s | %(levelname)-5s | "
    "%(filename)s:%(funcName)s:%(lineno)d | %(message)s"
)
_TEXT_DATEFMT = "%Y-%m-%d %H:%M:%S"

# Per-request / per-flow fields merged into every JSON log line on this
# thread/task (see bind_context / request_context).
_CONTEXT: contextvars.ContextVar[dict] = contextvars.ContextVar(
    "log_context", default={}
)

# Standard LogRecord attributes -- anything NOT here that lands on a record
# (i.e. passed via logging `extra=`) is treated as structured context.
_RESERVED = {
    "name",
    "msg",
    "args",
    "levelname",
    "levelno",
    "pathname",
    "filename",
    "module",
    "exc_info",
    "exc_text",
    "stack_info",
    "lineno",
    "funcName",
    "created",
    "msecs",
    "relativeCreated",
    "thread",
    "threadName",
    "processName",
    "process",
    "taskName",
    "message",
    "asctime",
}


def _service_name() -> str:
    return _SERVICE or os.getenv("SMARTHUB_SERVICE", "app")


def _error_block(exc_info) -> dict:
    """Build the structured error record (type/msg/fingerprint/stack)."""
    etype, evalue, tb = exc_info
    frames = traceback.extract_tb(tb)
    signature = f"{getattr(etype, '__name__', 'Error')}|" + "|".join(
        f"{os.path.basename(fr.filename)}:{fr.name}:{fr.lineno}" for fr in frames
    )
    fingerprint = hashlib.sha1(signature.encode("utf-8")).hexdigest()[:12]
    stack = "".join(traceback.format_exception(etype, evalue, tb))
    return {
        "type": getattr(etype, "__name__", "Error"),
        "msg": str(evalue),
        "fingerprint": fingerprint,
        "stack": stack,
    }


class JsonFormatter(logging.Formatter):
    """Render a LogRecord as a single JSON line matching the schema above."""

    def format(self, record: logging.LogRecord) -> str:  # noqa: A003
        ts = (
            _dt.datetime.fromtimestamp(record.created, _dt.timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%S."
            )
            + f"{int(record.msecs):03d}Z"
        )

        payload: dict = {
            "ts": ts,
            "level": record.levelname,
            "service": _service_name(),
            "logger": record.name,
            "msg": record.getMessage(),
        }

        for key, val in _CONTEXT.get().items():
            payload.setdefault(key, val)
        for key, val in record.__dict__.items():
            if key not in _RESERVED and not key.startswith("_"):
                payload[key] = val

        if record.exc_info:
            payload["error"] = _error_block(record.exc_info)

        return json.dumps(payload, default=str, ensure_ascii=False)


def configure_logging(
    level: str | None = None, *, fmt: str | None = None, service: str | None = None
) -> None:
    """Configure root logging once, idempotently. Subsequent calls are no-ops.

    Inputs
    ------
    level : str | None
        Log level name; falls back to the ``LOG_LEVEL`` env var, then ``INFO``.
    fmt : str | None
        ``"json"`` or ``"text"``; falls back to the ``LOG_FORMAT`` env var, then
        ``"text"``.
    service : str | None
        Service name for the ``service`` field in JSON logs; falls back to the
        ``SMARTHUB_SERVICE`` env var, then ``"app"``.
    """
    global _CONFIGURED, _SERVICE
    if service:
        _SERVICE = service
    if _CONFIGURED:
        return

    resolved = (level or os.getenv("LOG_LEVEL") or "INFO").upper()
    chosen_fmt = (fmt or os.getenv("LOG_FORMAT") or "text").lower()

    handler = logging.StreamHandler()
    if chosen_fmt == "json":
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter(fmt=_TEXT_FORMAT, datefmt=_TEXT_DATEFMT))

    root = logging.getLogger()
    root.handlers = [handler]  # own the output so the format is consistent
    root.setLevel(getattr(logging, resolved, logging.INFO))
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a logger for ``name``, ensuring logging is configured.

    Inputs
    ------
    name : str
        Logger name, typically ``__name__``.

    Returns
    -------
    logging.Logger
        The named logger.
    """
    configure_logging()
    return logging.getLogger(name)


# --------------------------------------------------------------------------- #
# structured-context helpers (additive; safe to ignore for plain logging)      #
# --------------------------------------------------------------------------- #
def new_request_id() -> str:
    """Short correlation id for a single request / flow run."""
    return uuid.uuid4().hex[:12]


def bind_context(**fields) -> contextvars.Token:
    """Merge ``fields`` into the log context for this thread/task.

    Every subsequent JSON log line picks them up automatically (e.g. a
    ``request_id`` set once at the top of a request appears on all its logs).
    Returns a token; pass it to :func:`reset_context`, or use the
    :func:`request_context` manager which handles that for you.
    """
    merged = dict(_CONTEXT.get())
    merged.update({k: v for k, v in fields.items() if v is not None})
    return _CONTEXT.set(merged)


def reset_context(token: contextvars.Token) -> None:
    """Undo a previous :func:`bind_context`."""
    _CONTEXT.reset(token)


def clear_context() -> None:
    """Drop all bound context fields."""
    _CONTEXT.set({})


@contextmanager
def request_context(**fields):
    """Bind ``fields`` for the duration of the block, then restore.

    Example::

        with request_context(request_id=new_request_id(), lead_type=6):
            logger.info("recommending bid", extra={"event": "bid_start"})
    """
    token = bind_context(**fields)
    try:
        yield
    finally:
        reset_context(token)


def log_event(
    logger: logging.Logger,
    event: str,
    msg: str,
    *,
    level: int = logging.INFO,
    **context,
) -> None:
    """Emit a log line carrying a machine ``event`` key plus context fields.

    Thin sugar over ``logger.log(level, msg, extra={"event": event, **context})``::

        log_event(log, "bid_recommended", "recommended bid 0.75",
                  bid=0.75, latency_ms=31)
    """
    logger.log(level, msg, extra={"event": event, **context})
