"""Centralised logging configuration."""

from __future__ import annotations

import logging
import os

_CONFIGURED = False


def configure_logging(level: str | None = None) -> None:
    """Configure root logging once, idempotently.

    Subsequent calls are no-ops.

    Inputs
    ------
    level : str | None
        Log level name; falls back to the ``LOG_LEVEL`` env var, then
        ``INFO``.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    resolved = (level or os.getenv("LOG_LEVEL") or "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, resolved, logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
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
