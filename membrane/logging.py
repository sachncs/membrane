"""Shared logging configuration for Membrane."""

import logging
import os

DEFAULT_FORMAT: str = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"


def configure_logging(
    level: int | str | None = None,
    fmt: str | None = None,
) -> None:
    """Configure root logging for Membrane.

    Uses the ``MEMBRANE_LOG_LEVEL`` environment variable when *level* is not
    provided. Defaults to ``INFO``.

    Args:
        level: Logging level (e.g. ``logging.DEBUG`` or ``"DEBUG"``).
        fmt: Log record format string. Uses :data:`DEFAULT_FORMAT` when
            omitted.
    """
    if level is None:
        level = os.environ.get("MEMBRANE_LOG_LEVEL", "INFO")
    if fmt is None:
        fmt = DEFAULT_FORMAT
    logging.basicConfig(level=level, format=fmt)


def get_logger(name: str) -> logging.Logger:
    """Return a logger with the given dotted name.

    Convenience wrapper around :func:`logging.getLogger` that ensures the
    logger is a descendant of ``membrane``.
    """
    return logging.getLogger(name)
