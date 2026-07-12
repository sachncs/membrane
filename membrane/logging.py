"""Shared logging configuration for Membrane.

This module is the canonical entry point for setting up logging in
both the library and CLI. It exposes:

* :data:`DEFAULT_FORMAT` — the default log record format.
* :func:`configure_logging` — idempotent root-logger configuration
  with environment-variable fallback.
* :func:`get_logger` — convenience wrapper around
  :func:`logging.getLogger` that returns a ``membrane.*`` logger.

Usage:
    >>> from membrane.logging import configure_logging, get_logger
    >>> configure_logging()  # respects MEMBRANE_LOG_LEVEL env var
    >>> logger = get_logger(__name__)
    >>> logger.info("ready")

Design notes:
    The module deliberately uses the *root* logger rather than a
    dedicated ``membrane`` logger. This makes it easier for library
    users to attach their own handlers without Membrane polluting
    their global logging configuration.
"""

import logging
import os

DEFAULT_FORMAT: str = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"


def configure_logging(
    level: int | str | None = None,
    fmt: str | None = None,
) -> None:
    """Configure root logging for Membrane.

    Calls :func:`logging.basicConfig` with the supplied (or
    defaulted) level and format. Subsequent calls are no-ops if
    the root logger already has handlers configured, mirroring the
    behavior of :func:`logging.basicConfig` itself.

    Args:
        level: Logging level. Accepts either a numeric
            :mod:`logging` constant (e.g., ``logging.DEBUG``) or a
            level name string (e.g., ``"DEBUG"``). When ``None``,
            the value of the ``MEMBRANE_LOG_LEVEL`` environment
            variable is used; if that is unset, ``"INFO"`` is used.
        fmt: Log record format string. Defaults to
            :data:`DEFAULT_FORMAT`.
    """
    if level is None:
        level = os.environ.get("MEMBRANE_LOG_LEVEL", "INFO")
    if fmt is None:
        fmt = DEFAULT_FORMAT
    # basicConfig is idempotent: if the root logger already has
    # handlers configured (e.g., by an embedding application), this
    # call is effectively a no-op.
    logging.basicConfig(level=level, format=fmt)


def get_logger(name: str) -> logging.Logger:
    """Return a logger with the given dotted name.

    Thin wrapper around :func:`logging.getLogger`. Callers should
    pass ``__name__`` so that the resulting logger is namespaced
    under the ``membrane`` package.

    Args:
        name: Dotted logger name. Conventionally ``__name__`` from
            the calling module.

    Returns:
        logging.Logger: A standard library logger that inherits
        the root configuration established by
        :func:`configure_logging`.
    """
    return logging.getLogger(name)
