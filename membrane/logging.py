"""Structured logging configuration.

Two formatters are provided:

* :class:`TextFormatter` — the historical human-readable format. Used
  by default for local development and by the CLI dashboard.
* :class:`JsonFormatter` — line-delimited JSON with standard fields
  (timestamp, level, logger, message) and any ``extra={...}`` keys
  supplied at log call sites. Use this for production: structured
  logs are far easier to ship to a log aggregator and to filter by
  request_id, node_id, peer, etc.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

from membrane.constants import DEFAULT_LOG_FORMAT, DEFAULT_LOG_LEVEL

configured = False


class TextFormatter(logging.Formatter):
    """Plain-text formatter (default)."""

    def __init__(self, fmt: str = DEFAULT_LOG_FORMAT) -> None:
        super().__init__(fmt=fmt)


class JsonFormatter(logging.Formatter):
    """JSON line formatter.

    Emits one JSON object per record. Standard fields are emitted
    explicitly; any ``extra={...}`` keys supplied at the log call
    site are merged into the output.
    """

    RESERVED: frozenset[str] = frozenset(
        {
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
            "message",
            "taskName",
        }
    )

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        for key, value in record.__dict__.items():
            if key in self.RESERVED or key in payload:
                continue
            payload[key] = value
        return json.dumps(payload, default=str)


def configure_logging(
    level: str | None = None,
    fmt: str | None = None,
    json_mode: bool = False,
) -> None:
    """Configure the root logger.

    Args:
        level: Log level name (DEBUG, INFO, WARNING, ERROR, CRITICAL).
            Defaults to ``MEMBRANE_LOG_LEVEL`` env var or ``INFO``.
        fmt: Format string for text mode. Ignored in JSON mode.
        json_mode: When ``True``, emit JSON lines instead of text.
    """
    global configured
    if configured:
        return
    effective_level = (level or DEFAULT_LOG_LEVEL).upper()
    handler = logging.StreamHandler()
    if json_mode:
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(TextFormatter(fmt or DEFAULT_LOG_FORMAT))
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(effective_level)
    configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a logger, ensuring logging is configured.

    Args:
        name: Logger name (typically ``__name__``).

    Returns:
        logging.Logger: Configured logger.
    """
    if not configured:
        configure_logging()
    return logging.getLogger(name)


__all__ = ["JsonFormatter", "TextFormatter", "configure_logging", "get_logger"]
