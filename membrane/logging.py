"""Structured logging for the Membrane runtime and CLI.

Everything Membrane writes goes through :mod:`logging`; nothing uses
``print``.

* Diagnostics go to **stderr** through :class:`TextFormatter` (human
  readable) or :class:`JsonFormatter` (one JSON object per line, for
  log aggregators).
* Command results of the CLI go to **stdout** through the dedicated
  :data:`OUTPUT_LOGGER_NAME` logger with a bare ``%(message)s`` format,
  so ``membrane client inventory | jq`` keeps working.
* :func:`log_event` takes a template string (PEP 750) and records each
  interpolated value as a structured field, so JSON logs are queryable
  by ``content_hash``, ``node_id`` and so on without parsing messages.
* :data:`request_id` carries the current HTTP request's ID; the
  :class:`RequestContextFilter` stamps it on every record emitted while
  the request is handled.
"""

import json
import logging
import sys
from contextvars import ContextVar
from datetime import UTC, datetime
from string.templatelib import Interpolation, Template
from typing import override

from membrane.constants import DEFAULT_LOG_FORMAT, DEFAULT_LOG_LEVEL

#: Name of the logger whose records are a CLI command's result.
OUTPUT_LOGGER_NAME = "membrane.cli.output"

#: ID of the request being handled in the current context ("" outside one).
request_id: ContextVar[str] = ContextVar("membrane_request_id", default="")

#: Standard :class:`logging.LogRecord` attributes, never emitted as fields.
RESERVED_ATTRIBUTES: frozenset[str] = frozenset(
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


class LoggingState:
    """Process-wide logging configuration flag.

    Attributes:
        configured: Whether :func:`configure_logging` has run.
    """

    configured: bool = False


class TextFormatter(logging.Formatter):
    """Human-readable formatter; appends ``[request_id]`` when one is set."""

    def __init__(self, fmt: str = DEFAULT_LOG_FORMAT) -> None:
        """Initialize the formatter.

        Args:
            fmt: :mod:`logging` format string.
        """
        super().__init__(fmt=fmt)

    @override
    def format(self, record: logging.LogRecord) -> str:
        """Format ``record``, suffixing the request ID when present.

        Args:
            record: The record to format.

        Returns:
            str: The formatted line.
        """
        line = super().format(record)
        rid = getattr(record, "request_id", "")
        return f"{line} [{rid}]" if rid else line


class JsonFormatter(logging.Formatter):
    """JSON line formatter.

    Emits one JSON object per record: ``ts``, ``level``, ``logger``,
    ``message``, ``exc`` (when present), and every non-standard record
    attribute, which includes ``extra={...}`` keys, :func:`log_event`
    fields, and ``request_id``.
    """

    RESERVED: frozenset[str] = RESERVED_ATTRIBUTES

    @override
    def format(self, record: logging.LogRecord) -> str:
        """Serialize ``record`` as one JSON object.

        Args:
            record: The record to format.

        Returns:
            str: A single line of JSON.
        """
        payload: dict[str, object] = {
            "ts": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        for key, value in record.__dict__.items():
            if key in self.RESERVED or key in payload:
                continue
            if key == "request_id" and not value:
                continue  # outside a request
            payload[key] = value
        return json.dumps(payload, default=str)


class StdStreamHandler(logging.StreamHandler):  # type: ignore[type-arg]
    """Stream handler bound to ``sys.stdout``/``sys.stderr`` at emit time.

    Resolving the stream on every record keeps output correct when the
    process redirects its standard streams after logging is configured
    (test runners, daemonizers).
    """

    def __init__(self, stream_name: str) -> None:
        """Initialize the handler.

        Args:
            stream_name: ``"stdout"`` or ``"stderr"``.
        """
        super().__init__()
        self.stream_name = stream_name

    @override
    def emit(self, record: logging.LogRecord) -> None:
        """Write ``record`` to the current standard stream.

        Args:
            record: The record to emit.
        """
        self.stream = getattr(sys, self.stream_name)
        super().emit(record)


class RequestContextFilter(logging.Filter):
    """Attach the current :data:`request_id` to every record."""

    @override
    def filter(self, record: logging.LogRecord) -> bool:
        """Stamp ``record.request_id`` and keep the record.

        Args:
            record: The record being emitted.

        Returns:
            bool: Always ``True``.
        """
        if not getattr(record, "request_id", ""):
            record.request_id = request_id.get()
        return True


def render_template(template: Template) -> tuple[str, dict[str, object]]:
    """Render a template string and collect its interpolated values.

    Args:
        template: A ``t"..."`` template string.

    Returns:
        tuple[str, dict[str, object]]: The rendered text, and a mapping
        from each interpolation's expression (e.g. ``"node_id"``) to its
        value. Expressions that collide with standard record attributes
        are prefixed with ``field_``.
    """
    parts: list[str] = []
    fields: dict[str, object] = {}
    for item in template:
        if isinstance(item, Interpolation):
            value = item.value
            match item.conversion:
                case "r":
                    shown: object = repr(value)
                case "s":
                    shown = str(value)
                case "a":
                    shown = ascii(value)
                case _:
                    shown = value
            parts.append(format(shown, item.format_spec))
            key = item.expression.strip()
            fields[f"field_{key}" if key in RESERVED_ATTRIBUTES else key] = value
        else:
            parts.append(item)
    return "".join(parts), fields


def log_event(logger: logging.Logger, level: int, template: Template, **extra: object) -> None:
    """Log a template string with its values as structured fields.

    ``log_event(logger, logging.INFO, t"stored {content_hash} on {node_id}")``
    logs the text ``stored ab12… on n1`` and, in JSON mode, the fields
    ``content_hash`` and ``node_id``.

    Args:
        logger: Logger to emit on.
        level: :mod:`logging` level.
        template: The ``t"..."`` message.
        **extra: Additional structured fields.
    """
    if not logger.isEnabledFor(level):
        return
    message, fields = render_template(template)
    logger.log(level, "%s", message, extra={**fields, **extra}, stacklevel=2)


def output_logger() -> logging.Logger:
    """Return the logger for CLI command results (stdout, bare messages).

    Returns:
        logging.Logger: The :data:`OUTPUT_LOGGER_NAME` logger.
    """
    return logging.getLogger(OUTPUT_LOGGER_NAME)


def configure_logging(
    level: str | None = None,
    fmt: str | None = None,
    json_mode: bool = False,
    force: bool = False,
) -> None:
    """Configure diagnostics (stderr) and CLI output (stdout) logging.

    Args:
        level: Log level name (DEBUG, INFO, WARNING, ERROR, CRITICAL).
            Defaults to ``MEMBRANE_LOG_LEVEL`` or ``INFO``.
        fmt: Format string for text mode. Ignored in JSON mode.
        json_mode: Emit diagnostics as JSON lines instead of text.
        force: Reconfigure even if logging was already configured.
    """
    if LoggingState.configured and not force:
        return
    diagnostics = StdStreamHandler("stderr")
    diagnostics.setFormatter(JsonFormatter() if json_mode else TextFormatter(fmt or DEFAULT_LOG_FORMAT))
    diagnostics.addFilter(RequestContextFilter())
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(diagnostics)
    root.setLevel((level or DEFAULT_LOG_LEVEL).upper())

    results = StdStreamHandler("stdout")
    results.setFormatter(logging.Formatter("%(message)s"))
    output = output_logger()
    output.handlers.clear()
    output.addHandler(results)
    output.setLevel(logging.INFO)
    output.propagate = False
    LoggingState.configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a logger, configuring logging on first use.

    Args:
        name: Logger name (typically ``__name__``).

    Returns:
        logging.Logger: The logger.
    """
    if not LoggingState.configured:
        configure_logging()
    return logging.getLogger(name)


__all__ = [
    "OUTPUT_LOGGER_NAME",
    "JsonFormatter",
    "LoggingState",
    "RequestContextFilter",
    "StdStreamHandler",
    "TextFormatter",
    "configure_logging",
    "get_logger",
    "log_event",
    "output_logger",
    "render_template",
    "request_id",
]
