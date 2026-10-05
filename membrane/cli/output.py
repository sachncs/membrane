"""CLI output through :mod:`logging`.

The CLI never prints. Command **results** (JSON documents, tables) are
logged on the :data:`~membrane.logging.OUTPUT_LOGGER_NAME` logger, which
writes bare messages to stdout so they can be piped; **status** and
**errors** are logged on the ``membrane.cli`` logger, which writes to
stderr through the configured diagnostics formatter.

Rich markup and renderables (tables) are rendered to text first, with
colour only when the destination is a terminal.
"""

import io
import json
import logging
import sys
from typing import Any

from rich.console import Console

from membrane.logging import LoggingState, configure_logging, output_logger

status_logger = logging.getLogger("membrane.cli")


def ready() -> None:
    """Configure logging on first use, so output never disappears.

    Commands invoked without the root ``membrane`` callback (tests,
    embedding a sub-app) would otherwise log into an unconfigured
    logging system that drops ``INFO`` records.
    """
    if not LoggingState.configured:
        configure_logging()


def render(renderable: Any, *, for_stdout: bool) -> str:
    """Render Rich markup or a renderable to a string.

    Args:
        renderable: A string with Rich markup, or a Rich renderable
            such as a :class:`rich.table.Table`.
        for_stdout: Whether the text is destined for stdout (colour is
            kept only when that stream is a terminal).

    Returns:
        str: The rendered text without a trailing newline.
    """
    stream = sys.stdout if for_stdout else sys.stderr
    buffer = io.StringIO()
    renderer = Console(file=buffer, force_terminal=stream.isatty(), width=120, highlight=False, soft_wrap=True)
    renderer.print(renderable)
    return buffer.getvalue().rstrip("\n")


def result(renderable: Any) -> None:
    """Log a command result to stdout.

    Args:
        renderable: Text, Rich markup, or a Rich renderable.
    """
    ready()
    output_logger().info(render(renderable, for_stdout=True))


def result_json(payload: object) -> None:
    """Log a JSON document as a command result.

    Args:
        payload: Any JSON-serializable value.
    """
    ready()
    output_logger().info(json.dumps(payload, indent=2, sort_keys=True))


def info(renderable: Any) -> None:
    """Log a status message (stderr).

    Args:
        renderable: Text or Rich markup.
    """
    ready()
    status_logger.info(render(renderable, for_stdout=False))


def error(renderable: Any) -> None:
    """Log an error message (stderr).

    Args:
        renderable: Text or Rich markup.
    """
    ready()
    status_logger.error(render(renderable, for_stdout=False))


__all__ = ["error", "info", "ready", "render", "result", "result_json", "status_logger"]
