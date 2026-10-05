"""Membrane CLI: production command-line interface with live dashboard.

Commands:

* ``membrane serve`` — start a Membrane server (commands/serve.py).
* ``membrane dashboard`` — open a live TUI dashboard against a remote
  server (poll.py).
* ``membrane cluster-status`` — show cluster membership (commands/cluster.py).
* ``membrane llm-status`` — show LLM backend status (commands/llm.py).
* ``membrane config`` — show static configuration (commands/config.py).

Example::

    membrane serve --node-id n1 --port 8080 --transport http --compute gpu
    membrane dashboard --host localhost --port 8080

The CLI is built on :mod:`typer` (commands and option parsing) and
:mod:`rich` (TUI rendering).
"""

import os

import typer

from membrane.cli import output
from membrane.cli.commands import admin, client, cluster, config, dashboard, llm, serve
from membrane.logging import configure_logging

app = typer.Typer(
    name="membrane",
    help="Membrane — Global Contextual Memory Fabric CLI",
    no_args_is_help=True,
)


def ensure_cli_logging() -> None:
    """Route CLI output and diagnostics through logging.

    Honours ``MEMBRANE_LOG_LEVEL`` and ``MEMBRANE_LOG_FORMAT`` (``json``
    for one JSON object per diagnostic line); ``membrane serve``
    reconfigures from its own flags.
    """
    configure_logging(
        level=os.environ.get("MEMBRANE_LOG_LEVEL", "INFO"),
        json_mode=os.environ.get("MEMBRANE_LOG_FORMAT", "text") == "json",
    )


def version_callback(value: bool) -> None:
    """Show the version and exit when ``--version`` is given.

    Args:
        value: Whether the flag was passed.

    Raises:
        typer.Exit: After printing the version.
    """
    if value:
        ensure_cli_logging()
        from membrane import __version__

        output.result(f"membrane {__version__}")
        raise typer.Exit()


@app.callback()
def root(
    version: bool = typer.Option(
        False, "--version", "-V", callback=version_callback, is_eager=True, help="Show the version and exit."
    ),
) -> None:
    """Membrane — Global Contextual Memory Fabric CLI."""
    ensure_cli_logging()


# Register subcommands. Each is a typer.command function from
# ``membrane.cli.commands.*`` exposed as ``main`` for uniformity.
app.command(name="serve", help="Start a Membrane production server.")(serve.main)
app.command(name="dashboard", help="Open a live TUI dashboard against a remote server.")(dashboard.main)
app.command(name="cluster-status", help="Show cluster membership and peer health.")(cluster.main)
app.command(name="llm-status", help="Show active LLM backend status and model info.")(llm.main)
app.command(name="config", help="Show Membrane configuration and environment.")(config.main)
# admin and client are command groups; registering their ``main``
# as a plain command would hide every subcommand.
app.add_typer(admin.admin_app, name="admin", help="Admin operations against a running Membrane node.")
app.add_typer(client.client_app, name="client", help="One-off interactions with a running Membrane server.")


def main() -> None:
    """CLI entry point registered as the ``membrane`` console script."""
    app()


__all__ = ["app", "main"]
