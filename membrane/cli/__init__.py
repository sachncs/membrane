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

from __future__ import annotations

import typer

from membrane.cli.commands import admin, client, cluster, config, dashboard, llm, serve

app = typer.Typer(
    name="membrane",
    help="Membrane — Global Contextual Memory Fabric CLI",
    no_args_is_help=True,
)

# Register subcommands. Each is a typer.command function from
# ``membrane.cli.commands.*`` exposed as ``main`` for uniformity.
app.command(name="serve", help="Start a Membrane production server.")(serve.main)
app.command(name="dashboard", help="Open a live TUI dashboard against a remote server.")(dashboard.main)
app.command(name="cluster-status", help="Show cluster membership and peer health.")(cluster.main)
app.command(name="llm-status", help="Show active LLM backend status and model info.")(llm.main)
app.command(name="config", help="Show Membrane configuration and environment.")(config.main)
app.command(name="admin", help="Admin operations against a running Membrane node.")(admin.main)
app.command(name="client", help="One-off interactions with a running Membrane server.")(client.main)


def main() -> None:
    """CLI entry point registered as the ``membrane`` console script."""
    app()


__all__ = ["app", "main"]
