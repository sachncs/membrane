"""``membrane dashboard`` command.

Connects to a remote Membrane server's ``/heartbeat`` endpoint
and renders a live Rich dashboard. Use this when the dashboard
runs in a separate process from the server; for an in-process
dashboard, see ``membrane serve`` (no ``--daemon``).
"""

from __future__ import annotations

import typer

from membrane.cli.dashboard import run_remote_dashboard


def main(
    host: str = typer.Option("localhost", "--host", help="Server host to monitor"),
    port: int = typer.Option(8080, "--port", "-p", help="Server port to monitor"),
    refresh: float = typer.Option(1.0, "--refresh", help="Refresh interval seconds"),
) -> None:
    """Open a live Rich dashboard connected to a running Membrane server."""
    run_remote_dashboard(host=host, port=port, refresh=refresh)


__all__ = ["main"]
