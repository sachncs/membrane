"""Standalone HTTP-polling TUI dashboard.

Connects to a remote Membrane server's ``/heartbeat`` endpoint at a
configurable interval and renders a Rich full-screen layout. Use
this when the dashboard runs in a separate process from the server
(typical for production observability workflows).

For the local in-process dashboard, see :mod:`membrane.cli.dashboard`.
"""

from __future__ import annotations

import json
import time
import urllib.request

import typer
from rich.align import Align
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from membrane.cli.formatters import fmt_bytes

console = Console()


def _fetch_json(host: str, port: int, path: str, timeout: float = 2.0) -> dict:
    """Fetch a JSON payload from the remote server.

    Args:
        host: Server hostname or IP.
        port: Server listen port.
        path: URL path appended to ``http://<host>:<port>``.
        timeout: Connection timeout in seconds.

    Returns:
        dict: Parsed JSON payload, or an empty dict on any error
        (network failure, non-JSON response).
    """
    url = f"http://{host}:{port}{path}"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except Exception:
        return {}


def _header(data: dict) -> Panel:
    """Render the standalone-dashboard header panel."""
    node_id = data.get("node_id", "unknown")
    healthy = data.get("healthy", False)
    status = "[green]HEALTHY[/green]" if healthy else "[red]UNHEALTHY[/red]"
    text = Text.assemble(
        "Membrane Dashboard  |  ",
        f"Node: {node_id}  |  ",
        f"Status: {status}",
    )
    return Panel(Align.center(text), style="bold white on blue")


def _metrics(data: dict) -> Panel:
    """Render the standalone-dashboard metrics panel."""
    table = Table(show_header=False, box=None)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")
    table.add_row("Memory Used", fmt_bytes(data.get("memory_used_bytes", 0)))
    table.add_row("Memory Limit", fmt_bytes(data.get("memory_limit_bytes", 0)))
    table.add_row("Fragment Count", str(data.get("fragment_count", 0)))
    table.add_row("Load", f"{data.get('load', 0.0):.2%}")
    return Panel(table, title="[bold]Metrics[/bold]", border_style="green")


def _diagnostics() -> Panel:
    """Render the diagnostics panel for standalone mode."""
    text = Text("Connect to a local server with 'membrane serve' for full diagnostics.")
    return Panel(text, title="[bold]Diagnostics[/bold]", border_style="yellow")


def _footer() -> Panel:
    """Render the standalone-dashboard footer."""
    text = Text("[Q]uit  |  Refresh: ", style="dim")
    return Panel(Align.center(text), style="dim")


def main(
    host: str = typer.Option("localhost", "--host", help="Server host to monitor"),
    port: int = typer.Option(8080, "--port", "-p", help="Server port to monitor"),
    refresh: float = typer.Option(1.0, "--refresh", help="Refresh interval seconds"),
) -> None:
    """Open a live TUI dashboard connected to a running Membrane server.

    The dashboard polls the ``/heartbeat`` endpoint at the configured
    interval and renders metrics in a Rich-based full-screen layout.
    Exit with Ctrl+C.
    """
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="main", ratio=1),
        Layout(name="footer", size=3),
    )
    layout["main"].split_row(
        Layout(name="left", ratio=1),
        Layout(name="right", ratio=1),
    )

    console.print("[bold cyan]Connecting to Membrane server...[/bold cyan]")
    with Live(layout, refresh_per_second=1 / refresh, screen=True):
        while True:
            data = _fetch_json(host, port, "/heartbeat")
            layout["header"].update(_header(data))
            layout["left"].update(_metrics(data))
            layout["right"].update(_diagnostics())
            layout["footer"].update(_footer())
            time.sleep(refresh)


__all__ = ["main"]
