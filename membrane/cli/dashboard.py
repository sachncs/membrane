"""Membrane CLI dashboard: in-process and remote (HTTP-polling) variants.

This module unifies two previously-separate dashboards behind a
single namespace:

* :func:`run_dashboard` — render a Rich full-screen layout against an
  in-process :class:`~membrane.server.Server`. Used by ``membrane
  serve`` when run without ``--daemon``.
* :func:`run_remote_dashboard` — connect to a remote Membrane server
  over HTTP and render the same layout from the ``/heartbeat``
  payload. Used by the ``membrane dashboard`` subcommand.
* :func:`fetch_json` — small helper used by the remote variant and
  by other CLI commands (``cluster``, ``llm``) that need a JSON
  snapshot from a remote Membrane server.

Both dashboards render the same header / metrics / panels; only
the data source differs.

Thread safety:
    The dashboards run as the foreground process of ``typer``. No
    shared mutable state is shared between iterations of the
    refresh loop.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.request
from typing import Any

from rich.align import Align
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from membrane.cli.formatters import fmt_bytes, fmt_duration
from membrane.server import Server, ServerDiagnostics

logger = logging.getLogger(__name__)

console = Console()


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------


def fetch_json(host: str, port: int, path: str, timeout: float = 2.0) -> dict[str, Any]:
    """Fetch a JSON payload from a remote Membrane server.

    Args:
        host: Server hostname or IP.
        port: Server listen port.
        path: URL path appended to ``http://<host>:<port>``.
        timeout: Connection timeout.

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


# ---------------------------------------------------------------------------
# Panel renderers
# ---------------------------------------------------------------------------


def _header_panel_inproc(diag: ServerDiagnostics) -> Panel:
    """Render the in-process dashboard header."""
    status = "[green]HEALTHY[/green]" if diag.load < 0.9 else "[yellow]WARNING[/yellow]"
    text = Text.assemble(
        "Membrane Server  |  ",
        f"Node: {diag.node_id}  |  ",
        f"Uptime: {fmt_duration(diag.uptime_seconds)}  |  ",
        f"Status: {status}",
    )
    return Panel(Align.center(text), style="bold white on blue")


def _header_panel_remote(data: dict[str, Any]) -> Panel:
    """Render the remote-polling dashboard header."""
    node_id = data.get("node_id", "unknown")
    healthy = data.get("healthy", False)
    status = "[green]HEALTHY[/green]" if healthy else "[red]UNHEALTHY[/red]"
    text = Text.assemble(
        "Membrane Dashboard  |  ",
        f"Node: {node_id}  |  ",
        f"Status: {status}",
    )
    return Panel(Align.center(text), style="bold white on blue")


def _metrics_panel(diag: ServerDiagnostics) -> Panel:
    """Render the metrics table from a ``ServerDiagnostics`` snapshot."""
    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Value", style="magenta", no_wrap=True)
    table.add_row("Memory Used", fmt_bytes(diag.memory_used_bytes))
    table.add_row("Memory Limit", fmt_bytes(diag.memory_limit_bytes))
    table.add_row("Fragments", str(diag.fragment_count))
    table.add_row("Primaries", str(diag.primary_count))
    table.add_row("Compute Backend", diag.backend_name)
    table.add_row("Redis", "[green]ON[/green]" if diag.redis_connected else "[red]OFF[/red]")
    table.add_row("Load", f"{diag.load:.2%}")
    table.add_row("Requests", str(diag.request_count))
    table.add_row("Errors", str(diag.error_count))
    return Panel(table, title="[bold]Server Metrics[/bold]", border_style="green")


def _metrics_panel_remote(data: dict[str, Any]) -> Panel:
    """Render the metrics table from a remote heartbeat payload."""
    table = Table(show_header=False, box=None)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")
    table.add_row("Memory Used", fmt_bytes(data.get("memory_used_bytes", 0)))
    table.add_row("Memory Limit", fmt_bytes(data.get("memory_limit_bytes", 0)))
    table.add_row("Fragment Count", str(data.get("fragment_count", 0)))
    table.add_row("Load", f"{data.get('load', 0.0):.2%}")
    return Panel(table, title="[bold]Metrics[/bold]", border_style="green")


def _peers_panel(server: Server) -> Panel:
    """Render the connected-peers panel (in-process only)."""
    if not server.connected_nodes:
        return Panel(
            "[dim]No peers connected[/dim]",
            title="[bold]Peers[/bold]",
            border_style="dim",
        )
    table = Table(show_header=True, box=None, padding=(0, 1))
    table.add_column("Node ID", style="cyan")
    table.add_column("Status", style="green")
    for nid in server.connected_nodes:
        table.add_row(nid, "connected")
    return Panel(table, title="[bold]Peers[/bold]", border_style="blue")


def _events_panel(server: Server) -> Panel:
    """Render the recent-events panel (in-process only)."""
    events = server.recent_events(n=15)
    if not events:
        return Panel(
            "[dim]No events yet[/dim]",
            title="[bold]Event Log[/bold]",
            border_style="dim",
        )
    table = Table(show_header=True, box=None, padding=(0, 1))
    table.add_column("Time", style="dim", no_wrap=True)
    table.add_column("Level", style="bold", no_wrap=True)
    table.add_column("Message", style="white")
    for ev in reversed(events):
        ts = time.strftime("%H:%M:%S", time.localtime(ev.timestamp))
        color = {
            "error": "red",
            "warn": "yellow",
            "info": "green",
            "debug": "dim",
        }.get(ev.level, "white")
        table.add_row(ts, f"[{color}]{ev.level.upper()}[/{color}]", ev.message)
    return Panel(table, title="[bold]Event Log[/bold]", border_style="yellow")


def _diagnostics_panel() -> Panel:
    """Render the diagnostics placeholder panel (remote only)."""
    text = Text("Connect to a local server with 'membrane serve' for full diagnostics.")
    return Panel(text, title="[bold]Diagnostics[/bold]", border_style="yellow")


def _footer_inproc() -> Panel:
    """Footer for the in-process dashboard."""
    return Panel(
        Align.center(Text("[Ctrl+C] Stop server  |  Live Dashboard", style="dim")),
        style="dim",
    )


def _footer_remote() -> Panel:
    """Footer for the remote dashboard."""
    return Panel(
        Align.center(Text("[Q]uit  |  Refresh: ", style="dim")),
        style="dim",
    )


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def run_dashboard(server: Server) -> None:
    """Render a live Rich dashboard for the local ``Server``.

    Unlike :func:`run_remote_dashboard` (which polls a remote server
    over HTTP), this variant reads :meth:`Server.diagnostics` and
    :meth:`Server.recent_events` directly. It is the default when
    ``membrane serve`` is invoked without ``--daemon``.
    """
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="main", ratio=1),
        Layout(name="footer", size=8),
    )
    layout["main"].split_row(
        Layout(name="left", ratio=2),
        Layout(name="right", ratio=3),
    )
    layout["left"].split_column(
        Layout(name="metrics", ratio=1),
        Layout(name="peers", size=10),
    )

    with Live(layout, refresh_per_second=2, screen=True):
        try:
            while server.running:
                diag = server.diagnostics()
                layout["header"].update(_header_panel_inproc(diag))
                layout["metrics"].update(_metrics_panel(diag))
                layout["peers"].update(_peers_panel(server))
                layout["right"].update(_events_panel(server))
                layout["footer"].update(_footer_inproc())
                time.sleep(0.5)
        except KeyboardInterrupt:
            pass
        finally:
            server.stop()
            console.print("\n[bold red]Server stopped.[/bold red]")


def run_remote_dashboard(
    host: str = "localhost",
    port: int = 8080,
    refresh: float = 1.0,
) -> None:
    """Render a live Rich dashboard polling a remote Membrane server.

    Args:
        host: Server host to monitor.
        port: Server listen port.
        refresh: Seconds between polls.
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
            data = fetch_json(host, port, "/heartbeat")
            layout["header"].update(_header_panel_remote(data))
            layout["left"].update(_metrics_panel_remote(data))
            layout["right"].update(_diagnostics_panel())
            layout["footer"].update(_footer_remote())
            time.sleep(refresh)


__all__ = ["fetch_json", "run_dashboard", "run_remote_dashboard"]
