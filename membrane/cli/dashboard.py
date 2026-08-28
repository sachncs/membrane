"""Local TUI dashboard for an in-process :class:`Server`.

Renders a Rich layout (header / metrics / peers / event-log / footer)
with a 0.5 s refresh tick. Reads :meth:`Server.diagnostics` and
:meth:`Server.recent_events` directly so no HTTP polling is required
when the dashboard is bound to the same process as the server.
"""

from __future__ import annotations

import time

from rich.align import Align
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from membrane.cli.formatters import fmt_bytes, fmt_duration
from membrane.server import Server

console = Console()


def _header(diag) -> Panel:
    """Render the local-dashboard header."""
    status = "[green]HEALTHY[/green]" if diag.load < 0.9 else "[yellow]WARNING[/yellow]"
    text = Text.assemble(
        "Membrane Server  |  ",
        f"Node: {diag.node_id}  |  ",
        f"Uptime: {fmt_duration(diag.uptime_seconds)}  |  ",
        f"Status: {status}",
    )
    return Panel(Align.center(text), style="bold white on blue")


def _metrics(diag) -> Panel:
    """Render the local-dashboard metrics table."""
    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Value", style="magenta", no_wrap=True)
    table.add_row("Memory Used", fmt_bytes(diag.memory_used_bytes))
    table.add_row("Memory Limit", fmt_bytes(diag.memory_limit_bytes))
    table.add_row("Fragments", str(diag.fragment_count))
    table.add_row("Primaries", str(diag.primary_count))
    table.add_row("Connected Peers", str(diag.connected_nodes))
    table.add_row("Compute Backend", diag.backend_name)
    table.add_row("Redis", "[green]ON[/green]" if diag.redis_connected else "[red]OFF[/red]")
    table.add_row("Load", f"{diag.load:.2%}")
    table.add_row("Requests", str(diag.request_count))
    table.add_row("Errors", str(diag.error_count))
    return Panel(table, title="[bold]Server Metrics[/bold]", border_style="green")


def _peers(server: Server) -> Panel:
    """Render the connected-peers panel."""
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


def _events(server: Server) -> Panel:
    """Render the recent-events panel (newest first)."""
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
    # Reverse so the newest event is at the top of the table.
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


def _footer() -> Panel:
    """Render the local-dashboard footer."""
    text = Text("[Ctrl+C] Stop server  |  Live Dashboard", style="dim")
    return Panel(Align.center(text), style="dim")


def run_dashboard(server: Server) -> None:
    """Render a live Rich dashboard for the local ``Server``.

    Unlike :mod:`membrane.cli.poll` (which polls a remote server over
    HTTP), this version has full access to the in-process
    :class:`Server`'s diagnostics and event log. It is the default
    when ``membrane serve`` is invoked without ``--daemon``.
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
            while server._running:
                diag = server.diagnostics()
                layout["header"].update(_header(diag))
                layout["metrics"].update(_metrics(diag))
                layout["peers"].update(_peers(server))
                layout["right"].update(_events(server))
                layout["footer"].update(_footer())
                time.sleep(0.5)
        except KeyboardInterrupt:
            pass
        finally:
            server.stop()
            console.print("\n[bold red]Server stopped.[/bold red]")


__all__ = ["run_dashboard"]
