"""``membrane llm-status`` command.

Connects to a remote Membrane server's ``/metrics.json`` endpoint and
renders a summary of the active compute backend.
"""

from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

from membrane.cli.poll import _fetch_json

console = Console()


def main(
    host: str = typer.Option("localhost", "--host", help="Server host"),
    port: int = typer.Option(8080, "--port", "-p", help="Server port"),
) -> None:
    """Show active LLM backend status and model info."""
    data = _fetch_json(host, port, "/metrics.json")
    if not data:
        console.print(f"[red]Could not fetch LLM status from http://{host}:{port}/metrics.json[/red]")
        raise typer.Exit(1)

    table = Table(title="LLM Backend Status", box=None)
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="magenta")
    table.add_row("Backend", data.get("backend_name", "unknown"))
    table.add_row("Node ID", data.get("node_id", "unknown"))
    table.add_row("Load", f"{data.get('load', 0.0):.2%}")
    table.add_row("Fragments", str(data.get("fragment_count", 0)))
    console.print(table)


__all__ = ["main"]
