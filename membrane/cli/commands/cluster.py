"""``membrane cluster-status`` command.

Connects to a remote Membrane server's ``/peers`` endpoint and
renders the cluster membership as a Rich table.
"""

from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

from membrane.cli.poll import fetch_json

console = Console()


def main(
    host: str = typer.Option("localhost", "--host", help="Server host"),
    port: int = typer.Option(8080, "--port", "-p", help="Server port"),
) -> None:
    """Show cluster membership and peer health."""
    data = fetch_json(host, port, "/peers")
    if not data:
        console.print(f"[red]Could not fetch cluster status from http://{host}:{port}/peers[/red]")
        raise typer.Exit(1)

    peers = data.get("peers", [])
    if not peers:
        console.print("[dim]No peers connected.[/dim]")
        return

    table = Table(title="Cluster Peers", box=None)
    table.add_column("Node ID", style="cyan")
    table.add_column("Host", style="magenta")
    table.add_column("Port", style="magenta")
    table.add_column("Healthy", style="green")
    for p in peers:
        is_healthy = "[green]YES[/green]" if p.get("healthy") else "[red]NO[/red]"
        table.add_row(
            p.get("node_id", "?"),
            p.get("host", "?"),
            str(p.get("port", "?")),
            is_healthy,
        )
    console.print(table)


__all__ = ["main"]
