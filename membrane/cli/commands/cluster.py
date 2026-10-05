"""``membrane cluster-status`` command.

Connects to a remote Membrane server's ``/peers`` endpoint and
renders the cluster membership as a Rich table.
"""

import typer
from rich.table import Table

from membrane.cli import output
from membrane.cli.dashboard import fetch_json


def main(
    host: str = typer.Option("localhost", "--host", help="Server host"),
    port: int = typer.Option(8080, "--port", "-p", help="Server port"),
) -> None:
    """Show cluster membership and peer health.

    Args:
        host: Server host.
        port: Server port.
    """
    data = fetch_json(host, port, "/peers")
    if not data:
        output.error(f"Could not fetch cluster status from http://{host}:{port}/peers")
        raise typer.Exit(1)

    peers = data.get("peers", [])
    if not peers:
        output.result("No peers connected.")
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
    output.result(table)


__all__ = ["main"]
