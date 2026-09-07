"""``membrane config`` command.

Prints the static configuration (defaults) and runtime environment.
Useful for confirming that the installed package matches expectations
on a target host.
"""

from __future__ import annotations

import sys

import typer
from rich.console import Console
from rich.table import Table

from membrane.cli.formatters import fmt_bytes

console = Console()


def main(
    show: bool = typer.Option(True, "--show", help="Display current config"),
) -> None:
    """Show Membrane configuration and environment."""
    table = Table(title="Membrane Configuration", box=None)
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="magenta")
    table.add_row("Package", "membrane")
    table.add_row("Version", "0.1.0")
    table.add_row("Python", sys.version.split()[0])
    table.add_row("Platform", sys.platform)
    table.add_row("Max Memory Default", fmt_bytes(1 << 30))
    table.add_row("Default Transport", "http")
    table.add_row("Default Compute", "cpu")
    table.add_row("Default Port", "8080")
    console.print(table)


__all__ = ["main"]
