"""Membrane CLI: production command-line interface with live dashboard.

Commands:
  membrane serve        Start a Membrane server
  membrane status       Show server status and metrics
  membrane dashboard    Open live TUI dashboard
  membrane config       Show current configuration

Example:
  membrane serve --node-id n1 --port 8080 --transport http --compute gpu
  membrane dashboard --host localhost --port 8080
"""

import logging
import sys
import time
from pathlib import Path
from typing import Any

import typer
from rich.align import Align
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TextColumn
from rich.table import Table
from rich.text import Text

from membrane.membrane_node import MembraneNode
from membrane.network.config import ClusterConfig
from membrane.server import MembraneServer

app = typer.Typer(
    name="membrane",
    help="Membrane — Global Contextual Memory Fabric CLI",
    no_args_is_help=True,
)
console = Console()


# ------------------------------------------------------------------
# Helper: pretty byte formatting
# ------------------------------------------------------------------

def fmt_bytes(n: int) -> str:
    for unit in ["B", "KiB", "MiB", "GiB", "TiB"]:
        if abs(n) < 1024.0:
            return f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} PiB"


def fmt_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.0f}m"
    return f"{seconds / 3600:.1f}h"


# ------------------------------------------------------------------
# Interactive setup wizard
# ------------------------------------------------------------------

def _interactive_setup() -> dict[str, Any]:
    """Ask the user configuration questions interactively."""
    console.print("[bold cyan]Membrane Setup Wizard[/bold cyan]")
    console.print("Press Enter to accept defaults (shown in brackets).\n")

    def ask(prompt: str, default: str = "") -> str:
        full = f"{prompt} [{default}]: " if default else f"{prompt}: "
        val = input(full).strip()
        return val if val else default

    def ask_bool(prompt: str, default: bool = True) -> bool:
        suffix = "Y/n" if default else "y/N"
        val = input(f"{prompt} [{suffix}]: ").strip().lower()
        if not val:
            return default
        return val in ("y", "yes", "true", "1")

    node_id = ask("Node ID", "membrane-0")
    host = ask("Bind host", "0.0.0.0")
    port = int(ask("Listen port", "8080"))

    transport = ask("Transport (http/grpc)", "http")
    while transport not in ("http", "grpc"):
        console.print("[red]Invalid transport. Choose 'http' or 'grpc'.[/red]")
        transport = ask("Transport (http/grpc)", "http")

    compute = ask("Compute backend (cpu/gpu/ollama/openai/anthropic/transformers)", "cpu")
    valid_backends = ("cpu", "gpu", "ollama", "openai", "anthropic", "transformers")
    while compute not in valid_backends:
        console.print("[red]Invalid compute. Choose one of: cpu, gpu, ollama, openai, anthropic, transformers.[/red]")
        compute = ask("Compute backend", "cpu")

    llm_url = ""
    llm_model = ""
    api_key = ""
    if compute == "ollama":
        llm_url = ask("Ollama URL", "http://localhost:11434")
        llm_model = ask("Ollama model", "llama3.2")
    elif compute == "openai":
        api_key = ask("OpenAI API key", "")
        llm_model = ask("OpenAI model", "gpt-4o-mini")
    elif compute == "anthropic":
        api_key = ask("Anthropic API key", "")
        llm_model = ask("Anthropic model", "claude-3-sonnet-20240229")
    elif compute == "transformers":
        llm_model = ask("HuggingFace model ID", "gpt2")

    use_redis = ask_bool("Use Redis persistence?")
    redis_url = ""
    if use_redis:
        redis_url = ask("Redis URL", "redis://localhost:6379/0")

    peers_input = ask("Seed peers (comma-separated host:port, or leave empty)", "")
    peers = [p.strip() for p in peers_input.split(",") if p.strip()]

    max_memory = int(ask("Max memory (bytes)", str(1 << 30)))
    log_level = ask("Log level", "INFO")

    return {
        "node_id": node_id,
        "host": host,
        "port": port,
        "transport": transport,
        "compute": compute,
        "llm_url": llm_url,
        "llm_model": llm_model,
        "api_key": api_key,
        "redis_url": redis_url,
        "peers": peers,
        "max_memory": max_memory,
        "log_level": log_level,
    }


# ------------------------------------------------------------------
# Command: serve
# ------------------------------------------------------------------

@app.command()
def serve(
    node_id: str = typer.Option("membrane-0", "--node-id", "-n", help="Node identifier"),
    host: str = typer.Option("0.0.0.0", "--host", "-h", help="Bind address"),
    port: int = typer.Option(8080, "--port", "-p", help="Listen port"),
    transport: str = typer.Option("http", "--transport", "-t", help="Transport: http or grpc"),
    compute: str = typer.Option("cpu", "--compute", "-c", help="Compute: cpu, gpu, ollama, openai, anthropic, transformers"),
    redis_url: str = typer.Option("", "--redis", "-r", help="Redis URL (e.g. redis://localhost:6379/0)"),
    max_memory: int = typer.Option(1 << 30, "--max-memory", "-m", help="Max memory bytes"),
    log_level: str = typer.Option("INFO", "--log-level", "-l", help="Logging level"),
    daemon: bool = typer.Option(False, "--daemon", "-d", help="Run as daemon (no dashboard)"),
    interactive: bool = typer.Option(False, "--interactive", "-i", help="Interactive setup wizard"),
    peer: list[str] = typer.Option([], "--peer", help="Seed peer host:port (repeatable)"),
    heartbeat_interval: float = typer.Option(2.0, "--heartbeat-interval", help="Heartbeat interval seconds"),
    gossip_interval: float = typer.Option(5.0, "--gossip-interval", help="Gossip interval seconds"),
    replica_count: int = typer.Option(2, "--replica-count", help="Replicas per fragment"),
    failure_remove_threshold: int = typer.Option(4, "--failure-remove-threshold", help="Missed heartbeats before removing peer"),
    llm_url: str = typer.Option("", "--llm-url", help="Base URL for Ollama or custom OpenAI endpoint"),
    llm_model: str = typer.Option("", "--llm-model", help="Model name (e.g. llama3.2, gpt-4o-mini, claude-3-sonnet)"),
    api_key: str = typer.Option("", "--api-key", help="API key for OpenAI / Anthropic"),
) -> None:
    """Start a Membrane production server."""
    if interactive or (sys.stdin.isatty() and all(v == default for v, default in [
        (node_id, "membrane-0"),
        (host, "0.0.0.0"),
        (port, 8080),
        (transport, "http"),
        (compute, "cpu"),
        (redis_url, ""),
        (max_memory, 1 << 30),
        (log_level, "INFO"),
        (llm_url, ""),
        (llm_model, ""),
        (api_key, ""),
    ])):
        cfg = _interactive_setup()
        node_id = cfg["node_id"]
        host = cfg["host"]
        port = cfg["port"]
        transport = cfg["transport"]
        compute = cfg["compute"]
        llm_url = cfg.get("llm_url", "")
        llm_model = cfg.get("llm_model", "")
        api_key = cfg.get("api_key", "")
        redis_url = cfg["redis_url"]
        peer = cfg.get("peers", [])
        max_memory = cfg["max_memory"]
        log_level = cfg["log_level"]

    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    cluster_config = None
    if peer:
        cluster_config = ClusterConfig(
            node_id=node_id,
            host=host,
            port=port,
            peers=peer,
            heartbeat_interval_sec=heartbeat_interval,
            gossip_interval_sec=gossip_interval,
            replica_count=replica_count,
            failure_remove_threshold=failure_remove_threshold,
        )

    node = MembraneNode(node_id=node_id, max_memory_bytes=max_memory)
    server = MembraneServer(
        node=node,
        transport=transport,
        compute=compute,
        redis_url=redis_url,
        host=host,
        port=port,
        cluster_config=cluster_config,
        llm_url=llm_url,
        llm_model=llm_model,
        api_key=api_key,
    )

    server.start()
    console.print(f"[bold green]Membrane server started[/bold green] on {host}:{port}")
    console.print(f"  Node ID : {node_id}")
    console.print(f"  Transport: {transport}")
    console.print(f"  Compute  : {compute}")
    console.print(f"  LLM URL  : {llm_url or 'default'}")
    console.print(f"  LLM Model: {llm_model or 'default'}")
    console.print(f"  Redis    : {redis_url or 'disabled (in-memory)'}")
    console.print(f"  Peers    : {', '.join(peer) if peer else 'none'}")
    console.print(f"  Max Mem  : {fmt_bytes(max_memory)}")

    if daemon:
        console.print("[dim]Running in daemon mode. Press Ctrl+C to stop.[/dim]")
        try:
            server.join()
        except KeyboardInterrupt:
            server.stop()
            console.print("[bold red]Server stopped.[/bold red]")
    else:
        # Launch live dashboard
        _run_dashboard(server)


# ------------------------------------------------------------------
# Command: dashboard (standalone)
# ------------------------------------------------------------------

@app.command()
def dashboard(
    host: str = typer.Option("localhost", "--host", help="Server host to monitor"),
    port: int = typer.Option(8080, "--port", "-p", help="Server port to monitor"),
    refresh: float = typer.Option(1.0, "--refresh", help="Refresh interval seconds"),
) -> None:
    """Open a live TUI dashboard connected to a running Membrane server."""
    import urllib.request

    def fetch_json(path: str) -> dict:
        url = f"http://{host}:{port}{path}"
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                import json
                return json.loads(resp.read().decode())
        except Exception:
            return {}

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

    def make_header(data: dict) -> Panel:
        node_id = data.get("node_id", "unknown")
        healthy = data.get("healthy", False)
        status = "[green]HEALTHY[/green]" if healthy else "[red]UNHEALTHY[/red]"
        text = Text.assemble(
            "Membrane Dashboard  |  ",
            f"Node: {node_id}  |  ",
            f"Status: {status}",
        )
        return Panel(Align.center(text), style="bold white on blue")

    def make_metrics(data: dict) -> Panel:
        table = Table(show_header=False, box=None)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")
        table.add_row("Memory Used", fmt_bytes(data.get("memory_used_bytes", 0)))
        table.add_row("Memory Limit", fmt_bytes(data.get("memory_limit_bytes", 0)))
        table.add_row("Fragment Count", str(data.get("fragment_count", 0)))
        table.add_row("Load", f"{data.get('load', 0.0):.2%}")
        return Panel(table, title="[bold]Metrics[/bold]", border_style="green")

    def make_connected(data: dict) -> Panel:
        # In standalone mode we don't have a local event log;
        # just show a placeholder or try to infer from metrics
        text = Text("Connect to a local server with 'membrane serve' for full diagnostics.")
        return Panel(text, title="[bold]Diagnostics[/bold]", border_style="yellow")

    def make_footer() -> Panel:
        text = Text("[Q]uit  |  Refresh: ", style="dim")
        return Panel(Align.center(text), style="dim")

    console.print("[bold cyan]Connecting to Membrane server...[/bold cyan]")
    with Live(layout, refresh_per_second=1 / refresh, screen=True) as live:
        while True:
            data = fetch_json("/heartbeat")
            layout["header"].update(make_header(data))
            layout["left"].update(make_metrics(data))
            layout["right"].update(make_connected(data))
            layout["footer"].update(make_footer())
            time.sleep(refresh)


# ------------------------------------------------------------------
# Internal: dashboard for local server
# ------------------------------------------------------------------

def _run_dashboard(server: MembraneServer) -> None:
    """Render a live Rich dashboard for the local MembraneServer."""
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

    def make_header(diag) -> Panel:
        status = "[green]HEALTHY[/green]" if diag.load < 0.9 else "[yellow]WARNING[/yellow]"
        text = Text.assemble(
            "Membrane Server  |  ",
            f"Node: {diag.node_id}  |  ",
            f"Uptime: {fmt_duration(diag.uptime_seconds)}  |  ",
            f"Status: {status}",
        )
        return Panel(Align.center(text), style="bold white on blue")

    def make_metrics(diag) -> Panel:
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

    def make_peers(server: MembraneServer) -> Panel:
        if not server.connected_nodes:
            return Panel("[dim]No peers connected[/dim]", title="[bold]Peers[/bold]", border_style="dim")
        table = Table(show_header=True, box=None, padding=(0, 1))
        table.add_column("Node ID", style="cyan")
        table.add_column("Status", style="green")
        for nid in server.connected_nodes:
            table.add_row(nid, "connected")
        return Panel(table, title="[bold]Peers[/bold]", border_style="blue")

    def make_events(server: MembraneServer) -> Panel:
        events = server.recent_events(n=15)
        if not events:
            return Panel("[dim]No events yet[/dim]", title="[bold]Event Log[/bold]", border_style="dim")
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

    def make_footer() -> Panel:
        text = Text("[Ctrl+C] Stop server  |  Live Dashboard", style="dim")
        return Panel(Align.center(text), style="dim")

    with Live(layout, refresh_per_second=2, screen=True) as live:
        try:
            while server._running:
                diag = server.diagnostics()
                layout["header"].update(make_header(diag))
                layout["metrics"].update(make_metrics(diag))
                layout["peers"].update(make_peers(server))
                layout["right"].update(make_events(server))
                layout["footer"].update(make_footer())
                time.sleep(0.5)
        except KeyboardInterrupt:
            pass
        finally:
            server.stop()
            console.print("\n[bold red]Server stopped.[/bold red]")


# ------------------------------------------------------------------
# Command: cluster status
# ------------------------------------------------------------------

@app.command(name="cluster-status")
def cluster_status(
    host: str = typer.Option("localhost", "--host", help="Server host"),
    port: int = typer.Option(8080, "--port", "-p", help="Server port"),
) -> None:
    """Show cluster membership and peer health."""
    import urllib.request

    url = f"http://{host}:{port}/peers"
    try:
        with urllib.request.urlopen(url, timeout=2) as resp:
            data = json.loads(resp.read().decode())
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
            healthy = "[green]YES[/green]" if p.get("healthy") else "[red]NO[/red]"
            table.add_row(p.get("node_id", "?"), p.get("host", "?"), str(p.get("port", "?")), healthy)
        console.print(table)
    except Exception as exc:
        console.print(f"[red]Could not fetch cluster status: {exc}[/red]")


# ------------------------------------------------------------------
# Command: llm-status
# ------------------------------------------------------------------

@app.command(name="llm-status")
def llm_status(
    host: str = typer.Option("localhost", "--host", help="Server host"),
    port: int = typer.Option(8080, "--port", "-p", help="Server port"),
) -> None:
    """Show active LLM backend status and model info."""
    import json
    import urllib.request

    url = f"http://{host}:{port}/metrics"
    try:
        with urllib.request.urlopen(url, timeout=2) as resp:
            data = json.loads(resp.read().decode())
        table = Table(title="LLM Backend Status", box=None)
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="magenta")
        table.add_row("Backend", data.get("backend_name", "unknown"))
        table.add_row("Node ID", data.get("node_id", "unknown"))
        table.add_row("Load", f"{data.get('load', 0.0):.2%}")
        table.add_row("Fragments", str(data.get("fragment_count", 0)))
        console.print(table)
    except Exception as exc:
        console.print(f"[red]Could not fetch LLM status: {exc}[/red]")


# ------------------------------------------------------------------
# Command: config
# ------------------------------------------------------------------

@app.command()
def config(
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


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

def main() -> None:
    app()


if __name__ == "__main__":
    main()
