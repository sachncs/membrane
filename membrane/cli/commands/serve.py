"""``membrane serve`` command.

Starts a production :class:`~membrane.server.Server` and (by default)
launches the local TUI dashboard. The interactive setup wizard is
invoked automatically when ``membrane serve`` is run with all
defaults in a TTY.
"""

from __future__ import annotations

import logging
import sys
from typing import Annotated

import typer
from rich.console import Console

from membrane.cli.dashboard import run_dashboard
from membrane.cli.formatters import fmt_bytes
from membrane.cli.wizard import interactive_setup
from membrane.network.config import ClusterConfig
from membrane.node import Node
from membrane.server import Server

console = Console()


def main(
    node_id: str = typer.Option("membrane-0", "--node-id", "-n", help="Node identifier"),
    host: str = typer.Option("0.0.0.0", "--host", "-h", help="Bind address"),
    port: int = typer.Option(8080, "--port", "-p", help="Listen port"),
    transport: str = typer.Option("http", "--transport", "-t", help="Transport: http or grpc"),
    compute: str = typer.Option(
        "cpu", "--compute", "-c", help="Compute: cpu, gpu, ollama, openai, anthropic, transformers"
    ),
    redis_url: str = typer.Option("", "--redis", "-r", help="Redis URL (e.g. redis://localhost:6379/0)"),
    max_memory: int = typer.Option(1 << 30, "--max-memory", "-m", help="Max memory bytes"),
    log_level: str = typer.Option("INFO", "--log-level", "-l", help="Logging level"),
    daemon: bool = typer.Option(False, "--daemon", "-d", help="Run as daemon (no dashboard)"),
    interactive: bool = typer.Option(False, "--interactive", "-i", help="Interactive setup wizard"),
    peer: Annotated[list[str] | None, typer.Option(help="Seed peer host:port (repeatable)")] = None,
    heartbeat_interval: float = typer.Option(2.0, "--heartbeat-interval", help="Heartbeat interval seconds"),
    gossip_interval: float = typer.Option(5.0, "--gossip-interval", help="Gossip interval seconds"),
    replica_count: int = typer.Option(2, "--replica-count", help="Replicas per fragment"),
    failure_remove_threshold: int = typer.Option(
        4, "--failure-remove-threshold", help="Missed heartbeats before removing peer"
    ),
    llm_url: str = typer.Option("", "--llm-url", help="Base URL for Ollama or custom OpenAI endpoint"),
    llm_model: str = typer.Option("", "--llm-model", help="Model name (e.g. llama3.2, gpt-4o-mini, claude-3-sonnet)"),
    api_key: str = typer.Option("", "--api-key", help="API key for OpenAI / Anthropic"),
) -> None:
    """Start a Membrane production server.

    When invoked with all defaults in a TTY, the interactive setup
    wizard is launched automatically. Pass ``--interactive`` explicitly
    to force the wizard; pass ``--daemon`` to run without the TUI
    dashboard.
    """
    # Decide whether to launch the interactive wizard.
    defaults_match = all(
        v == default
        for v, default in [
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
        ]
    )
    if interactive or (sys.stdin.isatty() and defaults_match):
        cfg = interactive_setup()
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

    peer_list: list[str] = peer or []

    # Only build cluster config when at least one seed peer is
    # supplied — single-node mode skips cluster bootstrapping.
    cluster_config = None
    if peer_list:
        cluster_config = ClusterConfig(
            node_id=node_id,
            host=host,
            port=port,
            peers=peer_list,
            heartbeat_interval_sec=heartbeat_interval,
            gossip_interval_sec=gossip_interval,
            replica_count=replica_count,
            failure_remove_threshold=failure_remove_threshold,
        )

    node = Node(node_id=node_id, max_memory_bytes=max_memory)
    server = Server(
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
    console.print(f"  Peers    : {', '.join(peer_list) if peer_list else 'none'}")
    console.print(f"  Max Mem  : {fmt_bytes(max_memory)}")

    if daemon:
        console.print("[dim]Running in daemon mode. Press Ctrl+C to stop.[/dim]")
        try:
            server.join()
        except KeyboardInterrupt:
            server.stop()
            console.print("[bold red]Server stopped.[/bold red]")
    else:
        # Launch the local TUI dashboard.
        run_dashboard(server)


__all__ = ["main"]
