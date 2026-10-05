"""``membrane serve`` command.

Starts a production :class:`~membrane.server.Server` and (by default)
launches the local TUI dashboard. The interactive setup wizard is
invoked automatically when ``membrane serve`` is run with all
defaults in a TTY.

Every option can also be set through the ``MEMBRANE_*`` environment
variable shown in ``--help``, which is how the container image and
the Kubernetes / Compose manifests configure the server. Secrets
(API keys, TLS keys) are read from files so they never appear in the
process arguments.

Security posture: the server refuses to listen on a non-loopback
address without inbound authentication (``--api-key-file`` or mTLS)
unless ``--allow-unauthenticated`` is passed explicitly.
"""

from __future__ import annotations

import ipaddress
import logging
import signal
import sys
from pathlib import Path
from types import FrameType
from typing import Annotated

import typer
from rich.console import Console

from membrane.cli.dashboard import run_dashboard
from membrane.cli.formatters import fmt_bytes
from membrane.cli.wizard import interactive_setup
from membrane.network.config import ClusterConfig
from membrane.node import Node
from membrane.server import Server
from membrane.transport.tls import MTLSConfig

console = Console()
logger = logging.getLogger(__name__)


def _is_loopback(host: str) -> bool:
    """Return True when ``host`` only accepts local connections."""
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _read_secret(path: str, what: str) -> str:
    """Read a secret file, exiting with a clear message on failure."""
    try:
        return Path(path).read_text()
    except OSError as exc:
        console.print(f"[bold red]Cannot read {what} file {path!r}: {exc}[/bold red]")
        raise typer.Exit(2) from exc


def _split_list(values: list[str] | None) -> list[str]:
    """Flatten repeatable options that may also carry comma-separated env values."""
    result: list[str] = []
    for value in values or []:
        result.extend(item.strip() for item in value.split(",") if item.strip())
    return result


def _build_tls(cert: str, key: str, ca: str, allowed_cns: list[str], allow_any_cn: bool) -> MTLSConfig | None:
    """Build the mTLS configuration from file paths, or ``None`` when unset."""
    if not (cert or key or ca):
        return None
    if not (cert and key and ca):
        console.print("[bold red]--tls-cert, --tls-key and --tls-ca must be given together.[/bold red]")
        raise typer.Exit(2)
    cert_pem = _read_secret(cert, "TLS certificate")
    key_pem = _read_secret(key, "TLS key")
    ca_pem = _read_secret(ca, "TLS CA bundle")
    if allow_any_cn:
        return MTLSConfig.allow_all_signed_by_ca(
            server_cert_pem=cert_pem,
            server_key_pem=key_pem,
            ca_bundle_pem=ca_pem,
            client_cert_pem=cert_pem,
            client_key_pem=key_pem,
        )
    if not allowed_cns:
        console.print("[bold red]mTLS needs --tls-allowed-cn (repeatable) or --tls-allow-any-cn.[/bold red]")
        raise typer.Exit(2)
    return MTLSConfig(
        server_cert_pem=cert_pem,
        server_key_pem=key_pem,
        ca_bundle_pem=ca_pem,
        allowed_cns=frozenset(allowed_cns),
        client_cert_pem=cert_pem,
        client_key_pem=key_pem,
    )


def main(
    node_id: str = typer.Option("membrane-0", "--node-id", "-n", envvar="MEMBRANE_NODE_ID", help="Node identifier"),
    host: str = typer.Option(
        "127.0.0.1", "--host", "-h", envvar="MEMBRANE_HOST", help="Bind address (0.0.0.0 for all interfaces)"
    ),
    port: int = typer.Option(8080, "--port", "-p", envvar="MEMBRANE_PORT", help="Listen port"),
    transport: str = typer.Option(
        "http", "--transport", "-t", envvar="MEMBRANE_TRANSPORT", help="Transport (only 'http' is supported)"
    ),
    compute: str = typer.Option(
        "cpu",
        "--compute",
        "-c",
        envvar="MEMBRANE_COMPUTE",
        help="Compute: cpu, gpu, ollama, openai, anthropic, transformers",
    ),
    redis_url: str = typer.Option(
        "", "--redis", "-r", envvar="MEMBRANE_REDIS_URL", help="Redis URL (e.g. redis://localhost:6379/0)"
    ),
    data_dir: str = typer.Option(
        "",
        "--data-dir",
        envvar="MEMBRANE_DATA_DIR",
        help="Keep KV bytes on disk (encrypted) here so they survive restarts",
    ),
    data_key_file: str = typer.Option(
        "",
        "--data-key-file",
        envvar="MEMBRANE_DATA_KEY_FILE",
        help="32-byte (or 64-hex) key for --data-dir; default: generated into <data-dir>/master.key",
    ),
    max_memory: int = typer.Option(
        1 << 30, "--max-memory", "-m", envvar="MEMBRANE_MAX_MEMORY", help="Max memory bytes"
    ),
    log_level: str = typer.Option("INFO", "--log-level", "-l", envvar="MEMBRANE_LOG_LEVEL", help="Logging level"),
    daemon: bool = typer.Option(
        False,
        "--daemon",
        "-d",
        envvar="MEMBRANE_DAEMON",
        help="Run without the dashboard (implied when stdout is not a TTY)",
    ),
    interactive: bool = typer.Option(False, "--interactive", "-i", help="Interactive setup wizard"),
    peer: Annotated[
        list[str] | None,
        typer.Option(envvar="MEMBRANE_PEERS", help="Seed peer host:port (repeatable or comma-separated)"),
    ] = None,
    advertise_host: str = typer.Option(
        "",
        "--advertise-host",
        envvar="MEMBRANE_ADVERTISE_HOST",
        help="Host peers use to reach this node (default: bind host, or FQDN for 0.0.0.0)",
    ),
    peer_network: Annotated[
        list[str] | None,
        typer.Option(
            envvar="MEMBRANE_PEER_NETWORKS",
            help="CIDR the cluster's peers live in, exempt from the SSRF private-IP block (repeatable)",
        ),
    ] = None,
    heartbeat_interval: float = typer.Option(
        2.0, "--heartbeat-interval", envvar="MEMBRANE_HEARTBEAT_INTERVAL", help="Heartbeat interval seconds"
    ),
    gossip_interval: float = typer.Option(
        5.0, "--gossip-interval", envvar="MEMBRANE_GOSSIP_INTERVAL", help="Gossip interval seconds"
    ),
    replica_count: int = typer.Option(
        2, "--replica-count", envvar="MEMBRANE_REPLICA_COUNT", help="Replicas per fragment"
    ),
    failure_remove_threshold: int = typer.Option(
        4,
        "--failure-remove-threshold",
        envvar="MEMBRANE_FAILURE_REMOVE_THRESHOLD",
        help="Missed heartbeats before removing peer",
    ),
    consistency: str = typer.Option(
        "strong",
        "--consistency",
        envvar="MEMBRANE_CONSISTENCY",
        help="Default write consistency: strong, quorum or eventual",
    ),
    quorum_count: int = typer.Option(
        2,
        "--quorum-count",
        envvar="MEMBRANE_QUORUM_COUNT",
        help="Peer acks a strong/quorum write waits for (cluster needs quorum_count + 1 nodes)",
    ),
    api_key_file: str = typer.Option(
        "",
        "--api-key-file",
        envvar="MEMBRANE_API_KEY_FILE",
        help="Inbound API keyfile; one '<key>:<subject>:<scope,...>' per line",
    ),
    peer_api_key_file: str = typer.Option(
        "",
        "--peer-api-key-file",
        envvar="MEMBRANE_PEER_API_KEY_FILE",
        help="File holding the bearer key this node presents to its peers (needs admin scope)",
    ),
    tls_cert: str = typer.Option("", "--tls-cert", envvar="MEMBRANE_TLS_CERT_FILE", help="mTLS certificate PEM file"),
    tls_key: str = typer.Option("", "--tls-key", envvar="MEMBRANE_TLS_KEY_FILE", help="mTLS private key PEM file"),
    tls_ca: str = typer.Option("", "--tls-ca", envvar="MEMBRANE_TLS_CA_FILE", help="mTLS CA bundle PEM file"),
    tls_allowed_cn: Annotated[
        list[str] | None,
        typer.Option(envvar="MEMBRANE_TLS_ALLOWED_CNS", help="Peer certificate CN to accept (repeatable)"),
    ] = None,
    tls_allow_any_cn: bool = typer.Option(
        False,
        "--tls-allow-any-cn",
        envvar="MEMBRANE_TLS_ALLOW_ANY_CN",
        help="Accept any CN signed by the CA (development only)",
    ),
    allow_unauthenticated: bool = typer.Option(
        False,
        "--allow-unauthenticated",
        envvar="MEMBRANE_ALLOW_UNAUTHENTICATED",
        help="Serve without authentication on a non-loopback address (unsafe)",
    ),
    llm_url: str = typer.Option(
        "", "--llm-url", envvar="MEMBRANE_LLM_URL", help="Base URL for Ollama or custom OpenAI endpoint"
    ),
    llm_model: str = typer.Option(
        "", "--llm-model", envvar="MEMBRANE_LLM_MODEL", help="Model name (e.g. llama3.2, gpt-4o-mini)"
    ),
    api_key: str = typer.Option(
        "", "--api-key", envvar="MEMBRANE_LLM_API_KEY", help="API key for the OpenAI / Anthropic compute backend"
    ),
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
            (host, "127.0.0.1"),
            (port, 8080),
            (transport, "http"),
            (compute, "cpu"),
            (redis_url, ""),
            (max_memory, 1 << 30),
            (log_level, "INFO"),
            (llm_url, ""),
            (llm_model, ""),
            (api_key, ""),
            (api_key_file, ""),
            (tls_cert, ""),
            (allow_unauthenticated, False),
        ]
    )
    if interactive or (sys.stdin.isatty() and sys.stdout.isatty() and defaults_match and not daemon):
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

    peer_list = _split_list(peer)
    tls = _build_tls(tls_cert, tls_key, tls_ca, _split_list(tls_allowed_cn), tls_allow_any_cn)

    authenticator = None
    if api_key_file:
        from membrane.auth.apikey import APIKeyAuthenticator

        authenticator = APIKeyAuthenticator(_read_secret(api_key_file, "API keyfile"))
        if not authenticator.keys:
            console.print(f"[bold red]API keyfile {api_key_file!r} contains no valid keys.[/bold red]")
            raise typer.Exit(2)

    if authenticator is None and tls is None and not _is_loopback(host):
        if not allow_unauthenticated:
            console.print(
                f"[bold red]Refusing to serve unauthenticated on {host}.[/bold red] "
                "Configure --api-key-file or mTLS (--tls-cert/--tls-key/--tls-ca), "
                "bind to 127.0.0.1, or pass --allow-unauthenticated."
            )
            raise typer.Exit(2)
        logger.warning("Serving WITHOUT authentication on %s:%s (--allow-unauthenticated)", host, port)

    peer_api_key = _read_secret(peer_api_key_file, "peer API key").strip() if peer_api_key_file else ""
    if peer_list and authenticator is not None and tls is None:
        if not peer_api_key:
            console.print(
                "[bold red]A cluster using API keys needs --peer-api-key-file so peers can authenticate.[/bold red]"
            )
            raise typer.Exit(2)
        # Peers replicate every tenant's fragments and propagate
        # deletes, which the receiving node only allows for admin.
        peer_record = authenticator.keys.get(peer_api_key)
        if peer_record is None:
            logger.warning("Peer API key is not in the local keyfile; peers must share a keyfile containing it")
        elif "admin" not in peer_record.scopes:
            console.print("[bold red]The peer API key must carry the 'admin' scope.[/bold red]")
            raise typer.Exit(2)

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
            mtls=tls,
            advertise_host=advertise_host,
            default_consistency=consistency,
            quorum_count=quorum_count,
        )

    content_store = None
    if data_dir:
        from membrane.server import build_content_store

        try:
            content_store = build_content_store(data_dir, data_key_file)
        except (OSError, ValueError) as exc:
            console.print(f"[bold red]Cannot open data directory {data_dir!r}: {exc}[/bold red]")
            raise typer.Exit(2) from exc
    node = Node(node_id=node_id, max_memory_bytes=max_memory, content_store=content_store)
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
        authenticator=authenticator,
        peer_api_key=peer_api_key,
        peer_networks=tuple(_split_list(peer_network)),
        tls=tls,
    )

    if redis_url and not server.durable:
        console.print(
            f"[bold red]Redis at {redis_url} is unreachable; refusing to start without the requested durability.[/bold red]"
        )
        raise typer.Exit(2)

    server.start()
    if tls is not None:
        auth_mode = "mTLS"
    elif authenticator is not None:
        auth_mode = "API key"
    else:
        auth_mode = "none (loopback only)" if _is_loopback(host) else "NONE (--allow-unauthenticated)"
    console.print(f"[bold green]Membrane server started[/bold green] on {host}:{port}")
    console.print(f"  Node ID : {node_id}")
    console.print(f"  Transport: {transport}")
    console.print(f"  Auth     : {auth_mode}")
    console.print(f"  Compute  : {compute}")
    console.print(f"  LLM URL  : {llm_url or 'default'}")
    console.print(f"  LLM Model: {llm_model or 'default'}")
    console.print(f"  Redis    : {redis_url or 'disabled (in-memory)'}")
    console.print(f"  Data dir : {data_dir or 'none (KV bytes in memory)'}")
    console.print(f"  Peers    : {', '.join(peer_list) if peer_list else 'none'}")
    console.print(f"  Max Mem  : {fmt_bytes(max_memory)}")

    if daemon or not sys.stdout.isatty():
        # Containers and service managers stop the process with
        # SIGTERM; shut down gracefully instead of dying mid-write.
        def _on_sigterm(_signum: int, _frame: FrameType | None) -> None:
            logger.info("SIGTERM received; shutting down")
            server.stop()

        signal.signal(signal.SIGTERM, _on_sigterm)
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
