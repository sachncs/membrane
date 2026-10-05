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

import logging
import sys
from typing import Annotated

import typer
from rich.markup import escape

from membrane.cli import output
from membrane.cli.dashboard import run_dashboard
from membrane.cli.formatters import fmt_bytes
from membrane.cli.wizard import interactive_setup
from membrane.logging import configure_logging
from membrane.runtime.lifecycle import run_until_signalled
from membrane.runtime.settings import ServerSettings, SettingsError, build_server
from membrane.transport.limits import TransportLimits

logger = logging.getLogger(__name__)


def split_list(values: list[str] | None) -> list[str]:
    """Flatten repeatable options that may also carry comma-separated env values.

    Args:
        values: Option values; each may hold several comma-separated items.

    Returns:
        list[str]: The individual, stripped items.
    """
    result: list[str] = []
    for value in values or []:
        result.extend(item.strip() for item in value.split(",") if item.strip())
    return result


def fail(message: str) -> typer.Exit:
    """Log a startup error and return the exit to raise.

    Args:
        message: Plain-text error message.

    Returns:
        typer.Exit: Exit with status 2 (configuration error).
    """
    output.error(f"[bold red]{escape(message)}[/bold red]")
    return typer.Exit(2)


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
        help="Compute plugin: cpu, gpu, ollama, openai, anthropic, transformers, or an installed plugin",
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
    content_store: str = typer.Option(
        "filesystem",
        "--content-store",
        envvar="MEMBRANE_CONTENT_STORE",
        help="Content-store plugin for --data-dir (filesystem, memory, or an installed plugin)",
    ),
    persistence: str = typer.Option(
        "",
        "--persistence",
        envvar="MEMBRANE_PERSISTENCE",
        help="Persistence plugin (redis, memory, or installed); default: redis with --redis, else memory",
    ),
    eviction: str = typer.Option(
        "weighted-lru",
        "--eviction",
        envvar="MEMBRANE_EVICTION",
        help="Eviction policy plugin: weighted-lru, tinylfu, or installed",
    ),
    no_hooks: bool = typer.Option(
        False, "--no-hooks", envvar="MEMBRANE_NO_HOOKS", help="Do not run installed membrane.hooks plugins"
    ),
    max_memory: int = typer.Option(
        1 << 30, "--max-memory", "-m", envvar="MEMBRANE_MAX_MEMORY", help="Max memory bytes"
    ),
    log_level: str = typer.Option("INFO", "--log-level", "-l", envvar="MEMBRANE_LOG_LEVEL", help="Logging level"),
    log_format: str = typer.Option(
        "text",
        "--log-format",
        envvar="MEMBRANE_LOG_FORMAT",
        help="Diagnostics format: text or json (one object per line)",
    ),
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
        help="Inbound API keyfile; one 'sha256:<digest>:<subject>:<scope,...>' per line (membrane keys generate)",
    ),
    authenticator: str = typer.Option(
        "apikey",
        "--authenticator",
        envvar="MEMBRANE_AUTHENTICATOR",
        help="Authenticator plugin used with --auth-config",
    ),
    auth_config: str = typer.Option(
        "", "--auth-config", envvar="MEMBRANE_AUTH_CONFIG", help="Configuration file for the authenticator plugin"
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
    drain_timeout: float = typer.Option(
        30.0,
        "--drain-timeout",
        envvar="MEMBRANE_DRAIN_TIMEOUT",
        help="Seconds SIGTERM may spend draining (503 readiness, hand off primaries) before exit",
    ),
    max_concurrency: int = typer.Option(
        64,
        "--max-concurrency",
        envvar="MEMBRANE_MAX_CONCURRENCY",
        help="Requests handled at once; excess requests get 503 + Retry-After (0: unbounded)",
    ),
    rate_limit: float = typer.Option(
        0.0,
        "--rate-limit",
        envvar="MEMBRANE_RATE_LIMIT",
        help="Requests per second per credential; excess requests get 429 (0: off)",
    ),
    rate_limit_burst: int = typer.Option(
        0, "--rate-limit-burst", envvar="MEMBRANE_RATE_LIMIT_BURST", help="Rate-limit burst size (0: twice the rate)"
    ),
    max_connections: int = typer.Option(
        0, "--max-connections", envvar="MEMBRANE_MAX_CONNECTIONS", help="Open connections accepted (0: unlimited)"
    ),
    keep_alive_timeout: float = typer.Option(
        5.0, "--keep-alive-timeout", envvar="MEMBRANE_KEEP_ALIVE_TIMEOUT", help="Idle keep-alive timeout seconds"
    ),
    enable_api_docs: bool = typer.Option(
        False,
        "--enable-api-docs",
        envvar="MEMBRANE_ENABLE_API_DOCS",
        help="Serve /openapi.json (requires the read scope)",
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

    Args:
        node_id: Node identifier.
        host: Bind address (0.0.0.0 for all interfaces).
        port: Listen port.
        transport: Transport (only 'http' is supported).
        compute: Compute plugin name.
        redis_url: Redis URL (e.g. redis://localhost:6379/0).
        data_dir: Keep KV bytes on disk (encrypted) here so they survive
            restarts.
        data_key_file: 32-byte (or 64-hex) key for --data-dir; default:
            generated into <data-dir>/master.key.
        content_store: Content-store plugin for --data-dir.
        persistence: Persistence plugin.
        eviction: Eviction policy plugin.
        no_hooks: Do not run installed hook plugins.
        max_memory: Max memory bytes.
        log_level: Logging level.
        log_format: Diagnostics format: text or json (one object per line).
        daemon: Run without the dashboard (implied when stdout is not a
            TTY).
        interactive: Interactive setup wizard.
        peer: Seed peer host:port (repeatable or comma-separated).
        advertise_host: Host peers use to reach this node (default: bind
            host, or FQDN for 0.0.0.0).
        peer_network: CIDR the cluster's peers live in, exempt from the SSRF
            private-IP block (repeatable).
        heartbeat_interval: Heartbeat interval seconds.
        gossip_interval: Gossip interval seconds.
        replica_count: Replicas per fragment.
        failure_remove_threshold: Missed heartbeats before removing peer.
        consistency: Default write consistency: strong, quorum or eventual.
        quorum_count: Peer acks a strong/quorum write waits for (cluster
            needs quorum_count + 1 nodes).
        api_key_file: Inbound API keyfile; one
            'sha256:<digest>:<subject>:<scope,...>' per line.
        authenticator: Authenticator plugin used with --auth-config.
        auth_config: Configuration file for the authenticator plugin.
        peer_api_key_file: File holding the bearer key this node presents to
            its peers (needs admin scope).
        tls_cert: mTLS certificate PEM file.
        tls_key: mTLS private key PEM file.
        tls_ca: mTLS CA bundle PEM file.
        tls_allowed_cn: Peer certificate CN to accept (repeatable).
        tls_allow_any_cn: Accept any CN signed by the CA (development only).
        allow_unauthenticated: Serve without authentication on a
            non-loopback address (unsafe).
        drain_timeout: Seconds SIGTERM may spend draining.
        max_concurrency: Requests handled at once (0: unbounded).
        rate_limit: Requests per second per credential (0: off).
        rate_limit_burst: Rate-limit burst size (0: twice the rate).
        max_connections: Open connections accepted (0: unlimited).
        keep_alive_timeout: Idle keep-alive timeout seconds.
        enable_api_docs: Serve /openapi.json behind the read scope.
        llm_url: Base URL for Ollama or custom OpenAI endpoint.
        llm_model: Model name (e.g. llama3.2, gpt-4o-mini).
        api_key: API key for the OpenAI / Anthropic compute backend.
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

    if log_format not in ("text", "json"):
        raise fail("--log-format must be 'text' or 'json'.")
    configure_logging(level=log_level, json_mode=log_format == "json", force=True)

    try:
        settings = ServerSettings(
            node_id=node_id,
            host=host,
            port=port,
            transport=transport,
            compute=compute,
            llm_url=llm_url,
            llm_model=llm_model,
            llm_api_key=api_key,
            redis_url=redis_url,
            data_dir=data_dir,
            data_key_file=data_key_file,
            content_store=content_store,
            persistence=persistence,
            eviction=eviction,
            load_hooks=not no_hooks,
            max_memory=max_memory,
            peers=tuple(split_list(peer)),
            advertise_host=advertise_host,
            peer_networks=tuple(split_list(peer_network)),
            heartbeat_interval=heartbeat_interval,
            gossip_interval=gossip_interval,
            replica_count=replica_count,
            failure_remove_threshold=failure_remove_threshold,
            consistency=consistency,
            quorum_count=quorum_count,
            api_key_file=api_key_file,
            authenticator=authenticator,
            auth_config=auth_config,
            peer_api_key_file=peer_api_key_file,
            tls_cert=tls_cert,
            tls_key=tls_key,
            tls_ca=tls_ca,
            tls_allowed_cns=tuple(split_list(tls_allowed_cn)),
            tls_allow_any_cn=tls_allow_any_cn,
            allow_unauthenticated=allow_unauthenticated,
            drain_timeout=drain_timeout,
            limits=TransportLimits(
                max_concurrency=max_concurrency,
                rate_limit_per_sec=rate_limit,
                rate_limit_burst=rate_limit_burst,
                max_connections=max_connections or None,
                keep_alive_timeout_sec=keep_alive_timeout,
                enable_api_docs=enable_api_docs,
            ),
        )
        server, auth_mode = build_server(settings)
    except SettingsError as exc:
        raise fail(str(exc)) from exc

    server.start()
    output.info(
        "\n".join(
            [
                f"[bold green]Membrane server started[/bold green] on {host}:{port}",
                f"  Node ID  : {node_id}",
                f"  Auth     : {auth_mode}",
                f"  Compute  : {compute}",
                f"  LLM      : {llm_url or 'default'} / {llm_model or 'default'}",
                f"  Redis    : {redis_url or 'disabled (in-memory)'}",
                f"  Data dir : {data_dir or 'none (KV bytes in memory)'}",
                f"  Peers    : {', '.join(settings.peers) or 'none'}",
                f"  Max Mem  : {fmt_bytes(max_memory)}",
                f"  Limits   : {max_concurrency or 'unbounded'} concurrent, "
                f"{f'{rate_limit:g}/s per key' if rate_limit else 'no rate limit'}",
            ]
        )
    )

    if daemon or not sys.stdout.isatty():
        # Containers and service managers stop the process with SIGTERM:
        # drain (readiness 503, hand off primaries, leave) before exiting.
        output.info("[dim]Running in daemon mode. Send SIGTERM or press Ctrl+C to drain and stop.[/dim]")
        run_until_signalled(server, settings.drain_timeout)
        output.info("Server stopped.")
    else:
        # Launch the local TUI dashboard.
        run_dashboard(server)


__all__ = ["main"]
