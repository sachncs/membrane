"""Validated server settings and the startup policy that turns them into a server.

:class:`ServerSettings` is the single source of truth for a node's
configuration. ``membrane serve`` builds it from flags and ``MEMBRANE_*``
environment variables; tests and embedding code can build it directly.
Invalid values fail in the constructor, before anything is started.

:func:`build_server` applies the security policy:

* refuse to listen beyond loopback without inbound authentication unless
  ``allow_unauthenticated`` is set;
* refuse secret files that other users can read;
* require a peer key with the ``admin`` scope in an API-key cluster.

Every refusal raises :class:`SettingsError` with an actionable message.
"""

import ipaddress
import logging
from dataclasses import dataclass, field
from pathlib import Path

from membrane.auth import Authenticator
from membrane.auth.apikey import APIKeyAuthenticator
from membrane.network.config import CONSISTENCY_LEVELS, ClusterConfig
from membrane.node import Node
from membrane.runtime.plugins import (
    AUTHENTICATORS,
    COMPUTE_BACKENDS,
    CONTENT_STORES,
    EVICTION,
    PERSISTENCE,
    UnknownPluginError,
)
from membrane.security.files import InsecureFileError, require_private_file
from membrane.server import Server
from membrane.transport.limits import TransportLimits
from membrane.transport.tls import MTLSConfig

logger = logging.getLogger(__name__)


class SettingsError(ValueError):
    """Raised when settings are invalid or violate the startup policy."""


@dataclass(frozen=True, slots=True, kw_only=True)
class ServerSettings:
    """Complete configuration of one Membrane node.

    Attributes:
        node_id: Node identifier.
        host: Bind address.
        port: Listen port (``0`` picks a free port).
        transport: Transport name; only ``"http"``.
        compute: Compute plugin name.
        llm_url: Base URL for the LLM compute backend.
        llm_model: Model name for the LLM compute backend.
        llm_api_key: API key for the LLM compute backend.
        redis_url: Redis URL for fragment metadata; empty for none.
        data_dir: Directory for KV bytes; empty keeps them in memory.
        data_key_file: Master key for ``data_dir``.
        content_store: Content-store plugin used with ``data_dir``.
        persistence: Persistence plugin; empty means ``redis`` with a
            ``redis_url`` and ``memory`` otherwise.
        eviction: Eviction-policy plugin.
        load_hooks: Run every installed ``membrane.hooks`` entry point.
        max_memory: Node memory limit in bytes.
        peers: Seed peers as ``host:port``; empty for a single node.
        advertise_host: Host peers use to reach this node.
        peer_networks: CIDRs exempt from the SSRF private-address block.
        heartbeat_interval: Seconds between heartbeats.
        gossip_interval: Seconds between gossip rounds.
        replica_count: Replicas per fragment.
        failure_remove_threshold: Missed heartbeats before a peer is removed.
        consistency: Default write consistency.
        quorum_count: Copies a strong or quorum write waits for.
        api_key_file: API keyfile (selects the ``apikey`` authenticator).
        authenticator: Authenticator plugin used with ``auth_config``.
        auth_config: Configuration file passed to the authenticator plugin.
        peer_api_key_file: File with the bearer key presented to peers.
        tls_cert: mTLS certificate PEM file.
        tls_key: mTLS private key PEM file.
        tls_ca: mTLS CA bundle PEM file.
        tls_allowed_cns: Client certificate CNs to accept.
        tls_allow_any_cn: Accept any CN signed by the CA (development only).
        allow_unauthenticated: Serve without authentication beyond loopback.
        drain_timeout: Seconds a SIGTERM drain may take.
        limits: HTTP capacity settings.
    """

    node_id: str = "membrane-0"
    host: str = "127.0.0.1"
    port: int = 8080
    transport: str = "http"
    compute: str = "cpu"
    llm_url: str = ""
    llm_model: str = ""
    llm_api_key: str = field(default="", repr=False)
    redis_url: str = ""
    data_dir: str = ""
    data_key_file: str = ""
    content_store: str = "filesystem"
    persistence: str = ""
    eviction: str = "weighted-lru"
    load_hooks: bool = True
    max_memory: int = 1 << 30
    peers: tuple[str, ...] = ()
    advertise_host: str = ""
    peer_networks: tuple[str, ...] = ()
    heartbeat_interval: float = 2.0
    gossip_interval: float = 5.0
    replica_count: int = 2
    failure_remove_threshold: int = 4
    consistency: str = "strong"
    quorum_count: int = 2
    api_key_file: str = ""
    authenticator: str = "apikey"
    auth_config: str = ""
    peer_api_key_file: str = ""
    tls_cert: str = ""
    tls_key: str = ""
    tls_ca: str = ""
    tls_allowed_cns: tuple[str, ...] = ()
    tls_allow_any_cn: bool = False
    allow_unauthenticated: bool = False
    drain_timeout: float = 30.0
    limits: TransportLimits = field(default_factory=TransportLimits)

    def __post_init__(self) -> None:
        """Validate every field.

        Raises:
            SettingsError: On the first invalid value.
        """
        checks: list[tuple[bool, str]] = [
            (bool(self.node_id), "node id must not be empty"),
            (0 <= self.port <= 65535, f"port {self.port} is out of range"),
            (self.transport == "http", f"transport {self.transport!r} is not supported (only 'http')"),
            (self.compute in COMPUTE_BACKENDS, f"unknown compute backend {self.compute!r}"),
            (not self.persistence or self.persistence in PERSISTENCE, f"unknown persistence backend {self.persistence!r}"),
            (self.eviction in EVICTION, f"unknown eviction policy {self.eviction!r}"),
            (self.max_memory > 0, "max memory must be positive"),
            (self.consistency in CONSISTENCY_LEVELS, f"consistency must be one of {sorted(CONSISTENCY_LEVELS)}"),
            (self.drain_timeout >= 0, "drain timeout must not be negative"),
            (self.limits.max_concurrency >= 0, "max concurrency must not be negative"),
            (self.limits.rate_limit_per_sec >= 0, "rate limit must not be negative"),
            (not (self.api_key_file and self.auth_config), "use --api-key-file or --auth-config, not both"),
        ]
        for ok, message in checks:
            if not ok:
                raise SettingsError(message)
        try:
            self.cluster_config(tls=None)
        except ValueError as exc:
            raise SettingsError(str(exc)) from exc
        for cidr in self.peer_networks:
            try:
                ipaddress.ip_network(cidr, strict=False)
            except ValueError as exc:
                raise SettingsError(f"peer network {cidr!r} is not a CIDR") from exc

    def cluster_config(self, tls: MTLSConfig | None) -> ClusterConfig:
        """Derive the cluster configuration from these settings.

        Args:
            tls: The mTLS configuration peers use, if any.

        Returns:
            ClusterConfig: The validated cluster configuration.
        """
        return ClusterConfig(
            node_id=self.node_id,
            host=self.host,
            port=self.port,
            peers=list(self.peers),
            heartbeat_interval_sec=self.heartbeat_interval,
            gossip_interval_sec=self.gossip_interval,
            replica_count=self.replica_count,
            failure_remove_threshold=self.failure_remove_threshold,
            mtls=tls,
            advertise_host=self.advertise_host,
            default_consistency=self.consistency,
            quorum_count=self.quorum_count,
        )

    @property
    def loopback_only(self) -> bool:
        """Whether the bind address only accepts local connections."""
        return is_loopback_host(self.host)


def is_loopback_host(host: str) -> bool:
    """Return True when ``host`` only accepts local connections.

    Args:
        host: Host name or IP address.

    Returns:
        bool: True for ``localhost`` and loopback addresses.
    """
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def read_secret(path: str, what: str) -> str:
    """Read a secret file after checking that other users cannot read it.

    Args:
        path: Path of the file.
        what: Human-readable name used in messages.

    Returns:
        str: The file's contents.

    Raises:
        SettingsError: When the file is unreadable or not private.
    """
    try:
        require_private_file(path, what)
        return Path(path).read_text()
    except InsecureFileError as exc:
        raise SettingsError(str(exc)) from exc
    except OSError as exc:
        raise SettingsError(f"cannot read {what} file {path!r}: {exc}") from exc


def build_tls(settings: ServerSettings) -> MTLSConfig | None:
    """Build the mTLS configuration, or ``None`` when TLS is not configured.

    Args:
        settings: Server settings.

    Returns:
        MTLSConfig | None: The mTLS configuration.

    Raises:
        SettingsError: When the TLS settings are incomplete or unreadable.
    """
    files = (settings.tls_cert, settings.tls_key, settings.tls_ca)
    if not any(files):
        return None
    if not all(files):
        raise SettingsError("--tls-cert, --tls-key and --tls-ca must be given together")
    try:
        cert_pem, ca_pem = Path(settings.tls_cert).read_text(), Path(settings.tls_ca).read_text()
    except OSError as exc:
        raise SettingsError(f"cannot read TLS file: {exc}") from exc
    key_pem = read_secret(settings.tls_key, "TLS key")
    if settings.tls_allow_any_cn:
        return MTLSConfig.allow_all_signed_by_ca(
            server_cert_pem=cert_pem,
            server_key_pem=key_pem,
            ca_bundle_pem=ca_pem,
            client_cert_pem=cert_pem,
            client_key_pem=key_pem,
        )
    if not settings.tls_allowed_cns:
        raise SettingsError("mTLS needs --tls-allowed-cn (repeatable) or --tls-allow-any-cn")
    return MTLSConfig(
        server_cert_pem=cert_pem,
        server_key_pem=key_pem,
        ca_bundle_pem=ca_pem,
        allowed_cns=frozenset(settings.tls_allowed_cns),
        client_cert_pem=cert_pem,
        client_key_pem=key_pem,
    )


def build_inbound_authenticator(settings: ServerSettings) -> Authenticator | None:
    """Build the authenticator selected by the settings, if any.

    Args:
        settings: Server settings.

    Returns:
        Authenticator | None: The authenticator, or ``None`` when neither
        ``api_key_file`` nor ``auth_config`` is set.

    Raises:
        SettingsError: When the configuration is unreadable or holds no keys.
    """
    if settings.api_key_file:
        authenticator = APIKeyAuthenticator(read_secret(settings.api_key_file, "API keyfile"))
        if not authenticator.keys:
            raise SettingsError(f"API keyfile {settings.api_key_file!r} contains no valid keys")
        return authenticator
    if not settings.auth_config:
        return None
    try:
        require_private_file(settings.auth_config, "authenticator config")
        return AUTHENTICATORS.get(settings.authenticator)(settings.auth_config)
    except (InsecureFileError, UnknownPluginError) as exc:
        raise SettingsError(str(exc)) from exc
    except OSError as exc:
        raise SettingsError(f"cannot read authenticator config {settings.auth_config!r}: {exc}") from exc


def auth_mode(settings: ServerSettings, authenticator: Authenticator | None, tls: MTLSConfig | None) -> str:
    """Describe the inbound authentication, enforcing the loopback policy.

    Args:
        settings: Server settings.
        authenticator: The inbound authenticator, if any.
        tls: The mTLS configuration, if any.

    Returns:
        str: A short description for the startup banner.

    Raises:
        SettingsError: When serving unauthenticated beyond loopback without
            ``allow_unauthenticated``.
    """
    if tls is not None:
        return "mTLS"
    if authenticator is not None:
        return "API key" if isinstance(authenticator, APIKeyAuthenticator) else settings.authenticator
    if settings.loopback_only:
        return "none (loopback only)"
    if not settings.allow_unauthenticated:
        raise SettingsError(
            f"Refusing to serve unauthenticated on {settings.host}. Configure --api-key-file or mTLS "
            "(--tls-cert/--tls-key/--tls-ca), bind to 127.0.0.1, or pass --allow-unauthenticated."
        )
    logger.warning("Serving WITHOUT authentication on %s:%s (--allow-unauthenticated)", settings.host, settings.port)
    return "NONE (--allow-unauthenticated)"


def peer_key(settings: ServerSettings, authenticator: Authenticator | None, tls: MTLSConfig | None) -> str:
    """Read and check the key this node presents to its peers.

    Args:
        settings: Server settings.
        authenticator: The inbound authenticator, if any.
        tls: The mTLS configuration, if any.

    Returns:
        str: The peer bearer key, or ``""``.

    Raises:
        SettingsError: When an API-key cluster lacks a peer key, or the
            key does not carry the ``admin`` scope.
    """
    key = read_secret(settings.peer_api_key_file, "peer API key").strip() if settings.peer_api_key_file else ""
    if not settings.peers or authenticator is None or tls is not None:
        return key
    if not key:
        raise SettingsError("A cluster using API keys needs --peer-api-key-file so peers can authenticate")
    if isinstance(authenticator, APIKeyAuthenticator):
        # Peers replicate every tenant's fragments and propagate deletes,
        # which the receiving node only allows for admin.
        record = authenticator.lookup(key)
        if record is None:
            logger.warning("Peer API key is not in the local keyfile; peers must share a keyfile containing it")
        elif "admin" not in record.scopes:
            raise SettingsError("The peer API key must carry the 'admin' scope")
    return key


def build_server(settings: ServerSettings) -> tuple[Server, str]:
    """Build (but do not start) a server, enforcing the startup policy.

    Args:
        settings: Server settings.

    Returns:
        tuple[Server, str]: The server and a description of its inbound
        authentication.

    Raises:
        SettingsError: When the settings violate the startup policy or a
            component cannot be built.
    """
    tls = build_tls(settings)
    authenticator = build_inbound_authenticator(settings)
    mode = auth_mode(settings, authenticator, tls)
    peer_api_key = peer_key(settings, authenticator, tls)

    cluster_config = settings.cluster_config(tls) if settings.peers else None

    content_store = None
    if settings.data_dir:
        try:
            factory = CONTENT_STORES.get(settings.content_store)
            content_store = factory(settings.data_dir, settings.data_key_file)
        except (OSError, ValueError) as exc:
            raise SettingsError(f"Cannot open data directory {settings.data_dir!r}: {exc}") from exc

    node = Node(
        node_id=settings.node_id,
        max_memory_bytes=settings.max_memory,
        content_store=content_store,
        eviction_policy=EVICTION.get(settings.eviction)(),
    )
    server = Server(
        node=node,
        transport=settings.transport,
        compute=settings.compute,
        redis_url=settings.redis_url,
        host=settings.host,
        port=settings.port,
        cluster_config=cluster_config,
        llm_url=settings.llm_url,
        llm_model=settings.llm_model,
        api_key=settings.llm_api_key,
        authenticator=authenticator,
        peer_api_key=peer_api_key,
        peer_networks=settings.peer_networks,
        tls=tls,
        limits=settings.limits,
        persistence=settings.persistence,
        load_hooks=settings.load_hooks,
    )
    if settings.redis_url and not server.durable:
        raise SettingsError(
            f"Redis at {settings.redis_url} is unreachable; refusing to start without the requested durability"
        )
    return server, mode


__all__ = [
    "ServerSettings",
    "SettingsError",
    "auth_mode",
    "build_inbound_authenticator",
    "build_server",
    "build_tls",
    "is_loopback_host",
    "peer_key",
    "read_secret",
]
