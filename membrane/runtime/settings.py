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

import importlib.util
import ipaddress
import logging
from dataclasses import dataclass, field
from pathlib import Path

from membrane.auth import Authenticator
from membrane.auth.apikey import APIKeyAuthenticator
from membrane.auth.spiffe import SPIFFEAuthenticator, parse_id_scopes
from membrane.codec import method_available
from membrane.network.config import CONSISTENCY_LEVELS, ClusterConfig
from membrane.node import Node, NodeAttributes
from membrane.otel_tracer.otel import TRACING
from membrane.replica import Replica
from membrane.runtime.components import load_data_key
from membrane.runtime.plugins import (
    AUTHENTICATORS,
    COMPUTE_BACKENDS,
    CONTENT_STORES,
    EVICTION,
    PERSISTENCE,
    PLACEMENT,
    SECRET_PROVIDERS,
    UnknownPluginError,
)
from membrane.secrets import (
    SecretBackendError,
    SecretNotFoundError,
    is_secret_ref,
    resolve_secret,
    set_default_provider,
)
from membrane.security.files import InsecureFileError, require_private_file
from membrane.server import Server
from membrane.services import ServiceOptions
from membrane.store.quantizing import FORMATS as QUANTIZATION_FORMATS
from membrane.store.quantizing import QuantizingStore
from membrane.store.tiered import WarmTier
from membrane.transport.acme import LETS_ENCRYPT, ACMEConfig, ACMEError, ensure_certificate
from membrane.transport.limits import TransportLimits
from membrane.transport.spiffe import SPIFFEClient, SPIFFEConfig
from membrane.transport.tls import MTLSConfig
from membrane.transport.tls_rotation import enforce_not_after

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
        data_key_file: Master key for ``data_dir`` (a file, or ``secret://name``).
        secret_provider: Secret-provider plugin that resolves ``secret://``
            references in any secret setting.
        content_store: Content-store plugin used with ``data_dir``.
        persistence: Persistence plugin; empty means ``redis`` with a
            ``redis_url`` and ``memory`` otherwise.
        eviction: Eviction-policy plugin.
        load_hooks: Run every installed ``membrane.hooks`` entry point.
        warm_tier_bytes: Keep up to this many bytes of fragments evicted from
            memory in an encrypted on-disk warm tier (``<data-dir>/warm``);
            0 disables it.
        kv_quantization: Quantize KV tensors at rest: ``none``, ``int8``,
            ``fp8_e4m3``, ``fp8_e5m2``, or ``nf4`` (lossy; needs numpy).
        transfer_compression: How KV bytes travel to peers: ``zstd``,
            ``lz4``, ``deflate``, or ``raw``.
        otel_endpoint: OTLP/gRPC endpoint for traces; empty uses
            ``OTEL_EXPORTER_OTLP_ENDPOINT`` when set, else tracing is off.
        placement: Placement policy plugin answering ``/route``.
        route_threshold: Uncached prompt tokens above which ``/route`` says to
            offload prefill to Membrane; adapts to load. ``0`` leaves it out.
        promote_replicas: Copies a frequently read fragment may reach;
            ``0`` disables promotion.
        dynamic_roles: Re-evaluate the node's role from load and advertise it.
        region: Region this node runs in (advertised; used by routing and
            replica placement).
        origin: ``host:port`` of an origin this node caches for; local misses
            are read through to it and kept as non-primary copies.
        require_compat: ``MODEL[:DTYPE]``: refuse fragments stamped for
            another model; prefilled fragments are stamped.
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
        tls_spiffe_socket: Take the node's certificate, key, and trust bundle
            from this SPIFFE Workload API socket.
        tls_spiffe_allow: ``spiffe://.../path=scope,...`` entries: SPIFFE IDs
            accepted as peers or clients, and their scopes.
        tls_acme_domains: Obtain and renew the listener certificate from an
            ACME CA for these DNS names (single-node public listeners).
        tls_acme_directory: ACME directory URL.
        tls_acme_email: Contact email for the ACME account.
        tls_acme_state_dir: Where the account key and certificate live;
            defaults to ``{data_dir}/acme``.
        tls_acme_http_port: Port the HTTP-01 responder listens on (the
            domain's port 80 must reach it).
        tls_acme_ca_bundle: CA bundle trusted for the ACME directory
            (private or test CAs).
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
    secret_provider: str = "env"
    content_store: str = "filesystem"
    persistence: str = ""
    eviction: str = "weighted-lru"
    load_hooks: bool = True
    otel_endpoint: str = ""
    placement: str = "ring"
    route_threshold: int = 0
    promote_replicas: int = 0
    dynamic_roles: bool = False
    region: str = ""
    origin: str = ""
    require_compat: str = ""
    transfer_compression: str = "zstd"
    kv_quantization: str = "none"
    warm_tier_bytes: int = 0
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
    tls_spiffe_socket: str = ""
    tls_spiffe_allow: tuple[str, ...] = ()
    tls_acme_domains: tuple[str, ...] = ()
    tls_acme_directory: str = LETS_ENCRYPT
    tls_acme_email: str = ""
    tls_acme_state_dir: str = ""
    tls_acme_http_port: int = 80
    tls_acme_ca_bundle: str = ""
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
            (
                not self.persistence or self.persistence in PERSISTENCE,
                f"unknown persistence backend {self.persistence!r}",
            ),
            (self.eviction in EVICTION, f"unknown eviction policy {self.eviction!r}"),
            (self.secret_provider in SECRET_PROVIDERS, f"unknown secret provider {self.secret_provider!r}"),
            (self.placement in PLACEMENT, f"unknown placement policy {self.placement!r}"),
            (self.route_threshold >= 0, "route threshold must not be negative"),
            (self.promote_replicas >= 0, "promote replicas must not be negative"),
            (not (self.origin and self.peers), "a regional cache (--origin) runs without --peer"),
            (not self.origin or ":" in self.origin, "--origin must be HOST:PORT"),
            (self.warm_tier_bytes >= 0, "warm tier bytes must not be negative"),
            (not self.warm_tier_bytes or bool(self.data_dir), "the warm tier needs --data-dir"),
            (
                self.kv_quantization == "none" or self.kv_quantization in QUANTIZATION_FORMATS,
                f"KV quantization must be none or one of {sorted(QUANTIZATION_FORMATS)}",
            ),
            (
                self.kv_quantization == "none" or importlib.util.find_spec("numpy") is not None,
                "KV quantization needs numpy: install membrane[transfer]",
            ),
            (
                method_available(self.transfer_compression),
                f"transfer compression {self.transfer_compression!r} is unknown or not installed",
            ),
            (self.max_memory > 0, "max memory must be positive"),
            (self.consistency in CONSISTENCY_LEVELS, f"consistency must be one of {sorted(CONSISTENCY_LEVELS)}"),
            (self.drain_timeout >= 0, "drain timeout must not be negative"),
            (self.limits.max_concurrency >= 0, "max concurrency must not be negative"),
            (self.limits.rate_limit_per_sec >= 0, "rate limit must not be negative"),
            (not (self.api_key_file and self.auth_config), "use --api-key-file or --auth-config, not both"),
            (
                not (self.tls_acme_domains and (self.tls_cert or self.tls_key or self.tls_ca)),
                "use --tls-acme-domain or --tls-cert/--tls-key/--tls-ca, not both",
            ),
            (not (self.tls_acme_domains and self.peers), "ACME certificates are for single-node public listeners"),
            (
                not (self.tls_spiffe_socket and (self.tls_cert or self.tls_key or self.tls_acme_domains)),
                "use --tls-spiffe-socket alone, without --tls-cert or --tls-acme-domain",
            ),
            (
                not self.tls_spiffe_socket or bool(self.tls_spiffe_allow),
                "SPIFFE needs --tls-spiffe-allow spiffe://.../path=scope (repeatable)",
            ),
            (
                not self.tls_acme_domains or bool(self.tls_acme_state_dir or self.data_dir),
                "ACME needs --tls-acme-state-dir or --data-dir to keep its account and certificate",
            ),
        ]
        for ok, message in checks:
            if not ok:
                raise SettingsError(message)
        try:
            self.cluster_config(tls=None)
        except ValueError as exc:
            raise SettingsError(str(exc)) from exc
        try:
            parse_id_scopes(self.tls_spiffe_allow)
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
    """Read a secret from a private file, or from the secret provider.

    Args:
        path: Path of the file, or ``secret://name`` to ask the configured
            secret provider (``--secret-provider``) instead.
        what: Human-readable name used in messages.

    Returns:
        str: The secret.

    Raises:
        SettingsError: When the file is unreadable or not private, or the
            provider cannot return the secret.
    """
    if is_secret_ref(path):
        try:
            return resolve_secret(path)
        except (SecretNotFoundError, SecretBackendError, ValueError) as exc:
            raise SettingsError(f"cannot resolve {what} {path!r}: {exc!r}") from exc
    try:
        require_private_file(path, what)
        return Path(path).read_text()
    except InsecureFileError as exc:
        raise SettingsError(str(exc)) from exc
    except OSError as exc:
        raise SettingsError(f"cannot read {what} file {path!r}: {exc}") from exc


def spiffe_config(settings: ServerSettings) -> SPIFFEConfig | None:
    """Return the SPIFFE configuration, or ``None`` when SPIFFE is not used.

    Args:
        settings: Server settings.

    Returns:
        SPIFFEConfig | None: The configuration.
    """
    if not settings.tls_spiffe_socket:
        return None
    return SPIFFEConfig(
        socket_path=settings.tls_spiffe_socket,
        allowed_ids=frozenset(parse_id_scopes(settings.tls_spiffe_allow)),
    )


def acme_config(settings: ServerSettings) -> ACMEConfig | None:
    """Return the ACME configuration, or ``None`` when ACME is not used.

    Args:
        settings: Server settings.

    Returns:
        ACMEConfig | None: The configuration.
    """
    if not settings.tls_acme_domains:
        return None
    return ACMEConfig(
        directory_url=settings.tls_acme_directory,
        domains=list(settings.tls_acme_domains),
        state_dir=settings.tls_acme_state_dir or str(Path(settings.data_dir) / "acme"),
        contact=[f"mailto:{settings.tls_acme_email}"] if settings.tls_acme_email else None,
        http_port=settings.tls_acme_http_port,
        ca_bundle=settings.tls_acme_ca_bundle,
    )


def build_tls(settings: ServerSettings) -> MTLSConfig | None:
    """Build the mTLS configuration, or ``None`` when TLS is not configured.

    Args:
        settings: Server settings.

    Returns:
        MTLSConfig | None: The mTLS configuration.

    Raises:
        SettingsError: When the TLS settings are incomplete or unreadable.
    """
    spiffe = spiffe_config(settings)
    if spiffe is not None:
        try:
            return SPIFFEClient(spiffe).fetch_mtls_config()
        except Exception as exc:
            raise SettingsError(f"SPIFFE Workload API at {settings.tls_spiffe_socket!r}: {exc}") from exc
    acme = acme_config(settings)
    if acme is not None:
        try:
            ensure_certificate(acme)
        except (ACMEError, OSError) as exc:
            raise SettingsError(f"ACME certificate for {', '.join(acme.domains)} failed: {exc}") from exc
        return MTLSConfig.server_only(acme.cert_path.read_text(), acme.key_path.read_text())
    files = (settings.tls_cert, settings.tls_key, settings.tls_ca)
    if not any(files):
        return None
    if not all(files):
        raise SettingsError("--tls-cert, --tls-key and --tls-ca must be given together")
    try:
        cert_pem, ca_pem = (
            resolve_secret(ref) if is_secret_ref(ref) else Path(ref).read_text()
            for ref in (settings.tls_cert, settings.tls_ca)
        )
    except (OSError, SecretNotFoundError, SecretBackendError) as exc:
        raise SettingsError(f"cannot read TLS file: {exc!r}") from exc
    key_pem = read_secret(settings.tls_key, "TLS key")
    try:
        enforce_not_after(cert_pem)
    except RuntimeError as exc:
        raise SettingsError(str(exc)) from exc
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
    if settings.tls_spiffe_socket and not settings.api_key_file:
        return SPIFFEAuthenticator(parse_id_scopes(settings.tls_spiffe_allow))
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
    if settings.tls_spiffe_socket:
        return "SPIFFE"
    if tls is not None and tls.require_client_cert:
        return "mTLS"
    if authenticator is not None:
        kind = "API key" if isinstance(authenticator, APIKeyAuthenticator) else settings.authenticator
        return f"{kind} over TLS" if tls is not None else kind
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
    if not settings.peers or authenticator is None or (tls is not None and tls.require_client_cert):
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


def tls_files(settings: ServerSettings, tls: MTLSConfig | None) -> tuple[str, str] | None:
    """Return the certificate and key files the server should watch for rotation.

    Args:
        settings: Server settings.
        tls: The TLS configuration in use.

    Returns:
        tuple[str, str] | None: ``(cert, key)`` paths, or ``None`` when TLS
        is off or the material came from the secret provider.
    """
    acme = acme_config(settings)
    if acme is not None:
        return str(acme.cert_path), str(acme.key_path)
    if settings.tls_spiffe_socket:
        return None  # refreshed from the Workload API instead
    if tls is None or is_secret_ref(settings.tls_cert) or is_secret_ref(settings.tls_key):
        return None
    return settings.tls_cert, settings.tls_key


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
    try:
        set_default_provider(SECRET_PROVIDERS.get(settings.secret_provider)())
    except Exception as exc:
        raise SettingsError(f"cannot start secret provider {settings.secret_provider!r}: {exc}") from exc
    TRACING.configure(settings.otel_endpoint, node_id=settings.node_id)
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

    if settings.kv_quantization != "none":
        from membrane.content_store import InProcessBytes

        content_store = QuantizingStore(content_store or InProcessBytes(), settings.kv_quantization)
    # A regional cache never owns primaries: it holds read-through copies.
    node_class = Replica if settings.origin else Node
    node = node_class(
        node_id=settings.node_id,
        max_memory_bytes=settings.max_memory,
        content_store=content_store,
        eviction_policy=EVICTION.get(settings.eviction)(),
        attributes=NodeAttributes(region=settings.region) if settings.region else None,
    )
    if settings.warm_tier_bytes:
        from membrane.content_store import FilesystemBlob

        root = Path(settings.data_dir)
        warm_store = FilesystemBlob(
            root / "warm", tenant_id="membrane-warm", key_provider=load_data_key(root, settings.data_key_file)
        )
        node.lower_tier = WarmTier(warm_store, settings.warm_tier_bytes)
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
        api_key=read_secret(settings.llm_api_key, "LLM API key")
        if is_secret_ref(settings.llm_api_key)
        else settings.llm_api_key,
        authenticator=authenticator,
        peer_api_key=peer_api_key,
        peer_networks=settings.peer_networks,
        tls=tls,
        limits=settings.limits,
        persistence=settings.persistence,
        load_hooks=settings.load_hooks,
        tls_files=tls_files(settings, tls),
        acme=acme_config(settings),
        spiffe=spiffe_config(settings),
        audit_path=str(Path(settings.data_dir) / "audit.jsonl") if settings.data_dir else None,
        transfer_compression=settings.transfer_compression,
        services=ServiceOptions(
            placement=settings.placement,
            route_threshold=settings.route_threshold,
            promote_replicas=settings.promote_replicas,
            dynamic_roles=settings.dynamic_roles,
            origin=settings.origin,
            require_compat=settings.require_compat,
        ),
    )
    if settings.redis_url and not server.durable:
        raise SettingsError(
            f"Redis at {settings.redis_url} is unreachable; refusing to start without the requested durability"
        )
    return server, mode


__all__ = [
    "ServerSettings",
    "SettingsError",
    "acme_config",
    "auth_mode",
    "build_inbound_authenticator",
    "build_server",
    "build_tls",
    "is_loopback_host",
    "peer_key",
    "read_secret",
    "spiffe_config",
    "tls_files",
]
