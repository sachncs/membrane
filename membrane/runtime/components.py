"""Builders for the server's pluggable components.

Each function builds one subsystem from plain configuration, so
:class:`~membrane.server.Server` only wires them together and tests can
build any piece in isolation.
"""

import ipaddress
import logging
import os
import secrets
import socket
from pathlib import Path
from typing import Any

from membrane.auth import Authenticator
from membrane.network.config import ClusterConfig
from membrane.persistence.cache import CachingPersistence
from membrane.persistence.memory import Memory
from membrane.transport.tls import MTLSConfig

logger = logging.getLogger(__name__)


def build_content_store(data_dir: str, key_file: str = "") -> Any:
    """Return an encrypted on-disk content store rooted at ``data_dir``.

    KV bytes live in ``{data_dir}/blobs`` (AES-256-GCM,
    :class:`~membrane.content_store.FilesystemBlob`), so they survive a
    restart. The 32-byte master key comes from ``key_file`` when given
    (mount it from a secret manager in production); otherwise it is
    generated once into ``{data_dir}/master.key`` with mode 0600.

    Args:
        data_dir: Node data directory; created when missing.
        key_file: Optional path to a file holding the raw 32-byte key
            or its 64-character hex encoding.

    Returns:
        FilesystemBlob: The content store.

    Raises:
        ValueError: When the key file is missing, readable by other
            users, or does not hold a 32-byte key.
    """
    from membrane.content_store import FilesystemBlob
    from membrane.security.encryption import StaticKeyProvider
    from membrane.security.files import require_private_file

    root = Path(data_dir)
    root.mkdir(parents=True, exist_ok=True)
    key_path = Path(key_file) if key_file else root / "master.key"
    if not key_path.exists():
        if key_file:
            raise ValueError(f"data key file {key_file!r} does not exist")
        fd = os.open(key_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb") as handle:
            handle.write(secrets.token_bytes(32))
    require_private_file(key_path, "data key")
    raw = key_path.read_bytes()
    if len(raw) == 32:
        key = raw  # raw key bytes: never strip, they may look like whitespace
    else:
        try:
            key = bytes.fromhex(raw.decode("ascii").strip())
        except UnicodeDecodeError, ValueError:
            key = b""
    if len(key) != 32:
        raise ValueError(f"data key in {str(key_path)!r} must be 32 bytes (or 64 hex characters)")
    return FilesystemBlob(root / "blobs", tenant_id="membrane", key_provider=StaticKeyProvider(key=key))


def resolves_to_loopback(host: str) -> bool:
    """Return True when ``host`` resolves only to loopback addresses.

    Args:
        host: Host name or address to resolve.

    Returns:
        bool: True when ``host`` resolves only to loopback addresses.
    """
    try:
        infos = socket.getaddrinfo(host, None)
    except OSError:
        return False
    addresses = {ipaddress.ip_address(info[4][0]) for info in infos}
    return bool(addresses) and all(addr.is_loopback for addr in addresses)


def build_persistence(redis_url: str, name: str = "") -> CachingPersistence:
    """Build the persistence layer from the ``membrane.persistence`` registry.

    Args:
        redis_url: Backend URL (``redis://...`` for the built-in Redis
            backend); empty for none.
        name: Persistence plugin; defaults to ``redis`` when a URL is given
            and ``memory`` otherwise.

    Returns:
        CachingPersistence: The backend, fronted by a read cache. When the
        Redis backend is unreachable this falls back to in-memory
        persistence (and :func:`is_durable` reports False).
    """
    from membrane.runtime.plugins import PERSISTENCE

    name = name or ("redis" if redis_url else "memory")
    backend: Any = Memory()
    try:
        candidate = PERSISTENCE.get(name)(redis_url)
        if name != "redis" or candidate.ping():
            backend = candidate
            logger.info("persistence: %s%s", name, f" at {redis_url}" if redis_url else "")
        else:
            logger.warning("Redis at %s unreachable; using in-memory persistence", redis_url)
    except Exception as exc:
        logger.warning("persistence backend %s failed (%s); using in-memory persistence", name, exc)
    return CachingPersistence(backend)


def is_durable(persistence: Any) -> bool:
    """Whether ``persistence`` keeps fragments beyond this process.

    Args:
        persistence: The persistence layer.

    Returns:
        bool: True unless the backend is the in-process :class:`Memory`.
    """
    inner = getattr(persistence, "inner", None)
    return inner is not None and not isinstance(inner, Memory)


def build_authenticator(mtls: MTLSConfig | None) -> Authenticator | None:
    """Return an mTLS authenticator when client certs are required.

    Args:
        mtls: mTLS configuration, or ``None`` when TLS is off.

    Returns:
        Authenticator | None: An mTLS authenticator when client certs are
        required.
    """
    if mtls is None or not mtls.require_client_cert:
        return None
    from membrane.auth.mtls import MTLSAuthenticator

    return MTLSAuthenticator(mtls)


def configure_peer_access(
    cluster_config: ClusterConfig,
    mtls: MTLSConfig | None,
    peer_api_key: str,
    peer_networks: tuple[str, ...],
) -> None:
    """Install process-wide peer credentials and the outbound URL policy.

    Seed peer hosts are always allowed; ``peer_networks`` admits peers
    learned later through join responses and gossip.

    Args:
        cluster_config: Cluster configuration (its seed peers are allowed by
            name).
        mtls: mTLS configuration, or ``None`` when TLS is off.
        peer_api_key: Bearer key this node presents to its peers (needs the
            ``admin`` scope).
        peer_networks: CIDR ranges of the peer network, exempt from the SSRF
            private-address block.
    """
    from membrane.network.peer import PeerCredentials, set_default_peer_credentials
    from membrane.security.url_allowlist import configure as configure_allowlist
    from membrane.transport.tls import build_client_context

    set_default_peer_credentials(
        PeerCredentials(
            scheme="https" if mtls is not None else "http",
            bearer_token=peer_api_key,
            ssl_context=build_client_context(mtls) if mtls is not None else None,
        )
    )
    seed_hosts = [seed.rsplit(":", 1)[0].strip("[]") for seed in cluster_config.peers]
    networks = list(peer_networks)
    # A local (loopback) cluster advertises 127.0.0.1 / ::1 rather than
    # the seed hostnames, so admit loopback peers when the operator
    # seeded the cluster with loopback addresses.
    if any(resolves_to_loopback(host) for host in seed_hosts):
        networks += ["127.0.0.0/8", "::1/128"]
    configure_allowlist(allowlist=seed_hosts, allowed_networks=networks)


__all__ = [
    "build_authenticator",
    "build_content_store",
    "build_persistence",
    "configure_peer_access",
    "is_durable",
    "resolves_to_loopback",
]
