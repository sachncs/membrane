"""Peer: HTTP client for inter-node communication.

Speaks the same REST surface as
:class:`~membrane.transport.fastapi.FastAPIServer`, exposing methods for
the cluster-management verbs (``join``, ``leave``, ``heartbeat``,
``gossip``) and the fragment-management verbs (``store``,
``retrieve``, ``replicate``).

The wire-level HTTP work is delegated to a pluggable
:class:`Transport` (default: :class:`HTTPTransport`). The default
transport uses ``urllib.request``; tests use
``unittest.mock.patch`` on the transport instance directly.

Thread safety:
    The class is **not** explicitly thread-safe; in practice a
    client is bound to a single peer and shared across the
    background threads that talk to that peer. The default
    :class:`HTTPTransport` handles concurrent sockets internally.
"""

import json
import logging
import ssl
import time
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from membrane.codec import CompressionTransport
from membrane.errors import NetworkError
from membrane.fragment import Fragment
from membrane.otel_tracer.otel import TRACING
from membrane.resilience import CircuitBreaker, CircuitBreakerPolicy, RetryPolicy, compute_backoff
from membrane.serialization import JsonDict, from_dict, to_dict
from membrane.wire.v3.chunks import sha256_hex

logger = logging.getLogger(__name__)

#: Monotonic deadline for peer requests made in the current context.
#: Callers with a budget (e.g. the quorum fan-out) set it so retries stop
#: once the result can no longer be used; ``None`` means no deadline.
peer_deadline: ContextVar[float | None] = ContextVar("membrane_peer_deadline", default=None)

#: Request/response header naming the compression of a blob body.
COMPRESSION_HEADER = "X-Membrane-Compression"
#: Request header: the compression the caller accepts for a blob download.
ACCEPT_COMPRESSION_HEADER = "X-Membrane-Accept-Compression"
#: Blobs smaller than this travel uncompressed.
COMPRESS_MIN_BYTES = 1024
#: Upper bound for a decompressed blob (matches the server's body limit).
MAX_BLOB_BYTES = 100 << 20


@dataclass(frozen=True)
class PeerCredentials:
    """How this node reaches and authenticates to its peers.

    Attributes:
        scheme: ``"http"`` or ``"https"`` for peer URLs built from
            a bare ``host:port``.
        bearer_token: API key sent as ``Authorization: Bearer``
            on every peer request. Empty to send none.
        ssl_context: Client TLS context (CA bundle plus, for mTLS,
            this node's client certificate). ``None`` uses the
            system trust store.
        compression: How KV bytes travel between peers: ``zstd``
            (default), ``lz4``, ``deflate``, or ``raw``.
    """

    scheme: str = "http"
    bearer_token: str = ""
    ssl_context: ssl.SSLContext | None = None
    compression: str = "zstd"


DEFAULT_CREDENTIALS = PeerCredentials()


def get_default_peer_credentials() -> PeerCredentials:
    """Return the process-wide :class:`PeerCredentials`.

    Returns:
        PeerCredentials: The process-wide :class:`PeerCredentials`.
    """
    return DEFAULT_CREDENTIALS


def set_default_peer_credentials(credentials: PeerCredentials) -> None:
    """Install the process-wide :class:`PeerCredentials`.

    :class:`~membrane.server.Server` calls this at startup so every
    :class:`Peer` the cluster layer creates uses the same scheme and
    credentials.

    Args:
        credentials: Credentials to install.
    """
    global DEFAULT_CREDENTIALS
    DEFAULT_CREDENTIALS = credentials


def peer_url(host_port: str) -> str:
    """Return the base URL for a bare ``host:port`` peer address.

    Args:
        host_port: Peer address as ``host:port`` (or a full URL, returned
            unchanged).

    Returns:
        str: The base URL for a bare ``host:port`` peer address.
    """
    if "://" in host_port:
        return host_port
    return f"{get_default_peer_credentials().scheme}://{host_port}"


@dataclass(frozen=True, slots=True)
class RawResponse:
    """An HTTP response whose body is returned as bytes, not parsed JSON.

    Attributes:
        status: HTTP status code (2xx or 404; other errors raise).
        headers: Response headers, lower-cased names.
        body: Response body.
    """

    status: int
    headers: dict[str, str]
    body: bytes


@runtime_checkable
class Transport(Protocol):
    """Pluggable wire-level HTTP transport.

    Implementations must return a ``dict`` (parsed JSON body) on a
    successful 2xx response and ``None`` on any non-retryable
    failure. Retry semantics are the caller's responsibility; this
    protocol is intentionally minimal.
    """

    def request(
        self,
        method: str,
        url: str,
        body: bytes | None,
        headers: dict[str, str],
        timeout_sec: float,
    ) -> JsonDict | None:
        """Issue one HTTP request and return the parsed JSON body.

        Args:
            method: HTTP method.
            url: Full URL.
            body: Request body bytes or ``None``.
            headers: Request headers.
            timeout_sec: Per-request timeout in seconds.

        Returns:
            JsonDict | None: Parsed JSON body on success,
            ``None`` on non-retryable failure.
        """
        ...

    def request_bytes(
        self,
        method: str,
        url: str,
        body: bytes | None,
        headers: dict[str, str],
        timeout_sec: float,
    ) -> RawResponse | None:
        """Issue one HTTP request and return the raw response.

        Args:
            method: HTTP method.
            url: Full URL.
            body: Request body bytes or ``None``.
            headers: Request headers.
            timeout_sec: Per-request timeout in seconds.

        Returns:
            RawResponse | None: The response for 2xx and 404, or ``None``
            when the URL is rejected by the outbound policy.
        """
        ...


class HTTPTransport:
    """Default :class:`Transport` backed by an :mod:`httpx` connection pool.

    The transport:

    * Validates every outbound URL against the SSRF allow-list
      (:func:`membrane.security.validate_outbound_url`) and pins
      the resolved IP at the socket layer so a DNS-rebinding
      attack cannot smuggle a private address into the second
      resolution (the original :mod:`urllib` path re-resolved
      on its own).
    * Disables automatic redirect-following; a 3xx response
      surfaces to the caller as a typed redirect so the caller
      can re-validate the target URL rather than the cluster
      trusting it.
    * Maintains a pooled ``httpx.Client`` (``max_keepalive_connections=10``,
      ``max_connections=100``) so amortised TCP / TLS setup
      costs do not dominate the cluster's p50 latency.
    """

    def __init__(self, ssl_context: ssl.SSLContext | None = None) -> None:
        """Create the transport; the pooled HTTP client is built on first use.

        Args:
            ssl_context: Client TLS context for HTTPS peers; ``None`` uses the
                system trust store.
        """
        self.__client: Any | None = None
        self.ssl_context = ssl_context

    def __get_client(self) -> Any:
        """Return the pooled HTTP client, creating it on first use.

        Returns:
            Any: The pooled HTTP client, creating it on first use.
        """
        if self.__client is None:
            try:
                import httpx

                limits = httpx.Limits(
                    max_keepalive_connections=10,
                    max_connections=100,
                )
                self.__client = httpx.Client(
                    timeout=httpx.Timeout(60.0),
                    limits=limits,
                    follow_redirects=False,
                    verify=self.ssl_context if self.ssl_context is not None else True,
                )
            except ImportError:
                logger.warning("HTTPTransport: httpx not installed; falling back to None")
                self.__client = None
        return self.__client

    def request(
        self,
        method: str,
        url: str,
        body: bytes | None,
        headers: dict[str, str],
        timeout_sec: float,
    ) -> JsonDict | None:
        """Issue an HTTP request via the pooled client.

        Args:
            method: HTTP method.
            url: Full URL.
            body: Request body bytes or ``None``.
            headers: Request headers.
            timeout_sec: Per-request timeout in seconds.

        Returns:
            JsonDict | None: The parsed JSON body, or ``None`` when the SSRF
            policy rejects the URL.

        Raises:
            NetworkError: On transport failure, a redirect, or an HTTP error.
        """
        resp = self.__send(method, url, body, headers, timeout_sec)
        if resp is None:
            return None
        if resp.status_code >= 400:
            raise NetworkError(f"HTTP {resp.status_code} from {method} {url}")
        raw = resp.text
        return json.loads(raw) if raw else {}

    def request_bytes(
        self,
        method: str,
        url: str,
        body: bytes | None,
        headers: dict[str, str],
        timeout_sec: float,
    ) -> RawResponse | None:
        """Issue an HTTP request and return the raw response.

        Args:
            method: HTTP method.
            url: Full URL.
            body: Request body bytes or ``None``.
            headers: Request headers.
            timeout_sec: Per-request timeout in seconds.

        Returns:
            RawResponse | None: The response for 2xx and 404, or ``None``
            when the SSRF policy rejects the URL.

        Raises:
            NetworkError: On transport failure, a redirect, or another HTTP
                error.
        """
        resp = self.__send(method, url, body, headers, timeout_sec)
        if resp is None:
            return None
        if resp.status_code >= 400 and resp.status_code != 404:
            raise NetworkError(f"HTTP {resp.status_code} from {method} {url}")
        return RawResponse(
            status=resp.status_code,
            headers={k.lower(): v for k, v in resp.headers.items()},
            body=resp.content,
        )

    def __send(
        self,
        method: str,
        url: str,
        body: bytes | None,
        headers: dict[str, str],
        timeout_sec: float,
    ) -> Any:
        """Validate ``url``, pin its address, and send the request.

        Args:
            method: HTTP method.
            url: Full URL.
            body: Request body bytes or ``None``.
            headers: Request headers.
            timeout_sec: Per-request timeout in seconds.

        Returns:
            Any: The ``httpx.Response``, or ``None`` when the SSRF policy
            rejects the URL.

        Raises:
            NetworkError: On transport failure or a redirect.
        """
        from urllib.parse import urlparse

        from membrane.security import validate_outbound_url
        from membrane.security.url_allowlist import SSRFError, get_default_allowlist, resolve_addresses

        try:
            validate_outbound_url(url)
        except SSRFError as exc:
            logger.warning("HTTPTransport %s %s rejected by SSRF policy: %s", method, url, exc)
            return None

        client = self.__get_client()
        if client is None:
            raise NetworkError("HTTPTransport: httpx not installed")

        parsed = urlparse(url)
        extensions: dict[str, Any] = {}
        policy = get_default_allowlist()
        if policy.block_private and parsed.hostname and not policy.is_host_allowed(parsed.hostname.lower()):
            try:
                addresses = resolve_addresses(parsed.hostname)
            except OSError:
                addresses = []
            if addresses:
                pinned_ip = str(addresses[0])
                # Replace the host with the pinned IP and set the Host header
                # to the original hostname so SNI and certificate
                # verification still use the user-supplied hostname.
                host_header = parsed.hostname
                port = parsed.port
                netloc = pinned_ip if port is None else f"{pinned_ip}:{port}"
                url = parsed._replace(netloc=netloc).geturl()
                headers = {**headers, "Host": host_header}
                extensions["sni_hostname"] = host_header

        try:
            resp = client.request(
                method,
                url,
                content=body,
                headers=headers,
                timeout=timeout_sec,
                extensions=extensions or None,
            )
        except Exception as exc:
            raise NetworkError(f"transport failure on {method} {parsed.hostname or url}: {exc}") from exc

        if 300 <= resp.status_code < 400:
            raise NetworkError(
                f"redirect not followed ({resp.status_code}) on {method} {url}; re-validate the target before retrying"
            )
        return resp


class Peer:
    """HTTP client for a single Membrane peer.

    Args:
        base_url: Peer URL (e.g., ``http://192.168.1.2:8080``).
        transport: :class:`Transport` instance. Defaults to a
            shared :class:`HTTPTransport`.
        timeout_sec: Request timeout.
        max_retries: Max retry attempts.
        retry_delay_sec: Base delay between retries.
    """

    def __init__(
        self,
        base_url: str,
        transport: Transport | None = None,
        timeout_sec: float = 5.0,
        max_retries: int = 3,
        retry_delay_sec: float = 1.0,
        local_peer_cn: str = "",
        credentials: PeerCredentials | None = None,
        breaker_policy: CircuitBreakerPolicy | None = None,
    ) -> None:
        """Initialize the client.

        Args:
            base_url: Peer URL. Trailing slashes are stripped.
            transport: Optional :class:`Transport`; defaults to
                :class:`HTTPTransport`.
            timeout_sec: Per-request timeout in seconds.
            max_retries: Maximum number of attempts before
                giving up.
            retry_delay_sec: Base delay used as ``base *
                2 ** attempt`` for exponential backoff.
            local_peer_cn: Common Name this client presents as
                the verified peer on the wire. When non-empty,
                every outbound request carries an
                ``X-Local-Peer-CN`` header so the receiving
                :class:`~membrane.network.membership.Membership`
                records and validates the caller's identity at
                the membership layer.
            credentials: Scheme / bearer token / TLS context for
                this peer. Defaults to
                :func:`get_default_peer_credentials`.
            breaker_policy: When to stop calling this peer; the default
                opens after 5 consecutive failed calls for 30 s.
        """
        self.credentials = credentials or get_default_peer_credentials()
        self.base_url = peer_url(base_url).rstrip("/")
        self.transport = transport or HTTPTransport(ssl_context=self.credentials.ssl_context)
        self.timeout_sec = timeout_sec
        self.max_retries = max_retries
        self.retry_delay_sec = retry_delay_sec
        self.local_peer_cn = local_peer_cn
        # One breaker per peer: a dead peer costs one failed call per
        # cool-down, not every caller's full retry budget.
        self.breaker = CircuitBreaker(breaker_policy or CircuitBreakerPolicy())
        self.retry = RetryPolicy(
            max_attempts=max_retries, base_delay=retry_delay_sec, max_delay=max(retry_delay_sec * 8, 0.0)
        )

    @property
    def base_headers(self) -> dict[str, str]:
        """Headers attached to every outbound request.

        Returns:
            dict[str, str]: ``User-Agent`` plus the optional
            ``X-Local-Peer-CN`` header. The receiving
            :class:`~membrane.transport.ops.op_heartbeat` reads
            the latter to update the membership's per-peer CN.
        """
        headers = {"User-Agent": "membrane-peer/2.0"}
        if self.local_peer_cn:
            headers["X-Local-Peer-CN"] = self.local_peer_cn
        if self.credentials.bearer_token:
            headers["Authorization"] = f"Bearer {self.credentials.bearer_token}"
        return headers

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def heartbeat(self) -> JsonDict | None:
        """Send ``GET /heartbeat`` to the peer.

        Returns:
            JsonDict | None: Parsed JSON response, or ``None`` on
            failure.
        """
        return self.request_with_retry("GET", "/heartbeat", extra_headers=self.base_headers)

    def get_inventory(self) -> JsonDict | None:
        """Send ``GET /inventory`` to the peer.

        Returns:
            JsonDict | None: The peer's inventory digest, or ``None`` on
            failure.
        """
        return self.request_with_retry("GET", "/inventory")

    def inventory_digest(self, page_size: int = 10_000) -> dict[str, int] | None:
        """Fetch the peer's whole inventory, one page at a time.

        Args:
            page_size: Hashes per request.

        Returns:
            dict[str, int] | None: ``content_hash -> version_id``, or
            ``None`` when a page could not be fetched.
        """
        from urllib.parse import quote

        digest: dict[str, int] = {}
        cursor = ""
        while True:
            page = self.request_with_retry("GET", f"/inventory?limit={page_size}&after={quote(cursor)}")
            if not isinstance(page, dict):
                return None
            digest.update(page.get("digest", {}))
            cursor = str(page.get("next", ""))
            if not cursor:
                return digest

    def store_fragment(self, fragment: Fragment, is_primary: bool = False) -> bool:
        """Send ``POST /store`` with ``fragment`` and ``is_primary``.

        Args:
            fragment: The fragment.
            is_primary: Whether this node owns the fragment's primary copy.

        Returns:
            bool: True when the peer stored the fragment.
        """
        payload = {"fragment": to_dict(fragment), "is_primary": is_primary}
        resp = self.request_with_retry("POST", "/store", payload)
        return resp is not None and resp.get("success", False)

    def retrieve_fragment(self, content_hash: str) -> Fragment | None:
        """Send ``GET /retrieve?content_hash=...``.

        Args:
            content_hash: Content hash of the fragment.

        Returns:
            Fragment | None: The fragment, or ``None`` when absent or
            unreachable.
        """
        resp = self.request_with_retry("GET", f"/retrieve?content_hash={content_hash}")
        if resp and resp.get("found"):
            return from_dict(resp["fragment"])
        return None

    def join_cluster(self, node_id: str, host: str, port: int) -> JsonDict | None:
        """Send ``POST /join`` to bootstrap into the cluster.

        Args:
            node_id: Id of this node, sent to the peer.
            host: Address the peer should use to reach this node.
            port: Port the peer should use to reach this node.

        Returns:
            JsonDict | None: The join response with the seed's peers, or
            ``None`` on failure.
        """
        return self.request_with_retry("POST", "/join", {"node_id": node_id, "host": host, "port": port})

    def leave_cluster(self, node_id: str) -> bool:
        """Send ``POST /leave`` to remove ``node_id`` from the cluster.

        Args:
            node_id: Id of this node, sent to the peer.

        Returns:
            bool: True when the peer acknowledged.
        """
        resp = self.request_with_retry("POST", "/leave", {"node_id": node_id})
        return resp is not None and resp.get("success", False)

    def gossip(self, state: JsonDict) -> JsonDict | None:
        """Send ``POST /gossip`` with the supplied state payload.

        Args:
            state: This node's gossip state.

        Returns:
            JsonDict | None: The peer's own gossip state, or ``None`` on
            failure.
        """
        return self.request_with_retry("POST", "/gossip", state)

    def request_replicate(self, fragment: Fragment, is_primary: bool = False) -> bool:
        """Send ``POST /replicate`` with ``fragment``'s metadata.

        The peer refuses (422) a fragment whose payload it does not
        hold, so send the bytes with :meth:`put_blob` first; use
        :func:`membrane.replication.replicate_fragment` for both steps.

        Args:
            fragment: The fragment.
            is_primary: Ask the peer to take over as the primary owner.

        Returns:
            bool: True when the peer stored the replica.
        """
        payload = {"fragment": to_dict(fragment), "is_primary": is_primary}
        resp = self.request_with_retry("POST", "/replicate", payload)
        return resp is not None and resp.get("success", False)

    def put_blob(self, payload_ref: str, data: bytes) -> bool:
        """Upload KV bytes with ``PUT /blobs/{payload_ref}``.

        The peer verifies the ``X-Content-SHA256`` header against the
        body before storing it.

        Args:
            payload_ref: Content-store key of the bytes.
            data: The bytes, exactly as the local content store holds them.

        Returns:
            bool: True when the peer stored (or already held) the bytes.
        """
        headers = {"Content-Type": "application/octet-stream", "X-Content-SHA256": sha256_hex(data)}
        body = data
        method = self.credentials.compression
        if method != "raw" and len(data) >= COMPRESS_MIN_BYTES:
            body = CompressionTransport(method).compress(data)
            headers[COMPRESSION_HEADER] = method
        resp = self.request_with_retry("PUT", f"/blobs/{payload_ref}", raw_body=body, extra_headers=headers)
        return resp is not None and bool(resp.get("stored", False))

    def get_blob(self, payload_ref: str) -> bytes | None:
        """Download KV bytes with ``GET /blobs/{payload_ref}``.

        Args:
            payload_ref: Content-store key of the bytes.

        Returns:
            bytes | None: The verified bytes, or ``None`` when the peer does
            not hold them, they fail verification, or the request fails.
        """
        resp = self.request_raw(
            "GET", f"/blobs/{payload_ref}", {ACCEPT_COMPRESSION_HEADER: self.credentials.compression}
        )
        if resp is None or resp.status != 200:
            return None
        body = resp.body
        if resp.headers.get(COMPRESSION_HEADER.lower()):
            try:
                body = CompressionTransport().decompress(body, max_size=MAX_BLOB_BYTES)
            except (ValueError, RuntimeError) as exc:
                logger.warning("blob %s from %s did not decompress: %s", payload_ref, self.base_url, exc)
                return None
        if resp.headers.get("x-content-sha256") != sha256_hex(body):
            logger.warning("blob %s from %s failed its digest check", payload_ref, self.base_url)
            return None
        return body

    def blob_digest(self, payload_ref: str) -> str | None:
        """Return the SHA-256 the peer reports for its copy of ``payload_ref``.

        Args:
            payload_ref: Content-store key of the bytes.

        Returns:
            str | None: The hex digest, or ``None`` when the peer does not
            hold the bytes or the request fails.
        """
        resp = self.request_raw("HEAD", f"/blobs/{payload_ref}")
        if resp is None or resp.status != 200:
            return None
        return resp.headers.get("x-content-sha256")

    def request_raw(self, method: str, path: str, extra_headers: dict[str, str] | None = None) -> RawResponse | None:
        """Issue a body-less request and return the raw response, with retries.

        Args:
            method: HTTP method.
            path: URL path appended to ``self.base_url``.
            extra_headers: Additional request headers.

        Returns:
            RawResponse | None: The response, or ``None`` on terminal failure.
        """
        url = f"{self.base_url}{path}"
        if not self.breaker.allow():
            return None
        for attempt in range(self.max_retries):
            try:
                headers = {**self.base_headers, **(extra_headers or {})}
                TRACING.inject(headers)
                response = self.transport.request_bytes(method, url, None, headers, self.timeout_sec)
                self.breaker.record_success()
                return response
            except NetworkError as exc:
                logger.debug("%s %s failed (attempt %s/%s): %s", method, url, attempt + 1, self.max_retries, exc)
                if attempt < self.max_retries - 1:
                    time.sleep(compute_backoff(self.retry, attempt))
        self.breaker.record_failure()
        return None

    def request_delete(
        self,
        content_hash: str,
        node_id: str,
        tombstone_until: float | None = None,
    ) -> bool:
        """Send ``POST /delete`` for ``content_hash``.

        Args:
            content_hash: Hash to soft-delete.
            node_id: Local node identifier propagated to the peer.
            tombstone_until: Optional Unix deadline.

        Returns:
            bool: ``True`` when the peer's :func:`op_delete`
            succeeded.
        """
        body: dict[str, object] = {
            "content_hash": content_hash,
            "node_id": node_id,
        }
        if tombstone_until is not None:
            body["tombstone_until"] = tombstone_until
        resp = self.request_with_retry("POST", "/delete", body)
        return resp is not None and bool(resp.get("success", False))

    def request_tombstone(
        self,
        content_hash: str,
        until: float,
        node_id: str,
    ) -> bool:
        """Send ``POST /tombstone`` marking a soft-delete.

        Args:
            content_hash: Hash to tombstone.
            until: Wall-clock deadline.
            node_id: Originating node identifier.

        Returns:
            bool: ``True`` when the peer's :func:`op_tombstone`
            succeeded.
        """
        body = {"content_hash": content_hash, "until": until, "node_id": node_id}
        resp = self.request_with_retry("POST", "/tombstone", body)
        return resp is not None and bool(resp.get("success", False))

    def request_verify_received(
        self,
        content_hash: str,
        claimed_size: int,
        claimed_sha256_hex: str,
    ) -> bool:
        """Send ``POST /verify`` confirming a stored fragment.

        The verified-migration flow on the new primary calls this
        to confirm the destination replica actually received the
        canonical bytes with the expected size before flipping
        the shard map.

        Args:
            content_hash: Hash of the fragment just received.
            claimed_size: Size the caller believes it stored.
            claimed_sha256_hex: Hex sha256 the caller computed.

        Returns:
            bool: ``True`` when the destination acks the bytes
            match. ``False`` when the destination reports a size
            mismatch, a missing fragment, or any transport
            failure.
        """
        body = {
            "content_hash": content_hash,
            "claimed_size": int(claimed_size),
            "claimed_sha256": str(claimed_sha256_hex),
        }
        resp = self.request_with_retry("POST", "/verify", body)
        return resp is not None and bool(resp.get("success", False))

    def get_peers(self) -> JsonDict | None:
        """Send ``GET /peers``.

        Returns:
            JsonDict | None: The peer's membership list, or ``None`` on failure.
        """
        return self.request_with_retry("GET", "/peers")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def request_with_retry(
        self,
        method: str,
        path: str,
        payload: JsonDict | None = None,
        extra_headers: dict[str, str] | None = None,
        raw_body: bytes | None = None,
    ) -> JsonDict | None:
        """Issue an HTTP request with retries and exponential backoff.

        A ``None`` response from the transport is retried up to
        ``max_retries`` times with ``retry_delay_sec * 2 ** attempt``
        seconds between attempts. Tests that want to simulate
        transient failures can return ``None`` from a mocked
        transport.

        Args:
            method: HTTP method.
            path: URL path appended to ``self.base_url``.
            payload: Optional JSON-serializable body.
            extra_headers: Optional additional headers merged
                with the per-request content-type. Used by
                :meth:`heartbeat` to attach the local
                ``X-Local-Peer-CN``.
            raw_body: Request body sent as-is instead of ``payload``.

        Returns:
            JsonDict | None: Parsed JSON response or ``None`` on
            terminal failure.
        """
        url = f"{self.base_url}{path}"
        data = raw_body if raw_body is not None else (json.dumps(payload).encode() if payload else None)
        headers = dict(self.base_headers)
        if payload:
            headers["Content-Type"] = "application/json"
        if extra_headers:
            headers.update(extra_headers)
        TRACING.inject(headers)
        last_error: Exception | None = None

        if not self.breaker.allow():
            logger.debug("circuit open for %s; failing fast on %s %s", self.base_url, method, path)
            return None
        deadline = peer_deadline.get()
        for attempt in range(self.max_retries):
            timeout = self.timeout_sec
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                timeout = min(timeout, remaining)
            try:
                resp = self.transport.request(
                    method=method,
                    url=url,
                    body=data,
                    headers=headers,
                    timeout_sec=timeout,
                )
                if resp is not None:
                    self.breaker.record_success()
                    return resp
            except NetworkError as exc:
                last_error = exc

            if attempt == self.max_retries - 1:
                break  # no point sleeping after the last attempt
            # Exponential backoff: 1x, 2x, 4x, ...
            delay = compute_backoff(self.retry, attempt)
            if deadline is not None and time.monotonic() + delay >= deadline:
                break  # the caller has already given up
            logger.debug(
                "Request to %s%s failed (attempt %s/%s), retrying in %.1fs",
                self.base_url,
                path,
                attempt + 1,
                self.max_retries,
                delay,
            )
            time.sleep(delay)

        self.breaker.record_failure()
        logger.warning(
            "Request to %s%s failed after %s retries: %s",
            self.base_url,
            path,
            self.max_retries,
            last_error,
        )
        return None


__all__ = [
    "HTTPTransport",
    "Peer",
    "PeerCredentials",
    "RawResponse",
    "Transport",
    "get_default_peer_credentials",
    "peer_deadline",
    "peer_url",
    "set_default_peer_credentials",
]
