"""GrpcServer: production gRPC server for Membrane nodes.

Requires ``grpcio`` to be installed. Falls back to HTTP-only mode
when ``grpcio`` is unavailable (a warning is logged at
construction time and :meth:`start` raises
:class:`RuntimeError`).

The gRPC service surface mirrors the HTTP endpoints defined in
:class:`~membrane.transport.fastapi.FastAPIServer`:

* ``StoreFragment`` — store a fragment (primary or replica).
* ``RetrieveFragment`` — retrieve a fragment by hash.
* ``SyncInventory`` — return the node's inventory digest.
* ``Prefill`` — run prefill and return fragments.
* ``Heartbeat`` — return node health and load.

Note:
    The generated ``membrane_pb2`` and ``membrane_pb2_grpc``
    modules are imported lazily because they require ``grpcio``
    and ``grpcio-tools`` at install time.
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from membrane.constants import DEFAULT_MAX_BODY_BYTES
from membrane.errors import SchemaError
from membrane.fragment import Fragment
from membrane.identity import PayloadIdentity
from membrane.transport.tls import MTLSConfig

logger = logging.getLogger(__name__)


SCHEMA_VERSION: int = 2


class GrpcServer:
    """gRPC server wrapper for Membrane.

    Args:
        node: Node to serve.
        host: Bind address.
        port: Listen port.
        compute_backend: Optional :class:`Backend` for the
            ``Prefill`` RPC.
        tls: Optional :class:`MTLSConfig`. When supplied, the
            server binds on the secure port with mTLS via
            ``grpc.ssl_server_credentials(root_certificates=...,
            certificate_chain=..., private_key=...)``. Callers
            must also configure ``Peer`` clients (Phase 1.5) with
            the matching :func:`membrane.transport.tls.build_client_context`.
    """

    def __init__(
        self,
        node,
        host: str = "0.0.0.0",
        port: int = 50051,
        compute_backend=None,
        tls: MTLSConfig | None = None,
    ) -> None:
        """Initialize the gRPC server wrapper.

        Args:
            node: Node to serve.
            host: Bind address.
            port: Listen port.
            compute_backend: Optional compute backend used by
                the ``Prefill`` RPC.
            tls: Optional :class:`MTLSConfig`. When supplied, the
                server enables mutual TLS on the bind port.
        """
        self.node = node
        self.host = host
        self.port = port
        self.compute_backend = compute_backend
        self.tls = tls
        self.grpc_server: Any | None = None
        self.grpc_module: Any | None = None
        try:
            # gRPC lacks type stubs, hence the type: ignore on the
            # import. The alias is captured for later use.
            import grpc as grpc_module  # type: ignore[import-untyped]

            self.grpc_module = grpc_module
        except ImportError:
            logger.warning(
                "grpcio not installed; GrpcServer will not function. Install with: pip install grpcio grpcio-tools"
            )

    def start(self) -> None:
        """Start the gRPC server (blocking).

        Raises:
            RuntimeError: When ``grpcio`` is not installed.
        """
        grpc_module = self.grpc_module
        if grpc_module is None:
            raise RuntimeError("grpcio is not installed")

        from membrane.transport.proto import membrane_pb2_grpc

        servicer = Handler(self.node, self.compute_backend)
        # The default gRPC receive/send limits are 4 MiB; bump them
        # to match the wire-config MAX_BODY_BYTES so a real
        # payload (~ a 1 GiB window per-layer fragment) can be
        # carried inline as ``bytes payload = 14``.
        server_options = [
            ("grpc.max_receive_message_length", DEFAULT_MAX_BODY_BYTES),
            ("grpc.max_send_message_length", DEFAULT_MAX_BODY_BYTES),
        ]
        self.grpc_server = grpc_module.server(
            thread_pool=ThreadPoolExecutor(max_workers=10),
            options=server_options,
        )
        membrane_pb2_grpc.add_MembraneServicer_to_server(servicer, self.grpc_server)
        if self.tls is not None:
            self._bind_secure_port()
        else:
            # Insecure port for local development only; production
            # callers must supply tls.
            self.grpc_server.add_insecure_port(f"{self.host}:{self.port}")
            logger.warning(
                "gRPC server bound on insecure port %s:%s; pass tls=MTLSConfig for mTLS",
                self.host,
                self.port,
            )
        self.grpc_server.start()
        logger.info("gRPC server started on %s:%s", self.host, self.port)
        self.grpc_server.wait_for_termination()

    def _bind_secure_port(self) -> None:
        """Bind the secure port when ``tls`` is configured.

        Reads the configured :class:`MTLSConfig` and turns each
        PEM block into a temporary on-disk file because
        ``ssl_server_credentials`` in this grpcio version reads
        certs from disk rather than from PEM bytes.
        """
        import contextlib
        import os
        import tempfile

        from membrane.transport.tls import build_server_context

        assert self.tls is not None
        with tempfile.NamedTemporaryFile(
            mode="w", delete=False, suffix=".pem"
        ) as cert_file, tempfile.NamedTemporaryFile(
            mode="w", delete=False, suffix=".pem"
        ) as key_file, tempfile.NamedTemporaryFile(
            mode="w", delete=False, suffix=".pem"
        ) as ca_file:
            cert_file.write(self.tls.server_cert_pem)
            cert_file.flush()
            key_file.write(self.tls.server_key_pem)
            key_file.flush()
            ca_file.write(self.tls.ca_bundle_pem)
            ca_file.flush()
            cert_path = cert_file.name
            key_path = key_file.name
            ca_path = ca_file.name
        # Verify the SSL context was buildable before invoking
        # grpc.ssl_server_credentials (which does not surface
        # clear error messages for malformed chains).
        build_server_context(self.tls)
        try:
            with open(cert_path, "rb") as f:
                cert_bytes = f.read()
            with open(key_path, "rb") as f:
                key_bytes = f.read()
            with open(ca_path, "rb") as f:
                ca_bytes = f.read()
            credentials = self.grpc_module.ssl_server_credentials(  # type: ignore[union-attr]
                root_certificates=ca_bytes,
                certificate_chain=cert_bytes,
                private_key=key_bytes,
                require_client_auth=self.tls.require_client_cert,
            )
            self.grpc_server.add_secure_port(  # type: ignore[union-attr]
                f"{self.host}:{self.port}", credentials
            )
        finally:
            for path in (cert_path, key_path, ca_path):
                with contextlib.suppress(OSError):
                    os.unlink(path)

    def stop(self) -> None:
        """Stop the gRPC server.

        Passes a zero grace period to :meth:`grpc.Server.stop` so
        shutdown is immediate. No-op when the server was never
        started.
        """
        if self.grpc_server:
            self.grpc_server.stop(0)
            logger.info("gRPC server stopped")


class Handler:
    """Implementation of the Membrane gRPC service.

    Attributes:
        node: Local :class:`Node` instance.
        compute_backend: Optional :class:`Backend`.
        _pb2: Lazily imported ``membrane_pb2`` module.
    """

    def __init__(self, node, compute_backend) -> None:
        """Initialize the servicer with the local node.

        Args:
            node: Local :class:`Node`.
            compute_backend: Optional compute backend.
        """
        self.node = node
        self.compute_backend = compute_backend
        from membrane.transport.proto import membrane_pb2

        self.pb2_module: Any = membrane_pb2

    def StoreFragment(self, request, context):
        """gRPC handler: store a fragment on the local node.

        Args:
            request: ``StoreRequest`` carrying the fragment
                message and ``is_primary`` flag.
            context: gRPC ``ServicerContext``.

        Returns:
            StoreResponse: ``success`` and ``content_hash``.
        """
        frag = self.pb_to_fragment(request.fragment)
        success = self.node.store(frag, is_primary=request.is_primary)
        return self.pb2_module.StoreResponse(
            success=success,
            content_hash=frag.identity.payload_hash,
        )

    def RetrieveFragment(self, request, context):
        """gRPC handler: retrieve a fragment by content hash.

        Args:
            request: ``RetrieveRequest`` carrying the content
                hash.
            context: gRPC ``ServicerContext``.

        Returns:
            RetrieveResponse: ``found`` flag plus the
            fragment message (when present).
        """
        frag = self.node.retrieve(request.content_hash)
        if frag is None:
            return self.pb2_module.RetrieveResponse(found=False)
        return self.pb2_module.RetrieveResponse(
            found=True,
            fragment=self.fragment_to_pb(frag),
        )

    def SyncInventory(self, request, context):
        """gRPC handler: return the node's inventory digest.

        Args:
            request: ``InventoryRequest`` (empty payload).
            context: gRPC ``ServicerContext``.

        Returns:
            InventoryResponse: ``digest`` map plus the
            ``node_id``.
        """
        digest = {h: frag.version_id for h, frag in self.node.fragments.items()}
        return self.pb2_module.InventoryResponse(
            digest=digest,
            node_id=self.node.node_id,
        )

    def Prefill(self, request, context):
        """gRPC handler: run prefill and return the resulting fragments.

        Args:
            request: ``PrefillRequest`` carrying prompt tokens
                and the model id.
            context: gRPC ``ServicerContext``.

        Returns:
            PrefillResponse: ``success``, the fragment
            messages, the total KV size in MiB, and the
            measured latency.
        """
        t0 = time.time()
        frags = self.compute_backend.prefill(list(request.prompt_tokens), request.model_id)
        latency = time.time() - t0
        return self.pb2_module.PrefillResponse(
            success=True,
            fragments=[self.fragment_to_pb(f) for f in frags],
            kv_size_mib=sum(f.payload_size for f in frags) / (1024 * 1024),
            latency_seconds=latency,
        )

    def Heartbeat(self, request, context):
        """gRPC handler: return node health and load.

        Args:
            request: ``HeartbeatRequest`` (empty payload).
            context: gRPC ``ServicerContext``.

        Returns:
            HeartbeatResponse: ``node_id``, ``load``,
            ``memory_used_bytes``, ``memory_limit_bytes``,
            ``fragment_count``, ``healthy``.
        """
        stats = self.node.get_stats()
        return self.pb2_module.HeartbeatResponse(
            node_id=self.node.node_id,
            load=self.node.heartbeat(),
            memory_used_bytes=stats.memory_used_bytes,
            memory_limit_bytes=stats.memory_limit_bytes,
            fragment_count=stats.fragment_count,
            healthy=True,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def pb_to_fragment(self, msg) -> Fragment:
        """Convert a protobuf FragmentMessage (schema v2) to a Fragment.

        Reads the full :class:`~membrane.identity.PayloadIdentity`
        sub-fields and the ``payload_ref`` / ``payload_size`` /
        ``payload`` fields from the wire message. The v1
        ``content_hash`` field is absent; the new
        ``payload_hash`` is a ``bytes`` field carrying the raw
        32-byte SHA-256 digest.

        Args:
            msg: ``FragmentMessage`` protobuf instance.

        Returns:
            Fragment: Reconstructed fragment.

        Raises:
            SchemaError: When ``schema_version`` is not 2.
        """
        if msg.schema_version != SCHEMA_VERSION:
            raise SchemaError(
                f"incompatible gRPC schema_version={msg.schema_version}; expected {SCHEMA_VERSION}"
            )
        payload_hash_hex = msg.payload_hash.hex() if msg.payload_hash else ""
        identity = PayloadIdentity(
            payload_hash=payload_hash_hex,
            model_id=msg.model_id,
            model_revision=msg.model_revision,
            tokenizer_name=msg.tokenizer_name or msg.model_id,
            tokenizer_revision=msg.tokenizer_revision,
            layer_range=(msg.layer_range[0], msg.layer_range[1]),
            head_range=(msg.head_range[0], msg.head_range[1]),
            token_span=(msg.token_span[0], msg.token_span[1]),
            dtype=msg.dtype or "float16",
            shape=tuple(msg.shape),
        )
        payload_ref = msg.payload_ref or None
        return Fragment(
            identity=identity,
            payload_ref=payload_ref,
            payload_size=msg.payload_size,
            ttl=msg.ttl,
            reuse_score=msg.reuse_score,
            version_id=msg.version_id,
        )

    def fragment_to_pb(self, frag: Fragment):
        """Convert a Fragment dataclass to a protobuf FragmentMessage (schema v2).

        Args:
            frag: Fragment to serialize.

        Returns:
            FragmentMessage: Protobuf message suitable for
            transport over gRPC.
        """
        identity = frag.identity
        payload_hash_bytes = bytes.fromhex(identity.payload_hash) if identity.payload_hash else b""
        return self.pb2_module.FragmentMessage(
            schema_version=SCHEMA_VERSION,
            payload_hash=payload_hash_bytes,
            model_id=identity.model_id,
            model_revision=identity.model_revision,
            tokenizer_name=identity.tokenizer_name,
            tokenizer_revision=identity.tokenizer_revision,
            layer_range=list(identity.layer_range),
            head_range=list(identity.head_range),
            token_span=list(identity.token_span),
            dtype=identity.dtype,
            shape=list(identity.shape),
            payload_ref=frag.payload_ref or "",
            payload_size=frag.payload_size,
            payload=b"",
            ttl=frag.ttl,
            reuse_score=frag.reuse_score,
            version_id=frag.version_id,
        )
