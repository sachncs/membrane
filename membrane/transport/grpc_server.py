"""GrpcServer: production gRPC server for Membrane nodes.

Requires ``grpcio`` to be installed. Falls back to HTTP-only mode
when ``grpcio`` is unavailable (a warning is logged at
construction time and :meth:`start` raises
:class:`RuntimeError`).

The gRPC service surface mirrors the HTTP endpoints defined in
:class:`~membrane.transport.http_server.HTTPServer`:

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
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from membrane.fragment import Fragment
from membrane.structural_signature import StructuralSignature

logger = logging.getLogger(__name__)


class GrpcServer:
    """gRPC server wrapper for Membrane.

    Args:
        node: MembraneNode to serve.
        host: Bind address.
        port: Listen port.
    """

    def __init__(
        self,
        node,
        host: str = "0.0.0.0",
        port: int = 50051,
        compute_backend=None,
    ) -> None:
        """Initialize the gRPC server wrapper.

        Args:
            node: MembraneNode to serve.
            host: Bind address.
            port: Listen port.
            compute_backend: Optional compute backend used by
                the ``Prefill`` RPC.
        """
        self.node = node
        self.host = host
        self.port = port
        self.compute_backend = compute_backend
        self._server: Any | None = None
        self._grpc: Any | None = None
        try:
            import grpc  # type: ignore[import-untyped]

            self._grpc = grpc
        except ImportError:
            logger.warning(
                "grpcio not installed; GrpcServer will not function. "
                "Install with: pip install grpcio grpcio-tools"
            )

    def start(self) -> None:
        """Start the gRPC server (blocking).

        Raises:
            RuntimeError: When ``grpcio`` is not installed.
        """
        grpc_module = self._grpc
        if grpc_module is None:
            raise RuntimeError("grpcio is not installed")

        from membrane.transport.proto import membrane_pb2
        from membrane.transport.proto import membrane_pb2_grpc

        servicer = MembraneServicer(self.node, self.compute_backend)
        self._server = grpc_module.server(thread_pool=ThreadPoolExecutor(max_workers=10))
        membrane_pb2_grpc.add_MembraneServicer_to_server(servicer, self._server)
        # Insecure port for local development. For production
        # use a TLS-enabled port via add_secure_port().
        self._server.add_insecure_port(f"{self.host}:{self.port}")
        self._server.start()
        logger.info("gRPC server started on %s:%s", self.host, self.port)
        self._server.wait_for_termination()

    def stop(self) -> None:
        """Stop the gRPC server.

        Passes a zero grace period to :meth:`grpc.Server.stop` so
        shutdown is immediate. No-op when the server was never
        started.
        """
        if self._server:
            self._server.stop(0)
            logger.info("gRPC server stopped")


class MembraneServicer:
    """Implementation of the Membrane gRPC service.

    Attributes:
        node: Local :class:`MembraneNode` instance.
        compute_backend: Optional :class:`ComputeBackend`.
        _pb2: Lazily imported ``membrane_pb2`` module.
    """

    def __init__(self, node, compute_backend) -> None:
        """Initialize the servicer with the local node.

        Args:
            node: Local :class:`MembraneNode`.
            compute_backend: Optional compute backend.
        """
        self.node = node
        self.compute_backend = compute_backend
        from membrane.transport.proto import membrane_pb2

        self._pb2: Any = membrane_pb2

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
        return self._pb2.StoreResponse(
            success=success,
            content_hash=frag.content_hash,
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
            return self._pb2.RetrieveResponse(found=False)
        return self._pb2.RetrieveResponse(
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
        stats = self.node.get_stats()
        digest = {h: frag.version_id for h, frag in self.node.fragments.items()}
        return self._pb2.InventoryResponse(
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
        return self._pb2.PrefillResponse(
            success=True,
            fragments=[self.fragment_to_pb(f) for f in frags],
            kv_size_mib=sum(f.size for f in frags) / (1024 * 1024),
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
        return self._pb2.HeartbeatResponse(
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
        """Convert a protobuf Fragment message to a Fragment dataclass.

        Args:
            msg: ``FragmentMessage`` protobuf instance.

        Returns:
            Fragment: Reconstructed fragment.
        """
        return Fragment(
            content_hash=msg.content_hash,
            embedding=tuple(msg.embedding),
            structural_signature=StructuralSignature(
                model_id=msg.model_id,
                layer_range=(msg.layer_start, msg.layer_end),
                token_span=(msg.token_start, msg.token_end),
            ),
            size=msg.size,
            ttl=msg.ttl,
            reuse_score=msg.reuse_score,
            version_id=msg.version_id,
        )

    def fragment_to_pb(self, frag: Fragment):
        """Convert a Fragment dataclass to a protobuf FragmentMessage.

        Args:
            frag: Fragment to serialize.

        Returns:
            FragmentMessage: Protobuf message suitable for
            transport over gRPC.
        """
        return self._pb2.FragmentMessage(
            content_hash=frag.content_hash,
            embedding=list(frag.embedding),
            model_id=frag.structural_signature.model_id,
            layer_start=frag.structural_signature.layer_range[0],
            layer_end=frag.structural_signature.layer_range[1],
            token_start=frag.structural_signature.token_span[0],
            token_end=frag.structural_signature.token_span[1],
            size=frag.size,
            ttl=frag.ttl,
            reuse_score=frag.reuse_score,
            version_id=frag.version_id,
        )
