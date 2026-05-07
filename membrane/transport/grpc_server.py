"""GrpcServer: production gRPC server for Membrane nodes.

Requires ``grpcio`` to be installed.  Falls back to HTTP-only mode
if grpc is unavailable.
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
        compute_backend = None,
    ) -> None:
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
        """Start the gRPC server (blocking)."""
        grpc_module = self._grpc
        if grpc_module is None:
            raise RuntimeError("grpcio is not installed")

        from membrane.transport.proto import membrane_pb2
        from membrane.transport.proto import membrane_pb2_grpc

        servicer = _MembraneServicer(self.node, self.compute_backend)
        from concurrent.futures import ThreadPoolExecutor
        self._server = grpc_module.server(thread_pool=ThreadPoolExecutor(max_workers=10))
        membrane_pb2_grpc.add_MembraneServicer_to_server(servicer, self._server)
        self._server.add_insecure_port(f"{self.host}:{self.port}")
        self._server.start()
        logger.info("gRPC server started on %s:%s", self.host, self.port)
        self._server.wait_for_termination()

    def stop(self) -> None:
        """Stop the gRPC server."""
        if self._server:
            self._server.stop(0)
            logger.info("gRPC server stopped")


class _MembraneServicer:
    """Implementation of the Membrane gRPC service."""

    def __init__(self, node, compute_backend) -> None:
        self.node = node
        self.compute_backend = compute_backend
        from membrane.transport.proto import membrane_pb2

        self._pb2: Any = membrane_pb2

    def StoreFragment(self, request, context):
        frag = self._to_fragment(request.fragment)
        success = self.node.store(frag, is_primary=request.is_primary)
        return self._pb2.StoreResponse(
            success=success,
            content_hash=frag.content_hash,
        )

    def RetrieveFragment(self, request, context):
        frag = self.node.retrieve(request.content_hash)
        if frag is None:
            return self._pb2.RetrieveResponse(found=False)
        return self._pb2.RetrieveResponse(
            found=True,
            fragment=self._to_message(frag),
        )

    def SyncInventory(self, request, context):
        stats = self.node.get_stats()
        digest = {h: frag.version_id for h, frag in self.node.fragments.items()}
        return self._pb2.InventoryResponse(
            digest=digest,
            node_id=self.node.node_id,
        )

    def Prefill(self, request, context):
        import time as _time
        t0 = _time.time()
        frags = self.compute_backend.prefill(list(request.prompt_tokens), request.model_id)
        latency = _time.time() - t0
        return self._pb2.PrefillResponse(
            success=True,
            fragments=[self._to_message(f) for f in frags],
            kv_size_mib=sum(f.size for f in frags) / (1024 * 1024),
            latency_seconds=latency,
        )

    def Heartbeat(self, request, context):
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

    def _to_fragment(self, msg) -> Fragment:
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

    def _to_message(self, frag: Fragment):
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
