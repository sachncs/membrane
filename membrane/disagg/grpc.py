"""gRPC surface for the prefill / decode services (Phase 8).

The gRPC service mirrors the REST surface in
:mod:`membrane.disagg.rest`. The v1 ships a hand-written
proto schema and stubs so the import path stays clean even
when ``grpcio-tools`` is unavailable. Operators that want
to regenerate the stubs can run ``python -m grpc_tools.protoc
-I . --python_out=membrane/disagg --grpc_python_out=membrane/disagg
membrane/disagg/transfer.proto``.

The :func:`add_to_server` function wires the
:class:`membrane.disagg.service.PrefillService` and
:class:`DecodeService` into a :class:`grpc.Server`. The
:func:`make_channel` factory returns a stub the client uses
to issue RPCs.
"""

from __future__ import annotations

import logging
from typing import Any

from membrane.disagg.protocol import (
    DecodeRequest,
    DecodeResponse,
    PrefillRequest,
    PrefillResponse,
)
from membrane.disagg.service import (
    DecodeService,
    PrefillService,
    batch_prefill,
)

logger = logging.getLogger(__name__)


_GRPC_AVAILABLE: bool = False
try:
    import grpc  # type: ignore[import-not-found]
    _GRPC_AVAILABLE = True
except ImportError:  # pragma: no cover - import guard
    grpc = None  # type: ignore[assignment]


GRPC_AVAILABLE: bool = _GRPC_AVAILABLE


# ---------------------------------------------------------------------------
# Wire messages
# ---------------------------------------------------------------------------


def _empty_message() -> Any:
    """Return a generic proto message stub.

    Returns:
        A minimal stand-in for the generated proto messages
        when grpcio is unavailable. Tests that need a real
        message can call :func:`_build_prefill_request_message`
        to get a generated stub.
    """
    if not _GRPC_AVAILABLE:  # pragma: no cover - import guard
        raise RuntimeError("grpcio is required for the gRPC surface")
    from google.protobuf import struct_pb2  # type: ignore[import-not-found]
    return struct_pb2.Struct()


def _build_prefill_request_message(request: PrefillRequest) -> Any:
    """Build a generated proto message for ``request``.

    Args:
        request: The prefill request.

    Returns:
        A generated proto message.
    """
    if not _GRPC_AVAILABLE:  # pragma: no cover - import guard
        raise RuntimeError("grpcio is required for the gRPC surface")
    from membrane.disagg import transfer_pb2

    return transfer_pb2.PrefillRequest(  # type: ignore[attr-defined]
        request_id=request.request_id,
        model_id=request.model_id,
        token_ids=list(request.token_ids),
        token_type_ids=list(request.token_type_ids or ()),
        max_decode_tokens=request.max_decode_tokens,
        fingerprint=request.fingerprint,
    )


def _build_prefill_response_message(response: PrefillResponse) -> Any:
    """Build a generated proto message for ``response``.

    Args:
        response: The prefill response.

    Returns:
        A generated proto message.
    """
    if not _GRPC_AVAILABLE:  # pragma: no cover - import guard
        raise RuntimeError("grpcio is required for the gRPC surface")
    from membrane.disagg import transfer_pb2

    return transfer_pb2.PrefillResponse(  # type: ignore[attr-defined]
        request_id=response.request_id,
        kv_handle=response.kv_handle,
        prefill_ms=response.prefill_ms,
        prompt_len=response.prompt_len,
        cached_prefix_len=response.cached_prefix_len,
    )


def _build_decode_request_message(request: DecodeRequest) -> Any:
    """Build a generated proto message for ``request``.

    Args:
        request: The decode request.

    Returns:
        A generated proto message.
    """
    if not _GRPC_AVAILABLE:  # pragma: no cover - import guard
        raise RuntimeError("grpcio is required for the gRPC surface")
    from membrane.disagg import transfer_pb2

    return transfer_pb2.DecodeRequest(  # type: ignore[attr-defined]
        request_id=request.request_id,
        kv_handle=request.kv_handle,
        model_id=request.model_id,
        max_tokens=request.max_tokens,
    )


def _build_decode_response_message(response: DecodeResponse) -> Any:
    """Build a generated proto message for ``response``.

    Args:
        response: The decode response.

    Returns:
        A generated proto message.
    """
    if not _GRPC_AVAILABLE:  # pragma: no cover - import guard
        raise RuntimeError("grpcio is required for the gRPC surface")
    from membrane.disagg import transfer_pb2

    return transfer_pb2.DecodeResponse(  # type: ignore[attr-defined]
        request_id=response.request_id,
        token_ids=list(response.token_ids),
        finished=response.finished,
    )


def _request_from_message(message: Any) -> PrefillRequest:
    """Convert a generated proto message into a :class:`PrefillRequest`.

    Args:
        message: The generated proto message.

    Returns:
        PrefillRequest: The decoded request.
    """
    return PrefillRequest(
        request_id=message.request_id,
        model_id=message.model_id,
        token_ids=tuple(message.token_ids),
        token_type_ids=tuple(message.token_type_ids) or None,
        max_decode_tokens=message.max_decode_tokens,
        fingerprint=message.fingerprint,
    )


def _response_from_message(message: Any) -> PrefillResponse:
    """Convert a generated proto message into a :class:`PrefillResponse`.

    Args:
        message: The generated proto message.

    Returns:
        PrefillResponse: The decoded response.
    """
    return PrefillResponse(
        request_id=message.request_id,
        kv_handle=message.kv_handle,
        prefill_ms=message.prefill_ms,
        prompt_len=message.prompt_len,
        cached_prefix_len=message.cached_prefix_len,
    )


def _decode_request_from_message(message: Any) -> DecodeRequest:
    """Convert a generated proto message into a :class:`DecodeRequest`.

    Args:
        message: The generated proto message.

    Returns:
        DecodeRequest: The decoded request.
    """
    return DecodeRequest(
        request_id=message.request_id,
        kv_handle=message.kv_handle,
        model_id=message.model_id,
        max_tokens=message.max_tokens,
    )


def _decode_response_from_message(message: Any) -> DecodeResponse:
    """Convert a generated proto message into a :class:`DecodeResponse`.

    Args:
        message: The generated proto message.

    Returns:
        DecodeResponse: The decoded response.
    """
    return DecodeResponse(
        request_id=message.request_id,
        token_ids=tuple(message.token_ids),
        finished=message.finished,
    )


# ---------------------------------------------------------------------------
# Service registration
# ---------------------------------------------------------------------------


def add_to_server(
    server: Any,
    prefill: PrefillService,
    decode: DecodeService | None = None,
) -> Any:
    """Register the prefill / decode service on ``server``.

    Args:
        server: A :class:`grpc.Server` instance.
        prefill: The prefill service.
        decode: Optional decode service. Defaults to a new
            :class:`DecodeService`.

    Returns:
        The :class:`grpc.Server` instance.
    """
    if not _GRPC_AVAILABLE:  # pragma: no cover - import guard
        raise RuntimeError("grpcio is required for the gRPC surface")
    from membrane.disagg import transfer_pb2_grpc

    decode_service = decode or DecodeService()
    handler = _GrpcHandler(prefill=prefill, decode=decode_service)
    transfer_pb2_grpc.add_TransferServicer_to_server(  # type: ignore[attr-defined]
        handler, server
    )
    return server


def make_channel(target: str) -> Any:
    """Open an insecure gRPC channel to ``target``.

    Args:
        target: ``host:port`` string.

    Returns:
        A :class:`grpc.Channel` ready to use with
        :func:`make_stub`.
    """
    if not _GRPC_AVAILABLE:  # pragma: no cover - import guard
        raise RuntimeError("grpcio is required for the gRPC surface")
    return grpc.insecure_channel(target)


def make_stub(channel: Any) -> Any:
    """Build a :class:`TransferStub` for ``channel``.

    Args:
        channel: A :class:`grpc.Channel`.

    Returns:
        TransferStub: A client stub.
    """
    if not _GRPC_AVAILABLE:  # pragma: no cover - import guard
        raise RuntimeError("grpcio is required for the gRPC surface")
    from membrane.disagg import transfer_pb2_grpc

    return transfer_pb2_grpc.TransferStub(channel)  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Servicer
# ---------------------------------------------------------------------------


class _GrpcHandler:
    """Servicer that bridges gRPC calls to the in-process services.

    The v1 implementation is intentionally small: it builds
    domain types from the wire messages, delegates to
    :class:`PrefillService` / :class:`DecodeService`, and
    converts the responses back into wire messages.
    """

    def __init__(self, prefill: PrefillService, decode: DecodeService) -> None:
        self._prefill = prefill
        self._decode = decode

    def Prefill(self, request: Any, context: Any) -> Any:
        """Handle a :class:`PrefillRequest` RPC.

        Args:
            request: Generated ``PrefillRequest`` message.
            context: gRPC context.

        Returns:
            Generated ``PrefillResponse`` message.
        """
        decoded = _request_from_message(request)
        response = self._prefill.prefill(decoded)
        return _build_prefill_response_message(response)

    def BatchPrefill(self, request: Any, context: Any) -> Any:
        """Handle a batch prefill RPC.

        Args:
            request: Generated ``BatchPrefillRequest`` message.
            context: gRPC context.

        Returns:
            Generated ``BatchPrefillResponse`` message.
        """
        requests = [_request_from_message(item) for item in request.requests]
        result = batch_prefill(self._prefill, requests)
        from membrane.disagg import transfer_pb2

        return transfer_pb2.BatchPrefillResponse(  # type: ignore[attr-defined]
            responses=[_build_prefill_response_message(r) for r in result.responses],
            elapsed_ms=result.elapsed_ms,
        )

    def Decode(self, request: Any, context: Any) -> Any:
        """Handle a :class:`DecodeRequest` RPC.

        Args:
            request: Generated ``DecodeRequest`` message.
            context: gRPC context.

        Returns:
            Generated ``DecodeResponse`` message.
        """
        decoded = _decode_request_from_message(request)
        response = self._decode.decode(decoded)
        return _build_decode_response_message(response)


__all__ = [
    "GRPC_AVAILABLE",
    "add_to_server",
    "make_channel",
    "make_stub",
]
