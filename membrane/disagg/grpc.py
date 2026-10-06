"""gRPC surface for the prefill / decode services.

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

import logging
from collections.abc import Callable
from concurrent import futures
from typing import Any

from membrane.auth import AuthBackendError, Authenticator, AuthForbiddenError
from membrane.disagg.protocol import (
    DecodeRequest,
    DecodeResponse,
    PrefillRequest,
    PrefillResponse,
)
from membrane.disagg.service import (
    CALLER,
    DecodeService,
    PrefillService,
    RoleUnavailableError,
    batch_prefill,
)
from membrane.transport.authz import enforce_route_scope
from membrane.transport.tls import MTLSConfig

logger = logging.getLogger(__name__)


GRPC_IMPORTED: bool = False
try:
    import grpc  # type: ignore[import-not-found]

    GRPC_IMPORTED = True
except ImportError:  # pragma: no cover - import guard
    grpc = None  # type: ignore[assignment]


GRPC_AVAILABLE: bool = GRPC_IMPORTED


# ---------------------------------------------------------------------------
# Wire messages
# ---------------------------------------------------------------------------


def empty_message() -> Any:
    """Return a generic proto message stub.

    Returns:
        A minimal stand-in for the generated proto messages
        when grpcio is unavailable. Tests that need a real
        message can call :func:`build_prefill_request_message`
        to get a generated stub.
    """
    if not GRPC_IMPORTED:  # pragma: no cover - import guard
        raise RuntimeError("grpcio is required for the gRPC surface")
    from google.protobuf import struct_pb2  # type: ignore[import-not-found]

    return struct_pb2.Struct()


def build_prefill_request_message(request: PrefillRequest) -> Any:
    """Build a generated proto message for ``request``.

    Args:
        request: The prefill request.

    Returns:
        A generated proto message.
    """
    if not GRPC_IMPORTED:  # pragma: no cover - import guard
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


def build_prefill_response_message(response: PrefillResponse) -> Any:
    """Build a generated proto message for ``response``.

    Args:
        response: The prefill response.

    Returns:
        A generated proto message.
    """
    if not GRPC_IMPORTED:  # pragma: no cover - import guard
        raise RuntimeError("grpcio is required for the gRPC surface")
    from membrane.disagg import transfer_pb2

    return transfer_pb2.PrefillResponse(  # type: ignore[attr-defined]
        request_id=response.request_id,
        kv_handle=response.kv_handle,
        prefill_ms=response.prefill_ms,
        prompt_len=response.prompt_len,
        cached_prefix_len=response.cached_prefix_len,
    )


def build_decode_request_message(request: DecodeRequest) -> Any:
    """Build a generated proto message for ``request``.

    Args:
        request: The decode request.

    Returns:
        A generated proto message.
    """
    if not GRPC_IMPORTED:  # pragma: no cover - import guard
        raise RuntimeError("grpcio is required for the gRPC surface")
    from membrane.disagg import transfer_pb2

    return transfer_pb2.DecodeRequest(  # type: ignore[attr-defined]
        request_id=request.request_id,
        kv_handle=request.kv_handle,
        model_id=request.model_id,
        max_tokens=request.max_tokens,
    )


def build_decode_response_message(response: DecodeResponse) -> Any:
    """Build a generated proto message for ``response``.

    Args:
        response: The decode response.

    Returns:
        A generated proto message.
    """
    if not GRPC_IMPORTED:  # pragma: no cover - import guard
        raise RuntimeError("grpcio is required for the gRPC surface")
    from membrane.disagg import transfer_pb2

    return transfer_pb2.DecodeResponse(  # type: ignore[attr-defined]
        request_id=response.request_id,
        token_ids=list(response.token_ids),
        finished=response.finished,
    )


def request_from_message(message: Any) -> PrefillRequest:
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


def response_from_message(message: Any) -> PrefillResponse:
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


def decode_request_from_message(message: Any) -> DecodeRequest:
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


def decode_response_from_message(message: Any) -> DecodeResponse:
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
    if not GRPC_IMPORTED:  # pragma: no cover - import guard
        raise RuntimeError("grpcio is required for the gRPC surface")
    from membrane.disagg import transfer_pb2_grpc

    decode_service = decode or DecodeService()
    handler = GrpcHandler(prefill=prefill, decode=decode_service)
    transfer_pb2_grpc.add_TransferServicer_to_server(  # type: ignore[attr-defined]
        handler, server
    )
    return server


def make_channel(target: str, tls: MTLSConfig | None = None) -> Any:
    """Open a gRPC channel to ``target``.

    Args:
        target: ``host:port`` string.
        tls: Present this client certificate and trust this CA; plaintext
            when ``None``.

    Returns:
        A :class:`grpc.Channel` ready to use with
        :func:`make_stub`.
    """
    if not GRPC_IMPORTED:  # pragma: no cover - import guard
        raise RuntimeError("grpcio is required for the gRPC surface")
    if tls is None:
        return grpc.insecure_channel(target)
    credentials = grpc.ssl_channel_credentials(
        root_certificates=tls.ca_bundle_pem.encode(),
        private_key=(tls.client_key_pem or tls.server_key_pem).encode(),
        certificate_chain=(tls.client_cert_pem or tls.server_cert_pem).encode(),
    )
    return grpc.secure_channel(target, credentials)


def make_stub(channel: Any) -> Any:
    """Build a :class:`TransferStub` for ``channel``.

    Args:
        channel: A :class:`grpc.Channel`.

    Returns:
        TransferStub: A client stub.
    """
    if not GRPC_IMPORTED:  # pragma: no cover - import guard
        raise RuntimeError("grpcio is required for the gRPC surface")
    from membrane.disagg import transfer_pb2_grpc

    return transfer_pb2_grpc.TransferStub(channel)  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Servicer
# ---------------------------------------------------------------------------


#: The HTTP route whose scope each RPC requires.
RPC_ROUTES = {
    "Prefill": "/disagg/prefill",
    "BatchPrefill": "/disagg/prefill/batch",
    "Decode": "/disagg/decode",
}


def call_service[T](context: Any, call: Callable[[], T]) -> T:
    """Run a service call, turning its errors into gRPC status codes.

    Args:
        context: gRPC servicer context.
        call: The service call.

    Returns:
        T: Its result (``context.abort`` raises otherwise).
    """
    try:
        return call()
    except RoleUnavailableError as exc:
        context.abort(grpc.StatusCode.FAILED_PRECONDITION, str(exc))
        raise
    except LookupError as exc:
        context.abort(grpc.StatusCode.NOT_FOUND, str(exc))
        raise
    except ValueError as exc:
        context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
        raise


def caller_headers(metadata: Any, context: Any) -> dict[str, str]:
    """Build authenticator headers from RPC metadata and the TLS peer.

    A client-sent ``x-ssl-client-cn`` is discarded: the only trusted CN is
    the one from the verified client certificate.

    Args:
        metadata: Invocation metadata (key/value pairs).
        context: gRPC servicer context.

    Returns:
        dict[str, str]: Lower-cased headers.
    """
    headers = {str(k).lower(): str(v) for k, v in metadata or () if isinstance(v, str)}
    headers.pop("x-ssl-client-cn", None)
    names = (context.auth_context() or {}).get("x509_common_name") or []
    if names:
        headers["x-ssl-client-cn"] = names[0].decode()
    return headers


class AuthInterceptor(grpc.ServerInterceptor if GRPC_IMPORTED else object):  # type: ignore[misc]
    """Authenticate every RPC and enforce the scope of its HTTP twin."""

    def __init__(self, authenticator: Authenticator | None) -> None:
        """Wrap ``authenticator``.

        Args:
            authenticator: Inbound authenticator; scope checks are skipped
                when ``None`` (authentication off).
        """
        self.authenticator = authenticator

    def intercept_service(self, continuation: Any, handler_call_details: Any) -> Any:
        """Wrap the RPC handler with the authentication check.

        Args:
            continuation: Resolves the next handler.
            handler_call_details: Method name and metadata.

        Returns:
            Any: The wrapped handler (or the original for non-unary RPCs).
        """
        handler = continuation(handler_call_details)
        if handler is None or handler.unary_unary is None:
            return handler
        route = RPC_ROUTES.get(handler_call_details.method.rsplit("/", 1)[-1], "/disagg/unknown")
        metadata = handler_call_details.invocation_metadata
        inner = handler.unary_unary
        authenticator = self.authenticator

        def authenticated(request: Any, context: Any) -> Any:
            try:
                caller = enforce_route_scope(authenticator, "POST", route, headers=caller_headers(metadata, context))
            except AuthForbiddenError:
                context.abort(grpc.StatusCode.PERMISSION_DENIED, "forbidden")
            except AuthBackendError:
                context.abort(grpc.StatusCode.UNAUTHENTICATED, "unauthorized")
            token = CALLER.set(caller if authenticator is not None else None)
            try:
                return inner(request, context)
            finally:
                CALLER.reset(token)

        return grpc.unary_unary_rpc_method_handler(
            authenticated,
            request_deserializer=handler.request_deserializer,
            response_serializer=handler.response_serializer,
        )


def serve(
    prefill: PrefillService,
    decode: DecodeService,
    host: str,
    port: int,
    tls: MTLSConfig | None = None,
    authenticator: Authenticator | None = None,
    workers: int = 16,
) -> tuple[Any, int]:
    """Start a gRPC server for the prefill and decode services.

    Args:
        prefill: The prefill service.
        decode: The decode service.
        host: Bind address.
        port: Port (``0`` picks a free one).
        tls: Serve TLS with this certificate; clients must present a
            certificate when ``tls.require_client_cert``.
        authenticator: Checks every RPC like the HTTP routes.
        workers: Concurrent RPCs.

    Returns:
        tuple[Any, int]: The started :class:`grpc.Server` and its port.

    Raises:
        RuntimeError: When grpcio is not installed.
    """
    if not GRPC_IMPORTED:
        raise RuntimeError("the gRPC surface needs grpcio: install membrane[disagg]")
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=workers), interceptors=[AuthInterceptor(authenticator)])
    add_to_server(server, prefill, decode)
    address = f"[{host}]:{port}" if ":" in host else f"{host}:{port}"
    if tls is None:
        bound = server.add_insecure_port(address)
    else:
        credentials = grpc.ssl_server_credentials(
            [(tls.server_key_pem.encode(), tls.server_cert_pem.encode())],
            root_certificates=tls.ca_bundle_pem.encode(),
            require_client_auth=tls.require_client_cert,
        )
        bound = server.add_secure_port(address, credentials)
    server.start()
    logger.info("gRPC prefill/decode listening on %s:%s%s", host, bound, " (TLS)" if tls else "")
    return server, bound


class GrpcHandler:
    """Servicer that bridges gRPC calls to the in-process services.

    The v1 implementation is intentionally small: it builds
    domain types from the wire messages, delegates to
    :class:`PrefillService` / :class:`DecodeService`, and
    converts the responses back into wire messages.
    """

    def __init__(self, prefill: PrefillService, decode: DecodeService) -> None:
        """Bind the gRPC servicer to the prefill and decode services.

        Args:
            prefill: The prefill service.
            decode: Optional decode service. Defaults to a new
                :class:`DecodeService`.
        """
        self.__prefill = prefill
        self.__decode = decode

    def Prefill(self, request: Any, context: Any) -> Any:
        """Handle a :class:`PrefillRequest` RPC.

        Args:
            request: Generated ``PrefillRequest`` message.
            context: gRPC context.

        Returns:
            Generated ``PrefillResponse`` message.
        """
        decoded = request_from_message(request)
        response = call_service(context, lambda: self.__prefill.prefill(decoded))
        return build_prefill_response_message(response)

    def BatchPrefill(self, request: Any, context: Any) -> Any:
        """Handle a batch prefill RPC.

        Args:
            request: Generated ``BatchPrefillRequest`` message.
            context: gRPC context.

        Returns:
            Generated ``BatchPrefillResponse`` message.
        """
        requests = [request_from_message(item) for item in request.requests]
        result = call_service(context, lambda: batch_prefill(self.__prefill, requests))
        from membrane.disagg import transfer_pb2

        return transfer_pb2.BatchPrefillResponse(  # type: ignore[attr-defined]
            responses=[build_prefill_response_message(r) for r in result.responses],
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
        decoded = decode_request_from_message(request)
        response = call_service(context, lambda: self.__decode.decode(decoded))
        return build_decode_response_message(response)


__all__ = [
    "GRPC_AVAILABLE",
    "RPC_ROUTES",
    "AuthInterceptor",
    "add_to_server",
    "call_service",
    "caller_headers",
    "make_channel",
    "make_stub",
    "serve",
]
