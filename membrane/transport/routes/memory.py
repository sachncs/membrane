"""Memory API and routing routes: reconstruct, prefix lookup, sessions, objects, KV bundles, route."""

import asyncio

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

from membrane.services import Services
from membrane.services.memory import SESSION_HEADER, StoreRejectedError
from membrane.transport.context import app_context
from membrane.transport.metrics import record_transport
from membrane.transport.ops import MAX_BODY_BYTES
from membrane.transport.routes.common import read_limited_body, respond, route_scope, transport_metrics_for
from membrane.transport.routes.models import PrefixLookupRequest, ReconstructRequest, RouteRequest

MAX_QUERY_TOKENS = 4096


def services_for(app: FastAPI) -> Services | None:
    """Return the app's services, if the server built them.

    Args:
        app: The FastAPI application.

    Returns:
        Services | None: The services.
    """
    return app_context(app).services


def unavailable() -> Response:
    """Answer for an app without services (a bare test app).

    Returns:
        Response: ``503``.
    """
    return JSONResponse({"error": "memory services unavailable"}, status_code=503)


def handle_reconstruct(app: FastAPI, req: ReconstructRequest, request: Request) -> Response:
    """Serve ``POST /reconstruct``.

    Args:
        app: The FastAPI application.
        req: Validated body.
        request: The inbound request.

    Returns:
        Response: Fragments covering the prompt, coverage, and prefetch hints.
    """
    context = route_scope(request, "POST", "/reconstruct")
    if req.prefill:
        context = route_scope(request, "POST", "/prefill")  # computing KV is a write
    services = services_for(app)
    if services is None:
        return unavailable()
    session_id = request.headers.get(SESSION_HEADER, "")[:256]
    status, body = record_transport(
        transport_metrics_for(app),
        "reconstruct",
        "POST",
        lambda: (
            200,
            services.memory.reconstruct(req.tokens, req.model_id, context, prefill=req.prefill, session_id=session_id),
        ),
    )
    return respond(status, body)


def handle_prefix_lookup(app: FastAPI, tokens: list[int], model_id: str, request: Request) -> Response:
    """Serve ``GET`` and ``POST /prefix/lookup``.

    Args:
        app: The FastAPI application.
        tokens: Prompt tokens.
        model_id: Model identifier.
        request: The inbound request.

    Returns:
        Response: How many leading tokens are cached here, and by which fragments.
    """
    context = route_scope(request, "GET", "/prefix/lookup")
    services = services_for(app)
    if services is None:
        return unavailable()
    status, body = record_transport(
        transport_metrics_for(app),
        "prefix_lookup",
        request.method,
        lambda: (200, services.memory.prefix_lookup(tokens, model_id, context)),
    )
    return respond(status, body)


def parse_tokens(raw: str) -> list[int] | None:
    """Parse ``?tokens=1,2,3``.

    Args:
        raw: The query value.

    Returns:
        list[int] | None: The tokens, or ``None`` when malformed or too many.
    """
    parts = [p for p in raw.split(",") if p.strip()]
    if len(parts) > MAX_QUERY_TOKENS:
        return None
    try:
        return [int(p) for p in parts]
    except ValueError:
        return None


def handle_session(app: FastAPI, session_id: str, request: Request) -> Response:
    """Serve ``GET`` / ``DELETE /sessions/{session_id}``.

    Args:
        app: The FastAPI application.
        session_id: Session identifier.
        request: The inbound request.

    Returns:
        Response: The session's read history, or ``{"deleted": bool}``.
    """
    context = route_scope(request, request.method, "/sessions")
    services = services_for(app)
    if services is None:
        return unavailable()
    if request.method == "DELETE":
        return JSONResponse({"deleted": services.memory.forget_session(session_id, context)})
    return JSONResponse(services.memory.session(session_id, context))


async def handle_put_object(app: FastAPI, request: Request) -> Response:
    """Serve ``POST /objects``: store a typed memory object.

    Args:
        app: The FastAPI application.
        request: Body as described by
            :meth:`~membrane.services.memory.MemoryService.put_object`.

    Returns:
        Response: ``{"content_hash", "kind"}``; ``400`` for a bad body, ``409``
        for a refused write.
    """
    context = route_scope(request, "POST", "/objects")
    services = services_for(app)
    if services is None:
        return unavailable()
    if app_context(app).draining:
        return JSONResponse({"error": "node draining"}, status_code=503, headers={"Retry-After": "1"})
    try:
        body = await request.json()
    except ValueError:
        return JSONResponse({"error": "body must be JSON"}, status_code=400)
    if not isinstance(body, dict):
        return JSONResponse({"error": "body must be a JSON object"}, status_code=400)
    try:
        result = await asyncio.to_thread(services.memory.put_object, body, context)
    except StoreRejectedError as exc:
        return JSONResponse({"error": "store refused", "detail": str(exc)}, status_code=exc.status)
    return JSONResponse(result)


def handle_get_object(app: FastAPI, content_hash: str, request: Request) -> Response:
    """Serve ``GET /objects/{content_hash}``.

    Args:
        app: The FastAPI application.
        content_hash: The object's hash.
        request: The inbound request.

    Returns:
        Response: The typed object, or ``404``.
    """
    context = route_scope(request, "GET", "/objects")
    services = services_for(app)
    if services is None:
        return unavailable()
    found = services.memory.get_object(content_hash, context)
    if found is None:
        return JSONResponse({"error": "not found"}, status_code=404)
    return JSONResponse(found)


def handle_route(app: FastAPI, req: RouteRequest, request: Request) -> Response:
    """Serve ``POST /route``: where to fetch, prefill, and store.

    Args:
        app: The FastAPI application.
        req: Validated body.
        request: The inbound request.

    Returns:
        Response: The placement, the cached fragments with their locations,
        and (with ``--route-threshold``) the offload decision.
    """
    context = route_scope(request, "POST", "/route")
    services = services_for(app)
    if services is None:
        return unavailable()
    if not req.content_hash and not req.tokens:
        return JSONResponse({"error": "send content_hash or tokens"}, status_code=400)
    placement = services.placement

    def route() -> tuple[int, object]:
        answer = placement.route(
            content_hash=req.content_hash,
            tokens=req.tokens,
            model_id=req.model_id,
            local_cached_tokens=req.local_cached_tokens,
            auth_context=context,
        )
        return 200, placement.body(answer)

    status, body = record_transport(transport_metrics_for(app), "route", "POST", route)
    return respond(status, body)


async def handle_put_kv(app: FastAPI, handle: str, model_id: str, request: Request) -> Response:
    """Serve ``PUT /kv/{handle}?model_id=``: store engine KV bytes under a name.

    Args:
        app: The FastAPI application.
        handle: Client-chosen name.
        model_id: Model the bytes were produced by.
        request: The raw bytes.

    Returns:
        Response: ``{"content_hash"}``; ``400`` for a bad name, ``413`` when too large.
    """
    context = route_scope(request, "PUT", "/kv")
    services = services_for(app)
    if services is None:
        return unavailable()
    if app_context(app).draining:
        return JSONResponse({"error": "node draining"}, status_code=503, headers={"Retry-After": "1"})
    data = await read_limited_body(request, MAX_BODY_BYTES)
    if data is None:
        return JSONResponse({"error": "payload too large", "limit": MAX_BODY_BYTES}, status_code=413)
    try:
        content_hash = await asyncio.to_thread(services.bundles.put, model_id, handle, data, context)
    except StoreRejectedError as exc:
        return JSONResponse({"error": str(exc)}, status_code=exc.status)
    return JSONResponse({"content_hash": content_hash})


async def handle_get_kv(app: FastAPI, handle: str, model_id: str, request: Request) -> Response:
    """Serve ``GET`` / ``HEAD /kv/{handle}?model_id=``.

    Args:
        app: The FastAPI application.
        handle: Client-chosen name.
        model_id: Model the bytes were produced by.
        request: The inbound request.

    Returns:
        Response: The bytes (``HEAD``: headers only), or ``404``.
    """
    context = route_scope(request, request.method, "/kv")
    services = services_for(app)
    if services is None:
        return unavailable()
    data = await asyncio.to_thread(services.bundles.get, model_id, handle, context)
    if data is None:
        return JSONResponse({"error": "not found"}, status_code=404)
    if request.method == "HEAD":
        return Response(status_code=200, headers={"Content-Length": str(len(data))})
    return Response(content=data, media_type="application/octet-stream")


def register_memory_routes(app: FastAPI) -> None:
    """Register the memory API and routing routes.

    Args:
        app: The FastAPI application.
    """

    def reconstruct_handler(req: ReconstructRequest, request: Request):
        return handle_reconstruct(app, req, request)

    def prefix_lookup_get(request: Request, tokens: str = "", model_id: str = "default"):
        parsed = parse_tokens(tokens)
        if parsed is None:
            return JSONResponse(
                {"error": f"tokens must be up to {MAX_QUERY_TOKENS} comma-separated integers; POST longer prompts"},
                status_code=400,
            )
        return handle_prefix_lookup(app, parsed, model_id[:256], request)

    def prefix_lookup_post(req: PrefixLookupRequest, request: Request):
        return handle_prefix_lookup(app, req.tokens, req.model_id, request)

    def session_handler(session_id: str, request: Request):
        return handle_session(app, session_id[:256], request)

    async def put_object_handler(request: Request):
        return await handle_put_object(app, request)

    def get_object_handler(content_hash: str, request: Request):
        return handle_get_object(app, content_hash, request)

    def route_handler(req: RouteRequest, request: Request):
        return handle_route(app, req, request)

    async def put_kv_handler(handle: str, request: Request, model_id: str = "default"):
        return await handle_put_kv(app, handle, model_id, request)

    async def get_kv_handler(handle: str, request: Request, model_id: str = "default"):
        return await handle_get_kv(app, handle, model_id, request)

    app.add_api_route("/reconstruct", reconstruct_handler, methods=["POST"], response_model=None)
    app.add_api_route("/prefix/lookup", prefix_lookup_get, methods=["GET"], response_model=None)
    app.add_api_route(
        "/prefix/lookup",
        prefix_lookup_post,
        methods=["POST"],
        response_model=None,
        name="prefix_lookup_post",
        operation_id="prefix_lookup_post",
    )
    app.add_api_route("/sessions/{session_id}", session_handler, methods=["GET"], response_model=None)
    app.add_api_route(
        "/sessions/{session_id}",
        session_handler,
        methods=["DELETE"],
        response_model=None,
        name="delete_session",
        operation_id="delete_session",
    )
    app.add_api_route("/objects", put_object_handler, methods=["POST"], response_model=None)
    app.add_api_route("/objects/{content_hash}", get_object_handler, methods=["GET"], response_model=None)
    app.add_api_route("/route", route_handler, methods=["POST"], response_model=None)
    app.add_api_route("/kv/{handle}", put_kv_handler, methods=["PUT"], response_model=None)
    app.add_api_route("/kv/{handle}", get_kv_handler, methods=["GET"], response_model=None)
    app.add_api_route(
        "/kv/{handle}", get_kv_handler, methods=["HEAD"], response_model=None, name="head_kv", operation_id="head_kv"
    )


__all__ = [
    "MAX_QUERY_TOKENS",
    "handle_get_kv",
    "handle_get_object",
    "handle_prefix_lookup",
    "handle_put_kv",
    "handle_put_object",
    "handle_reconstruct",
    "handle_route",
    "handle_session",
    "parse_tokens",
    "register_memory_routes",
]
