"""Membrane HTTP routes, grouped by area.

Every route except the ``/livez`` and ``/readyz`` probes authenticates the
caller and enforces its scope (:mod:`membrane.transport.authz`) before
delegating to a transport-agnostic operation in :mod:`membrane.transport.ops`.

* :mod:`~membrane.transport.routes.probes`: probes, heartbeat, metrics.
* :mod:`~membrane.transport.routes.data`: fragments, blobs, prefill, sync.
* :mod:`~membrane.transport.routes.cluster`: replication, membership, gossip.
* :mod:`~membrane.transport.routes.deletes`: delete, tombstone, purge, verify.
* :mod:`~membrane.transport.routes.models`: request bodies.
* :mod:`~membrane.transport.routes.common`: shared helpers.
"""

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from membrane.auth import AuthBackendError
from membrane.openapi import generate_spec
from membrane.transport.context import app_context
from membrane.transport.routes.cluster import handle_gossip, handle_join, handle_leave, handle_peers, handle_replicate
from membrane.transport.routes.common import auth_error_response, route_scope
from membrane.transport.routes.data import (
    handle_get_blob,
    handle_inventory,
    handle_prefill,
    handle_put_blob,
    handle_retrieve,
    handle_store,
    handle_sync,
)
from membrane.transport.routes.deletes import handle_delete, handle_purge, handle_tombstone, handle_verify
from membrane.transport.routes.models import (
    DeleteRequest,
    GossipRequest,
    JoinRequest,
    LeaveRequest,
    PrefillRequest,
    PurgeRequest,
    ReplicateRequest,
    StoreRequest,
    SyncRequest,
    TombstoneRequest,
    VerifyRequest,
)
from membrane.transport.routes.probes import handle_heartbeat, handle_metrics, livez, metrics_json, readyz


def register_routes(app: FastAPI) -> None:
    """Register every Membrane HTTP route on ``app``.

    Each route delegates to the corresponding operation in
    :mod:`membrane.transport.ops` after running the per-route
    scope check in :mod:`membrane.transport.authz`. The Pydantic
    models at the top of this module validate request bodies; the
    inline lambdas capture ``app`` so the registered functions
    take only the request body.

    Args:
        app: The FastAPI app to configure.
    """
    app.add_exception_handler(AuthBackendError, auth_error_response)

    # GET endpoints that have no body.
    def heartbeat_handler(request: Request):
        return handle_heartbeat(app, request)

    app.add_api_route("/heartbeat", heartbeat_handler, methods=["GET"])
    app.add_api_route("/livez", lambda: livez(app), methods=["GET"])
    app.add_api_route("/readyz", lambda: readyz(app), methods=["GET"])

    def metrics_handler(request: Request):
        route_scope(request, "GET", "/metrics")
        return handle_metrics(app)

    def metrics_json_handler(request: Request):
        route_scope(request, "GET", "/metrics.json")
        return metrics_json(app)

    def retrieve_handler(content_hash: str, request: Request):
        return handle_retrieve(app, content_hash, route_scope(request, "GET", "/retrieve"))

    def inventory_handler(request: Request):
        route_scope(request, "GET", "/inventory")
        return handle_inventory(app)

    def peers_handler(request: Request):
        route_scope(request, "GET", "/peers")
        return handle_peers(app)

    app.add_api_route("/metrics", metrics_handler, methods=["GET"])
    app.add_api_route("/metrics.json", metrics_json_handler, methods=["GET"])
    app.add_api_route("/retrieve", retrieve_handler, methods=["GET"])
    app.add_api_route("/inventory", inventory_handler, methods=["GET"])
    app.add_api_route("/peers", peers_handler, methods=["GET"])

    limits = app_context(app).limits
    if limits is not None and limits.enable_api_docs:

        def openapi_handler(request: Request):
            route_scope(request, "GET", "/openapi.json")
            return JSONResponse(generate_spec(app))

        app.add_api_route("/openapi.json", openapi_handler, methods=["GET"], include_in_schema=False)

    # POST endpoints.
    def store_handler(req: StoreRequest, request: Request):
        return handle_store(app, req, request)

    def replicate_handler(req: ReplicateRequest, request: Request):
        return handle_replicate(app, req, request)

    def sync_handler(req: SyncRequest, request: Request):
        return handle_sync(app, req, request)

    def prefill_handler(req: PrefillRequest, request: Request):
        return handle_prefill(app, req, request)

    def join_handler(req: JoinRequest, request: Request):
        return handle_join(app, req, request)

    def leave_handler(req: LeaveRequest, request: Request):
        return handle_leave(app, req, request)

    def gossip_handler(req: GossipRequest, request: Request):
        return handle_gossip(app, req, request)

    def delete_handler(req: DeleteRequest, request: Request):
        return handle_delete(app, req, request)

    def tombstone_handler(req: TombstoneRequest, request: Request):
        return handle_tombstone(app, req, request)

    def purge_handler(req: PurgeRequest, request: Request):
        return handle_purge(app, req, request)

    def verify_handler(req: VerifyRequest, request: Request):
        return handle_verify(app, req, request)

    async def put_blob_handler(payload_ref: str, request: Request):
        return await handle_put_blob(app, payload_ref, request)

    async def get_blob_handler(payload_ref: str, request: Request):
        return await handle_get_blob(app, payload_ref, request)

    app.add_api_route("/blobs/{payload_ref}", put_blob_handler, methods=["PUT"], response_model=None)
    app.add_api_route("/blobs/{payload_ref}", get_blob_handler, methods=["GET"], response_model=None)
    app.add_api_route(
        "/blobs/{payload_ref}",
        get_blob_handler,
        methods=["HEAD"],
        response_model=None,
        name="head_blob_handler",
        operation_id="head_blob",
    )
    app.add_api_route("/store", store_handler, methods=["POST"], response_model=None)
    app.add_api_route("/replicate", replicate_handler, methods=["POST"], response_model=None)
    app.add_api_route("/sync", sync_handler, methods=["POST"], response_model=None)
    app.add_api_route("/prefill", prefill_handler, methods=["POST"], response_model=None)
    app.add_api_route("/join", join_handler, methods=["POST"], response_model=None)
    app.add_api_route("/leave", leave_handler, methods=["POST"], response_model=None)
    app.add_api_route("/gossip", gossip_handler, methods=["POST"], response_model=None)
    app.add_api_route("/delete", delete_handler, methods=["POST"], response_model=None)
    app.add_api_route("/tombstone", tombstone_handler, methods=["POST"], response_model=None)
    app.add_api_route("/purge", purge_handler, methods=["POST"], response_model=None)
    app.add_api_route("/verify", verify_handler, methods=["POST"], response_model=None)


__all__ = ["register_routes"]
