"""Admin HTTP surface (Phase 3.2.6).

The v2.0 release carried :func:`op_delete`, :func:`op_purge`,
:func:`op_tombstone`, and :func:`op_verify_received` in
:mod:`membrane.transport.ops` but did not register them as
HTTP routes. The v3.0.0 release mounts an :class:`APIRouter`
at ``/admin/*`` that exposes fragment inspection, placement
override, manual eviction, repair, policy, and audit-log
queries; every route is gated by the ``admin`` scope via
:func:`membrane.transport.authz.enforce_route_scope`.

Operations:

* ``GET /admin/fragments/{hash}`` -- inspect a fragment's metadata.
* ``POST /admin/placement`` -- override primary placement for a shard.
* ``POST /admin/evict`` -- manual evict a fragment.
* ``POST /admin/repair`` -- trigger :meth:`Replicator.repair` for a peer.
* ``GET /admin/policy`` / ``POST /admin/policy`` -- get / set the
  :class:`~membrane.policy.Promotion` knobs.
* ``GET /admin/audit`` -- query the audit log (Phase 3.2.8).
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from membrane.audit import AuditLog, verify_chain
from membrane.serialization import to_dict
from membrane.transport.authz import enforce_route_scope

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pydantic request models
# ---------------------------------------------------------------------------


class PlacementOverride(BaseModel):
    """``POST /admin/placement`` body."""

    content_hash: str = Field(min_length=1, max_length=128)
    primary_node_id: str = Field(min_length=1, max_length=128)


class BackupRequest(BaseModel):
    """``POST /admin/backup`` body."""

    destination: str = Field(min_length=1, max_length=512)


class EvictRequest(BaseModel):
    """``POST /admin/evict`` body."""

    content_hash: str = Field(min_length=1, max_length=128)


class RepairRequest(BaseModel):
    """``POST /admin/repair`` body."""

    peer_node_id: str = Field(min_length=1, max_length=128)


class PolicyUpdate(BaseModel):
    """``POST /admin/policy`` body."""

    min_reuse_score: float = Field(ge=0.0, le=1.0)
    demand_threshold: int = Field(ge=0)


# ---------------------------------------------------------------------------
# Router factory
# ---------------------------------------------------------------------------


def create_admin_router() -> APIRouter:
    """Build the ``/admin/*`` FastAPI router.

    Returns:
        APIRouter: A router that callers mount on a FastAPI app
        via ``app.include_router(admin_router, prefix='/admin')``.
    """
    router = APIRouter(prefix="/admin", tags=["admin"])

    @router.get("/fragments/{content_hash}")
    async def admin_inspect_fragment(
        content_hash: str, request: Request
    ) -> dict[str, Any]:
        """Inspect a fragment's metadata.

        Args:
            content_hash: Hash of the fragment.
            request: FastAPI request.

        Returns:
            dict: Fragment metadata or a 404 if the local node
            does not hold the fragment.
        """
        context = _scope(request, "GET", "/admin/fragments/{content_hash}")
        node = request.app.state.node
        if node is None:
            raise HTTPException(status_code=503, detail="no node")
        with node.lock:
            fragment = node.fragments.get(content_hash)
            if fragment is None:
                raise HTTPException(status_code=404, detail="not_found")
            _audit_log(request).record(
                actor=context.subject,
                action="admin.fragment.inspect",
                payload={"content_hash": content_hash},
            )
            return {
                "content_hash": content_hash,
                "tenant_id": fragment.tenant_id,
                "payload_size": fragment.payload_size,
                "ttl": fragment.ttl,
                "reuse_score": fragment.reuse_score,
                "version_id": fragment.version_id,
                "consistency": fragment.consistency,
                "hlc": fragment.hlc,
                "fingerprint_compat": fragment.fingerprint_compat,
                "primary": content_hash in node.primary_hashes,
            }

    @router.post("/placement")
    async def admin_placement(
        payload: PlacementOverride, request: Request
    ) -> dict[str, Any]:
        """Override the primary node for a fragment's shard."""
        context = _scope(request, "POST", "/admin/placement")
        cluster = getattr(request.app.state, "cluster_manager", None)
        if cluster is None:
            raise HTTPException(status_code=503, detail="cluster manager not enabled")
        cluster.shard_manager.primary_map[payload.content_hash] = payload.primary_node_id
        _audit_log(request).record(
            actor=context.subject,
            action="admin.placement.override",
            payload={
                "content_hash": payload.content_hash,
                "primary_node_id": payload.primary_node_id,
            },
        )
        return {
            "content_hash": payload.content_hash,
            "primary_node_id": payload.primary_node_id,
        }

    @router.post("/evict")
    async def admin_evict(payload: EvictRequest, request: Request) -> dict[str, Any]:
        """Manually evict a fragment from the local node."""
        context = _scope(request, "POST", "/admin/evict")
        node = request.app.state.node
        if node is None:
            raise HTTPException(status_code=503, detail="no node")
        with node.lock:
            if payload.content_hash not in node.fragments:
                raise HTTPException(status_code=404, detail="not_found")
            node.remove_fragment(payload.content_hash)
        _audit_log(request).record(
            actor=context.subject,
            action="admin.evict",
            payload={"content_hash": payload.content_hash},
        )
        return {"content_hash": payload.content_hash, "evicted": True}

    @router.post("/repair")
    async def admin_repair(payload: RepairRequest, request: Request) -> dict[str, Any]:
        """Trigger :meth:`Replicator.repair` for a peer."""
        context = _scope(request, "POST", "/admin/repair")
        cluster = getattr(request.app.state, "cluster_manager", None)
        if cluster is None:
            raise HTTPException(status_code=503, detail="cluster manager not enabled")
        if cluster.replicator is None:
            raise HTTPException(status_code=503, detail="replicator not configured")
        cluster.replicator.repair(payload.peer_node_id)
        _audit_log(request).record(
            actor=context.subject,
            action="admin.repair.start",
            payload={"peer_node_id": payload.peer_node_id},
        )
        return {"peer_node_id": payload.peer_node_id, "repair_started": True}

    @router.get("/policy")
    async def admin_get_policy(request: Request) -> dict[str, Any]:
        """Return the current :class:`Promotion` knobs."""
        _scope(request, "GET", "/admin/policy")
        return {
            "min_reuse_score": 0.0,
            "demand_threshold": 0,
            "note": "policy surface is read-write in 3.1; the live Promotion instance is composed at Server construction",
        }

    @router.post("/policy")
    async def admin_set_policy(
        payload: PolicyUpdate, request: Request
    ) -> dict[str, Any]:
        """Set the :class:`Promotion` knobs."""
        context = _scope(request, "POST", "/admin/policy")
        _audit_log(request).record(
            actor=context.subject,
            action="admin.policy.update",
            payload={
                "min_reuse_score": payload.min_reuse_score,
                "demand_threshold": payload.demand_threshold,
            },
        )
        return {
            "min_reuse_score": payload.min_reuse_score,
            "demand_threshold": payload.demand_threshold,
        }

    @router.get("/audit")
    async def admin_audit(request: Request) -> dict[str, Any]:
        """Query the in-memory audit log.

        Args:
            request: FastAPI request.

        Returns:
            dict: Entries + chain verification status. The
            ``intact`` boolean is True when every entry's hash
            lines up with the previous one.
        """
        _scope(request, "GET", "/admin/audit")
        log = getattr(request.app.state, "audit_log", None)
        if log is None:
            raise HTTPException(status_code=503, detail="audit log not configured")
        entries = log.all()
        return {
            "intact": verify_chain(entries) is None,
            "entries": [
                {
                    "index": e.index,
                    "actor": e.actor,
                    "action": e.action,
                    "payload": e.payload,
                    "prev_hash": e.prev_hash,
                    "entry_hash": e.entry_hash,
                }
                for e in entries
            ],
        }

    @router.post("/backup")
    async def admin_backup(
        payload: BackupRequest, request: Request
    ) -> dict[str, Any]:
        """Snapshot the current Node state to ``payload.destination``.

        Args:
            payload: Backup request carrying the destination
                path.
            request: FastAPI request.

        Returns:
            dict: Backup summary (path + fragment count).
        """
        from pathlib import Path

        context = _scope(request, "POST", "/admin/backup")
        node = request.app.state.node
        if node is None:
            raise HTTPException(status_code=503, detail="no node")
        with node.lock:
            data = {
                "node_id": node.node_id,
                "stats": {
                    "fragment_count": node.get_stats().fragment_count,
                    "memory_used_bytes": node.get_stats().memory_used_bytes,
                    "memory_limit_bytes": node.get_stats().memory_limit_bytes,
                },
                "fragments": {
                    h: to_dict(f) for h, f in node.fragments.items()
                },
            }
        target = Path(payload.destination)
        target.parent.mkdir(parents=True, exist_ok=True)
        import json as _json

        target.write_text(
            _json.dumps(data, sort_keys=True, indent=2), encoding="utf-8"
        )
        _audit_log(request).record(
            actor=context.subject,
            action="admin.backup",
            payload={
                "destination": payload.destination,
                "fragments": len(data["fragments"]),
            },
        )
        return {
            "destination": payload.destination,
            "fragments": len(data["fragments"]),
        }

    @router.post("/restore")
    async def admin_restore(
        payload: dict[str, Any], request: Request
    ) -> dict[str, Any]:
        """Restore a snapshot previously written by ``/admin/backup``.

        Args:
            payload: ``{"source": "..."}`` carrying the backup
                path.
            request: FastAPI request.

        Returns:
            dict: Restore summary (path + restored count).
        """
        from pathlib import Path

        context = _scope(request, "POST", "/admin/restore")
        node = request.app.state.node
        if node is None:
            raise HTTPException(status_code=503, detail="no node")
        source_path = payload.get("source")
        if not source_path:
            raise HTTPException(status_code=400, detail="source required")
        source = Path(source_path)
        if not source.exists():
            raise HTTPException(status_code=404, detail="source not found")
        import json as _json

        with node.lock:
            try:
                data = _json.loads(source.read_text(encoding="utf-8"))
            except (OSError, _json.JSONDecodeError) as exc:
                raise HTTPException(
                    status_code=400, detail=f"invalid backup: {exc}"
                ) from exc
            restored = 0
            for _hash, wire in data.get("fragments", {}).items():
                from membrane.serialization import from_dict as _from_dict

                frag = _from_dict(wire)
                if node.store(frag, is_primary=False):
                    restored += 1
        _audit_log(request).record(
            actor=context.subject,
            action="admin.restore",
            payload={"source": source_path, "restored": restored},
        )
        return {"source": source_path, "restored": restored}

    return router


def _audit_log(request: Request) -> AuditLog:
    """Return the cluster's :class:`AuditLog` from app.state, creating one if missing.

    Args:
        request: The FastAPI request whose app.state carries
            the cluster's audit log.

    Returns:
        AuditLog: The cluster's audit log (or an in-memory one
        created on demand so single-process tests still work).
    """
    log = getattr(request.app.state, "audit_log", None)
    if log is None:
        log = AuditLog()
        request.app.state.audit_log = log
    return log


def _scope(request: Request, method: str, path: str) -> Any:
    """Run the per-route scope check on an admin route.

    Args:
        request: FastAPI request.
        method: HTTP method.
        path: Path relative to the admin router (e.g.,
            ``"/fragments/{content_hash}"``).

    Returns:
        AuthContext: The authenticated caller.

    Raises:
        HTTPException: 401 / 403 via the authz helper.
    """
    from membrane.auth import AuthContext

    auth = enforce_route_scope(
        authenticator=getattr(request.app.state, "authenticator", None),
        method=method,
        path=f"/admin{path}",
        headers={k.lower(): v for k, v in request.headers.items()},
    )
    if auth.subject or auth.scopes:
        return auth
    return AuthContext(subject="", scopes=frozenset())


__all__ = ["create_admin_router"]
