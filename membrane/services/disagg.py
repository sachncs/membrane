"""Prefill / decode disaggregation on a running node.

With ``--role prefill``, a node computes KV for prompts sent to
``POST /disagg/prefill`` (or the ``Prefill`` RPC on ``--grpc-port``): it
reuses what is cached, prefills the rest with its compute backend, stores
the fragments, and records a *KV manifest* naming them under the returned
``kv_handle``. With ``--role decode``, a node answers ``POST /disagg/decode``
for a handle: it finds the manifest (locally, through the gossiped
location directory, or by asking prefill-capable peers), copies the
fragments it lacks from the prefill node with their bytes verified, and
generates with its compute backend. ``--role both`` (the default) serves
both phases.

Manifests are per tenant: a handle prefilled by one tenant is not visible
to another, even for an identical prompt.
"""

import hashlib
import json
import logging
import time
from typing import Any

from membrane.auth import AuthContext
from membrane.disagg.protocol import DecodeRequest, DecodeResponse, PrefillRequest, make_handle_for
from membrane.disagg.service import CALLER, DecodeService, PrefillService, RoleUnavailableError
from membrane.fragment import Fragment
from membrane.fragment_kind import FragmentKind
from membrane.identity import PayloadIdentity
from membrane.services.memory import MemoryService, caller_of
from membrane.services.policies import copy_fragment
from membrane.store.tenant_guard import TenantGuard

logger = logging.getLogger(__name__)

ROLES = frozenset({"prefill", "decode", "both"})
#: Roles (static or dynamic) of peers worth asking for a manifest.
PREFILL_CAPABLE = frozenset({"prefill", "both", "prefill_worker", ""})
#: Seconds a manifest lives (as long as the KV it names, by default).
MANIFEST_TTL_SEC = 3600.0


def manifest_hash(kv_handle: str, tenant: str) -> str:
    """The content hash a tenant's manifest for ``kv_handle`` is stored under.

    Args:
        kv_handle: The prefill handle.
        tenant: The caller's tenant.

    Returns:
        str: SHA-256 hex digest.
    """
    return hashlib.sha256(f"kv-manifest\x00{tenant}\x00{kv_handle}".encode()).hexdigest()


class Disaggregation:
    """Prefill and decode services bound to this node.

    Attributes:
        memory: Memory service (reuse, prefill, tenant checks).
        view: Cluster state (peers to fetch from).
        backend: Compute backend that generates on decode.
        role: ``prefill``, ``decode``, or ``both``.
        prefill_service: Serves ``/disagg/prefill``.
        decode_service: Serves ``/disagg/decode``.
        fetched: Fragments copied from prefill nodes so far.
    """

    def __init__(self, memory: MemoryService, view: Any, backend: Any, role: str = "both") -> None:
        """Bind the services.

        Args:
            memory: The node's memory service.
            view: :class:`~membrane.services.placement.ClusterView`.
            backend: The node's compute backend.
            role: Which phases this node serves.

        Raises:
            ValueError: On an unknown role.
        """
        if role not in ROLES:
            raise ValueError(f"role must be one of {sorted(ROLES)}")
        self.memory = memory
        self.view = view
        self.backend = backend
        self.role = role
        self.fetched = 0
        self.prefill_service = PrefillService(backend=self, cached_prefix=self.cached_prefix)
        self.decode_service = DecodeService(backend=self)

    @property
    def node(self) -> Any:
        """The local node."""
        return self.memory.node

    # ------------------------------------------------------------------
    # Prefill
    # ------------------------------------------------------------------

    def cached_prefix(self, request: PrefillRequest) -> int:
        """Leading prompt tokens already cached for the caller.

        Args:
            request: The prefill request.

        Returns:
            int: Matched tokens.
        """
        lookup = self.memory.prefix_lookup(list(request.token_ids), request.model_id, CALLER.get())
        return int(lookup["matched_tokens"])

    def run_prefill(self, request: PrefillRequest, cached_prefix_len: int) -> tuple[int, float]:
        """Reuse cached KV, prefill the rest, and record the manifest.

        Args:
            request: The prefill request.
            cached_prefix_len: Tokens already cached (informational).

        Returns:
            tuple[int, float]: ``(prompt_len, prefill_ms)``.

        Raises:
            RoleUnavailableError: On a decode-only node.
            ValueError: When the prompt could not be fully covered.
        """
        if self.role == "decode":
            raise RoleUnavailableError("this node only decodes (--role decode)")
        started = time.perf_counter()
        caller = CALLER.get()
        result = self.memory.reconstruct(list(request.token_ids), request.model_id, caller, prefill=True)
        if result["coverage"] < 1.0:
            raise ValueError(f"prefill covered {result['coverage']:.0%} of the prompt")
        hashes = [f["identity"]["payload_hash"] for f in result["fragments"]]
        self.store_manifest(request, hashes, caller)
        return len(request.token_ids), (time.perf_counter() - started) * 1000.0

    def store_manifest(self, request: PrefillRequest, hashes: list[str], caller: AuthContext | None) -> Fragment:
        """Store the manifest naming a prefill's fragments.

        Args:
            request: The prefill request.
            hashes: Fragments covering the prompt, in token order.
            caller: Authenticated caller.

        Returns:
            Fragment: The stored manifest fragment.
        """
        tenant, scopes = caller_of(caller)
        kv_handle = make_handle_for(request).handle
        content_hash = manifest_hash(kv_handle, tenant)
        payload = json.dumps(
            {
                "kv_handle": kv_handle,
                "model_id": request.model_id,
                "token_ids": list(request.token_ids),
                "fragments": hashes,
                "node_id": self.view.local_id,
            }
        ).encode()
        identity = PayloadIdentity(
            payload_hash=content_hash,
            model_id=FragmentKind.KV_MANIFEST,
            model_revision="",
            tokenizer_name=FragmentKind.KV_MANIFEST,
            tokenizer_revision="",
            layer_range=(0, 0),
            head_range=(-1, -1),
            token_span=(0, max(0, len(request.token_ids) - 1)),
            dtype="uint8",
            shape=(len(payload),),
        )
        manifest = Fragment(
            identity=identity,
            payload_ref=content_hash,
            payload_size=len(payload),
            ttl=MANIFEST_TTL_SEC,
            reuse_score=0.5,
            version_id=1,
        )
        if tenant:
            manifest = manifest.with_tenant(tenant)
        self.node.content_store.put(content_hash, payload)
        self.node.store(manifest, is_primary=True, caller_tenant=tenant, caller_scopes=scopes)
        return manifest

    # ------------------------------------------------------------------
    # Decode
    # ------------------------------------------------------------------

    def decode(self, request: DecodeRequest) -> DecodeResponse:
        """Fetch the handle's KV if needed, then generate.

        Args:
            request: The decode request.

        Returns:
            DecodeResponse: Generated tokens (``finished`` once generation
            stops; the compute backend generates up to ``max_tokens`` at once).

        Raises:
            RoleUnavailableError: On a prefill-only node.
            LookupError: When no node knows the handle, or its KV is gone.
            ValueError: When the handle was prefilled for another model.
        """
        if self.role == "prefill":
            raise RoleUnavailableError("this node only prefills (--role prefill)")
        caller = CALLER.get()
        manifest = self.find_manifest(request.kv_handle, caller)
        if manifest is None:
            raise LookupError(f"unknown kv_handle {request.kv_handle}")
        if manifest["model_id"] != request.model_id:
            raise ValueError(f"kv_handle was prefilled for model {manifest['model_id']!r}")
        self.ensure_local(manifest, caller)
        generated = self.backend.generate(list(manifest["token_ids"]), request.model_id, request.max_tokens)
        tokens = tuple(int(t) for t in generated.get("tokens", ()))[: request.max_tokens]
        return DecodeResponse(request_id=request.request_id, token_ids=tokens, finished=True)

    def find_manifest(self, kv_handle: str, caller: AuthContext | None) -> dict[str, Any] | None:
        """Find the caller's manifest for ``kv_handle``, fetching it from a peer if needed.

        Args:
            kv_handle: The prefill handle.
            caller: Authenticated caller.

        Returns:
            dict[str, Any] | None: The manifest, or ``None`` when no reachable
            node has it for this tenant.
        """
        tenant, scopes = caller_of(caller)
        content_hash = manifest_hash(kv_handle, tenant)
        fragment = self.node.retrieve(content_hash, caller_tenant=tenant, caller_scopes=scopes)
        if fragment is None:
            for peer in self.candidates(content_hash):
                copied = copy_fragment(peer, self.node, content_hash)
                if copied is not None:
                    fragment = copied
                    break
        if fragment is None or not TenantGuard.can_read(fragment.tenant_id, tenant, scopes):
            return None
        payload = self.node.content_store.get(content_hash)
        return json.loads(payload) if payload is not None else None

    def candidates(self, content_hash: str) -> list[Any]:
        """Peers that may hold ``content_hash``: known holders, then prefill-capable peers.

        Args:
            content_hash: Content hash.

        Returns:
            list[Any]: Peer clients, best first.
        """
        cluster = self.view.cluster
        if cluster is None:
            return []
        ordered = [n for n in self.view.holders(content_hash) if n != self.view.local_id]
        ordered += [p.node_id for p in cluster.membership.healthy() if p.role in PREFILL_CAPABLE]
        clients = [cluster.membership.get_client(n) for n in dict.fromkeys(ordered)]
        return [c for c in clients if c is not None]

    def ensure_local(self, manifest: dict[str, Any], caller: AuthContext | None) -> None:
        """Copy the manifest's fragments this node lacks from the prefill node.

        Args:
            manifest: The manifest.
            caller: Authenticated caller.

        Raises:
            LookupError: When a fragment cannot be fetched or is not the
                caller's to read.
        """
        tenant, scopes = caller_of(caller)
        cluster = self.view.cluster
        for content_hash in manifest["fragments"]:
            if self.node.locate(content_hash, tenant, scopes) is not None:
                continue
            sources = []
            if cluster is not None:
                origin = cluster.membership.get_client(manifest["node_id"])
                sources = ([origin] if origin is not None else []) + self.candidates(content_hash)
            copied = next((f for f in (copy_fragment(p, self.node, content_hash) for p in sources) if f), None)
            if copied is None:
                raise LookupError(f"KV fragment {content_hash} of this handle is no longer available")
            self.fetched += 1
        for content_hash in manifest["fragments"]:
            if self.node.retrieve(content_hash, caller_tenant=tenant, caller_scopes=scopes) is None:
                raise LookupError(f"KV fragment {content_hash} is not readable")


def bind_caller(context: AuthContext | None) -> Any:
    """Set the disaggregation caller for the current request.

    Args:
        context: Authenticated caller.

    Returns:
        Any: Token for :meth:`ContextVar.reset`.
    """
    return CALLER.set(context)


__all__ = [
    "MANIFEST_TTL_SEC",
    "PREFILL_CAPABLE",
    "ROLES",
    "Disaggregation",
    "bind_caller",
    "manifest_hash",
]
