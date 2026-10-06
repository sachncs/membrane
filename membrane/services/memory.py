"""The memory API a running node serves: reconstruction, prefixes, sessions, typed objects.

:class:`MemoryService` puts the node's indexes to work for clients:

* ``POST /reconstruct`` assembles the cached fragments that cover a
  prompt (:class:`~membrane.reconstructor.Reconstructor`), optionally
  prefilling the gaps with the node's compute backend. Every fragment
  returned hashes to the prompt tokens it covers and is visible to the
  caller's tenant.
* ``GET /prefix/lookup`` answers how many leading tokens of a prompt
  are cached, memoized in a :class:`~membrane.prefix_cache.PrefixCache`
  that forgets an entry as soon as one of its fragments leaves the node.
* ``/sessions/{id}`` returns the fragments a session read, recorded by
  :class:`~membrane.sessions.Sessions` from the ``X-Membrane-Session``
  header.
* ``/objects`` stores and reads the typed memory objects
  (:class:`~membrane.prefix.Prefix`, :class:`~membrane.segment.Segment`,
  :class:`~membrane.artifact.Artifact`, :class:`~membrane.trace.Trace`)
  as fragments whose bytes the server addresses itself.

It also guards writes: with a compatibility fingerprint configured
(``--require-compat MODEL``), a fragment stamped for a different model
or tokenizer is refused (:class:`~membrane.compat.MembraneValidator`),
and a fragment whose hash is already bound to a different identity
(:class:`~membrane.identity_index.IdentityIndex`) is refused rather than
aliasing two different KV tensors.
"""

import base64
import binascii
import dataclasses
import hashlib
import itertools
import json
import logging
import re
import threading
from collections import Counter, deque
from typing import Any, override

from membrane.auth import AuthContext
from membrane.compat import MembraneIncompatibleError, MembraneValidator, ModelCompatibilityFingerprint
from membrane.compute.base import Backend
from membrane.constants import SESSION_HEADER
from membrane.fragment import Fragment
from membrane.fragment_kind import FragmentKind
from membrane.identity import PayloadIdentity
from membrane.identity_index import IdentityIndex
from membrane.prefilling import Adapter, PrefillResult
from membrane.prefix import Prefix
from membrane.prefix_cache import KVHandle, PrefixCache
from membrane.reconstructor import Reconstructor, ReconstructorConfig
from membrane.serialization import to_dict
from membrane.sessions import Sessions
from membrane.store.tenant_guard import TenantGuard
from membrane.weighted import Weighted

logger = logging.getLogger(__name__)

#: Tokens accepted in one reconstruction or lookup.
MAX_PROMPT_TOKENS = 1 << 20
#: Distinct hashes whose hit counts are tracked between promotion passes.
MAX_TRACKED_HITS = 100_000
#: Recent reads kept for value estimates (most recent last).
RECENT_READS = 1024
#: Neighbours returned as prefetch hints.
PREFETCH_HINTS = 8
OBJECT_KINDS = {
    "prefix": FragmentKind.PREFIX,
    "segment": FragmentKind.KV,
    "artifact": FragmentKind.ARTIFACT,
    "trace": FragmentKind.TRACE,
}


class StoreRejectedError(ValueError):
    """A write was refused by the compatibility or identity check.

    Attributes:
        status: HTTP status to answer with.
    """

    def __init__(self, message: str, status: int = 409) -> None:
        """Create the error.

        Args:
            message: What was wrong.
            status: HTTP status to answer with.
        """
        super().__init__(message)
        self.status = status


def caller_of(auth_context: AuthContext | None) -> tuple[str, frozenset[str]]:
    """Return the caller's tenant and scopes.

    Args:
        auth_context: Authenticated caller; ``None`` when authentication is off.

    Returns:
        tuple[str, frozenset[str]]: ``(tenant, scopes)``.
    """
    if auth_context is None:
        return "", frozenset()
    return auth_context.subject, auth_context.scopes


class BackendPrefill(Adapter):
    """Prefill adapter that runs the node's compute backend."""

    def __init__(self, backend: Backend) -> None:
        """Wrap ``backend``.

        Args:
            backend: The node's compute backend.
        """
        super().__init__()
        self.backend = backend

    @override
    def prefill(self, prompt_tokens: list[int], model_id: str) -> PrefillResult:
        """Run the backend on ``prompt_tokens``.

        Args:
            prompt_tokens: Tokens of one gap.
            model_id: Model identifier.

        Returns:
            PrefillResult: The backend's fragments (spans relative to the gap).
        """
        fragments = self.backend.prefill(prompt_tokens, model_id)
        return PrefillResult(kv_size=0.0, latency_seconds=0.0, routing_decision=None, fragments=fragments)


class MemoryService:
    """Memory API over one node.

    Attributes:
        node: The local node.
        backend: Compute backend used to prefill gaps.
        sessions: Per-session read history.
        prefix_cache: Memoized prefix lookups.
        coaccess: Fragments read together, for prefetch hints.
        identities: Identity bound to every stored hash.
        validator: Compatibility check for writes (``None``: off).
        hits: Reads per hash since the last promotion pass.
        recent: The latest reads, most recent last.
    """

    def __init__(
        self,
        node: Any,
        backend: Backend | None = None,
        compat: ModelCompatibilityFingerprint | None = None,
        prefix_cache_capacity: int = 4096,
        sessions: Sessions | None = None,
    ) -> None:
        """Create the service.

        Args:
            node: The local :class:`~membrane.node.Node`.
            backend: Compute backend for ``prefill`` reconstructions.
            compat: Fingerprint every written fragment must carry; ``None``
                accepts any.
            prefix_cache_capacity: Prefix lookups memoized.
            sessions: Session tracker; a bounded one by default.
        """
        self.node = node
        self.backend = backend
        self.sessions = sessions or Sessions()
        self.prefix_cache = PrefixCache(capacity=prefix_cache_capacity)
        self.coaccess = Weighted()
        self.identities = IdentityIndex()
        self.validator = MembraneValidator(compat) if compat is not None else None
        self.hits: Counter[str] = Counter()
        self.recent: deque[str] = deque(maxlen=RECENT_READS)
        self.__lock = threading.Lock()
        # Prefix-cache entries that depend on each fragment, and the
        # fragments each entry covers.
        self.__dependents: dict[str, set[str]] = {}
        self.__covers: dict[str, tuple[str, ...]] = {}

    # ------------------------------------------------------------------
    # Node hooks
    # ------------------------------------------------------------------

    def on_stored(self, fragment: Fragment) -> None:
        """Track a fragment the node just stored (called under the node lock).

        Args:
            fragment: The stored fragment.
        """
        content_hash = fragment.identity.payload_hash
        for entry in self.identities.lookup_by_hash(content_hash):
            self.identities.remove(entry.identity)
        self.identities.insert(fragment.identity, content_hash)

    def on_removed(self, content_hash: str) -> None:
        """Forget a fragment that left the node, and every lookup that used it.

        Args:
            content_hash: The removed fragment's hash.
        """
        for entry in self.identities.lookup_by_hash(content_hash):
            self.identities.remove(entry.identity)
        with self.__lock:
            handles = self.__dependents.pop(content_hash, set())
            self.hits.pop(content_hash, None)
            for handle in handles:
                self.__covers.pop(handle, None)
        for handle in handles:
            self.prefix_cache.by_handle.pop(handle, None)

    # ------------------------------------------------------------------
    # Write guard
    # ------------------------------------------------------------------

    def check_store(self, fragment: Fragment) -> None:
        """Refuse a fragment built for another model, or one that aliases another identity.

        Args:
            fragment: The fragment about to be stored.

        Raises:
            StoreRejectedError: On a compatibility mismatch or identity conflict.
        """
        if self.validator is not None:
            try:
                self.validator.validate(fragment)
            except MembraneIncompatibleError as exc:
                raise StoreRejectedError(str(exc)) from exc
        for entry in self.identities.lookup_by_hash(fragment.identity.payload_hash):
            if entry.identity != fragment.identity:
                raise StoreRejectedError(
                    f"payload hash {fragment.identity.payload_hash} is already bound to a different identity"
                )

    def stamp(self, fragment: Fragment) -> Fragment:
        """Stamp the configured compatibility fingerprint on a fragment the node produced.

        Args:
            fragment: A fragment from the compute backend.

        Returns:
            Fragment: The fragment, carrying the fingerprint when one is set.
        """
        if self.validator is None or fragment.fingerprint_compat:
            return fragment
        return dataclasses.replace(fragment, fingerprint_compat=self.validator.hash)

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    @staticmethod
    def session_key(session_id: str, tenant: str) -> str:
        """Key a session by tenant so one tenant cannot read another's history.

        Args:
            session_id: Client-chosen session identifier.
            tenant: Caller's tenant.

        Returns:
            str: The tracker key.
        """
        return f"{tenant}\x00{session_id}"

    def record_access(self, content_hash: str, session_id: str = "", tenant: str = "") -> None:
        """Record a read: reuse score, hit count, and the session's history.

        Args:
            content_hash: The fragment read.
            session_id: Session it belongs to (``""``: none).
            tenant: Caller's tenant (scopes the session).
        """
        self.node.record_hit(content_hash)
        with self.__lock:
            self.recent.append(content_hash)
            self.hits[content_hash] += 1
            if len(self.hits) > MAX_TRACKED_HITS:
                for key, _count in self.hits.most_common()[MAX_TRACKED_HITS // 2 :]:
                    del self.hits[key]
        if session_id:
            self.sessions.record_access(self.session_key(session_id, tenant), content_hash)

    def take_hits(self) -> Counter[str]:
        """Return and reset the read counts since the last call.

        Returns:
            Counter[str]: Reads per hash.
        """
        with self.__lock:
            hits, self.hits = self.hits, Counter()
        return hits

    def reconstructor(self, auth_context: AuthContext | None, prefill: bool) -> Reconstructor:
        """Build a reconstructor scoped to the caller.

        Args:
            auth_context: Authenticated caller.
            prefill: Whether gaps are prefilled.

        Returns:
            Reconstructor: Over the node's index, filtered to the caller's tenant.
        """
        tenant, scopes = caller_of(auth_context)

        def visible(fragment: Fragment) -> bool:
            if fragment.identity.payload_hash not in self.node.fragments:
                return False  # in the index but evicted
            return TenantGuard.can_read(fragment.tenant_id, tenant, scopes)

        def store_prefilled(fragment: Fragment) -> None:
            fragment = self.stamp(fragment.with_tenant(tenant) if tenant else fragment)
            ref = fragment.payload_ref
            if ref is not None and self.backend is not None and not self.node.content_store.has(ref):
                payload = self.backend.simulated_payload(fragment)
                if payload is not None:
                    self.node.content_store.put(ref, payload)
            self.node.store(fragment, is_primary=True, caller_tenant=tenant, caller_scopes=scopes)

        adapter: Adapter = BackendPrefill(self.backend) if prefill and self.backend is not None else NoPrefill()
        config = ReconstructorConfig(max_gap_tokens=0 if prefill else 1 << 30, index_prefilled=False)
        return Reconstructor(
            self.node.index_system,
            adapter,
            config=config,
            visible=visible,
            on_prefilled=store_prefilled if prefill else None,
        )

    def reconstruct(
        self,
        tokens: list[int],
        model_id: str,
        auth_context: AuthContext | None = None,
        prefill: bool = False,
        session_id: str = "",
    ) -> dict[str, Any]:
        """Assemble the fragments that cover ``tokens``.

        Args:
            tokens: Prompt token IDs.
            model_id: Model identifier.
            auth_context: Authenticated caller.
            prefill: Prefill uncovered spans with the compute backend.
            session_id: Session the read belongs to.

        Returns:
            dict[str, Any]: ``fragments`` (in token order), ``coverage``,
            ``missing`` spans, ``prefilled``, and ``prefetch`` hints.
        """
        result = self.reconstructor(auth_context, prefill).rebuild_context(list(tokens), model_id)
        hashes = [f.identity.payload_hash for f in result.fragments]
        tenant, _scopes = caller_of(auth_context)
        for content_hash in hashes:
            self.record_access(content_hash, session_id, tenant)
        for first, second in itertools.pairwise(hashes):
            previous = self.coaccess.get_edge_weight(first, second, "co_access")
            self.coaccess.add_weighted_edge(first, second, "co_access", min(1.0, previous + 0.1))
        prefetch = self.prefetch_hints(hashes[-1]) if hashes else []
        return {
            "fragments": [to_dict(f) for f in result.fragments],
            "coverage": result.coverage_ratio,
            "missing": [list(span) for span in result.missing_segments],
            "prefilled": result.prefill_invoked,
            "prefetch": [h for h in prefetch if h not in hashes],
        }

    def prefetch_hints(self, content_hash: str) -> list[str]:
        """Fragments often read after ``content_hash`` that the node holds.

        Args:
            content_hash: The last fragment read.

        Returns:
            list[str]: Up to :data:`PREFETCH_HINTS` hashes.
        """
        if not self.coaccess.has_node(content_hash):
            return []
        neighbours = self.coaccess.get_strong_neighbors(content_hash, "co_access", min_weight=0.3)
        return sorted(h for h in neighbours if h in self.node.fragments)[:PREFETCH_HINTS]

    def prefix_lookup(
        self, tokens: list[int], model_id: str, auth_context: AuthContext | None = None
    ) -> dict[str, Any]:
        """How many leading tokens of a prompt this node has cached.

        Args:
            tokens: Prompt token IDs.
            model_id: Model identifier.
            auth_context: Authenticated caller.

        Returns:
            dict[str, Any]: ``matched_tokens``, ``total_tokens``, ``full``,
            ``fragments`` (hashes covering the matched prefix), and ``cached``
            (answered from the prefix cache).
        """
        tenant, _scopes = caller_of(auth_context)
        key = f"{tenant}\x00{model_id}"
        prompt = tuple(tokens)
        match = self.prefix_cache.lookup(key, prompt)
        if match.handle is not None:
            with self.__lock:
                hashes = list(self.__covers.get(match.handle.handle, ()))
            if hashes and all(h in self.node.fragments for h in hashes):
                return self.lookup_answer(match.token_len, len(prompt), hashes, cached=True)
        result = self.reconstructor(auth_context, prefill=False).rebuild_context(list(prompt), model_id)
        matched, hashes = 0, []
        for fragment in result.fragments:  # sorted by start: walk the contiguous prefix
            start, end = fragment.identity.token_span
            if start != matched:
                break
            matched = end + 1
            hashes.append(fragment.identity.payload_hash)
        if matched:
            handle = self.prefix_cache.insert(key, prompt[:matched], layer_range=(0, 0))
            self.remember(handle, hashes)
        return self.lookup_answer(matched, len(prompt), hashes, cached=False)

    def remember(self, handle: KVHandle, hashes: list[str]) -> None:
        """Record which fragments a memoized prefix depends on.

        Args:
            handle: The prefix-cache entry.
            hashes: Fragments covering it.
        """
        with self.__lock:
            self.__covers[handle.handle] = tuple(hashes)
            for content_hash in hashes:
                self.__dependents.setdefault(content_hash, set()).add(handle.handle)
            if len(self.__covers) > 2 * max(1, self.prefix_cache.capacity):
                # Drop bookkeeping for entries the cache evicted on its own.
                live = set(self.prefix_cache.by_handle)
                self.__covers = {h: c for h, c in self.__covers.items() if h in live}
                for content_hash in list(self.__dependents):
                    self.__dependents[content_hash] &= live
                    if not self.__dependents[content_hash]:
                        del self.__dependents[content_hash]

    @staticmethod
    def lookup_answer(matched: int, total: int, hashes: list[str], cached: bool) -> dict[str, Any]:
        """Build the ``/prefix/lookup`` body.

        Args:
            matched: Leading tokens cached.
            total: Prompt length.
            hashes: Fragments covering the matched tokens.
            cached: Whether the prefix cache answered.

        Returns:
            dict[str, Any]: The response body.
        """
        return {
            "matched_tokens": matched,
            "total_tokens": total,
            "full": total > 0 and matched == total,
            "fragments": hashes,
            "cached": cached,
        }

    def session(self, session_id: str, auth_context: AuthContext | None = None) -> dict[str, Any]:
        """Return a session's read history.

        Args:
            session_id: Session identifier.
            auth_context: Authenticated caller (sessions are per tenant).

        Returns:
            dict[str, Any]: ``history`` (oldest first) and ``unique`` count.
        """
        tenant, _scopes = caller_of(auth_context)
        history = self.sessions.get_session_history(self.session_key(session_id, tenant))
        return {"session_id": session_id, "history": history, "unique": len(set(history))}

    def forget_session(self, session_id: str, auth_context: AuthContext | None = None) -> bool:
        """Drop a session's history.

        Args:
            session_id: Session identifier.
            auth_context: Authenticated caller.

        Returns:
            bool: True when the session existed.
        """
        tenant, _scopes = caller_of(auth_context)
        return self.sessions.forget(self.session_key(session_id, tenant))

    # ------------------------------------------------------------------
    # Typed memory objects
    # ------------------------------------------------------------------

    def put_object(self, body: dict[str, Any], auth_context: AuthContext | None = None) -> dict[str, Any]:
        """Store a typed memory object: its bytes and the fragment describing them.

        The server derives the content hash from the bytes, so a client
        cannot bind a hash to content it does not hold.

        Args:
            body: ``{"kind": "prefix", "tokens": [...]}``,
                ``{"kind": "segment", "layer", "head", "token_span", "tensor_shape", "data": base64}``,
                ``{"kind": "artifact", "source_url", "data": base64, "token_count"}``, or
                ``{"kind": "trace", "tool_name", "input", "output"}``; each may
                carry ``reuse_score`` and ``ttl``.
            auth_context: Authenticated caller.

        Returns:
            dict[str, Any]: ``{"content_hash", "kind"}``.

        Raises:
            StoreRejectedError: On an invalid body (``status`` 400) or a refused write.
        """
        try:
            fragment, payload = self.build_object(body)
        except (KeyError, TypeError, ValueError, binascii.Error) as exc:
            raise StoreRejectedError(f"invalid {body.get('kind', 'object')!r}: {exc}", status=400) from exc
        tenant, scopes = caller_of(auth_context)
        if tenant:
            fragment = fragment.with_tenant(tenant)
        self.check_store(fragment)
        if fragment.payload_ref is not None:
            self.node.content_store.put(fragment.payload_ref, payload)
        self.node.store(fragment, is_primary=True, caller_tenant=tenant, caller_scopes=scopes)
        return {"content_hash": fragment.identity.payload_hash, "kind": body["kind"]}

    @staticmethod
    def build_object(body: dict[str, Any]) -> tuple[Fragment, bytes]:
        """Turn a request body into a fragment and its bytes.

        Args:
            body: The request body (see :meth:`put_object`).

        Returns:
            tuple[Fragment, bytes]: The fragment and the bytes to store.

        Raises:
            ValueError: On an unknown kind or invalid field.
        """
        from membrane.artifact import Artifact
        from membrane.segment import Segment
        from membrane.trace import Trace

        kind = str(body["kind"])
        reuse = float(body.get("reuse_score", 0.5))
        obj: Prefix | Segment | Artifact | Trace
        if kind == "prefix":
            tokens = tuple(int(t) for t in body["tokens"])
            payload = json.dumps(list(tokens)).encode()
            digest = hashlib.sha256(payload).hexdigest()
            obj = Prefix(
                tokens=tokens,
                content_hash=digest,
                semantic_hash=digest[:16],
                size_bytes=len(payload),
                token_count=len(tokens),
                reuse_score=reuse,
            )
        elif kind == "segment":
            payload = base64.b64decode(body["data"], validate=True)
            digest = hashlib.sha256(payload).hexdigest()
            span = body["token_span"]
            obj = Segment(
                layer=int(body["layer"]),
                head=int(body.get("head", 0)),
                token_span=(int(span[0]), int(span[1])),
                tensor_shape=tuple(int(d) for d in body["tensor_shape"]),
                content_hash=digest,
                semantic_hash=digest[:16],
                size_bytes=len(payload),
                reuse_score=reuse,
            )
        elif kind == "artifact":
            payload = base64.b64decode(body["data"], validate=True)
            digest = hashlib.sha256(payload).hexdigest()
            obj = Artifact(
                source_url=str(body.get("source_url", "")),
                text_hash=digest,
                embedding=(),
                content_hash=digest,
                semantic_hash=digest[:16],
                size_bytes=len(payload),
                token_count=int(body.get("token_count", 0)),
                reuse_score=reuse,
            )
        elif kind == "trace":
            tool, tool_input, output = str(body["tool_name"]), str(body.get("input", "")), str(body["output"])
            payload = json.dumps({"tool_name": tool, "input": tool_input, "output": output}, sort_keys=True).encode()
            digest = hashlib.sha256(payload).hexdigest()
            obj = Trace(
                tool_name=tool,
                input_hash=hashlib.sha256(tool_input.encode()).hexdigest(),
                output_hash=hashlib.sha256(output.encode()).hexdigest(),
                structured_output=output,
                content_hash=digest,
                semantic_hash=digest[:16],
                size_bytes=len(payload),
                reuse_score=reuse,
            )
        else:
            raise ValueError(f"kind must be one of {sorted(OBJECT_KINDS)}")
        fragment = obj.materialize()
        if "ttl" in body:
            fragment = dataclasses.replace(fragment, ttl=float(body["ttl"]))
        return fragment, payload

    def get_object(self, content_hash: str, auth_context: AuthContext | None = None) -> dict[str, Any] | None:
        """Read a typed memory object.

        Args:
            content_hash: The object's hash.
            auth_context: Authenticated caller.

        Returns:
            dict[str, Any] | None: ``kind``, its fields, and ``data`` (base64
            bytes), or ``None`` when absent, not visible, or not a typed object.
        """
        from membrane.artifact import Artifact
        from membrane.segment import Segment
        from membrane.trace import Trace

        tenant, scopes = caller_of(auth_context)
        fragment = self.node.retrieve(content_hash, caller_tenant=tenant, caller_scopes=scopes)
        if fragment is None:
            return None
        kinds: dict[str, tuple[str, Any]] = {
            FragmentKind.PREFIX: ("prefix", Prefix),
            FragmentKind.KV: ("segment", Segment),
            FragmentKind.ARTIFACT: ("artifact", Artifact),
            FragmentKind.TRACE: ("trace", Trace),
        }
        found = kinds.get(fragment.identity.model_id)
        if found is None:
            return None
        kind, cls = found
        payload = self.node.content_store.get(fragment.payload_ref) if fragment.payload_ref else None
        fields = dataclasses.asdict(cls.from_fragment(fragment))
        if kind == "prefix" and payload is not None:
            fields["tokens"] = json.loads(payload)
        if kind == "trace" and payload is not None:
            fields.update(json.loads(payload))
        self.record_access(content_hash, tenant=tenant)
        return {
            "kind": kind,
            "content_hash": content_hash,
            "object": fields,
            "data": base64.b64encode(payload).decode() if payload is not None else None,
        }


#: Characters allowed in a KV bundle handle.
BUNDLE_HANDLE = re.compile(r"[A-Za-z0-9._:-]{1,256}")


def bundle_hash(model_id: str, handle: str, tenant: str) -> str:
    """The content hash a tenant's named KV bundle is stored under.

    Args:
        model_id: Model the bytes were produced by.
        handle: Client-chosen name.
        tenant: Caller's tenant.

    Returns:
        str: SHA-256 hex digest.
    """
    return hashlib.sha256(f"kv-bundle\x00{tenant}\x00{model_id}\x00{handle}".encode()).hexdigest()


class BundleStore:
    """Engine KV bytes stored under a client-chosen name, per tenant and model.

    The engine adapters (vLLM, SGLang, TensorRT-LLM) address KV by their
    own handles; a bundle maps such a handle to bytes held like any other
    fragment (replicated, evicted, persisted).

    Attributes:
        memory: The memory service (node access and write guard).
    """

    def __init__(self, memory: MemoryService) -> None:
        """Create the store.

        Args:
            memory: The memory service.
        """
        self.memory = memory

    def put(
        self, model_id: str, handle: str, data: bytes, auth_context: AuthContext | None = None, ttl: float = 3600.0
    ) -> str:
        """Store ``data`` under ``handle``, replacing an earlier bundle.

        Args:
            model_id: Model the bytes were produced by.
            handle: Client-chosen name.
            data: The bytes.
            auth_context: Authenticated caller.
            ttl: Seconds the bundle lives.

        Returns:
            str: The bundle's content hash.

        Raises:
            StoreRejectedError: On an invalid handle or model (``status`` 400).
        """
        if not BUNDLE_HANDLE.fullmatch(handle) or not model_id or len(model_id) > 256:
            raise StoreRejectedError("handle must be 1-256 of [A-Za-z0-9._:-] and model_id 1-256 chars", status=400)
        tenant, scopes = caller_of(auth_context)
        content_hash = bundle_hash(model_id, handle, tenant)
        identity = PayloadIdentity(
            payload_hash=content_hash,
            model_id=FragmentKind.KV_BUNDLE,
            model_revision="",
            tokenizer_name=model_id,
            tokenizer_revision="",
            layer_range=(0, 0),
            head_range=(-1, -1),
            token_span=(0, 0),
            dtype="uint8",
            shape=(len(data),),
        )
        fragment = Fragment(
            identity=identity,
            payload_ref=content_hash,
            payload_size=len(data),
            ttl=ttl,
            reuse_score=0.5,
            version_id=1,
        )
        if tenant:
            fragment = fragment.with_tenant(tenant)
        node = self.memory.node
        with node.lock:
            if content_hash in node.fragments:
                node.remove_fragment(content_hash)  # a new version replaces the old
        node.content_store.put(content_hash, data)
        node.store(fragment, is_primary=True, caller_tenant=tenant, caller_scopes=scopes)
        return content_hash

    def get(self, model_id: str, handle: str, auth_context: AuthContext | None = None) -> bytes | None:
        """Read a bundle.

        Args:
            model_id: Model the bytes were produced by.
            handle: Client-chosen name.
            auth_context: Authenticated caller.

        Returns:
            bytes | None: The bytes, or ``None`` when absent or not visible.
        """
        if not BUNDLE_HANDLE.fullmatch(handle):
            return None
        tenant, scopes = caller_of(auth_context)
        content_hash = bundle_hash(model_id, handle, tenant)
        node = self.memory.node
        if node.retrieve(content_hash, caller_tenant=tenant, caller_scopes=scopes) is None:
            return None
        data = node.content_store.get(content_hash)
        if data is not None:
            self.memory.record_access(content_hash, tenant=tenant)
        return data


class NoPrefill(Adapter):
    """Adapter that never prefills (lookups and reads only)."""

    @override
    def prefill(self, prompt_tokens: list[int], model_id: str) -> PrefillResult:
        """Return no fragments.

        Args:
            prompt_tokens: Ignored.
            model_id: Ignored.

        Returns:
            PrefillResult: Empty.
        """
        return PrefillResult(kv_size=0.0, latency_seconds=0.0, routing_decision=None, fragments=[])


__all__ = [
    "BUNDLE_HANDLE",
    "MAX_PROMPT_TOKENS",
    "OBJECT_KINDS",
    "RECENT_READS",
    "SESSION_HEADER",
    "BackendPrefill",
    "BundleStore",
    "MemoryService",
    "NoPrefill",
    "StoreRejectedError",
    "bundle_hash",
    "caller_of",
]
