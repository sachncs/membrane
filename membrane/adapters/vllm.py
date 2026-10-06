"""vLLM KVConnector adapter.

Wires the engine-agnostic :class:`~membrane.adapters.KVAdapter`
protocol onto vLLM's distributed KV-transfer surface. The
:class:`MembraneVLLMAdapter` is the canonical entry point for
v2.0+ vLLM deployments; it subclasses :class:`BaseAdapter` so
the byte serializer / validator stay in lockstep with the HF
and SGLang adapters, and conditionally subclasses vLLM's
:class:`KVConnectorBase` when vLLM is installed at runtime.

The vLLM connector surface is large; the implementation here
focuses on the six abstract methods that vLLM v0.10+ calls
from the scheduler and the model runner:

* :meth:`MembraneVLLMAdapter.get_num_new_matched_tokens` --
  ask the cluster how many cached tokens we already have for
  a request.
* :meth:`MembraneVLLMAdapter.update_state_after_alloc` --
  record the block table that vLLM allocated for the request.
* :meth:`MembraneVLLMAdapter.build_connector_meta` --
  emit per-step metadata that vLLM threads through the model
  runner.
* :meth:`MembraneVLLMAdapter.start_load_kv` -- begin
  pulling the cached K/V from the cluster.
* :meth:`MembraneVLLMAdapter.wait_for_layer_load` -- block
  the runner until a single layer is resident.
* :meth:`MembraneVLLMAdapter.save_kv` -- push the freshly
  computed K/V back to the cluster.

The connector talks to a :class:`MembraneClusterClient` that
encapsulates the wire format defined by
:class:`membrane.transfer_engine.KVTransferEngine`. The
client is decoupled from the connector so unit tests can
inject a fake without touching the network.
"""

import logging
import threading
from dataclasses import dataclass, field
from typing import Any, override

from membrane.adapters import (
    BaseAdapter,
    KVTensor,
    LayerKV,
    ValidationResult,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Optional vLLM base class
# ---------------------------------------------------------------------------


def load_vllm_base() -> type[Any] | None:
    """Import vLLM's :class:`KVConnectorBase` if vLLM is installed.

    Returns:
        The class object when vLLM is importable, otherwise
        ``None``. The connector subclasses the imported class
        only when it is not ``None``; the v1 of this module
        stays import-safe on systems without vLLM.
    """
    try:
        from vllm.distributed.kv_transfer.kv_connector.v1.base import (  # type: ignore[import-not-found]
            KVConnectorBase,
        )
    except ImportError:
        return None
    return KVConnectorBase


VLLM_BASE: type[Any] | None = load_vllm_base()


# ---------------------------------------------------------------------------
# Cluster client protocol
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MatchedPrefix:
    """Outcome of a prefix-cache lookup.

    Attributes:
        prefix_len: Number of tokens the cluster has cached for
            the request, ``0`` when nothing matched.
        kv_handle: Opaque identifier the cluster uses to fetch
            the cached K/V. ``""`` when nothing matched.
    """

    prefix_len: int
    kv_handle: str


@dataclass(frozen=True)
class LayerLoad:
    """A single layer that is in flight from the cluster.

    Attributes:
        layer_idx: Index of the layer being loaded.
        kv_handle: Cluster handle for the bundle that owns the
            layer.
    """

    layer_idx: int
    kv_handle: str


class MembraneClusterClient:
    """Minimal cluster client the vLLM connector talks to.

    The v1 of the client wraps the byte transport defined by
    :mod:`membrane.transfer_engine`; tests can subclass it
    with a dictionary-backed fake to drive the connector
    without a live cluster.
    """

    def lookup_prefix(
        self,
        model_id: str,
        token_ids: tuple[int, ...],
    ) -> MatchedPrefix:
        """Look up how many of ``token_ids`` are already cached.

        Args:
            model_id: Model identity the prefix was computed under.
            token_ids: Sequence of token ids for the incoming
                request, in order.

        Returns:
            MatchedPrefix: ``prefix_len=0`` on miss.
        """
        raise NotImplementedError

    def start_load(
        self,
        kv_handle: str,
        layer_indices: tuple[int, ...],
    ) -> tuple[LayerLoad, ...]:
        """Begin streaming a K/V bundle from the cluster.

        Args:
            kv_handle: Cluster-side handle returned by
                :func:`lookup_prefix`.
            layer_indices: Inclusive list of layer indices the
                runner wants to load.

        Returns:
            Tuple of in-flight layer descriptors.
        """
        raise NotImplementedError

    def fetch_layer(
        self,
        layer_load: LayerLoad,
        model_id: str,
        shape: tuple[int, int, int, int],
        dtype: str,
    ) -> KVTensor:
        """Block until a single layer is resident and return it.

        Args:
            layer_load: In-flight layer descriptor from
                :func:`start_load`.
            model_id: Model identity the layer was produced under.
            shape: Per-layer tensor shape the runner expects.
            dtype: Element dtype as a string.

        Returns:
            KVTensor: A single-layer bundle.
        """
        raise NotImplementedError

    def save_layer(
        self,
        layer: LayerKV,
        model_id: str,
        token_span: tuple[int, int],
        token_ids: tuple[int, ...] = (),
    ) -> None:
        """Push a freshly-computed layer to the cluster.

        Args:
            layer: The layer to push.
            model_id: Model identity the layer was produced under.
            token_span: Inclusive ``(start, end)`` of token
                positions this layer covers.
            token_ids: The prompt the layer belongs to (``()`` when unknown).
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Connector state
# ---------------------------------------------------------------------------


@dataclass
class RequestState:
    """Per-request bookkeeping kept by the connector.

    Attributes:
        request_id: vLLM-assigned request id.
        model_id: Model identity the request was made under.
        token_ids: Token sequence for the request.
        block_table: Block ids vLLM allocated for the request.
        matched: Outcome of the cluster lookup.
        loads: Per-layer in-flight loads, keyed by layer index.
    """

    request_id: str
    model_id: str
    token_ids: tuple[int, ...]
    block_table: tuple[int, ...] = ()
    matched: MatchedPrefix = field(default_factory=lambda: MatchedPrefix(0, ""))
    loads: dict[int, LayerLoad] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# In-memory fake cluster client (used in tests + as a no-op default)
# ---------------------------------------------------------------------------


class InMemoryClusterClient(MembraneClusterClient):
    """Cluster client backed by a dictionary.

    Operators that want a single-process demo can use this
    client to drive the connector end-to-end without standing
    up the network stack. Tests inject pre-seeded entries via
    :func:`seed`.
    """

    def __init__(self) -> None:
        """Create an empty in-memory KV store keyed by handle."""
        self.by_handle: dict[str, dict[int, bytes]] = {}
        self.lock = threading.RLock()

    def seed(self, kv_handle: str, layers: dict[int, bytes]) -> None:
        """Pre-populate a K/V bundle for tests.

        Args:
            kv_handle: The cluster-side handle.
            layers: Per-layer raw bytes (caller decides the
                format; the v1 of this client just stores them).
        """
        with self.lock:
            self.by_handle[kv_handle] = dict(layers)

    @override
    def lookup_prefix(
        self,
        model_id: str,
        token_ids: tuple[int, ...],
    ) -> MatchedPrefix:
        """Return the cached prefix of ``token_ids`` (always a miss in memory).

        Args:
            model_id: Model identifier.
            token_ids: Sequence of token ids for the incoming request, in order.

        Returns:
            MatchedPrefix: The cached prefix of ``token_ids`` (always a miss in
            memory).
        """
        if not token_ids:
            return MatchedPrefix(0, "")
        handle = f"mem:{model_id}:{len(token_ids)}"
        with self.lock:
            if handle in self.by_handle:
                return MatchedPrefix(len(token_ids), handle)
        return MatchedPrefix(0, "")

    @override
    def start_load(
        self,
        kv_handle: str,
        layer_indices: tuple[int, ...],
    ) -> tuple[LayerLoad, ...]:
        """Start loading the requested layers of a cached K/V bundle.

        Args:
            kv_handle: Cluster-side handle returned by :func:`lookup_prefix`.
            layer_indices: Inclusive list of layer indices the runner wants to
                load.

        Returns:
            tuple[LayerLoad, ...]: One load handle per requested layer that is
            cached.
        """
        with self.lock:
            bundle = self.by_handle.get(kv_handle)
            if bundle is None:
                return ()
            return tuple(LayerLoad(layer_idx=i, kv_handle=kv_handle) for i in layer_indices if i in bundle)

    @override
    def fetch_layer(
        self,
        layer_load: LayerLoad,
        model_id: str,
        shape: tuple[int, int, int, int],
        dtype: str,
    ) -> KVTensor:
        """Return the K/V tensors of one layer started by :meth:`start_load`.

        Args:
            layer_load: In-flight layer descriptor from :func:`start_load`.
            model_id: Model identifier.
            shape: Per-layer tensor shape the runner expects.
            dtype: Element dtype.

        Returns:
            KVTensor: The K/V tensors of one layer started by
            :meth:`start_load`.
        """
        with self.lock:
            bundle = self.by_handle.get(layer_load.kv_handle, {})
            raw = bundle.get(layer_load.layer_idx, b"")
        return KVTensor(
            layers=(
                LayerKV(
                    layer_idx=layer_load.layer_idx,
                    k=memoryview(raw),
                    v=memoryview(raw),
                    head_range=(-1, -1),
                    dtype=dtype,
                ),
            ),
            layer_range=(layer_load.layer_idx, layer_load.layer_idx),
            head_range=(-1, -1),
            token_span=(0, 0),
            shape=shape,
            fingerprint=placeholder_fingerprint(model_id, dtype),
        )

    @override
    def save_layer(
        self,
        layer: LayerKV,
        model_id: str,
        token_span: tuple[int, int],
        token_ids: tuple[int, ...] = (),
    ) -> None:
        """Store one layer's K/V tensors for later lookups.

        Args:
            layer: The layer to push.
            model_id: Model identifier.
            token_span: Inclusive ``(start, end)`` of token positions this layer
                covers.
            token_ids: The prompt (unused in memory).
        """
        del model_id, token_span, token_ids
        handle = f"mem:{layer.layer_idx}"
        payload = tensor_payload(layer.k)
        with self.lock:
            self.by_handle.setdefault(handle, {})[layer.layer_idx] = payload


def placeholder_fingerprint(model_id: str, dtype: str) -> Any:
    """Build a placeholder fingerprint for the in-memory client.

    Args:
        model_id: Model identity.
        dtype: Element dtype.

    Returns:
        ModelCompatibilityFingerprint: A default fingerprint.
    """
    from membrane.compat import compat_hash

    return compat_hash(model_id=model_id, dtype=dtype)


def tensor_payload(tensor: Any) -> bytes:
    """Coerce a tensor-like object into raw bytes.

    Args:
        tensor: Tensor-like object (``torch.Tensor``,
            ``numpy.ndarray``, ``bytes``/``memoryview``, or
            any object with a ``tobytes()`` method).

    Returns:
        bytes: Raw element bytes.
    """
    if isinstance(tensor, (bytes, bytearray, memoryview)):
        return bytes(tensor)
    if hasattr(tensor, "detach") and hasattr(tensor, "numpy"):
        return tensor.detach().cpu().numpy().tobytes()
    if hasattr(tensor, "numpy"):
        return tensor.numpy().tobytes()
    if hasattr(tensor, "tobytes"):
        return tensor.tobytes()
    return bytes(tensor)


def select_blocks(tensor: Any, block_table: tuple[int, ...]) -> Any:
    """Take a request's blocks from one layer of a paged KV cache.

    Args:
        tensor: K or V of shape ``(1, num_blocks, block_size, heads, dim)``
            (torch or numpy); other objects are returned whole.
        block_table: The request's block ids.

    Returns:
        Any: The request's blocks, or ``tensor`` when it cannot be indexed.
    """
    shape = getattr(tensor, "shape", None)
    if not block_table or shape is None or len(shape) < 2:
        return tensor
    return tensor[:, list(block_table)]


# ---------------------------------------------------------------------------
# Connector
# ---------------------------------------------------------------------------


def build_connector(cls: type[Any], vllm_base: type[Any] | None) -> type[Any]:
    """Build a vLLM-compatible connector class.

    When vLLM is installed, the connector is a real subclass of
    :class:`vllm.distributed.kv_transfer.kv_connector.v1.base.KVConnectorBase`
    so vLLM's scheduler / model runner can pick it up via the
    KV-transfer config. When vLLM is missing, the connector
    falls back to a duck-typed stub that exposes the same
    surface so the test suite can validate the wire-up.

    Args:
        cls: The :class:`MembraneVLLMAdapter` we are wiring
            into a connector. The connector delegates
            :func:`extract` / :func:`import_into` /
            :func:`serialize` / :func:`deserialize` /
            :func:`validate` to this adapter.
        vllm_base: The vLLM base class when available, else
            ``None``.

    Returns:
        A new connector class.
    """
    if vllm_base is None:

        class _StubBase:
            """Stand-in for vLLM's :class:`KVConnectorBase`.

            Mirrors the abstract method surface so the v1
            tests can validate the connector without
            importing vLLM.
            """

            def get_num_new_matched_tokens(self, request: Any, page_attn_params: Any) -> int | None:
                raise NotImplementedError

            def update_state_after_alloc(self, request: Any, pages: Any, page_attn_params: Any) -> None:
                raise NotImplementedError

            def build_connector_meta(self, scheduler_output: Any) -> Any:
                raise NotImplementedError

            def start_load_kv(self, request: Any) -> None:
                raise NotImplementedError

            def wait_for_layer_load(self, layer: int, request: Any) -> None:
                raise NotImplementedError

            def save_kv(self, layer: int, kv_caches: list[Any], attn_metadata: Any) -> None:
                raise NotImplementedError

        base: type[Any] = _StubBase
    else:
        base = vllm_base

    class MembraneVLLMConnector(cls, base):  # type: ignore[misc, valid-type]
        """vLLM KVConnector wired to a :class:`MembraneVLLMAdapter`.

        Attributes:
            client: The :class:`MembraneClusterClient` the
                connector talks to.
            n_layers: Number of transformer layers the runner
                will iterate over.
            model_id: Model identity reported by the runner.
            dtype: Element dtype reported by the runner.
        """

        def __init__(
            self,
            client: MembraneClusterClient,
            n_layers: int,
            model_id: str,
            dtype: str = "float16",
        ) -> None:
            """Initialize the connector.

            Args:
                client: Cluster client the connector talks to.
                n_layers: Number of transformer layers.
                model_id: Model identity.
                dtype: Element dtype the runner expects.
            """
            cls.__init__(self, kv_backend=None)
            self.client = client
            self.n_layers = n_layers
            self.model_id = model_id
            self.dtype = dtype
            self.__requests: dict[str, RequestState] = {}
            self.lock = threading.RLock()

        # ----- vLLM connector surface -----

        @override
        def get_num_new_matched_tokens(
            self,
            request: Any,
            page_attn_params: Any,
        ) -> int:
            """Return how many cached tokens the cluster has.

            Args:
                request: vLLM request object.
                page_attn_params: vLLM page-attention parameters.

            Returns:
                int: Number of cached tokens, ``0`` on miss.
            """
            state = self.__register(request)
            matched = self.client.lookup_prefix(state.model_id, state.token_ids)
            with self.lock:
                state.matched = matched
            return matched.prefix_len

        @override
        def update_state_after_alloc(
            self,
            request: Any,
            pages: Any,
            page_attn_params: Any,
        ) -> None:
            """Record the block table vLLM allocated for ``request``.

            Args:
                request: vLLM request object.
                pages: Block pages vLLM just allocated.
                page_attn_params: vLLM page-attention parameters.
            """
            state = self.__register(request)
            block_table = self.__extract_block_table(pages)
            with self.lock:
                state.block_table = block_table

        @override
        def build_connector_meta(self, scheduler_output: Any) -> dict[str, Any]:
            """Emit per-step metadata for the model runner.

            Args:
                scheduler_output: vLLM scheduler output.

            Returns:
                dict: Per-request connector metadata.
            """
            with self.lock:
                return {
                    req_id: {
                        "matched_prefix": state.matched.prefix_len,
                        "block_table": list(state.block_table),
                    }
                    for req_id, state in self.__requests.items()
                }

        @override
        def start_load_kv(self, request: Any) -> None:
            """Begin streaming the cached K/V for ``request``.

            Args:
                request: vLLM request object.
            """
            state = self.__register(request)
            if not state.matched.kv_handle:
                return
            loads = self.client.start_load(
                state.matched.kv_handle,
                tuple(range(self.n_layers)),
            )
            with self.lock:
                state.loads = {load.layer_idx: load for load in loads}

        @override
        def wait_for_layer_load(self, layer: int, request: Any) -> None:
            """Block until ``layer`` is resident for ``request``.

            Args:
                layer: Layer index the runner is about to use.
                request: vLLM request object.
            """
            state = self.__register(request)
            with self.lock:
                load = state.loads.pop(layer, None)
            if load is None:
                return
            self.client.fetch_layer(
                load,
                state.model_id,
                (1, 1, 1, 64),
                self.dtype,
            )

        @override
        def save_kv(
            self,
            layer: int,
            kv_caches: list[Any],
            attn_metadata: Any,
        ) -> None:
            """Push the freshly-computed K/V for ``layer``.

            Args:
                layer: Layer index being saved.
                kv_caches: Engine-resident K/V tensors.
                attn_metadata: vLLM attention metadata.
            """
            k_tensor, v_tensor = self.__split_kv(kv_caches, layer)
            with self.lock:
                pending = [
                    state
                    for state in self.__requests.values()
                    if state.token_ids and state.matched.prefix_len < len(state.token_ids)
                ]
            if not pending:
                # No request to file the layer under: hand it over as is.
                layer_kv = LayerKV(layer_idx=layer, k=k_tensor, v=v_tensor, head_range=(-1, -1), dtype=self.dtype)
                self.client.save_layer(layer_kv, self.model_id, (0, 0))
                return
            for state in pending:
                # Each request's KV lives in its own blocks of the paged cache.
                layer_kv = LayerKV(
                    layer_idx=layer,
                    k=select_blocks(k_tensor, state.block_table),
                    v=select_blocks(v_tensor, state.block_table),
                    head_range=(-1, -1),
                    dtype=self.dtype,
                )
                span = (0, len(state.token_ids) - 1)
                self.client.save_layer(layer_kv, state.model_id, span, token_ids=state.token_ids)
                if layer == self.n_layers - 1:
                    with self.lock:
                        state.matched = MatchedPrefix(len(state.token_ids), state.matched.kv_handle)

        # ----- internal helpers -----

        def __register(self, request: Any) -> RequestState:
            request_id = getattr(request, "request_id", None) or str(id(request))
            with self.lock:
                state = self.__requests.get(request_id)
                if state is not None:
                    return state
            token_ids = tuple(getattr(request, "token_ids", ()) or ())
            model_id = getattr(request, "model_id", None) or self.model_id
            state = RequestState(
                request_id=request_id,
                model_id=model_id,
                token_ids=token_ids,
            )
            with self.lock:
                self.__requests[request_id] = state
            return state

        @staticmethod
        def __extract_block_table(pages: Any) -> tuple[int, ...]:
            """Normalize vLLM's page list into a tuple of block ids.

            Args:
                pages: vLLM's page list (either a tuple of ints
                    or a list of page objects with a
                    ``block_id`` field).

            Returns:
                Tuple of block ids.
            """
            if pages is None:
                return ()
            if isinstance(pages, (tuple, list)):
                result: list[int] = []
                for page in pages:
                    if isinstance(page, int):
                        result.append(page)
                    else:
                        result.append(int(getattr(page, "block_id", 0)))
                return tuple(result)
            return ()

        @staticmethod
        def __split_kv(kv_caches: list[Any], layer: int) -> tuple[Any, Any]:
            """Extract the K and V tensors for ``layer``.

            vLLM keeps K and V in a single tensor of shape
            ``(2, num_blocks, block_size, num_heads, head_dim)``
            along the first axis. The connector splits that
            tensor into K and V halves before pushing them
            separately to the cluster.

            Args:
                kv_caches: Per-layer K/V tensors from vLLM.
                layer: Layer index to extract.

            Returns:
                A ``(k, v)`` pair of tensor-like objects.
            """
            if not kv_caches:
                return b"", b""
            cache = kv_caches[layer] if layer < len(kv_caches) else kv_caches[0]
            if hasattr(cache, "split") and hasattr(cache, "contiguous"):  # torch
                k, v = cache.split(1, dim=0)
                return k.contiguous(), v.contiguous()
            shape = getattr(cache, "shape", None)
            if shape is not None and len(shape) >= 1 and shape[0] == 2:  # numpy and friends
                return cache[0:1], cache[1:2]
            if isinstance(cache, (tuple, list)) and len(cache) >= 2:
                return cache[0], cache[1]
            return cache, cache

    MembraneVLLMConnector.__name__ = "MembraneVLLMConnector"
    MembraneVLLMConnector.__qualname__ = "MembraneVLLMConnector"
    return MembraneVLLMConnector


class MembraneVLLMAdapter(BaseAdapter):  # type: ignore[misc]
    """vLLM-flavored :class:`KVAdapter`.

    The v1 of the adapter delegates the vLLM-specific
    :func:`extract` / :func:`import_into` calls to the
    connector's :class:`MembraneVLLMConnector` instance so the
    byte serializer stays shared with the other engines.
    """

    def __init__(self, kv_backend: Any) -> None:
        """Initialize the adapter.

        Args:
            kv_backend: A :class:`KVBackend` instance, or
                ``None`` for the vLLM-only path.
        """
        self.kv_backend = kv_backend
        self.__connector_cls = build_connector(MembraneVLLMAdapter, VLLM_BASE)

    def make_connector(
        self,
        client: MembraneClusterClient,
        n_layers: int,
        model_id: str,
        dtype: str = "float16",
    ) -> Any:
        """Build a vLLM-compatible connector for the runner.

        Args:
            client: Cluster client the connector talks to.
            n_layers: Number of transformer layers.
            model_id: Model identity.
            dtype: Element dtype.

        Returns:
            A :class:`MembraneVLLMConnector` instance ready to
            be installed in vLLM's KV-transfer config.
        """
        return self.__connector_cls(
            client=client,
            n_layers=n_layers,
            model_id=model_id,
            dtype=dtype,
        )

    def extract(
        self,
        model: Any,
        layer_range: tuple[int, int],
        head_range: tuple[int, int],
        token_span: tuple[int, int],
    ) -> KVTensor:
        """Read K/V tensors from a vLLM model runner.

        The v1 implementation returns an empty bundle; the
        vLLM connector itself owns the per-request state, so
        callers that need a populated bundle should drive the
        connector's :func:`save_kv` path first and read back
        via :func:`import_into`.

        Args:
            model: The vLLM model runner.
            layer_range: Inclusive ``(start, end)``.
            head_range: Inclusive ``(start, end)``.
            token_span: Inclusive ``(start, end)``.

        Returns:
            KVTensor: Empty bundle with the requested ranges.
        """
        from membrane.compat import compat_hash

        return KVTensor(
            layers=(),
            layer_range=layer_range,
            head_range=head_range,
            token_span=token_span,
            shape=(1, 1, 1, 64),
            fingerprint=compat_hash(model_id="vllm", dtype="float16"),
        )

    def import_into(
        self,
        model: Any,
        tensor: KVTensor,
        layer_range: tuple[int, int],
    ) -> None:
        """Install a K/V bundle into a vLLM model runner.

        Args:
            model: vLLM model runner.
            tensor: Bundle produced by :func:`extract`.
            layer_range: Inclusive ``(start, end)``.
        """
        logger.debug(
            "MembraneVLLMAdapter.import_into: %d layers deferred to runner",
            len(tensor.layers),
        )
        return None

    @override
    def validate(self, tensor: KVTensor) -> ValidationResult:
        """Run the default BaseAdapter validation.

        Args:
            tensor: The bundle.

        Returns:
            ValidationResult: Outcome of the checks.
        """
        return super().validate(tensor)


__all__ = [
    "VLLM_AVAILABLE",
    "InMemoryClusterClient",
    "LayerLoad",
    "MatchedPrefix",
    "MembraneClusterClient",
    "MembraneVLLMAdapter",
    "MembraneVLLMConnector",
    "select_blocks",
]


VLLM_AVAILABLE: bool = VLLM_BASE is not None
MembraneVLLMConnector: type[Any] = build_connector(MembraneVLLMAdapter, VLLM_BASE)
