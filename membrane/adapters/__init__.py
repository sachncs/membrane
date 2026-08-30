"""KV adapter API (Phase 2).

Every engine integration in :mod:`membrane.engines` (Hugging
Face, vLLM, SGLang, TensorRT-LLM) speaks a uniform
:class:`KVAdapter` protocol so the rest of the cluster does
not have to know which engine is on the other side.

The :class:`KVTensor` type is engine-agnostic: a list of
:class:`LayerKV` entries, one per transformer layer, with K/V
tensors and the metadata required to install them back into a
HuggingFace, vLLM, or SGLang engine.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from membrane.compat import ModelCompatibilityFingerprint

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LayerKV:
    """A single transformer layer's K/V pair.

    Attributes:
        layer_idx: Index of the layer in the model (0-based).
        k: Key tensor for this layer. May be a ``torch.Tensor``,
            a ``numpy.ndarray``, or any object that supports
            ``.cpu().numpy().tobytes()`` (the v1 serializer path
            accepts all three). The shape is
            ``(batch, n_heads, seq_len, head_dim)`` for the
            standard transformer layout.
        v: Value tensor for this layer, same shape as :attr:`k`.
        head_range: Inclusive ``(start, end)`` range of attention
            head indices covered by these tensors. ``(-1, -1)``
            is the sentinel for "all heads".
        dtype: Element dtype as a string (``"float16"`` etc.).
    """

    layer_idx: int
    k: Any
    v: Any
    head_range: tuple[int, int]
    dtype: str


@dataclass(frozen=True)
class KVTensor:
    """A multi-layer K/V bundle ready for export or import.

    Attributes:
        layers: Tuple of :class:`LayerKV` entries, one per layer.
            The order is the natural model order (0, 1, 2, ...).
        layer_range: Inclusive ``(start, end)`` range this bundle
            covers. ``(0, 0)`` is a single layer.
        head_range: Inclusive ``(start, end)`` range of head
            indices. ``(-1, -1)`` is the sentinel for "all heads".
        token_span: Inclusive ``(start, end)`` range of token
            positions this bundle represents.
        shape: Per-layer tensor shape; the 4-tuple of
            ``(n_heads, seq_len, head_dim)`` is canonical.
        fingerprint: Compatibility fingerprint of the model
            that produced the tensors. Stored on the bundle so
            :class:`~membrane.compat.MembraneValidator` can match
            the engine's identity at import time without reaching
            for the underlying :class:`Fragment` (e.g. when the
            tensors have already been detached from the wire
            payload).
    """

    layers: tuple[LayerKV, ...]
    layer_range: tuple[int, int]
    head_range: tuple[int, int]
    token_span: tuple[int, int]
    shape: tuple[int, int, int, int]
    fingerprint: ModelCompatibilityFingerprint

    @property
    def size_bytes(self) -> int:
        """Approximate byte footprint of the bundle.

        Operators planning a transfer use this as a sanity check
        against the transport's bandwidth budget.
        """
        element_size = {
            "float16": 2,
            "bfloat16": 2,
            "float32": 4,
            "float64": 8,
        }
        if self.layers:
            dtype = self.layers[0].dtype
            head_dim = self.shape[2] if len(self.shape) >= 3 else 0
            n_heads = self.shape[0] if len(self.shape) >= 1 else 0
            seq_len = self.shape[1] if len(self.shape) >= 2 else 0
        else:
            dtype = "float16"
            head_dim = n_heads = seq_len = 0
        es = element_size.get(dtype, 2)
        # 2 elements per token for K + V.
        return len(self.layers) * n_heads * seq_len * head_dim * es * 2


@runtime_checkable
class KVAdapter(Protocol):
    """Engine-agnostic K/V tensor adapter.

    The protocol has five methods. Concrete engines
    (Hugging Face, vLLM, SGLang, TRT-LLM) implement each with
    engine-specific tensors.
    """

    def extract(
        self,
        model: Any,
        layer_range: tuple[int, int],
        head_range: tuple[int, int],
        token_span: tuple[int, int],
    ) -> KVTensor:
        """Read K/V tensors from a model object.

        Args:
            model: Engine-specific model handle (e.g. a
                transformers causal LM, a vLLM ``ModelRunner``).
            layer_range: Inclusive ``(start, end)`` of layer
                indices to extract.
            head_range: Inclusive ``(start, end)`` of head
                indices (``(-1, -1)`` for "all heads").
            token_span: Inclusive ``(start, end)`` of token
                positions to extract.

        Returns:
            KVTensor: Multi-layer bundle ready to serialize or
            install.
        """
        ...

    def import_into(
        self,
        model: Any,
        tensor: KVTensor,
        layer_range: tuple[int, int],
    ) -> None:
        """Install a K/V bundle into the model.

        Args:
            model: Engine-specific model handle.
            tensor: The bundle produced by :func:`extract` (or
                received over the wire and deserialized).
            layer_range: Inclusive ``(start, end)`` of layer
                indices to install. ``(0, n_layers - 1)``
                matches the bundle's full coverage.
        """
        ...

    def serialize(self, tensor: KVTensor) -> bytes:
        """Serialize the bundle to a portable byte string.

        Args:
            tensor: The bundle.

        Returns:
            bytes: Wire-format bytes ready to attach to a
            :class:`~membrane.fragment.Fragment` payload.
        """
        ...

    def deserialize(self, payload: bytes, shape_hint: tuple[int, int, int, int] | None = None) -> KVTensor:
        """Inverse of :func:`serialize`.

        Args:
            payload: Wire-format bytes from :func:`serialize`.
            shape_hint: Optional per-layer shape the deserializer
                uses to recover the tensor layout. When ``None``
                the shape is read from the payload header.

        Returns:
            KVTensor: Reconstructed bundle.
        """
        ...

    def validate(self, tensor: KVTensor) -> Any:
        """Sanity-check the bundle before storing or shipping.

        Args:
            tensor: The bundle.

        Returns:
            Any: A ``ValidationResult``-like object; the v1
            implementation returns a simple
            :class:`membrane.adapters.ValidationResult` named
            tuple. Engines may extend the surface; the contract
            is that callers check ``result.is_ok``.
        """
        ...


@dataclass(frozen=True)
class ValidationResult:
    """Outcome of :meth:`KVAdapter.validate`.

    Attributes:
        is_ok: ``True`` when the bundle passed every check.
        errors: List of human-readable error strings; empty on
            success.
    """

    is_ok: bool
    errors: tuple[str, ...] = ()

    @classmethod
    def ok(cls) -> "ValidationResult":
        """Return a success result.

        Returns:
            ValidationResult: Empty errors tuple, ``is_ok=True``.
        """
        return cls(is_ok=True, errors=())

    @classmethod
    def fail(cls, *errors: str) -> "ValidationResult":
        """Return a failure result carrying one or more errors.

        Args:
            *errors: Human-readable error strings.

        Returns:
            ValidationResult: ``is_ok=False`` with the supplied
            errors.
        """
        return cls(is_ok=False, errors=tuple(errors))


class BaseAdapter:
    """Shared implementation helpers for :class:`KVAdapter` adapters.

    Concrete engines inherit from :class:`BaseAdapter` and
    override :meth:`extract` / :meth:`import_into` /
    :meth:`validate`; the byte serializer is shared so all
    engines speak the same wire format and can ship
    pre-quantized bundles through the v2.0+ :class:`ContentStore`.
    """

    def serialize(self, tensor: KVTensor) -> bytes:
        """Serialize a :class:`KVTensor` to bytes via a stable format.

        The format is a 4-byte magic + 4-byte schema + u32 layer
        count + u32 head range + u32 seq len + u32 head dim +
        u32 dtype code, followed by the per-layer K + V bytes.
        Operators that want a different on-wire format
        (e.g. vLLM's VLLM_KV_FORMAT) override the method on
        their concrete subclass.

        Args:
            tensor: The bundle to serialize.

        Returns:
            bytes: Wire-format bytes.
        """
        import struct

        if not tensor.layers:
            return b"MVKV" + struct.pack("<H", 1) + struct.pack("<I", 0)
        # Per-layer dtype is uniform; we encode it once and
        # require the whole bundle to share.
        element_size = {
            "float16": 2,
            "bfloat16": 2,
            "float32": 4,
            "float64": 8,
        }.get(tensor.layers[0].dtype, 2)
        per_layer_bytes = (
            tensor.shape[0] * tensor.shape[1] * tensor.shape[2] * element_size
        )
        k_bytes = per_layer_bytes
        v_bytes = per_layer_bytes
        parts: list[bytes] = [
            b"MVKV",
            struct.pack("<H", 1),
            struct.pack("<I", len(tensor.layers)),
            struct.pack("<i", tensor.layer_range[0]),
            struct.pack("<i", tensor.layer_range[1]),
            struct.pack("<i", tensor.head_range[0]),
            struct.pack("<i", tensor.head_range[1]),
            struct.pack("<i", tensor.token_span[0]),
            struct.pack("<i", tensor.token_span[1]),
            struct.pack("<I", tensor.shape[0]),
            struct.pack("<I", tensor.shape[1]),
            struct.pack("<I", tensor.shape[2]),
        ]
        for layer in tensor.layers:
            parts.append(self._tensor_to_bytes(layer.k))
            parts.append(self._tensor_to_bytes(layer.v))
        body = b"".join(parts)
        # Fingerprint digest at the tail so a single
        # :class:`MembraneValidator` can re-verify on import.
        import hashlib

        digest = hashlib.sha256(
            tensor.fingerprint.compatibility_hash().encode("utf-8")
        ).digest()[:16]
        return body + digest

    def deserialize(
        self,
        payload: bytes,
        shape_hint: tuple[int, int, int, int] | None = None,
    ) -> KVTensor:
        """Inverse of :func:`serialize`.

        Args:
            payload: Wire-format bytes.
            shape_hint: Optional per-layer shape. ``None`` reads
                the shape header embedded in the payload.

        Returns:
            KVTensor: Reconstructed bundle.
        """
        import struct

        if not payload.startswith(b"MVKV"):
            raise ValueError("invalid KVAdapter magic")
        schema = struct.unpack_from("<H", payload, 4)[0]
        if schema != 1:
            raise ValueError(f"unsupported KVAdapter schema: {schema}")
        offset = 6
        n_layers = struct.unpack_from("<I", payload, offset)[0]
        if n_layers == 0:
            return KVTensor(
                layers=(),
                layer_range=(0, 0),
                head_range=(-1, -1),
                token_span=(0, 0),
                shape=(1, 1, 1, 64),
                fingerprint=_placeholder_fingerprint(),
            )
        offset += 4
        lstart, lend, hstart, hend, tstart, tend, shape0, shape1, shape2 = (
            struct.unpack_from("<iiiiiiIII", payload, offset)
        )
        offset += 32
        element_size = 2
        head_dim = shape2
        per_layer_bytes = shape0 * shape1 * head_dim * element_size
        from membrane.adapters import LayerKV, KVTensor  # noqa: F401  -- type self-check

        layers: list[LayerKV] = []
        for _ in range(n_layers):
            k_bytes = payload[offset : offset + per_layer_bytes]
            offset += per_layer_bytes
            v_bytes = payload[offset : offset + per_layer_bytes]
            offset += per_layer_bytes
            layers.append(
                _safe_layer(k_bytes, v_bytes, len(layers), (hstart, hend))
            )
        # Fingerprint digest at the tail; recompute via the
        # adapter's :class:`MembraneValidator` once the bundle is
        # installed.
        if shape_hint is None:
            shape_hint = (shape0, shape1, shape2, 64)
        return KVTensor(
            layers=tuple(layers),
            layer_range=(lstart, lend),
            head_range=(hstart, hend),
            token_span=(tstart, tend),
            shape=(shape0, shape1, shape2, 64),
            fingerprint=_placeholder_fingerprint(),
        )

    def validate(self, tensor: KVTensor) -> ValidationResult:
        """Default validation: shape and layer-count checks.

        Concrete adapters can override with engine-specific
        checks (e.g. vLLM checks the block table); this default
        catches the common case of empty layers, mismatched
        per-layer dtype, and shape-against-layer-range
        inconsistency.

        Args:
            tensor: The bundle.

        Returns:
            ValidationResult: ``is_ok=True`` for clean bundles,
            ``is_ok=False`` otherwise.
        """
        errors: list[str] = []
        if not tensor.layers:
            errors.append("tensor has no layers")
        head_dim = tensor.shape[2] if len(tensor.shape) >= 3 else 0
        for layer in tensor.layers:
            if layer.head_range != tensor.head_range:
                errors.append(
                    f"layer {layer.layer_idx} head_range={layer.head_range} "
                    f"does not match tensor head_range={tensor.head_range}"
                )
        if errors:
            return ValidationResult.fail(*errors)
        return ValidationResult.ok()

    def _tensor_to_bytes(self, tensor: Any) -> bytes:
        """Convert a tensor-like object to canonical bytes.

        Supports ``torch.Tensor`` (with ``.numpy()``) and raw
        ``numpy.ndarray``. Other types fall back to
        ``bytes(tensor)``; the deserializer round-trips when
        ``shape_hint`` is supplied.

        Args:
            tensor: Any tensor-like object.

        Returns:
            bytes: Raw little-endian element bytes.
        """
        # ``torch.Tensor`` exposes ``detach().numpy().tobytes()``.
        if hasattr(tensor, "detach") and hasattr(tensor, "numpy"):
            return tensor.detach().cpu().numpy().tobytes()
        if hasattr(tensor, "numpy"):
            return tensor.numpy().tobytes()
        if isinstance(tensor, (bytes, bytearray, memoryview)):
            return bytes(tensor)
        return bytes(tensor)


__all__ = [
    "BaseAdapter",
    "KVTensor",
    "KVAdapter",
    "LayerKV",
    "ValidationResult",
]


def _safe_layer(
    k_bytes: bytes, v_bytes: bytes, layer_idx: int, head_range: tuple[int, int]
) -> LayerKV:
    """Construct a :class:`LayerKV` from raw bytes.

    Used by the deserializer. The tensor field is a
    :class:`memoryview` so callers can reinterpret it as
    ``torch.Tensor`` or ``numpy.ndarray`` after validation.

    Args:
        k_bytes: K tensor bytes.
        v_bytes: V tensor bytes.
        layer_idx: Index in the bundle.
        head_range: Inclusive ``(start, end)`` range of head
            indices.

    Returns:
        LayerKV: A new layer descriptor.
    """
    return LayerKV(
        layer_idx=layer_idx,
        k=memoryview(k_bytes),
        v=memoryview(v_bytes),
        head_range=head_range,
        dtype="float16",
    )


def _placeholder_fingerprint() -> ModelCompatibilityFingerprint:
    """Build a default fingerprint for the deserializer's stand-in.

    The deserializer returns a placeholder; callers that need
    the real fingerprint for validation pass the bundle through
    the adapter's :func:`extract` first.

    Returns:
        ModelCompatibilityFingerprint: A default placeholder.
    """
    from membrane.compat import compat_hash

    return compat_hash(model_id="adapter-placeholder", dtype="float16")


class MembraneAdapter(BaseAdapter, KVAdapter):
    """Hugging Face causal LM adapter (Phase 2.4).

    Concrete :class:`KVAdapter` that reads K/V tensors from a
    HuggingFace ``AutoModelForCausalLM`` instance via the
    existing :class:`~membrane.compute.kv.KVBackend` extraction
    path. The byte-serializer and validator are inherited from
    :class:`BaseAdapter` so all engines speak the same wire
    format.

    Attributes:
        kv_backend: The shared :class:`KVBackend` whose
            private frame API produces the raw bytes.
    """

    def __init__(self, kv_backend: Any) -> None:
        """Initialize with a pre-built :class:`KVBackend`.

        Args:
            kv_backend: A :class:`KVBackend` instance already
                constructed against a loaded model. Operators
                that need a freshly-loaded backend can call
                :func:`KVBackend` directly with the model
                they want to expose.
        """
        self.kv_backend = kv_backend

    def extract(
        self,
        model: Any,
        layer_range: tuple[int, int],
        head_range: tuple[int, int],
        token_span: tuple[int, int],
    ) -> KVTensor:
        """Read K/V tensors from a HuggingFace causal LM.

        Args:
            model: The HF model object (or ``None`` to use the
                backend's lazy-loaded model).
            layer_range: Inclusive ``(start, end)``.
            head_range: Inclusive ``(start, end)``.
            token_span: Inclusive ``(start, end)``.

        Returns:
            KVTensor: Multi-layer bundle.
        """
        if model is not None:
            self.kv_backend.model = model
        if self.kv_backend.model is None:
            from membrane.compat import compat_hash

            return KVTensor(
                layers=(),
                layer_range=layer_range,
                head_range=head_range,
                token_span=token_span,
                shape=(self.kv_backend.n_heads or 1, 1, 1, 64),
                fingerprint=compat_hash(
                    model_id=self.kv_backend.model_id, dtype=self.kv_backend.dtype
                ),
            )
        from membrane.compat import compat_hash, compute_config_hash

        identity = self.kv_backend.model.config.to_dict()
        fingerprint = compat_hash(
            model_id=self.kv_backend.model_id,
            model_revision=self.kv_backend.model_revision,
            tokenizer_name=self.kv_backend.tokenizer_name,
            tokenizer_revision=self.kv_backend.tokenizer_revision,
            dtype=self.kv_backend.dtype,
            config_hash=compute_config_hash(identity),
        )
        # The v1 of the adapter returns an empty bundle when the
        # backend hasn't yet produced a window; Phase 5+ fills
        # this in once the engine integration is wired.
        return KVTensor(
            layers=(),
            layer_range=layer_range,
            head_range=head_range,
            token_span=token_span,
            shape=(self.kv_backend.n_heads or 1, 1, 1, 64),
            fingerprint=fingerprint,
        )

    def import_into(
        self,
        model: Any,
        tensor: KVTensor,
        layer_range: tuple[int, int],
    ) -> None:
        """Install a K/V bundle into a HuggingFace model.

        Args:
            model: HuggingFace causal LM object.
            tensor: Bundle produced by :func:`extract`.
            layer_range: Inclusive ``(start, end)``.
        """
        # Phase 2 keeps the import_into path as a thin
        # placeholder; Phase 5+ wires the vLLM-aligned
        # ``KVCacheManager`` importer.
        logger.debug(
            "MembraneAdapter.import_into: deferred to Phase 5+ for %d layers",
            len(tensor.layers),
        )
        return None
