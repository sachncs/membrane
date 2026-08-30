"""Stable fragment identity (fingerprint).

This module defines :class:`PayloadIdentity`, the single fingerprint used
everywhere a fragment's *position in the computation graph* and *byte
content* must be addressed together. It supersedes the older
:class:`membrane.signature.Signature`, which carried only three of the
ten fields and could not disambiguate, for example, two windows of the
same model occupying the same layer range but different head ranges or
different quantization dtypes.

The fingerprint is the composite key that:

*   Drives content addressing in :class:`membrane.persistence.ContentStore`.
*   Drives reuse matching in :class:`membrane.index.Index`.
*   Is serialized in the wire format (:mod:`membrane.serialization`)
    and in gRPC payloads (:mod:`membrane.transport.proto`).
*   Anchors the canonical bytes (:mod:`membrane.canonical`).

All ten fields are required. The dataclass is ``frozen=True`` so the
fingerprint is hashable and comparable; equality on all ten fields is
the operational definition of "these two payloads are interchangeable".
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any

_DTYPE_VALUES: frozenset[str] = frozenset({"float16", "bfloat16", "float32", "float64"})


@dataclass(frozen=True)
class PayloadIdentity:
    """Immutable fragment fingerprint.

    Ten required fields; the ``frozen=True`` dataclass guarantees
    hashability, structural equality, and immutability across threads
    and processes.

    Attributes:
        payload_hash: Lower-case SHA-256 hex digest of the canonical
            bytes for the fragment. Two fragments collide only if the
            raw bytes collide, even when every other field matches.
        model_id: Stable identifier of the model that produced (or can
            consume) the fragment, e.g. ``"llama-3-8b"`` or
            ``"mistral-7b-instruct"``. Two fragments are only
            interchangeable for inference when their ``model_id``
            matches exactly.
        model_revision: Pinning of the model weights (commit hash,
            ``""`` when unpinned). Used to refuse inference against a
            drifted weight set.
        tokenizer_name: Tokenizer identifier (often equal to
            ``model_id``). Required even when the tokenizer ships with
            the weights, because token-id sequences are only meaningful
            when paired with their encoder.
        tokenizer_revision: Pinning of the tokenizer assets.
        layer_range: Inclusive ``(start, end)`` range of transformer
            layer indices covered by the fragment. ``start == end``
            denotes a single layer; ``start < end`` denotes a span.
        head_range: Inclusive ``(start, end)`` range of attention head
            indices. ``(-1, -1)`` is the sentinel meaning "all heads"
            and only valid when the fragment stores the full tensor.
        token_span: Inclusive ``(start, end)`` range of token positions
            within the prompt the fragment corresponds to.
        dtype: Element dtype of the underlying tensor. One of
            ``"float16"``, ``"bfloat16"``, ``"float32"``, ``"float64"``.
        shape: Tensor shape as a tuple of integers. The first dimension
            is conventionally batch; the last is head_dim.
    """

    payload_hash: str
    model_id: str
    model_revision: str
    tokenizer_name: str
    tokenizer_revision: str
    layer_range: tuple[int, int]
    head_range: tuple[int, int]
    token_span: tuple[int, int]
    dtype: str
    shape: tuple[int, ...]

    def __post_init__(self) -> None:
        """Validate invariants.

        Raises:
            ValueError: On any malformed field.
        """
        if not isinstance(self.payload_hash, str) or not self.payload_hash:
            raise ValueError(
                f"payload_hash must be a non-empty string, got {self.payload_hash!r}"
            )
        if self.dtype not in _DTYPE_VALUES:
            raise ValueError(
                f"dtype must be one of {sorted(_DTYPE_VALUES)}, got {self.dtype!r}"
            )
        for name, value in (
            ("layer_range", self.layer_range),
            ("token_span", self.token_span),
        ):
            if not isinstance(value, tuple) or len(value) != 2:
                raise ValueError(f"{name} must be a 2-tuple, got {value!r}")
            start, end = value
            if start < 0 or end < 0:
                raise ValueError(f"{name} bounds must be non-negative, got {value}")
            if start > end:
                raise ValueError(
                    f"{name} start must be <= end, got start={start}, end={end}"
                )
        head = self.head_range
        if not isinstance(head, tuple) or len(head) != 2:
            raise ValueError(f"head_range must be a 2-tuple, got {head!r}")
        if head != (-1, -1):
            start, end = head
            if start < 0 or end < 0:
                raise ValueError(f"head_range bounds must be non-negative, got {head}")
            if start > end:
                raise ValueError(
                    f"head_range start must be <= end, got start={start}, end={end}"
                )
        if self.head_range == (-1, -1) and not self.shape:
            raise ValueError(
                "shape must be non-empty when head_range is the all-heads sentinel"
            )
        if not self.shape:
            raise ValueError("shape must be a non-empty tuple of ints")

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible ``dict``.

        Tuples are encoded as lists because JSON has no tuple type.
        Round-trips through :func:`from_dict` deterministically.

        Returns:
            dict[str, Any]: A plain ``dict`` ready for ``json.dumps``.
        """
        return {
            "payload_hash": self.payload_hash,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "tokenizer_name": self.tokenizer_name,
            "tokenizer_revision": self.tokenizer_revision,
            "layer_range": list(self.layer_range),
            "head_range": list(self.head_range),
            "token_span": list(self.token_span),
            "dtype": self.dtype,
            "shape": list(self.shape),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PayloadIdentity:
        """Reconstruct from the output of :meth:`to_dict`.

        Args:
            data: A ``dict`` previously produced by :meth:`to_dict` or
                received over the wire.

        Returns:
            PayloadIdentity: The reconstructed fingerprint.

        Raises:
            ValueError: On a missing or malformed field. The wire
                format guarantees field presence; rejection on missing
                fields avoids silent fallback to ``""`` defaults.
        """
        required = (
            "payload_hash",
            "model_id",
            "model_revision",
            "tokenizer_name",
            "tokenizer_revision",
            "layer_range",
            "head_range",
            "token_span",
            "dtype",
            "shape",
        )
        missing = [k for k in required if k not in data]
        if missing:
            raise ValueError(f"PayloadIdentity missing required fields: {missing}")
        layer_range_raw: tuple[int, ...] = tuple(int(x) for x in data["layer_range"])
        head_range_raw: tuple[int, ...] = tuple(int(x) for x in data["head_range"])
        token_span_raw: tuple[int, ...] = tuple(int(x) for x in data["token_span"])
        shape_raw: tuple[int, ...] = tuple(int(x) for x in data["shape"])
        return cls(
            payload_hash=str(data["payload_hash"]),
            model_id=str(data["model_id"]),
            model_revision=str(data["model_revision"]),
            tokenizer_name=str(data["tokenizer_name"]),
            tokenizer_revision=str(data["tokenizer_revision"]),
            layer_range=(layer_range_raw[0], layer_range_raw[1]),
            head_range=(head_range_raw[0], head_range_raw[1]),
            token_span=(token_span_raw[0], token_span_raw[1]),
            dtype=str(data["dtype"]),
            shape=shape_raw,
        )

    def fingerprint(self) -> str:
        """Stable, content-independent fingerprint of the identity.

        Useful as a stable hash key for the identity itself (e.g. when
        bucketing by model or layer) when the bytes are already pinned
        via :attr:`payload_hash`. Computed as
        ``sha256(canonical_json(asdict(self)))``.

        Returns:
            str: 64-character lowercase hex digest.
        """
        canonical = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


__all__ = ["PayloadIdentity"]
