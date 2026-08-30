"""Prefill / decode disaggregation protocol (Phase 8).

The prefill service runs on a node optimized for the
compute-bound prefill phase; the decode service runs on a
node optimized for the memory-bound, low-latency decode
phase. The two services exchange K/V caches over a
:class:`membrane.transfer_engine.TransferEnvelope` instead
of the text stream, so the decode node picks up where the
prefill node left off without redoing the prefill.

The protocol is small on purpose: a :class:`PrefillRequest`
encodes a single prefill invocation; a :class:`PrefillResponse`
returns a stable :class:`KVHandle` that the decode node
uses to fetch the cached K/V. Both dataclasses are
JSON-serializable so the same types feed the REST and gRPC
surfaces.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any

from membrane.prefix_cache import KVHandle


@dataclass(frozen=True)
class PrefillRequest:
    """Request to run prefill on a tokenized prompt.

    Attributes:
        request_id: Client-side identifier echoed in the
            response.
        model_id: Model identity the prefill should run
            under. Used to gate the response with
            :class:`membrane.compat.MembraneValidator`.
        token_ids: Tokenized prompt.
        token_type_ids: Optional segment ids; ``None`` for
            single-segment prompts.
        max_decode_tokens: Maximum number of decode tokens
            to budget for the response.
        fingerprint: Pre-computed compatibility fingerprint
            for the model. ``""`` means "use the receiver's
            default".
    """

    request_id: str
    model_id: str
    token_ids: tuple[int, ...]
    token_type_ids: tuple[int, ...] | None = None
    max_decode_tokens: int = 256
    fingerprint: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dict.

        Returns:
            dict: ``token_ids`` and ``token_type_ids`` are
            serialized as lists.
        """
        payload = asdict(self)
        payload["token_ids"] = list(self.token_ids)
        if self.token_type_ids is not None:
            payload["token_type_ids"] = list(self.token_type_ids)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> PrefillRequest:
        """Deserialize from a JSON-safe dict.

        Args:
            payload: Dict produced by :func:`to_dict`.

        Returns:
            PrefillRequest: A new request.
        """
        return cls(
            request_id=payload["request_id"],
            model_id=payload["model_id"],
            token_ids=tuple(payload.get("token_ids", ())),
            token_type_ids=(
                tuple(payload["token_type_ids"])
                if payload.get("token_type_ids") is not None
                else None
            ),
            max_decode_tokens=int(payload.get("max_decode_tokens", 256)),
            fingerprint=payload.get("fingerprint", ""),
        )


@dataclass(frozen=True)
class PrefillResponse:
    """Response to a :class:`PrefillRequest`.

    Attributes:
        request_id: Echoed from the request.
        kv_handle: Stable :class:`KVHandle` hex digest the
            decode node uses to fetch the cached K/V.
        prefill_ms: Wall-clock prefill time, in milliseconds.
        prompt_len: Number of tokens in the original prompt.
        cached_prefix_len: Number of tokens served from the
            prefix cache, ``0`` on a cold prefill.
    """

    request_id: str
    kv_handle: str
    prefill_ms: float
    prompt_len: int
    cached_prefix_len: int

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dict.

        Returns:
            dict: Plain ``dict`` representation.
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> PrefillResponse:
        """Deserialize from a JSON-safe dict.

        Args:
            payload: Dict produced by :func:`to_dict`.

        Returns:
            PrefillResponse: A new response.
        """
        return cls(
            request_id=payload["request_id"],
            kv_handle=payload["kv_handle"],
            prefill_ms=float(payload["prefill_ms"]),
            prompt_len=int(payload["prompt_len"]),
            cached_prefix_len=int(payload.get("cached_prefix_len", 0)),
        )


@dataclass(frozen=True)
class DecodeRequest:
    """Request to continue generation from a prefill handle.

    Attributes:
        request_id: Client-side identifier echoed in the
            response.
        kv_handle: Hex digest of the prefill :class:`KVHandle`.
        model_id: Model identity to use for the decode pass.
        max_tokens: Maximum number of new tokens to emit.
    """

    request_id: str
    kv_handle: str
    model_id: str
    max_tokens: int = 256

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dict.

        Returns:
            dict: Plain ``dict`` representation.
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> DecodeRequest:
        """Deserialize from a JSON-safe dict.

        Args:
            payload: Dict produced by :func:`to_dict`.

        Returns:
            DecodeRequest: A new request.
        """
        return cls(
            request_id=payload["request_id"],
            kv_handle=payload["kv_handle"],
            model_id=payload["model_id"],
            max_tokens=int(payload.get("max_tokens", 256)),
        )


@dataclass(frozen=True)
class DecodeResponse:
    """Response to a :class:`DecodeRequest`.

    Attributes:
        request_id: Echoed from the request.
        token_ids: Tokens the decode node emitted in this
            step.
        finished: ``True`` when the decode pass hit a stop
            condition.
    """

    request_id: str
    token_ids: tuple[int, ...]
    finished: bool

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dict.

        Returns:
            dict: ``token_ids`` is a list.
        """
        payload = asdict(self)
        payload["token_ids"] = list(self.token_ids)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> DecodeResponse:
        """Deserialize from a JSON-safe dict.

        Args:
            payload: Dict produced by :func:`to_dict`.

        Returns:
            DecodeResponse: A new response.
        """
        return cls(
            request_id=payload["request_id"],
            token_ids=tuple(payload.get("token_ids", ())),
            finished=bool(payload.get("finished", False)),
        )


@dataclass
class _WallClock:
    """Helper that records elapsed milliseconds.

    Attributes:
        start: Monotonic start timestamp.
    """

    start: float = field(default_factory=time.monotonic)

    def elapsed_ms(self) -> float:
        """Return milliseconds elapsed since the start.

        Returns:
            float: Elapsed time in milliseconds.
        """
        return (time.monotonic() - self.start) * 1000.0


def make_handle_for(request: PrefillRequest) -> KVHandle:
    """Build the :class:`KVHandle` a :class:`PrefillResponse` will return.

    Args:
        request: The prefill request.

    Returns:
        KVHandle: A handle that fingerprints the prompt.
    """
    return KVHandle.for_tokens(request.model_id, request.token_ids)


__all__ = [
    "DecodeRequest",
    "DecodeResponse",
    "PrefillRequest",
    "PrefillResponse",
    "make_handle_for",
]
