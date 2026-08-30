"""Prefill / decode service (Phase 8).

The :class:`PrefillService` and :class:`DecodeService` are
the engine-agnostic building blocks the REST and gRPC
surfaces call. The v1 backs the prefill with a
:class:`membrane.prefix_cache.PrefixCache` so a repeat
prompt skips the heavy lift and returns immediately.

The :class:`PrefillService` does not own an actual model;
it expects a prefill backend supplied by the caller. The
backend protocol is intentionally small: the v1 only asks
the backend for the cached prefix length. A real backend
(a vLLM ModelRunner or an HF causal LM) plugs in here.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from membrane.disagg.protocol import (
    DecodeRequest,
    DecodeResponse,
    PrefillRequest,
    PrefillResponse,
    _WallClock,
)
from membrane.prefix_cache import KVHandle, PrefixCache, PrefixMatch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prefill backend
# ---------------------------------------------------------------------------


@runtime_checkable
class PrefillBackend(Protocol):
    """Engine-agnostic prefill backend.

    A real implementation reads the request's token_ids and
    runs prefill on the underlying model. The v1 of this
    module ships a no-op backend that the tests use to drive
    the wire format without standing up an actual model.
    """

    def run_prefill(
        self,
        request: PrefillRequest,
        cached_prefix_len: int,
    ) -> tuple[int, float]:
        """Run prefill on the request.

        Args:
            request: The prefill request.
            cached_prefix_len: Number of tokens the prefix
                cache already covered.

        Returns:
            Tuple of ``(prompt_len, prefill_ms)``.
        """
        ...


class NoopPrefillBackend:
    """Backend that records the call but does no compute.

    Tests use this backend to validate the wire format. A
    real deployment substitutes a vLLM, SGLang, or
    TensorRT-LLM adapter here.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    def run_prefill(
        self,
        request: PrefillRequest,
        cached_prefix_len: int,
    ) -> tuple[int, float]:
        """Record the call and return the prompt length.

        Args:
            request: The prefill request.
            cached_prefix_len: Tokens the cache already covered.

        Returns:
            Tuple of ``(prompt_len, prefill_ms)``.
        """
        self.calls.append((request.request_id, cached_prefix_len))
        return len(request.token_ids), 0.0


# ---------------------------------------------------------------------------
# Prefill service
# ---------------------------------------------------------------------------


class PrefillService:
    """Service that runs prefill and returns a :class:`PrefillResponse`.

    Attributes:
        cache: The :class:`PrefixCache` used to short-circuit
            repeat prompts.
        backend: The :class:`PrefillBackend` the service
            delegates prefill to.
        clock: Helper for elapsed-millisecond timing.
    """

    def __init__(
        self,
        cache: PrefixCache | None = None,
        backend: PrefillBackend | None = None,
    ) -> None:
        """Initialize the service.

        Args:
            cache: Optional prefix cache. Defaults to a new
                ``PrefixCache(capacity=4096)``.
            backend: Optional prefill backend. Defaults to
                :class:`NoopPrefillBackend`.
        """
        self.cache = cache or PrefixCache(capacity=4096)
        self.backend: PrefillBackend = backend or NoopPrefillBackend()
        self._lock = threading.RLock()

    def prefill(self, request: PrefillRequest) -> PrefillResponse:
        """Run prefill on ``request``.

        Args:
            request: The prefill request.

        Returns:
            PrefillResponse: A response carrying a stable
            :class:`KVHandle` for the cached K/V.
        """
        clock = _WallClock()
        match = self.cache.lookup(request.model_id, request.token_ids)
        cached_prefix_len = match.token_len
        with self._lock:
            handle = self.cache.insert(
                request.model_id,
                request.token_ids,
                layer_range=(0, 0),
            )
            prompt_len, backend_ms = self.backend.run_prefill(request, cached_prefix_len)
        total_ms = clock.elapsed_ms() + max(backend_ms, 0.0)
        return PrefillResponse(
            request_id=request.request_id,
            kv_handle=handle.handle,
            prefill_ms=total_ms,
            prompt_len=prompt_len,
            cached_prefix_len=cached_prefix_len,
        )


# ---------------------------------------------------------------------------
# Decode service
# ---------------------------------------------------------------------------


class DecodeService:
    """Service that continues generation from a prefill handle.

    The v1 implementation does not own a real model. It
    emits an empty token stream and reports ``finished=True``
    so callers can wire up the protocol and exercise the
    decode path. Operators substitute a real backend via
    :class:`DecodeBackend`.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()

    def decode(self, request: DecodeRequest) -> DecodeResponse:
        """Continue generation for ``request``.

        Args:
            request: The decode request.

        Returns:
            DecodeResponse: An empty response with
            ``finished=True``.
        """
        with self._lock:
            return DecodeResponse(
                request_id=request.request_id,
                token_ids=(),
                finished=True,
            )


# ---------------------------------------------------------------------------
# Helper: batch prefill
# ---------------------------------------------------------------------------


@dataclass
class BatchPrefillResult:
    """Aggregate outcome of a batch prefill.

    Attributes:
        responses: Per-request :class:`PrefillResponse` in
            input order.
        elapsed_ms: Total wall-clock time, in milliseconds.
    """

    responses: list[PrefillResponse] = field(default_factory=list)
    elapsed_ms: float = 0.0


def batch_prefill(
    service: PrefillService,
    requests: list[PrefillRequest],
) -> BatchPrefillResult:
    """Run prefill on a list of requests in input order.

    Args:
        service: The prefill service.
        requests: List of prefill requests.

    Returns:
        BatchPrefillResult: Aggregate outcome.
    """
    clock = _WallClock()
    responses = [service.prefill(req) for req in requests]
    return BatchPrefillResult(responses=responses, elapsed_ms=clock.elapsed_ms())


__all__ = [
    "BatchPrefillResult",
    "DecodeService",
    "NoopPrefillBackend",
    "PrefillBackend",
    "PrefillService",
    "batch_prefill",
]


# Re-export KVHandle + PrefixMatch to keep imports short.
__all__ += ["KVHandle", "PrefixMatch"]
