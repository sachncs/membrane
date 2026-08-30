"""Disaggregated prefill / decode services (Phase 8).

The ``membrane.disagg`` package is the v2.0+ implementation
of prefill / decode disaggregation. The package exposes:

* :mod:`membrane.disagg.protocol` -- the
  :class:`PrefillRequest` / :class:`PrefillResponse` /
  :class:`DecodeRequest` / :class:`DecodeResponse`
  dataclasses shared by the REST and gRPC surfaces.
* :mod:`membrane.disagg.service` -- the engine-agnostic
  :class:`PrefillService` / :class:`DecodeService` and the
  :class:`NoopPrefillBackend` test backend.
* :mod:`membrane.disagg.rest` -- the FastAPI router with
  ``/prefill``, ``/prefill/batch``, ``/decode``, and
  ``/healthz`` endpoints.
* :mod:`membrane.disagg.grpc` -- the gRPC servicer +
  :func:`add_to_server` / :func:`make_channel` /
  :func:`make_stub` factories.
"""

from __future__ import annotations

from membrane.disagg.protocol import (
    DecodeRequest,
    DecodeResponse,
    PrefillRequest,
    PrefillResponse,
)
from membrane.disagg.service import (
    BatchPrefillResult,
    DecodeService,
    NoopPrefillBackend,
    PrefillBackend,
    PrefillService,
    batch_prefill,
)

__all__ = [
    "BatchPrefillResult",
    "DecodeRequest",
    "DecodeResponse",
    "DecodeService",
    "NoopPrefillBackend",
    "PrefillBackend",
    "PrefillRequest",
    "PrefillResponse",
    "PrefillService",
    "batch_prefill",
]
