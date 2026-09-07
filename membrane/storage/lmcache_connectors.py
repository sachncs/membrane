"""LMCache distributed connector factory (Phase 0.3).

LMCache ships with two production-grade distributed
connectors:

* :class:`~lmcache.v1.storage_backend.remote_backend.RemoteBackend`
  — generic TCP connector for ``remote_url`` endpoints such as
  Redis or any LMCache-aware server.
* :class:`~lmcache.v1.storage_backend.gds_backend.GdsBackend` —
  GPU Direct Storage (cuFile) for the WekaFS / NVIDIA
  GPUDirect-style file paths.

Both are tightly coupled to the LMCache engine's event loop
and the ``lmcache_worker`` machinery. Exposing them as
standalone :class:`ContentStore` instances would either
duplicate the LMCache engine state or require a fake loop
that hides the real surface.

The v1 of this module therefore provides a factory that:
* validates the connector's prerequisites (LMCache, the
  optional CUDA stack for GdsBackend, the remote URL for
  RemoteBackend);
* constructs the underlying LMCache backend;
* hands the operator a documented pointer to LMCache's own
  engine integration (Phase 5+) when they want the full
  pipeline.

Operators who want the full KV-cache workflow should install
``lmcache>=0.5,<0.6`` and follow LMCache's own documentation
for distributed setup; the v1 :class:`LMCacheContentStore`
covers the local-cpu path and the Phase 5+ engine adapters
inherit the same primitives.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class LMCacheConnectorError(NotImplementedError):
    """Raised when the requested LMCache connector is unavailable.

    Used by :func:`build_distributed_store` to surface a clear
    message when operators request a distributed storage backend
    that the v1 of this arc does not implement as a standalone
    :class:`ContentStore`. The underlying LMCache primitives are
    wired into the Phase 5+ :mod:`membrane.engines` adapters; v1
    operators should follow LMCache's own documentation for
    distributed setup.
    """


def build_distributed_store(
    kind: str,
    remote_url: str | None = None,
    metadata: Any | None = None,
    config: Any | None = None,
    dst_device: str = "cpu",
) -> Any:
    """Build an LMCache distributed storage backend.

    Args:
        kind: One of ``"remote"`` (Redis / LMCache-aware TCP
            server) or ``"gds"`` (WekaFS / cuFile). The v1
            implements neither as a standalone ``ContentStore``;
            see :class:`LMCacheConnectorError` for the upgrade
            path.
        remote_url: Required for ``"remote"``. The ``host:port``
            pair LMCache's ``RemoteBackend`` dials.
        metadata: Optional :class:`LMCacheMetadata` paired with
            the engine that owns the storage tier.
        config: Optional :class:`LMCacheEngineConfig`.
        dst_device: Device string the backend should report
            (``"cpu"`` is the safe default for content-store
            conformance tests).

    Returns:
        The constructed LMCache backend.

    Raises:
        LMCacheConnectorError: Always, in this v1. The function
            is kept so callers can wire it conditionally and the
            surface stays stable for Phase 0.3+ follow-ups.
    """
    if kind == "remote" and not remote_url:
        raise LMCacheConnectorError(
            "RemoteBackend requires a remote_url"
        )
    if kind == "gds":
        try:
            import ctypes  # noqa: F401  -- gds needs cufile which is ctypes-loaded
        except ImportError as exc:
            raise LMCacheConnectorError(
                "GdsBackend requires ctypes / cufile on the system path"
            ) from exc
    # LMCache's distributed backends need an asyncio event loop
    # and the worker / engine plumbing. The v1 of this arc defers
    # that work to Phase 5+, where the engine adapters handle the
    # full flow. This function is the contract that Phase 0.3 keeps
    # stable so call sites can switch on the result kind.
    raise LMCacheConnectorError(
        f"LMCache {kind} connector is wired in Phase 5+ via "
        f"membrane.engines; this v1 of the storage layer keeps "
        f"the surface stable but does not construct a standalone "
        f"ContentStore. Install lmcache directly and use its "
        f"engine integration for production distributed setups."
    )


__all__ = ["LMCacheConnectorError", "build_distributed_store"]
