"""REST surface for the prefill / decode services (Phase 8).

The :func:`create_router` factory returns a FastAPI
:class:`APIRouter` with two endpoints:

* ``POST /prefill`` -- run prefill on a
  :class:`membrane.disagg.protocol.PrefillRequest`.
* ``POST /decode`` -- continue generation from a
  :class:`membrane.disagg.protocol.DecodeRequest`.
* ``GET /healthz`` -- liveness probe.

The router is a thin wrapper over the
:class:`membrane.disagg.service.PrefillService` and
:class:`DecodeService`; operators wire the services to a
real engine in production and call :func:`create_router`
on the resulting app.
"""

from __future__ import annotations

import logging
from typing import Any

from membrane.disagg.protocol import (
    DecodeRequest,
    DecodeResponse,
    PrefillRequest,
    PrefillResponse,
)
from membrane.disagg.service import (
    DecodeService,
    PrefillService,
    batch_prefill,
)

logger = logging.getLogger(__name__)


def create_router(
    prefill: PrefillService,
    decode: DecodeService | None = None,
) -> Any:
    """Build a FastAPI :class:`APIRouter` wired to the services.

    Args:
        prefill: The prefill service.
        decode: The decode service. Defaults to a new
            :class:`DecodeService`.

    Returns:
        APIRouter: A router ready to be mounted on a FastAPI
        app.
    """
    try:
        from fastapi import APIRouter, HTTPException
    except ImportError as exc:  # pragma: no cover - import guard
        raise RuntimeError("FastAPI is required for the REST surface") from exc
    decode_service = decode or DecodeService()

    router = APIRouter()

    @router.post("/prefill", response_model=None)
    async def post_prefill(payload: dict[str, Any]) -> dict[str, Any]:
        """Run prefill on a :class:`PrefillRequest`.

        Args:
            payload: JSON dict matching
                :class:`PrefillRequest.to_dict`.

        Returns:
            dict: JSON dict matching
            :class:`PrefillResponse.to_dict`.
        """
        try:
            request = PrefillRequest.from_dict(payload)
        except (KeyError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        response = prefill.prefill(request)
        return response.to_dict()

    @router.post("/prefill/batch", response_model=None)
    async def post_prefill_batch(payload: dict[str, Any]) -> dict[str, Any]:
        """Run prefill on a batch of requests.

        Args:
            payload: ``{"requests": [...]}`` list of
                :class:`PrefillRequest.to_dict` dicts.

        Returns:
            dict: ``{"responses": [...], "elapsed_ms": float}``.
        """
        raw = payload.get("requests", [])
        if not isinstance(raw, list):
            raise HTTPException(status_code=400, detail="requests must be a list")
        try:
            requests = [PrefillRequest.from_dict(item) for item in raw]
        except (KeyError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        result = batch_prefill(prefill, requests)
        return {
            "responses": [r.to_dict() for r in result.responses],
            "elapsed_ms": result.elapsed_ms,
        }

    @router.post("/decode", response_model=None)
    async def post_decode(payload: dict[str, Any]) -> dict[str, Any]:
        """Continue generation from a :class:`DecodeRequest`.

        Args:
            payload: JSON dict matching
                :class:`DecodeRequest.to_dict`.

        Returns:
            dict: JSON dict matching
            :class:`DecodeResponse.to_dict`.
        """
        try:
            request = DecodeRequest.from_dict(payload)
        except (KeyError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        response = decode_service.decode(request)
        return response.to_dict()

    @router.get("/healthz")
    async def get_healthz() -> dict[str, str]:
        """Liveness probe.

        Returns:
            dict: ``{"status": "ok"}``.
        """
        return {"status": "ok"}

    return router


__all__ = ["DecodeResponse", "DecodeService", "PrefillResponse", "PrefillService", "create_router"]
