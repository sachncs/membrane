"""Prefill / decode disaggregation routes, mounted under ``/disagg``."""

from fastapi import Depends, FastAPI, Request

from membrane.disagg.rest import create_router
from membrane.disagg.service import CALLER
from membrane.services.disagg import Disaggregation
from membrane.transport.context import app_context
from membrane.transport.routes.common import route_scope


async def disagg_caller(request: Request) -> None:
    """Authenticate a ``/disagg`` request and bind its caller for the services.

    Args:
        request: The inbound request.
    """
    context = route_scope(request, request.method, request.url.path)
    CALLER.set(context if app_context(request.app).authenticator is not None else None)


def mount_disagg(app: FastAPI, disagg: Disaggregation) -> None:
    """Mount the prefill / decode REST surface on ``app``.

    Args:
        app: The FastAPI application.
        disagg: The node's disaggregation services.
    """
    router = create_router(disagg.prefill_service, disagg.decode_service)
    app.include_router(router, prefix="/disagg", dependencies=[Depends(disagg_caller)])


__all__ = ["disagg_caller", "mount_disagg"]
