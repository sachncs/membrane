"""A server span per HTTP request, continuing the caller's trace.

:class:`TracingMiddleware` is a pass-through while tracing is off. When
on, it extracts the W3C ``traceparent`` from the request (so a peer's
replication call joins the writer's trace), opens a ``SERVER`` span
named ``"<METHOD> <path>"``, and records the status code and the request
ID.
"""

from membrane.logging import request_id
from membrane.otel_tracer.otel import TRACING
from membrane.transport.request_id import ASGIApp, Message, Receive, Scope, Send


class TracingMiddleware:
    """ASGI middleware that opens one span per HTTP request."""

    def __init__(self, app: ASGIApp) -> None:
        """Wrap ``app``.

        Args:
            app: The downstream ASGI application.
        """
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Handle one ASGI connection scope.

        Args:
            scope: ASGI scope.
            receive: ASGI receive callable.
            send: ASGI send callable.
        """
        if scope["type"] != "http" or not TRACING.enabled:
            await self.app(scope, receive, send)
            return
        headers = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope.get("headers") or []}
        method = scope.get("method", "")
        path = scope.get("path", "")
        with TRACING.span(
            f"{method} {path}",
            kind="server",
            context=TRACING.extract(headers),
            attributes={"http.request.method": method, "url.path": path, "membrane.request_id": request_id.get()},
        ) as span:

            async def send_with_status(message: Message) -> None:
                if message["type"] == "http.response.start" and span is not None:
                    span.set_attribute("http.response.status_code", int(message["status"]))
                await send(message)

            await self.app(scope, receive, send_with_status)


__all__ = ["TracingMiddleware"]
