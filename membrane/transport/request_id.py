"""Request IDs for every HTTP request.

:class:`RequestIdMiddleware` gives each request an ID, makes it the
current :data:`membrane.logging.request_id` (so every log line emitted
while handling the request carries it), and returns it in the
``X-Request-ID`` response header.

A caller-supplied ``X-Request-ID`` is reused when it is well formed
(at most 128 characters from ``[A-Za-z0-9._:-]``), so a proxy or client
can correlate its own logs; anything else is replaced. New IDs are
UUIDv7: unique and ordered by creation time.
"""

import re
import uuid
from collections.abc import Awaitable, Callable, MutableMapping
from typing import Any

from membrane.logging import request_id

type Scope = MutableMapping[str, Any]
type Message = MutableMapping[str, Any]
type Receive = Callable[[], Awaitable[Message]]
type Send = Callable[[Message], Awaitable[None]]
type ASGIApp = Callable[[Scope, Receive, Send], Awaitable[None]]

REQUEST_ID_HEADER = b"x-request-id"
VALID_REQUEST_ID = re.compile(r"[A-Za-z0-9._:-]{1,128}")


def choose_request_id(headers: list[tuple[bytes, bytes]]) -> str:
    """Return the caller's request ID when valid, else a new UUIDv7.

    Args:
        headers: Raw ASGI request headers.

    Returns:
        str: The request ID to use.
    """
    for name, value in headers:
        if name.lower() == REQUEST_ID_HEADER:
            candidate = value.decode("latin-1")
            if VALID_REQUEST_ID.fullmatch(candidate):
                return candidate
            break
    return str(uuid.uuid7())


class RequestIdMiddleware:
    """ASGI middleware that assigns and propagates request IDs."""

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
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        rid = choose_request_id(list(scope.get("headers") or []))
        token = request_id.set(rid)

        async def send_with_id(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = [h for h in message.get("headers", []) if h[0].lower() != REQUEST_ID_HEADER]
                headers.append((REQUEST_ID_HEADER, rid.encode("latin-1")))
                message["headers"] = headers
            await send(message)

        try:
            await self.app(scope, receive, send_with_id)
        finally:
            request_id.reset(token)


__all__ = ["REQUEST_ID_HEADER", "RequestIdMiddleware", "choose_request_id"]
