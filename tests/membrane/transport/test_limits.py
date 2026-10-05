"""Backpressure (503), rate limiting (429), and API-docs exposure."""

import asyncio

from fastapi.testclient import TestClient

from membrane.auth.apikey import APIKeyAuthenticator
from membrane.node import Node
from membrane.server import Server
from membrane.transport.limits import ConcurrencyLimitMiddleware, RateLimitMiddleware, TransportLimits

KEYFILE = "reader-key:svc:read\nother-key:svc2:read\n"


def bearer(key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {key}"}


def server_with(limits: TransportLimits) -> Server:
    return Server(node=Node("limits-0"), port=0, authenticator=APIKeyAuthenticator(KEYFILE), limits=limits)


async def call(app, path: str = "/inventory") -> tuple[int, dict[str, str]]:
    """Drive an ASGI app once and return (status, headers)."""
    sent: list[dict] = []

    async def receive() -> dict:
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message: dict) -> None:
        sent.append(message)

    scope = {"type": "http", "path": path, "method": "GET", "headers": [], "client": ("10.0.0.1", 1)}
    await app(scope, receive, send)
    start = sent[0]
    return start["status"], {k.decode(): v.decode() for k, v in start.get("headers", [])}


async def slow_app(scope, receive, send) -> None:
    await asyncio.sleep(0.3)
    await send({"type": "http.response.start", "status": 200, "headers": []})
    await send({"type": "http.response.body", "body": b""})


def test_saturation_returns_503_with_retry_after() -> None:
    reasons: list[str] = []
    limited = ConcurrencyLimitMiddleware(slow_app, max_concurrency=1, queue_timeout_sec=0.05, on_reject=reasons.append)

    async def scenario():
        return await asyncio.gather(call(limited), call(limited), call(limited, "/readyz"))

    (first, _), (second, headers), (probe, _) = asyncio.run(scenario())
    assert first == 200
    assert second == 503
    assert headers["retry-after"] == "1"
    assert probe == 200  # probes bypass the bound
    assert reasons == ["overloaded"]


def test_rate_limit_is_per_credential() -> None:
    client = TestClient(server_with(TransportLimits(rate_limit_per_sec=0.5, rate_limit_burst=2)).transport.app)
    assert client.get("/inventory", headers=bearer("reader-key")).status_code == 200
    assert client.get("/inventory", headers=bearer("reader-key")).status_code == 200
    limited = client.get("/inventory", headers=bearer("reader-key"))
    assert limited.status_code == 429
    assert int(limited.headers["retry-after"]) >= 1
    assert "x-request-id" in limited.headers
    assert client.get("/inventory", headers=bearer("other-key")).status_code == 200
    assert client.get("/livez").status_code == 200


def test_rate_limit_bucket_refills() -> None:
    limiter = RateLimitMiddleware(slow_app, rate_per_sec=10.0, burst=1)
    assert limiter.take("k", 0.0) == 0
    assert limiter.take("k", 0.0) > 0
    assert limiter.take("k", 0.2) == 0


def test_rejections_are_counted() -> None:
    server = server_with(TransportLimits(rate_limit_per_sec=0.1, rate_limit_burst=1))
    client = TestClient(server.transport.app)
    client.get("/inventory", headers=bearer("reader-key"))
    client.get("/inventory", headers=bearer("reader-key"))
    assert server.metrics_transport.rejected.get(reason="rate_limited") == 1


def test_api_docs_are_off_by_default() -> None:
    client = TestClient(server_with(TransportLimits()).transport.app)
    for path in ("/docs", "/redoc", "/openapi.json"):
        assert client.get(path, headers=bearer("reader-key")).status_code == 404


def test_api_schema_requires_read_scope_when_enabled() -> None:
    client = TestClient(server_with(TransportLimits(enable_api_docs=True)).transport.app)
    assert client.get("/openapi.json").status_code == 401
    response = client.get("/openapi.json", headers=bearer("reader-key"))
    assert response.status_code == 200
    assert "/store" in response.json()["paths"]
