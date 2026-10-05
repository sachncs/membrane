"""Request IDs: generated as UUIDv7, echoed, and stamped on logs."""

import json
import logging
import uuid

from fastapi.testclient import TestClient

from membrane.logging import JsonFormatter, RequestContextFilter, log_event, request_id
from membrane.node import Node
from membrane.transfer import TransferService
from membrane.transport.fastapi import create_app


def client() -> TestClient:
    app = create_app(
        node=Node("n1", max_memory_bytes=1000),
        compute_backend=None,
        transfer_service=TransferService(),
        cluster_manager=None,
    )
    return TestClient(app)


def test_generates_uuid7_request_id() -> None:
    rid = client().get("/livez").headers["x-request-id"]
    assert uuid.UUID(rid).version == 7


def test_reuses_well_formed_caller_id_and_replaces_bad_ones() -> None:
    c = client()
    assert c.get("/livez", headers={"X-Request-ID": "edge-42.a:b"}).headers["x-request-id"] == "edge-42.a:b"
    replaced = c.get("/livez", headers={"X-Request-ID": "bad id\n<script>"}).headers["x-request-id"]
    assert uuid.UUID(replaced).version == 7


def test_log_event_emits_template_fields_and_request_id() -> None:
    record_holder: list[logging.LogRecord] = []

    class Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            record_holder.append(record)

    logger = logging.getLogger("test.log_event")
    handler = Capture()
    handler.addFilter(RequestContextFilter())
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    token = request_id.set("req-1")
    try:
        content_hash, node_id = "ab12", "n1"
        log_event(logger, logging.INFO, t"stored {content_hash} on {node_id!r}")
    finally:
        request_id.reset(token)
        logger.removeHandler(handler)
    record = record_holder[0]
    assert record.getMessage() == "stored ab12 on 'n1'"
    payload = json.loads(JsonFormatter().format(record))
    assert payload["content_hash"] == "ab12"
    assert payload["node_id"] == "n1"
    assert payload["request_id"] == "req-1"
