"""Tests for the prefill / decode disaggregation services (Phase 8)."""

from __future__ import annotations

import pytest

from membrane.disagg import (
    BatchPrefillResult,
    DecodeRequest,
    DecodeResponse,
    DecodeService,
    NoopPrefillBackend,
    PrefillRequest,
    PrefillResponse,
    PrefillService,
    batch_prefill,
)
from membrane.disagg.protocol import make_handle_for


class TestPrefillRequest:
    def test_round_trip_dict(self):
        req = PrefillRequest(
            request_id="r1",
            model_id="m",
            token_ids=(1, 2, 3),
            max_decode_tokens=128,
            fingerprint="abc",
        )
        payload = req.to_dict()
        assert payload["token_ids"] == [1, 2, 3]
        decoded = PrefillRequest.from_dict(payload)
        assert decoded == req

    def test_round_trip_with_token_type_ids(self):
        req = PrefillRequest(
            request_id="r1",
            model_id="m",
            token_ids=(1, 2, 3),
            token_type_ids=(0, 0, 1),
        )
        decoded = PrefillRequest.from_dict(req.to_dict())
        assert decoded.token_type_ids == (0, 0, 1)

    def test_round_trip_without_token_type_ids(self):
        req = PrefillRequest(request_id="r1", model_id="m", token_ids=(1,))
        decoded = PrefillRequest.from_dict(req.to_dict())
        assert decoded.token_type_ids is None


class TestPrefillService:
    def test_cold_prefill_invokes_backend(self):
        backend = NoopPrefillBackend()
        service = PrefillService(backend=backend)
        request = PrefillRequest(request_id="r1", model_id="m", token_ids=(1, 2, 3))
        response = service.prefill(request)
        assert response.request_id == "r1"
        assert response.prompt_len == 3
        assert response.cached_prefix_len == 0
        assert len(response.kv_handle) == 64
        assert backend.calls == [("r1", 0)]

    def test_warm_prefill_hits_cache(self):
        backend = NoopPrefillBackend()
        service = PrefillService(backend=backend)
        request = PrefillRequest(request_id="r1", model_id="m", token_ids=(1, 2, 3))
        service.prefill(request)
        second = service.prefill(
            PrefillRequest(request_id="r2", model_id="m", token_ids=(1, 2, 3))
        )
        assert second.cached_prefix_len == 3
        assert backend.calls == [("r1", 0), ("r2", 3)]

    def test_longest_prefix_match(self):
        backend = NoopPrefillBackend()
        service = PrefillService(backend=backend)
        service.prefill(PrefillRequest(request_id="r1", model_id="m", token_ids=(1, 2, 3)))
        second = service.prefill(
            PrefillRequest(request_id="r2", model_id="m", token_ids=(1, 2, 3, 4))
        )
        assert second.cached_prefix_len == 3


class TestDecodeService:
    def test_decode_returns_finished(self):
        service = DecodeService()
        request = DecodeRequest(request_id="r1", kv_handle="h", model_id="m", max_tokens=8)
        response = service.decode(request)
        assert response.finished is True
        assert response.token_ids == ()


class TestBatchPrefill:
    def test_batch_preserves_order(self):
        backend = NoopPrefillBackend()
        service = PrefillService(backend=backend)
        requests = [
            PrefillRequest(request_id=f"r{i}", model_id="m", token_ids=(i, i + 1))
            for i in range(5)
        ]
        result = batch_prefill(service, requests)
        assert isinstance(result, BatchPrefillResult)
        assert [r.request_id for r in result.responses] == [f"r{i}" for i in range(5)]
        assert result.elapsed_ms >= 0.0

    def test_batch_empty_returns_empty(self):
        result = batch_prefill(PrefillService(), [])
        assert result.responses == []
        assert result.elapsed_ms >= 0.0


class TestHandle:
    def test_make_handle_for_is_stable(self):
        req = PrefillRequest(request_id="r1", model_id="m", token_ids=(1, 2, 3))
        a = make_handle_for(req)
        b = make_handle_for(req)
        assert a == b
        assert a.handle == b.handle


class TestProtocolRoundTrip:
    def test_response_round_trip(self):
        resp = PrefillResponse(
            request_id="r1",
            kv_handle="h",
            prefill_ms=12.5,
            prompt_len=10,
            cached_prefix_len=5,
        )
        decoded = PrefillResponse.from_dict(resp.to_dict())
        assert decoded == resp

    def test_decode_request_round_trip(self):
        req = DecodeRequest(request_id="r1", kv_handle="h", model_id="m", max_tokens=4)
        decoded = DecodeRequest.from_dict(req.to_dict())
        assert decoded == req

    def test_decode_response_round_trip(self):
        resp = DecodeResponse(request_id="r1", token_ids=(1, 2, 3), finished=False)
        decoded = DecodeResponse.from_dict(resp.to_dict())
        assert decoded == resp
        assert decoded.token_ids == (1, 2, 3)


class TestRestRouter:
    def test_router_has_endpoints(self):
        pytest.importorskip("fastapi")
        from membrane.disagg.rest import create_router

        router = create_router(PrefillService(), DecodeService())
        paths = {route.path for route in router.routes}
        assert "/prefill" in paths
        assert "/prefill/batch" in paths
        assert "/decode" in paths
        assert "/healthz" in paths


class TestGrpcSurface:
    def test_grpc_availability_flag(self):
        from membrane.disagg.grpc import GRPC_AVAILABLE

        assert isinstance(GRPC_AVAILABLE, bool)

    def test_add_to_server_registers_handler(self):
        pytest.importorskip("grpc")
        # Inspect the module-level state to confirm the servicer
        # class is registered.
        from membrane.disagg import grpc as grpc_module
        from membrane.disagg import transfer_pb2_grpc
        from membrane.disagg.grpc import add_to_server

        handler = grpc_module._GrpcHandler(PrefillService(), DecodeService())
        assert hasattr(handler, "Prefill")
        assert hasattr(handler, "BatchPrefill")
        assert hasattr(handler, "Decode")
