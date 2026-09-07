"""Tests for per-endpoint input-size limits (Phase 3.1.3).

The v2.0 release had a single 100 MiB body cap
(:data:`membrane.transport.ops.MAX_BODY_BYTES`) and unbounded
field lengths. The v3.0.0 release adds per-field limits via
Pydantic ``Field(max_length=...)`` so a hostile payload cannot
exhaust memory on a single field.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from membrane.transport.routes_fastapi import (
    DeleteRequest,
    FragmentPayload,
    GossipRequest,
    JoinRequest,
    LeaveRequest,
    PrefillRequest,
    PurgeRequest,
    ReplicateRequest,
    StoreRequest,
    SyncRequest,
    TombstoneRequest,
    VerifyRequest,
)


def _fragment(**overrides) -> dict:
    base = {
        "schema_version": 5,
        "tenant_id": "public",
        "identity": {"payload_hash": "h" * 4},
        "payload_ref": None,
        "payload_size": 0,
        "ttl": 60.0,
        "reuse_score": 0.5,
        "version_id": 1,
        "consistency": "strong",
        "hlc": 0,
        "fingerprint_compat": "",
    }
    base.update(overrides)
    return base


class TestFragmentPayloadLimits:
    def test_tenant_id_max_length(self):
        with pytest.raises(ValidationError):
            FragmentPayload(**_fragment(tenant_id="t" * 129))

    def test_payload_ref_max_length(self):
        with pytest.raises(ValidationError):
            FragmentPayload(**_fragment(payload_ref="x" * 513))

    def test_payload_size_non_negative(self):
        with pytest.raises(ValidationError):
            FragmentPayload(**_fragment(payload_size=-1))

    def test_payload_size_above_max(self):
        from membrane.transport.ops import MAX_BODY_BYTES

        with pytest.raises(ValidationError):
            FragmentPayload(**_fragment(payload_size=MAX_BODY_BYTES + 1))

    def test_fingerprint_compat_max_length(self):
        with pytest.raises(ValidationError):
            FragmentPayload(**_fragment(fingerprint_compat="f" * 129))

    def test_valid_payload(self):
        f = FragmentPayload(**_fragment())
        assert f.tenant_id == "public"
        assert f.schema_version == 5


class TestStoreRequest:
    def test_valid(self):
        StoreRequest(fragment=FragmentPayload(**_fragment()), is_primary=True)


class TestReplicateRequest:
    def test_valid(self):
        ReplicateRequest(fragment=FragmentPayload(**_fragment()))


class TestPrefillRequest:
    def test_prompt_tokens_max_length(self):
        with pytest.raises(ValidationError):
            PrefillRequest(prompt_tokens=[1] * 32769, model_id="m")

    def test_prompt_tokens_empty(self):
        r = PrefillRequest(prompt_tokens=[], model_id="m")
        assert r.prompt_tokens == []

    def test_model_id_max_length(self):
        with pytest.raises(ValidationError):
            PrefillRequest(prompt_tokens=[1], model_id="x" * 257)


class TestSyncRequest:
    def test_url_max_length(self):
        with pytest.raises(ValidationError):
            SyncRequest(source_url="http://x.example/" + "a" * 2048)


class TestJoinRequest:
    def test_valid(self):
        r = JoinRequest(node_id="n1", host="192.168.1.1", port=8080)
        assert r.port == 8080

    def test_port_out_of_range(self):
        with pytest.raises(ValidationError):
            JoinRequest(node_id="n1", host="192.168.1.1", port=0)
        with pytest.raises(ValidationError):
            JoinRequest(node_id="n1", host="192.168.1.1", port=70000)

    def test_host_max_length(self):
        with pytest.raises(ValidationError):
            JoinRequest(node_id="n1", host="a" * 256, port=8080)


class TestLeaveRequest:
    def test_node_id_required(self):
        with pytest.raises(ValidationError):
            LeaveRequest(node_id="")
        with pytest.raises(ValidationError):
            LeaveRequest(node_id="n" * 129)


class TestGossipRequest:
    def test_peers_max_length(self):
        with pytest.raises(ValidationError):
            GossipRequest(peers=[{} for _ in range(4097)])

    def test_default_payload_is_empty(self):
        r = GossipRequest()
        assert r.peers == []
        assert r.fragment_locations == {}
        assert r.inventory_digest == {}


class TestAdminRequests:
    def test_delete_request(self):
        DeleteRequest(content_hash="h" * 64, node_id="n1")

    def test_tombstone_request(self):
        TombstoneRequest(content_hash="h" * 64, until=time.time() + 60, node_id="n1")

    def test_purge_request(self):
        PurgeRequest(content_hash="h" * 64)

    def test_verify_request_sha256_length(self):
        from membrane.transport.ops import MAX_BODY_BYTES

        # SHA-256 hex is exactly 64 chars; a 63- or 65-char hex is rejected.
        with pytest.raises(ValidationError):
            VerifyRequest(content_hash="h" * 64, claimed_size=0, claimed_sha256_hex="a" * 63)
        with pytest.raises(ValidationError):
            VerifyRequest(content_hash="h" * 64, claimed_size=0, claimed_sha256_hex="a" * 65)

    def test_verify_size_out_of_range(self):
        from membrane.transport.ops import MAX_BODY_BYTES

        with pytest.raises(ValidationError):
            VerifyRequest(
                content_hash="h" * 64,
                claimed_size=MAX_BODY_BYTES + 1,
                claimed_sha256_hex="a" * 64,
            )


import time
