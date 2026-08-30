"""Security test suite (Phase 3.1.8).

The v3.0.0 release wires auth checks into every route, adds
the SSRF allow-list, deny-by-default mTLS, per-endpoint input
limits, and per-tenant scope. This module is the
consolidated test surface for those guarantees.
"""

from __future__ import annotations

import pytest

from membrane.auth import (
    AuthBackendError,
    AuthContext,
)
from membrane.auth.apikey import APIKeyAuthenticator
from membrane.fragment import Fragment
from membrane.identity import PayloadIdentity
from membrane.security import (
    SSRFError,
    TenantAuthorizer,
    URLAllowlist,
    can_read_tenant,
    can_write_tenant,
    validate_outbound_url,
)

# ---------------------------------------------------------------------------
# Auth-bypass attempts
# ---------------------------------------------------------------------------


class TestAuthBypass:
    def test_no_authenticator_bypasses_scope_check(self):
        """When no authenticator is configured, the scope check passes by design.

        The scope enforcement is real only when authentication is
        configured. Single-node / test deployments without
        authentication are unaffected.
        """
        from membrane.transport.authz import enforce_route_scope

        ctx = enforce_route_scope(None, "POST", "/store")
        assert ctx.subject == ""

    def test_admin_route_rejects_non_admin(self):
        from membrane.transport.authz import enforce_route_scope

        auth = APIKeyAuthenticator(keyfile_text="u:user1:read\n")
        with pytest.raises(AuthBackendError, match="missing required scope"):
            enforce_route_scope(auth, "POST", "/delete", headers={"authorization": "Bearer u"})

    def test_write_route_rejects_read_only_key(self):
        from membrane.transport.authz import enforce_route_scope

        auth = APIKeyAuthenticator(keyfile_text="r:user1:read\n")
        with pytest.raises(AuthBackendError):
            enforce_route_scope(auth, "POST", "/store", headers={"authorization": "Bearer r"})

    def test_unknown_route_defaults_to_read(self):
        """A new route registered without an entry in ROUTE_SCOPES
        requires the read scope (fail-closed)."""
        from membrane.transport.authz import required_scope

        assert required_scope("GET", "/brand-new-route") == "read"

    def test_public_probe_bypasses(self):
        from membrane.transport.authz import enforce_route_scope, required_scope

        assert required_scope("GET", "/livez") == "public"
        ctx = enforce_route_scope(None, "GET", "/livez")
        assert ctx.subject == ""


# ---------------------------------------------------------------------------
# SSRF attempts
# ---------------------------------------------------------------------------


class TestSSRF:
    def test_file_scheme_rejected(self):
        with pytest.raises(SSRFError, match="scheme not allowed"):
            validate_outbound_url("file:///etc/passwd")

    def test_local_ipv4_rejected(self):
        with pytest.raises(SSRFError, match="blocked address"):
            validate_outbound_url("http://127.0.0.1/x")

    def test_loopback_ipv6_rejected(self):
        with pytest.raises(SSRFError, match="blocked address"):
            validate_outbound_url("http://[::1]/x")

    def test_rfc1918_rejected(self):
        for url in (
            "http://10.0.0.1/x",
            "http://192.168.1.1/x",
            "http://172.16.0.1/x",
        ):
            with pytest.raises(SSRFError, match="blocked address"):
                validate_outbound_url(url)

    def test_link_local_rejected(self):
        """Cloud metadata service at 169.254.169.254 is blocked."""
        with pytest.raises(SSRFError, match="blocked address"):
            validate_outbound_url("http://169.254.169.254/latest/meta-data/")

    def test_allowlisted_host_bypasses(self):
        policy = URLAllowlist(allowlist=frozenset({"internal.svc.cluster"}))
        url = validate_outbound_url("http://internal.svc.cluster/x", allowlist=policy)
        assert url == "http://internal.svc.cluster/x"


# ---------------------------------------------------------------------------
# mTLS deny-by-default
# ---------------------------------------------------------------------------


class TestMTLSDenyByDefault:
    def test_empty_allowed_cns_denies_everyone(self):
        from membrane.transport.tls import MTLSConfig, peer_cn_allowed

        cfg = MTLSConfig(server_cert_pem="X", server_key_pem="X", ca_bundle_pem="X")
        assert peer_cn_allowed(cfg, "anyone") is False
        assert peer_cn_allowed(cfg, "admin-1") is False

    def test_explicit_allowlist_admits_only_listed(self):
        from membrane.transport.tls import MTLSConfig, peer_cn_allowed

        cfg = MTLSConfig(
            server_cert_pem="X",
            server_key_pem="X",
            ca_bundle_pem="X",
            allowed_cns=frozenset({"node-1"}),
        )
        assert peer_cn_allowed(cfg, "node-1") is True
        assert peer_cn_allowed(cfg, "node-2") is False

    def test_allow_all_helper_accepts_everyone(self):
        from membrane.transport.tls import MTLSConfig, peer_cn_allowed

        cfg = MTLSConfig.allow_all_signed_by_ca(
            server_cert_pem="X",
            server_key_pem="X",
            ca_bundle_pem="X",
        )
        assert peer_cn_allowed(cfg, "anyone") is True


# ---------------------------------------------------------------------------
# Tenant isolation
# ---------------------------------------------------------------------------


class TestTenantIsolation:
    def test_read_other_tenant_returns_none(self):
        from membrane.node import Node

        node = Node(node_id="n1", max_memory_bytes=1024)
        frag = Fragment(
            identity=PayloadIdentity(
                payload_hash="h" * 64,
                model_id="m",
                model_revision="",
                tokenizer_name="m",
                tokenizer_revision="",
                layer_range=(0, 1),
                head_range=(-1, -1),
                token_span=(0, 7),
                dtype="float16",
                shape=(1, 1, 1, 8, 64),
            ),
            payload_ref=None,
            payload_size=10,
            ttl=60.0,
            reuse_score=0.5,
            version_id=1,
            tenant_id="acme",
        )
        node.store(frag)
        result = node.retrieve(
            frag.identity.payload_hash,
            caller_tenant="globex",
            caller_scopes=frozenset({"read"}),
        )
        assert result is None

    def test_admin_can_read_other_tenant(self):
        from membrane.node import Node

        node = Node(node_id="n1", max_memory_bytes=1024)
        frag = Fragment(
            identity=PayloadIdentity(
                payload_hash="h" * 64,
                model_id="m",
                model_revision="",
                tokenizer_name="m",
                tokenizer_revision="",
                layer_range=(0, 1),
                head_range=(-1, -1),
                token_span=(0, 7),
                dtype="float16",
                shape=(1, 1, 1, 8, 64),
            ),
            payload_ref=None,
            payload_size=10,
            ttl=60.0,
            reuse_score=0.5,
            version_id=1,
            tenant_id="acme",
        )
        node.store(frag)
        result = node.retrieve(
            frag.identity.payload_hash,
            caller_tenant="ops",
            caller_scopes=frozenset({"admin"}),
        )
        assert result is not None

    def test_write_other_tenant_rejected(self):
        from membrane.errors import TenantScopeError
        from membrane.node import Node

        node = Node(node_id="n1", max_memory_bytes=1024)
        frag = Fragment(
            identity=PayloadIdentity(
                payload_hash="h" * 64,
                model_id="m",
                model_revision="",
                tokenizer_name="m",
                tokenizer_revision="",
                layer_range=(0, 1),
                head_range=(-1, -1),
                token_span=(0, 7),
                dtype="float16",
                shape=(1, 1, 1, 8, 64),
            ),
            payload_ref=None,
            payload_size=10,
            ttl=60.0,
            reuse_score=0.5,
            version_id=1,
            tenant_id="acme",
        )
        with pytest.raises(TenantScopeError):
            node.store(
                frag,
                caller_tenant="globex",
                caller_scopes=frozenset({"write"}),
            )


# ---------------------------------------------------------------------------
# Schema strictness
# ---------------------------------------------------------------------------


class TestSchemaStrictness:
    """The v3.0.0 release hard-fails old schemas; there is no compat shim."""

    def test_v4_canonical_frame_rejected(self):
        from membrane.canonical import canonicalize, parse_canonical
        from membrane.errors import SchemaError

        ident = PayloadIdentity(
            payload_hash="a" * 64,
            model_id="m",
            model_revision="",
            tokenizer_name="m",
            tokenizer_revision="",
            layer_range=(0, 1),
            head_range=(-1, -1),
            token_span=(0, 7),
            dtype="float16",
            shape=(1, 1, 1, 8, 64),
        )
        buf = canonicalize(ident, b"hello")
        # Mutate the last byte of the magic from 0x05 to 0x04 (v4).
        bad = buf[:3] + b"\x04" + buf[4:]
        with pytest.raises(SchemaError, match="bad magic"):
            parse_canonical(bad)

    def test_v4_wire_dict_rejected(self):
        from membrane.errors import SchemaError
        from membrane.serialization import from_dict

        with pytest.raises(SchemaError, match="incompatible schema_version=4"):
            from_dict({"schema_version": 4, "identity": {}, "payload_ref": None, "payload_size": 0, "ttl": 0, "reuse_score": 0, "version_id": 1})

    def test_v3_wire_dict_rejected(self):
        from membrane.errors import SchemaError
        from membrane.serialization import from_dict

        with pytest.raises(SchemaError, match="incompatible schema_version=3"):
            from_dict({"schema_version": 3, "identity": {}, "payload_ref": None, "payload_size": 0, "ttl": 0, "reuse_score": 0, "version_id": 1})


# ---------------------------------------------------------------------------
# Malformed payloads
# ---------------------------------------------------------------------------


class TestMalformedPayloads:
    def test_pydantic_rejects_oversized_prompt(self):
        from pydantic import ValidationError

        from membrane.transport.routes_fastapi import PrefillRequest

        with pytest.raises(ValidationError):
            PrefillRequest(prompt_tokens=list(range(0, 40000)), model_id="m")

    def test_pydantic_rejects_oversized_tenant_id(self):
        from pydantic import ValidationError

        from membrane.transport.routes_fastapi import FragmentPayload

        with pytest.raises(ValidationError):
            FragmentPayload(
                schema_version=5,
                tenant_id="t" * 200,
                identity={"h": "h"},
                payload_size=0,
                ttl=60,
                reuse_score=0.5,
                version_id=1,
            )

    def test_pydantic_rejects_negative_payload_size(self):
        from pydantic import ValidationError

        from membrane.transport.routes_fastapi import FragmentPayload

        with pytest.raises(ValidationError):
            FragmentPayload(
                schema_version=5,
                identity={"h": "h"},
                payload_size=-1,
                ttl=60,
                reuse_score=0.5,
                version_id=1,
            )


# ---------------------------------------------------------------------------
# Resource exhaustion guards
# ---------------------------------------------------------------------------


class TestResourceExhaustion:
    def test_mt_root_url_empty(self):
        with pytest.raises(SSRFError):
            validate_outbound_url("")

    def test_tenant_id_path_traversal_rejected(self):
        from membrane.fragment import _validate_tenant_id

        with pytest.raises(ValueError, match="forbidden"):
            _validate_tenant_id("../acme")

    def test_tenant_id_slash_rejected(self):
        from membrane.fragment import _validate_tenant_id

        with pytest.raises(ValueError, match="forbidden"):
            _validate_tenant_id("acme/co")
