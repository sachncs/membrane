"""Tests for mTLS-gated cluster joins and CN-derived scopes.

Phase 1.6 covers:

* op_join authenticates via MTLSAuthenticator and admits /
  rejects based on the CN allow-list.
* CN-vs-node_id mismatch is rejected.
* op_join without an authenticator is the legacy path (single
  node).
* op_heartbeat stamps the verified CN onto the membership
  record via Membership.record_peer_cn.
* Membership's record_peer_cn is a no-op on unknown peer ids.
* Peer constructor carries local_peer_cn into the outbound
  heartbeat header.
* Snapshot round-trip preserves peer_cn across restarts.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from membrane.auth import AuthBackendError, AuthRequest
from membrane.auth.mtls import (
    CN_SCOPE_PREFIXES,
    MTLSAuthenticator,
    PeerIdentity,
    scopes_for_cn,
)
from membrane.network.membership import Membership, PeerInfo
from membrane.network.peer import Peer
from membrane.ring import Ring
from membrane.shard import Shard
from membrane.transport.ops import op_heartbeat, op_join
from membrane.transport.tls import (
    MTLSConfig,
    parse_peer_cn_header,
    peer_cn_allowed,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cfg() -> MTLSConfig:
    return MTLSConfig(
        server_cert_pem="CERT",
        server_key_pem="KEY",
        ca_bundle_pem="CA",
        # `node-1` carries no scope prefix on purpose so we can
        # exercise the "admitted by allow-list but no scope" path
        # without a separate config.
        allowed_cns=frozenset({"admin-1", "write-2", "read-3", "node-1"}),
    )


def _membership() -> Membership:
    return Membership("local", Ring(), Shard())


def _request(cn: str | None, **extra: str) -> AuthRequest:
    headers: dict[str, str] = {}
    if cn is not None:
        headers["x-ssl-client-cn"] = cn
    headers.update(extra)
    return AuthRequest(method="POST", path="/join", headers=headers)


# ---------------------------------------------------------------------------
# Scope derivation
# ---------------------------------------------------------------------------


class TestScopesForCN:
    def test_admin_grants_all_scopes(self):
        assert scopes_for_cn("admin-1") == frozenset({"read", "write", "admin"})

    def test_write_inherits_read(self):
        assert scopes_for_cn("write-2") == frozenset({"read", "write"})

    def test_read_gets_only_read(self):
        assert scopes_for_cn("read-3") == frozenset({"read"})

    def test_unknown_cn_gets_no_scope(self):
        assert scopes_for_cn("no-prefix-here") == frozenset()

    def test_empty_cn_gets_no_scope(self):
        assert scopes_for_cn("") == frozenset()

    def test_prefix_table_is_well_ordered(self):
        # admin before write so ``startswith`` matches the longest
        # prefix first.
        prefixes = [prefix for prefix, _ in CN_SCOPE_PREFIXES]
        assert prefixes.index("admin-") < prefixes.index("write-")
        assert prefixes.index("write-") < prefixes.index("read-")


# ---------------------------------------------------------------------------
# Peer-CN allow-list
# ---------------------------------------------------------------------------


class TestPeerCNAllowed:
    def test_default_allowed_cns_is_empty_set_means_deny_all(self):
        """Empty frozenset = deny every CN. Operators must opt in."""
        cfg = MTLSConfig(server_cert_pem="X", server_key_pem="X", ca_bundle_pem="X")
        assert peer_cn_allowed(cfg, "anyone") is False

    def test_allow_all_signed_by_ca_helper_accepts_everyone(self):
        """The dev-only helper bypasses the allowlist."""
        cfg = MTLSConfig.allow_all_signed_by_ca(server_cert_pem="X", server_key_pem="X", ca_bundle_pem="X")
        assert peer_cn_allowed(cfg, "anyone") is True

    def test_when_listed_passes(self):
        assert peer_cn_allowed(_cfg(), "admin-1") is True

    def test_when_not_listed_rejects(self):
        assert peer_cn_allowed(_cfg(), "admin-999") is False

    def test_parse_header(self):
        assert parse_peer_cn_header({"x-ssl-client-cn": "  write-2  "}) == "write-2"
        assert parse_peer_cn_header({"X-SSL-Client-CN": "read-3"}) == "read-3"
        assert parse_peer_cn_header({}) is None
        assert parse_peer_cn_header({"x-ssl-client-cn": ""}) is None
        assert parse_peer_cn_header({"x-ssl-client-cn": "   "}) is None

    def test_parse_case_insensitive(self):
        assert parse_peer_cn_header({"X-ssl-CLIENT-cn": "alpha"}) == "alpha"


# ---------------------------------------------------------------------------
# MTLSAuthenticator
# ---------------------------------------------------------------------------


class TestMTLSAuthenticator:
    def test_admits_listed_cn_with_scope(self):
        auth = MTLSAuthenticator(_cfg())
        ctx = auth.authenticate(_request("admin-1"))
        assert ctx.subject == "admin-1"
        assert "admin" in ctx.scopes

    def test_rejects_unlisted_cn(self):
        auth = MTLSAuthenticator(_cfg())
        with pytest.raises(AuthBackendError):
            auth.authenticate(_request("random-node"))

    def test_rejects_missing_cn_header(self):
        auth = MTLSAuthenticator(_cfg())
        with pytest.raises(AuthBackendError):
            auth.authenticate(_request(None))

    def test_rejects_cn_with_no_known_prefix(self):
        auth = MTLSAuthenticator(_cfg())
        # `node-1` is in the allow-list but does not match any scope
        # prefix, so the authenticator must reject it after the
        # allow-list check passes.
        with pytest.raises(AuthBackendError, match="granted scope"):
            auth.authenticate(_request("node-1"))

    def test_constructor_rejects_non_mtls_mode(self):
        cfg = MTLSConfig(
            server_cert_pem="X",
            server_key_pem="X",
            ca_bundle_pem="X",
            require_client_cert=False,
        )
        with pytest.raises(ValueError, match="require_client_cert"):
            MTLSAuthenticator(cfg)


# ---------------------------------------------------------------------------
# PeerInfo peer_cn
# ---------------------------------------------------------------------------


class TestPeerInfoCN:
    def test_default_cn_is_empty(self):
        pi = PeerInfo(node_id="n1", host="h", port=8081)
        assert pi.peer_cn == ""

    def test_to_json_includes_cn(self):
        pi = PeerInfo(node_id="n1", host="h", port=8081, peer_cn="admin-1")
        body = pi.to_json()
        assert body["peer_cn"] == "admin-1"

    def test_membership_add_records_cn(self):
        mem = _membership()
        mem.add("n2", "127.0.0.1", 8081, peer_cn="admin-1")
        assert mem.peers["n2"].peer_cn == "admin-1"

    def test_membership_add_update_preserves_cn(self):
        mem = _membership()
        mem.add("n2", "127.0.0.1", 8081, peer_cn="admin-1")
        mem.add("n2", "127.0.0.1", 8082, peer_cn="admin-9")
        # An update refreshes host/port/CN; the most recent value
        # wins, which is what a CN rotation should look like.
        assert mem.peers["n2"].peer_cn == "admin-9"
        assert mem.peers["n2"].port == 8082


# ---------------------------------------------------------------------------
# Membership.record_peer_cn
# ---------------------------------------------------------------------------


class TestRecordPeerCN:
    def test_stamps_when_known(self):
        mem = _membership()
        mem.add("n2", "127.0.0.1", 8081, peer_cn="")
        mem.record_peer_cn("n2", "admin-1")
        assert mem.peers["n2"].peer_cn == "admin-1"

    def test_clear_to_empty(self):
        mem = _membership()
        mem.add("n2", "127.0.0.1", 8081, peer_cn="admin-1")
        mem.record_peer_cn("n2", "")
        assert mem.peers["n2"].peer_cn == ""

    def test_unknown_peer_is_no_op(self):
        mem = _membership()
        # No add(); should silently do nothing.
        mem.record_peer_cn("never-seen", "admin-1")
        assert "never-seen" not in mem.peers


# ---------------------------------------------------------------------------
# op_join mTLS gating
# ---------------------------------------------------------------------------


class TestOpJoinMTLS:
    def test_no_authenticator_is_legacy_path(self):
        """Single-node deployments skip the authenticator entirely."""
        cluster = MagicMock()
        cluster.membership.add.return_value = None
        cluster.membership.to_json.return_value = []
        status, body = op_join(cluster, "n2", "127.0.0.1", 8081)
        assert status == 200
        assert body["success"] is True

    def test_authenticator_admits_listed_cn(self):
        cluster = MagicMock()
        cluster.membership.add.return_value = None
        cluster.membership.to_json.return_value = []
        auth = MTLSAuthenticator(_cfg())
        status, body = op_join(
            cluster, "admin-1", "127.0.0.1", 8081,
            headers={"x-ssl-client-cn": "admin-1"},
            authenticator=auth,
        )
        assert status == 200
        assert body["success"] is True
        # Recorded the verified CN.
        _, kwargs = cluster.membership.add.call_args
        assert kwargs["peer_cn"] == "admin-1"

    def test_rejects_unlisted_cn_with_401(self):
        cluster = MagicMock()
        auth = MTLSAuthenticator(_cfg())
        status, body = op_join(
            cluster, "n99", "127.0.0.1", 8081,
            headers={"x-ssl-client-cn": "random"},
            authenticator=auth,
        )
        assert status == 401
        assert "error" in body
        cluster.membership.add.assert_not_called()

    def test_cn_prefix_must_match_node_id(self):
        """An admin CN claiming a different node id is rejected."""
        cluster = MagicMock()
        auth = MTLSAuthenticator(_cfg())
        status, body = op_join(
            cluster, "n2", "127.0.0.1", 8081,
            headers={"x-ssl-client-cn": "admin-1"},
            authenticator=auth,
        )
        assert status == 401
        assert "CN does not match" in body["error"]
        cluster.membership.add.assert_not_called()

    def test_cn_match_passes(self):
        cluster = MagicMock()
        auth = MTLSAuthenticator(_cfg())
        status, _body = op_join(
            cluster, "admin-1", "127.0.0.1", 8081,
            headers={"x-ssl-client-cn": "admin-1"},
            authenticator=auth,
        )
        assert status == 200

    def test_authenticator_exception_propagates_when_status_mismatched(self):
        """Authenticator raising is wrapped as 401."""
        from membrane.auth import AuthBackendError

        class _BoomAuth:
            def authenticate(self, _request):
                raise AuthBackendError("go away")

        cluster = MagicMock()
        status, body = op_join(
            cluster, "n2", "127.0.0.1", 8081,
            headers={"x-ssl-client-cn": "admin-1"},
            authenticator=_BoomAuth(),
        )
        assert status == 401
        assert "go away" in body["error"]


# ---------------------------------------------------------------------------
# op_heartbeat CN stamping
# ---------------------------------------------------------------------------


class TestOpHeartbeatCN:
    def test_records_cn_when_cluster_supplied(self):
        node = MagicMock()
        node.node_id = "self"
        node.heartbeat.return_value = 0.1
        # Stub stats with attributes the op touches.
        stats = MagicMock()
        stats.memory_used_bytes = 0
        stats.memory_limit_bytes = 1
        stats.fragment_count = 0
        stats.primary_count = 0
        node.get_stats.return_value = stats

        cluster = MagicMock()
        # The op accesses cluster.membership.record_peer_cn, so
        # we need the magic attr path to resolve. ``MagicMock``
        # (without spec) auto-creates attributes on access.
        status, _body = op_heartbeat(
            node,
            cluster=cluster,
            headers={"X-Local-Peer-CN": "admin-1", "x-ssl-client-cn": "ignored"},
        )
        assert status == 200
        cluster.membership.record_peer_cn.assert_called_once_with(
            "self", "admin-1"
        )

    def test_skips_when_cluster_missing(self):
        node = MagicMock()
        node.node_id = "self"
        node.heartbeat.return_value = 0.1
        node.get_stats.return_value = MagicMock(
            memory_used_bytes=0,
            memory_limit_bytes=1,
            fragment_count=0,
            primary_count=0,
        )
        status, _body = op_heartbeat(node, cluster=None, headers={"X-Local-Peer-CN": "x"})
        assert status == 200


# ---------------------------------------------------------------------------
# Peer local_peer_cn propagation
# ---------------------------------------------------------------------------


class TestPeerLocalCNPropagation:
    def test_constructor_carrying_local_cn(self):
        peer = Peer(
            "http://peer:8080",
            local_peer_cn="admin-1",
        )
        assert peer.local_peer_cn == "admin-1"
        assert peer.base_headers["X-Local-Peer-CN"] == "admin-1"
        assert "User-Agent" in peer.base_headers

    def test_no_cn_when_omitted(self):
        peer = Peer("http://peer:8080")
        assert peer.local_peer_cn == ""
        assert "X-Local-Peer-CN" not in peer.base_headers


# ---------------------------------------------------------------------------
# PeerIdentity type (and module export)
# ---------------------------------------------------------------------------


class TestPeerIdentity:
    def test_frozen_dataclass(self):
        pi = PeerIdentity(cn="admin-1", scopes=frozenset({"admin"}), verified_at=0.0)
        assert pi.cn == "admin-1"
        assert pi.scopes == frozenset({"admin"})

    def test_frozen_rejects_mutation(self):
        pi = PeerIdentity(cn="admin-1", scopes=frozenset({"admin"}), verified_at=0.0)
        with pytest.raises((AttributeError, dataclasses.FrozenInstanceError)):
            pi.cn = "x"  # type: ignore[misc]


import dataclasses

# ---------------------------------------------------------------------------
# Snapshot round-trip preserves peer_cn
# ---------------------------------------------------------------------------


class TestSnapshotCNCarry:
    def test_round_trip_preserves_cn(self):
        # First peer: 1.1+ payload format with peer_cn. Second peer:
        # legacy 1.0.x payload missing the field. Both should load
        # with peer_cn set correctly.
        legacy_payload = [
            {
                "node_id": "old",
                "host": "127.0.0.1",
                "port": 9090,
                # No "peer_cn" key -- simulates a 1.0.x snapshot file.
            }
        ]
        modern_payload = [
            {
                "node_id": "new",
                "host": "127.0.0.1",
                "port": 9091,
                "peer_cn": "admin-7",
            }
        ]
        mem = _membership()
        mem.load_snapshot(legacy_payload + modern_payload)
        assert mem.peers["old"].peer_cn == ""
        assert mem.peers["new"].peer_cn == "admin-7"
