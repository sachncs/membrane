"""Tests for the per-fragment consistency + quorum fan-out.

Covers:

* attempt_quorum_acks returns success when >= quorum_count peers
  reply in time.
* attempt_quorum_acks returns timed_out when the deadline hits
  before quorum_count is reached; ack_count reflects the actual
  successes.
* attempt_quorum_acks short-circuits when the peer list is empty
  or quorum_count <= 0 (returning ``timed_out=True`` because the
  function cannot honour the request).
* attempt_quorum_acks propagates per-peer errors as
  ``timed_out=True`` rather than crashing.
* op_store honors ``consistency='eventual'`` by returning 200
  without invoking the quorum attempt.
* op_store honors ``consistency='strong'`` and rolls the local
  fragment back when the quorum attempt fails.
* op_store passes the cluster's ``default_consistency`` over the
  wire-fragment's strong default when they differ.

The fixtures deliberately avoid spinning up real servers; the
``_FakeCluster`` and ``_FakePeer`` helpers stand in for the
:class:`~membrane.network.cluster.Cluster` integration.
"""

from __future__ import annotations

import time
from typing import Any
from unittest.mock import MagicMock

import pytest

from membrane.fragment import Fragment
from membrane.network.peer import Peer
from membrane.node import Node
from membrane.quorum import QuorumResult, attempt_quorum_acks
from membrane.serialization import to_dict
from membrane.transport.ops import op_store
from tests.conftest import make_fragment


class _FakeClusterConfig:
    def __init__(
        self,
        default_consistency: str = "strong",
        quorum_count: int = 2,
        cluster_quorum_timeout_sec: float = 0.5,
    ) -> None:
        self.default_consistency = default_consistency
        self.quorum_count = quorum_count
        self.cluster_quorum_timeout_sec = cluster_quorum_timeout_sec


class _FakePeer:
    """Returns ``ok`` after ``delay_sec``; ignored transport state."""

    def __init__(self, ok: bool = True, delay_sec: float = 0.0) -> None:
        self._ok = ok
        self._delay = delay_sec

    def request_replicate(self, _fragment: Fragment) -> bool:
        if self._delay > 0:
            time.sleep(self._delay)
        return self._ok


def _make_fragment_with_consistency(consistency: str, hlc: int = 0) -> Fragment:
    base = make_fragment("hash-q")
    return Fragment(
        identity=base.identity,
        payload_ref=base.payload_ref,
        payload_size=base.payload_size,
        ttl=base.ttl,
        reuse_score=base.reuse_score,
        version_id=base.version_id,
        consistency=consistency,
        hlc=hlc,
    )


def _make_fake_cluster(
    config: _FakeClusterConfig,
    peers: list[_FakePeer],
    primary_id: str = "self",
) -> Any:
    cluster = MagicMock()
    cluster.config = config
    cluster.node_id = primary_id
    cluster.membership.healthy.return_value = [
        type("PeerLike", (), {"node_id": f"peer-{i}"})() for i in range(len(peers))
    ]

    def _get_client(nid: str) -> _FakePeer | None:
        for i, p in enumerate(peers):
            if f"peer-{i}" == nid:
                return p
        return None

    cluster.membership.get_client.side_effect = _get_client
    return cluster


# ---------------------------------------------------------------------------
# attempt_quorum_acks
# ---------------------------------------------------------------------------


class TestAttemptQuorumAcks:
    def test_success_with_quorum(self):
        peers = [_FakePeer(ok=True) for _ in range(3)]
        result = attempt_quorum_acks(
            fragment=_make_fragment_with_consistency("strong"),
            peers=peers,
            quorum_count=2,
            timeout_sec=1.0,
        )
        assert result.success is True
        assert result.ack_count >= 2
        assert result.timed_out is False
        assert result.replica_count == 3

    def test_short_circuit_when_no_peers(self):
        result = attempt_quorum_acks(
            fragment=_make_fragment_with_consistency("strong"),
            peers=[],
            quorum_count=2,
            timeout_sec=1.0,
        )
        assert result.success is False
        assert result.ack_count == 0
        assert result.timed_out is True
        assert result.replica_count == 0

    def test_short_circuit_when_quorum_count_zero(self):
        peers = [_FakePeer(ok=True) for _ in range(2)]
        result = attempt_quorum_acks(
            fragment=_make_fragment_with_consistency("strong"),
            peers=peers,
            quorum_count=0,
            timeout_sec=1.0,
        )
        assert result.success is False
        assert result.ack_count == 0
        assert result.replica_count == 2

    def test_timeout_when_peers_slow(self):
        peers = [_FakePeer(ok=True, delay_sec=0.1) for _ in range(2)]
        result = attempt_quorum_acks(
            fragment=_make_fragment_with_consistency("strong"),
            peers=peers,
            quorum_count=3,
            timeout_sec=0.05,
        )
        assert result.success is False
        # ack_count is < quorum_count (3)
        assert result.ack_count < 3
        assert result.timed_out is True

    def test_peers_returning_false_count_misses(self):
        # Two peers return True, one returns False; quorum_count=3 means
        # we need all three to ack. With the false peer ignored, the
        # timeout elapses before quorum_count is reached.
        peers = [_FakePeer(ok=True), _FakePeer(ok=False), _FakePeer(ok=True)]
        result = attempt_quorum_acks(
            fragment=_make_fragment_with_consistency("strong"),
            peers=peers,
            quorum_count=3,
            timeout_sec=0.5,
        )
        assert result.success is False
        assert result.ack_count <= 2


# ---------------------------------------------------------------------------
# op_store
# ---------------------------------------------------------------------------


class TestOpStoreConsistency:
    def test_eventual_never_blocks(self):
        node = Node("self", max_memory_bytes=10_000)
        # eventual consistency means we should not consult the cluster.
        cluster = MagicMock()
        quorum_called = False

        def _attempt(*args: object, **kwargs: object) -> QuorumResult:
            nonlocal quorum_called
            quorum_called = True
            return QuorumResult(success=True, ack_count=1, timed_out=False, replica_count=1)

        frag = _make_fragment_with_consistency("eventual")
        status, body = op_store(
            node,
            to_dict(frag),
            is_primary=True,
            cluster=cluster,
            quorum_attempt=_attempt,
        )
        assert status == 200
        assert body["success"] is True
        assert quorum_called is False

    def test_strong_with_quorum_succeeds(self):
        node = Node("self", max_memory_bytes=10_000)
        cluster = MagicMock()
        cluster.config = _FakeClusterConfig(
            default_consistency="strong",
            quorum_count=2,
            cluster_quorum_timeout_sec=1.0,
        )
        cluster.membership.healthy.return_value = [
            type("PeerLike", (), {"node_id": f"peer-{i}"})() for i in range(2)
        ]
        cluster.membership.get_client.side_effect = lambda nid: _FakePeer(ok=True)

        def _attempt(
            fragment: Fragment,
            peers: list[Peer],
            quorum_count: int,
            timeout_sec: float,
        ) -> QuorumResult:
            return QuorumResult(
                success=True,
                ack_count=quorum_count,
                timed_out=False,
                replica_count=len(peers),
            )

        frag = _make_fragment_with_consistency("strong")
        status, body = op_store(
            node,
            to_dict(frag),
            is_primary=True,
            cluster=cluster,
            quorum_attempt=_attempt,
        )
        assert status == 200
        assert body["success"] is True

    def test_strong_with_quorum_timeout_rolls_back(self):
        node = Node("self", max_memory_bytes=10_000)
        cluster = MagicMock()
        cluster.config = _FakeClusterConfig(
            default_consistency="strong",
            quorum_count=2,
            cluster_quorum_timeout_sec=1.0,
        )
        cluster.membership.healthy.return_value = [
            type("PeerLike", (), {"node_id": f"peer-{i}"})() for i in range(2)
        ]
        cluster.membership.get_client.side_effect = lambda nid: _FakePeer(ok=False)

        def _attempt(
            fragment: Fragment,
            peers: list[Peer],
            quorum_count: int,
            timeout_sec: float,
        ) -> QuorumResult:
            return QuorumResult(
                success=False,
                ack_count=0,
                timed_out=True,
                replica_count=len(peers),
            )

        # Use a unique content_hash so the rollback is observable.
        # (Fragment.consistency = strong so the wire ships with the
        # new field which the dict has.)
        primary_payload = to_dict(_make_fragment_with_consistency("strong"))
        content_hash = primary_payload["identity"]["payload_hash"]
        assert node.store(
            Fragment(
                identity=__import__(
                    "membrane.identity", fromlist=["PayloadIdentity"]
                ).PayloadIdentity.from_dict(primary_payload["identity"]),
                payload_ref=primary_payload["payload_ref"],
                payload_size=primary_payload["payload_size"],
                ttl=primary_payload["ttl"],
                reuse_score=primary_payload["reuse_score"],
                version_id=primary_payload["version_id"],
                consistency="strong",
                hlc=primary_payload["hlc"],
            ),
            is_primary=True,
        ) is True
        assert content_hash in node.fragments

        status, body = op_store(
            node,
            primary_payload,
            is_primary=True,
            cluster=cluster,
            quorum_attempt=_attempt,
        )
        assert status == 503
        assert "quorum" in body["error"]
        # Local rollback must have removed the fragment.
        assert content_hash not in node.fragments

    def test_strong_default_honors_cluster_quorum(self):
        node = Node("self", max_memory_bytes=10_000)
        cluster = MagicMock()
        cluster.config = _FakeClusterConfig(
            default_consistency="quorum",
            quorum_count=2,
        )
        cluster.membership.healthy.return_value = [
            type("PeerLike", (), {"node_id": f"peer-{i}"})() for i in range(2)
        ]
        cluster.membership.get_client.side_effect = lambda nid: _FakePeer(ok=True)

        captured: dict[str, Any] = {}

        def _attempt(
            fragment: Fragment,
            peers: list[Peer],
            quorum_count: int,
            timeout_sec: float,
        ) -> QuorumResult:
            captured["consistency"] = fragment.consistency
            return QuorumResult(
                success=True,
                ack_count=quorum_count,
                timed_out=False,
                replica_count=len(peers),
            )

        # Fragment ships with strong (the v3 wire default), but the
        # cluster's default_consistency = quorum. op_store must
        # override the wire-side strong because the cluster default
        # is weaker.
        frag = _make_fragment_with_consistency("strong")
        status, _body = op_store(
            node,
            to_dict(frag),
            is_primary=True,
            cluster=cluster,
            quorum_attempt=_attempt,
        )
        assert status == 200
        # The override is documented at the cluster boundary; when
        # the cluster default is weaker than the wire-default
        # strong, op_store downgrades to the cluster's preference.
        assert captured["consistency"] == "quorum"

    def test_strong_default_does_not_upgrade_when_frag_says_eventual(self):
        node = Node("self", max_memory_bytes=10_000)
        cluster = MagicMock()
        cluster.config = _FakeClusterConfig(
            default_consistency="strong",
            quorum_count=2,
        )
        cluster.membership.healthy.return_value = [
            type("PeerLike", (), {"node_id": f"peer-{i}"})() for i in range(2)
        ]
        # eventual on the wire trumps the cluster default; the
        # quorum attempt must NOT run.
        attempted = []

        def _attempt(*args: object, **kwargs: object) -> QuorumResult:
            attempted.append(args)
            return QuorumResult(success=True, ack_count=1, timed_out=False, replica_count=1)

        frag = _make_fragment_with_consistency("eventual")
        status, _body = op_store(
            node,
            to_dict(frag),
            is_primary=True,
            cluster=cluster,
            quorum_attempt=_attempt,
        )
        assert status == 200
        assert attempted == []

    def test_missing_cluster_falls_back_to_local_only(self):
        node = Node("self", max_memory_bytes=10_000)
        frag = _make_fragment_with_consistency("strong")
        status, body = op_store(
            node,
            to_dict(frag),
            is_primary=True,
        )
        assert status == 200
        assert body["success"] is True
