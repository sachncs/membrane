from tests.conftest import make_fragment

"""Integration tests for Membrane."""

import pytest

from membrane.analytical import Isolation, Tenant, Workload
from membrane.canonical import canonicalize, parse_canonical
from membrane.chunks import Chunks
from membrane.delta import DeltaEncoder
from membrane.density import density
from membrane.economic import Economic
from membrane.fragment import Fragment
from membrane.graph import SubgraphRetrieval
from membrane.identity import PayloadIdentity
from membrane.joint import Joint
from membrane.kv import KVCache
from membrane.latency import Latency
from membrane.node import Node
from membrane.offload import Offload
from membrane.origin import Origin
from membrane.policy import Promotion
from membrane.prefilling import Prefiller as PrefillRemote
from membrane.replica import Replica
from membrane.replicator import Replicator
from membrane.ring import Ring
from membrane.roles import NodeRole, Roles, SystemState
from membrane.semantics import SemanticCluster
from membrane.sessions import Sessions
from membrane.telemetry import Telemetry
from membrane.versions import Versions
from membrane.weighted import Weighted


class TestMembraneIntegration:
    """End-to-end integration tests across all 10 phases."""

    def test_phase_1_cache_hit_tracking(self):
        mgr = KVCache()
        frag = make_fragment("h1")
        mgr.store_kv("p1", [frag])
        result = mgr.lookup_kv("p1")
        assert len(result) == 1
        assert result[0] == frag
        assert mgr.get_hit_rate() == 1.0

    def test_phase_2_origin_replica_promotion(self):
        origin = Origin("origin-1")
        replica = Replica("replica-1")
        frag = make_fragment("h1", size=50)
        origin.store(frag, is_primary=True)
        transferred = origin.bulk_promote(["h1"], replica)
        assert "h1" in transferred
        assert replica.retrieve("h1") == frag

    def test_phase_3_offload_and_ship(self):
        engine = Offload()
        local = Node("local")
        remote = Node("remote")
        decision = engine.decide(list(range(2048)), local, [remote])
        assert not decision.local_compute

        dispatcher = PrefillRemote()
        result = dispatcher.dispatch_sync(list(range(100)), "m", remote)
        assert result.kv_size > 0.0

    def test_phase_4_delta_roundtrip(self):
        enc = DeltaEncoder()
        base = tuple(range(10))
        new = (*tuple(range(10)), 99, 100)
        delta = enc.encode(base, new)
        assert enc.decode(base, delta) == new

    def test_phase_5_version_chain_ancestor(self):
        chain = Versions()
        chain.append_version("h1", 1)
        chain.append_version("h2", 2, parent_version=1)
        chain.append_version("h3", 3, parent_version=1)
        assert chain.get_common_ancestor(2, 3) == 1

    def test_phase_6_graph_cluster_replication(self):
        g = Weighted()
        g.add_weighted_edge("a", "b", "next", 0.9)
        g.add_weighted_edge("b", "c", "next", 0.9)
        sr = SubgraphRetrieval(g)
        comp = sr.retrieve_component("a", min_weight=0.5, max_depth=2)
        assert comp == {"a", "b", "c"}

        source = Node("source")
        t1 = Node("t1")
        for h in comp:
            source.store(make_fragment(h, size=10), is_primary=True)
        cr = Replicator()
        results = cr.replicate_cluster(comp, source, [t1])
        assert set(results["t1"]) == comp

    def test_phase_7_session_and_workload(self):
        st = Sessions()
        st.record_access("s1", "h1")
        st.record_access("s1", "h2")
        st.record_access("s1", "h1")
        assert st.get_unique_accesses("s1") == {"h1", "h2"}

        wa = Workload()
        log = st.get_session_history("s1")
        ratio = wa.reuse_ratio(log)
        assert ratio > 0.0

    def test_phase_8_economic_routing(self):
        router = Economic()
        frag = make_fragment("h1", reuse_score=0.9)
        telemetry = {
            "n1": Telemetry("n1", 1000.0, 0.5, 0.8, 0.8),
            "n2": Telemetry("n2", 10.0, 0.1, 0.1, 0.1),
        }
        best = router.route(frag, ["n1", "n2"], telemetry, [])
        assert best == "n2"

    def test_phase_9_tenant_canonical_store(self):
        ti = Isolation()
        frag = make_fragment("h1", reuse_score=0.9)
        assert ti.can_share(frag, "t1", "t2")

        # Use the new canonical byte-framing module in place of the
        # deleted :class:`Canonical` content-addressed cache.
        identity = PayloadIdentity(
            payload_hash=frag.identity.payload_hash,
            model_id=frag.identity.model_id,
            model_revision=frag.identity.model_revision,
            tokenizer_name=frag.identity.tokenizer_name,
            tokenizer_revision=frag.identity.tokenizer_revision,
            layer_range=frag.identity.layer_range,
            head_range=frag.identity.head_range,
            token_span=frag.identity.token_span,
            dtype=frag.identity.dtype,
            shape=frag.identity.shape,
        )
        payload = b"h1-payload"
        frame = canonicalize(identity, payload)
        parsed_identity, parsed_payload = parse_canonical(frame)
        assert parsed_identity == identity
        assert parsed_payload == payload

    def test_phase_10_role_and_joint_optimization(self):
        mgr = Roles()
        node = Node("n1", max_memory_bytes=100)

        f = make_fragment("x", size=80)
        node.store(f, is_primary=True)
        state = SystemState(average_gpu_load=0.1)
        role = mgr.evaluate_role(node, state)
        assert role == NodeRole.MEMORY_HOST

        opt = Joint()
        frag = make_fragment("h1")
        decision = opt.optimize(frag, [node], {"n1": Telemetry("n1", 10.0, 0.0, 0.0, 0.0)})
        assert decision.compute_node_id == "n1"
        assert decision.memory_node_id == "n1"
