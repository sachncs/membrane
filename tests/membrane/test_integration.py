"""Integration tests for Membrane."""

import pytest

from membrane.canonical_store import CanonicalStore
from membrane.chunked_transfer import ChunkedTransfer
from membrane.cluster_replicator import ClusterReplicator
from membrane.delta_encoder import DeltaEncoder
from membrane.distributed_directory import DistributedDirectory
from membrane.dynamic_role_manager import DynamicRoleManager, NodeRole, SystemState
from membrane.economic_router import EconomicRouter
from membrane.fragment import Fragment
from membrane.hash_ring import HashRing
from membrane.joint_optimizer import JointOptimizer
from membrane.kv_cache_manager import KVCacheManager
from membrane.latency_router import LatencyRouter
from membrane.membrane_node import MembraneNode
from membrane.node_telemetry import NodeTelemetry
from membrane.offload_decision_engine import OffloadDecisionEngine
from membrane.origin_node import OriginNode
from membrane.prefix_version_chain import PrefixVersionChain
from membrane.promotion_policy import PromotionPolicy
from membrane.remote_prefill_dispatcher import RemotePrefillDispatcher
from membrane.replica_node import ReplicaNode
from membrane.semantic_cluster import SemanticCluster
from membrane.session_tracker import SessionTracker
from membrane.structural_signature import StructuralSignature
from membrane.subgraph_retrieval import SubgraphRetrieval
from membrane.supernode import Supernode
from membrane.tenant_isolation import TenantIsolation, TenantPolicy
from membrane.value_density import ValueDensity
from membrane.weighted_graph import WeightedGraph
from membrane.workload_analyzer import WorkloadAnalyzer


def make_fragment(content_hash, embedding=(0.0, 0.0), reuse_score=0.5, size=10):
    return Fragment(
        content_hash=content_hash,
        embedding=embedding,
        structural_signature=StructuralSignature(
            model_id="m", layer_range=(0, 1), token_span=(0, 1)
        ),
        size=size,
        ttl=3600.0,
        reuse_score=reuse_score,
        version_id=1,
    )


class TestMembraneIntegration:
    """End-to-end integration tests across all 10 phases."""

    def test_phase_1_cache_hit_tracking(self):
        mgr = KVCacheManager()
        frag = make_fragment("h1")
        mgr.store_kv("p1", [frag])
        result = mgr.lookup_kv("p1")
        assert len(result) == 1
        assert result[0] == frag
        assert mgr.get_hit_rate() == 1.0

    def test_phase_2_origin_replica_promotion(self):
        origin = OriginNode("origin-1")
        replica = ReplicaNode("replica-1")
        frag = make_fragment("h1", size=50)
        origin.store(frag, is_primary=True)
        transferred = origin.bulk_promote(["h1"], replica)
        assert "h1" in transferred
        assert replica.retrieve("h1") == frag

    def test_phase_3_offload_and_ship(self):
        engine = OffloadDecisionEngine()
        local = MembraneNode("local")
        remote = MembraneNode("remote")
        decision = engine.decide(list(range(2048)), local, [remote])
        assert not decision.local_compute

        dispatcher = RemotePrefillDispatcher()
        result = dispatcher.dispatch(list(range(100)), "m", remote)
        assert result.kv_size_mib > 0.0

    def test_phase_4_directory_resolution(self):
        ring = HashRing()
        ring.add_node("n1")
        sn = Supernode("sn1", hash_ring=ring)
        sn.register_fragment("h1", "n1")
        dd = DistributedDirectory(hash_ring=ring)
        dd.register_supernode(sn)
        assert dd.locate("h1") == {"n1"}
        assert dd.locate_nearest("h1", "from") == "n1"

    def test_phase_5_delta_roundtrip(self):
        enc = DeltaEncoder()
        base = tuple(range(10))
        new = tuple(range(10)) + (99, 100)
        delta = enc.encode(base, new)
        assert enc.decode(base, delta) == new

    def test_phase_5_version_chain_ancestor(self):
        chain = PrefixVersionChain()
        chain.append_version("h1", 1)
        chain.append_version("h2", 2, parent_version=1)
        chain.append_version("h3", 3, parent_version=1)
        assert chain.get_common_ancestor(2, 3) == 1

    def test_phase_6_graph_cluster_replication(self):
        g = WeightedGraph()
        g.add_weighted_edge("a", "b", "next", 0.9)
        g.add_weighted_edge("b", "c", "next", 0.9)
        sr = SubgraphRetrieval(g)
        comp = sr.retrieve_component("a", min_weight=0.5, max_depth=2)
        assert comp == {"a", "b", "c"}

        source = MembraneNode("source")
        t1 = MembraneNode("t1")
        for h in comp:
            source.store(make_fragment(h, size=10), is_primary=True)
        cr = ClusterReplicator()
        results = cr.replicate_cluster(comp, source, [t1])
        assert set(results["t1"]) == comp

    def test_phase_7_session_and_workload(self):
        st = SessionTracker()
        st.record_access("s1", "h1")
        st.record_access("s1", "h2")
        st.record_access("s1", "h1")
        assert st.get_unique_accesses("s1") == {"h1", "h2"}

        wa = WorkloadAnalyzer()
        log = st.get_session_history("s1")
        ratio = wa.reuse_ratio(log)
        assert ratio > 0.0

    def test_phase_8_economic_routing(self):
        router = EconomicRouter()
        frag = make_fragment("h1", reuse_score=0.9)
        telemetry = {
            "n1": NodeTelemetry("n1", 1000.0, 0.5, 0.8, 0.8),
            "n2": NodeTelemetry("n2", 10.0, 0.1, 0.1, 0.1),
        }
        best = router.route(frag, ["n1", "n2"], telemetry, [])
        assert best == "n2"

    def test_phase_9_tenant_canonical_store(self):
        ti = TenantIsolation()
        frag = make_fragment("h1", reuse_score=0.9)
        assert ti.can_share(frag, "t1", "t2")

        cs = CanonicalStore()
        cs.store_canonical(frag, "t1")
        cs.store_canonical(frag, "t2")
        shared = cs.get_shared_fragments("t1")
        assert len(shared) == 1
        assert shared[0].content_hash == "h1"

    def test_phase_10_role_and_joint_optimization(self):
        mgr = DynamicRoleManager()
        node = MembraneNode("n1", max_memory_bytes=100)
        from tests.membrane.test_cluster_replicator import make_fragment as mkfrag

        f = mkfrag("x", size=80)
        node.store(f, is_primary=True)
        state = SystemState(average_gpu_load=0.1)
        role = mgr.evaluate_role(node, state)
        assert role == NodeRole.MEMORY_HOST

        opt = JointOptimizer()
        frag = make_fragment("h1")
        decision = opt.optimize(
            frag, [node], {"n1": NodeTelemetry("n1", 10.0, 0.0, 0.0, 0.0)}
        )
        assert decision.compute_node_id == "n1"
        assert decision.memory_node_id == "n1"
