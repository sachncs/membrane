#!/usr/bin/env python3
"""Demo script for Membrane."""

import logging

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

logger = logging.getLogger(__name__)


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


def main():
    logger.info("=== Membrane Demo ===\n")

    # Phase 1: Single-Region Cache
    logger.info("[Phase 1] KV Cache Manager")
    cache = KVCacheManager()
    frag = make_fragment("demo-1")
    cache.store_kv("prefix-1", [frag])
    hit = cache.lookup_kv("demo-1")
    logger.info(
        f"  Cache hit: {hit is not None}, hit_rate={cache.get_hit_rate():.2f}\n"
    )

    # Phase 2: Regional Replication
    logger.info("[Phase 2] Origin → Replica Promotion")
    origin = OriginNode("origin-us")
    replica = ReplicaNode("replica-eu")
    origin.store(frag, is_primary=True)
    transferred = origin.bulk_promote(["demo-1"], replica)
    logger.info(f"  Transferred to replica: {transferred}\n")

    # Phase 3: Selective KV Offload
    logger.info("[Phase 3] Offload Decision")
    engine = OffloadDecisionEngine()
    local = MembraneNode("local")
    remote = MembraneNode("remote")
    decision = engine.decide(list(range(2048)), local, [remote])
    logger.info(f"  Offload to {decision.target_node_id}, reason={decision.reason}\n")

    # Phase 4: Global Directory
    logger.info("[Phase 4] Distributed Directory")
    ring = HashRing()
    ring.add_node("n1")
    sn = Supernode("sn1", hash_ring=ring)
    sn.register_fragment("demo-1", "n1")
    dd = DistributedDirectory(hash_ring=ring)
    dd.register_supernode(sn)
    logger.info(f"  Locate demo-1: {dd.locate('demo-1')}\n")

    # Phase 5: Delta Transport
    logger.info("[Phase 5] Delta Encoding")
    enc = DeltaEncoder()
    base = tuple(range(10))
    new = tuple(range(10)) + (99, 100)
    delta = enc.encode(base, new)
    reconstructed = enc.decode(base, delta)
    logger.info(
        f"  Delta appended={delta.appended_tokens}, decode_ok={reconstructed == new}\n"
    )

    # Phase 6: Context Graph
    logger.info("[Phase 6] Weighted Graph + Cluster Replication")
    g = WeightedGraph()
    g.add_weighted_edge("a", "b", "next", 0.9)
    g.add_weighted_edge("b", "c", "next", 0.9)
    sr = SubgraphRetrieval(g)
    comp = sr.retrieve_component("a", min_weight=0.5, max_depth=2)
    logger.info(f"  Component from a: {comp}")
    source = MembraneNode("source")
    target = MembraneNode("target")
    for h in comp:
        source.store(make_fragment(h, size=10), is_primary=True)
    cr = ClusterReplicator()
    results = cr.replicate_cluster(comp, source, [target])
    logger.info(f"  Replicated to target: {set(results.get('target', []))}\n")

    # Phase 7: Predictive Routing
    logger.info("[Phase 7] Session Tracking + Workload Analysis")
    st = SessionTracker()
    st.record_access("session-1", "h1")
    st.record_access("session-1", "h2")
    st.record_access("session-1", "h1")
    wa = WorkloadAnalyzer()
    ratio = wa.reuse_ratio(st.get_session_history("session-1"))
    logger.info(f"  Session history: {st.get_session_history('session-1')}")
    logger.info(f"  Reuse ratio: {ratio:.2f}\n")

    # Phase 8: Economic Scheduler
    logger.info("[Phase 8] Economic Router")
    router = EconomicRouter()
    frag2 = make_fragment("high-value", reuse_score=0.9)
    telemetry = {
        "n1": NodeTelemetry("n1", 1000.0, 0.5, 0.8, 0.8),
        "n2": NodeTelemetry("n2", 10.0, 0.1, 0.1, 0.1),
    }
    best = router.route(frag2, ["n1", "n2"], telemetry, [])
    logger.info(f"  Best node for high-value fragment: {best}\n")

    # Phase 9: Multi-Tenant Deduplication
    logger.info("[Phase 9] Tenant Isolation + Canonical Store")
    ti = TenantIsolation(policy=TenantPolicy(allow_tool_traces=True))
    sharedmake_fragment = make_fragment("shared", reuse_score=0.9)
    logger.info(f"  Can share across tenants: {ti.can_share(sharedmake_fragment, 't1', 't2')}")
    cs = CanonicalStore()
    cs.store_canonical(sharedmake_fragment, "t1")
    cs.store_canonical(sharedmake_fragment, "t2")
    logger.info(f"  Shared fragments for t1: {len(cs.get_shared_fragments('t1'))}\n")

    # Phase 10: Compute-Memory Convergence
    logger.info("[Phase 10] Dynamic Role + Joint Optimization")
    mgr = DynamicRoleManager()
    node = MembraneNode("n1", max_memory_bytes=100)
    f = make_fragment("load", size=80)
    node.store(f, is_primary=True)
    state = SystemState(average_gpu_load=0.1)
    role = mgr.evaluate_role(node, state)
    logger.info(f"  Node role: {role.value}")
    opt = JointOptimizer()
    decision = opt.optimize(
        sharedmake_fragment, [node], {"n1": NodeTelemetry("n1", 10.0, 0.0, 0.0, 0.0)}
    )
    logger.info(
        f"  Placement: compute={decision.compute_node_id}, memory={decision.memory_node_id}"
    )

    logger.info("\n=== Demo Complete ===")


if __name__ == "__main__":
    main()
