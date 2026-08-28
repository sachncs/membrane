#!/usr/bin/env python3
"""Demo script for Membrane."""

import logging

from membrane.canonical import Canonical
from membrane.chunks import Chunks
from membrane.replicator import Replicator
from membrane.delta import DeltaEncoder
from membrane.directory import Directory
from membrane.roles import Roles, NodeRole, SystemState
from membrane.economic import Economic
from membrane.fragment import Fragment
from membrane.ring import Ring
from membrane.joint import Joint
from membrane.kv import KVCache
from membrane.latency import Latency
from membrane.node import Node
from membrane.telemetry import Telemetry
from membrane.offload import Offload
from membrane.origin import Origin
from membrane.versions import Versions
from membrane.policy import Promotion
from membrane.prefill_remote import PrefillRemote
from membrane.replica import Replica
from membrane.clusters import SemanticCluster
from membrane.sessions import Sessions
from membrane.signature import Signature
from membrane._subgraph_retrieval import _SubgraphRetrieval
from membrane.supernode import Supernode
from membrane.isolation import Isolation, Tenant
from membrane.density import density
from membrane.weighted import Weighted
from membrane.workload import Workload

logger = logging.getLogger(__name__)


def make_fragment(content_hash, embedding=(0.0, 0.0), reuse_score=0.5, size=10):
    return Fragment(
        content_hash=content_hash,
        embedding=embedding,
        structural_signature=Signature(
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
    cache = KVCache()
    frag = make_fragment("demo-1")
    cache.store_kv("prefix-1", [frag])
    hit = cache.lookup_kv("demo-1")
    logger.info(
        f"  Cache hit: {hit is not None}, hit_rate={cache.get_hit_rate():.2f}\n"
    )

    # Phase 2: Regional Replication
    logger.info("[Phase 2] Origin → Replica Promotion")
    origin = Origin("origin-us")
    replica = Replica("replica-eu")
    origin.store(frag, is_primary=True)
    transferred = origin.bulk_promote(["demo-1"], replica)
    logger.info(f"  Transferred to replica: {transferred}\n")

    # Phase 3: Selective KV Offload
    logger.info("[Phase 3] Offload Decision")
    engine = Offload()
    local = Node("local")
    remote = Node("remote")
    decision = engine.decide(list(range(2048)), local, [remote])
    logger.info(f"  Offload to {decision.target_node_id}, reason={decision.reason}\n")

    # Phase 4: Global Directory
    logger.info("[Phase 4] Distributed Directory")
    ring = Ring()
    ring.add_node("n1")
    sn = Supernode("sn1", hash_ring=ring)
    sn.register_fragment("demo-1", "n1")
    dd = Directory(hash_ring=ring)
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
    g = Weighted()
    g.add_weighted_edge("a", "b", "next", 0.9)
    g.add_weighted_edge("b", "c", "next", 0.9)
    sr = _SubgraphRetrieval(g)
    comp = sr.retrieve_component("a", min_weight=0.5, max_depth=2)
    logger.info(f"  Component from a: {comp}")
    source = Node("source")
    target = Node("target")
    for h in comp:
        source.store(make_fragment(h, size=10), is_primary=True)
    cr = Replicator()
    results = cr.replicate_cluster(comp, source, [target])
    logger.info(f"  Replicated to target: {set(results.get('target', []))}\n")

    # Phase 7: Predictive Routing
    logger.info("[Phase 7] Session Tracking + Workload Analysis")
    st = Sessions()
    st.record_access("session-1", "h1")
    st.record_access("session-1", "h2")
    st.record_access("session-1", "h1")
    wa = Workload()
    ratio = wa.reuse_ratio(st.get_session_history("session-1"))
    logger.info(f"  Session history: {st.get_session_history('session-1')}")
    logger.info(f"  Reuse ratio: {ratio:.2f}\n")

    # Phase 8: Economic Scheduler
    logger.info("[Phase 8] Economic Router")
    router = Economic()
    frag2 = make_fragment("high-value", reuse_score=0.9)
    telemetry = {
        "n1": Telemetry("n1", 1000.0, 0.5, 0.8, 0.8),
        "n2": Telemetry("n2", 10.0, 0.1, 0.1, 0.1),
    }
    best = router.route(frag2, ["n1", "n2"], telemetry, [])
    logger.info(f"  Best node for high-value fragment: {best}\n")

    # Phase 9: Multi-Tenant Deduplication
    logger.info("[Phase 9] Tenant Isolation + Canonical Store")
    ti = Isolation(policy=Tenant(allow_tool_traces=True))
    sharedmake_fragment = make_fragment("shared", reuse_score=0.9)
    logger.info(f"  Can share across tenants: {ti.can_share(sharedmake_fragment, 't1', 't2')}")
    cs = Canonical()
    cs.store_canonical(sharedmake_fragment, "t1")
    cs.store_canonical(sharedmake_fragment, "t2")
    logger.info(f"  Shared fragments for t1: {len(cs.get_shared_fragments('t1'))}\n")

    # Phase 10: Compute-Memory Convergence
    logger.info("[Phase 10] Dynamic Role + Joint Optimization")
    mgr = Roles()
    node = Node("n1", max_memory_bytes=100)
    f = make_fragment("load", size=80)
    node.store(f, is_primary=True)
    state = SystemState(average_gpu_load=0.1)
    role = mgr.evaluate_role(node, state)
    logger.info(f"  Node role: {role.value}")
    opt = Joint()
    decision = opt.optimize(
        sharedmake_fragment, [node], {"n1": Telemetry("n1", 10.0, 0.0, 0.0, 0.0)}
    )
    logger.info(
        f"  Placement: compute={decision.compute_node_id}, memory={decision.memory_node_id}"
    )

    logger.info("\n=== Demo Complete ===")


if __name__ == "__main__":
    main()
