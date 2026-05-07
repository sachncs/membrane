"""Runnable demo: multi-node Membrane simulation with reconstruction."""

import logging

from membrane.fragmentation_engine import FragmentationConfig, FragmentationEngine
from membrane.global_directory import GlobalDirectory
from membrane.membrane_node import MembraneNode
from membrane.prefill_adapter import PrefillAdapter
from membrane.reconstruction_engine import ReconstructionEngine
from membrane.transfer_service import TransferService

logger = logging.getLogger(__name__)


def main():
    """Run a three-node Membrane demo."""
    logger.info("=== Membrane Multi-Node Demo ===")

    # Create 3 nodes in different regions
    node0 = MembraneNode("us-east-1", max_memory_bytes=1 << 20)
    node1 = MembraneNode("us-west-2", max_memory_bytes=1 << 20)
    node2 = MembraneNode("eu-west-1", max_memory_bytes=1 << 20)

    # Global directory tracks locations
    directory = GlobalDirectory()
    directory.register_node(node0)
    directory.register_node(node1)
    directory.register_node(node2)

    # Fragment a prompt into windows
    engine = FragmentationEngine(FragmentationConfig(window_size=128))
    prompt_tokens = list(range(512))
    frags = engine.create_windows(prompt_tokens, model_id="kimi-linear-1t")
    logger.info("Prompt fragmented into %s windows", len(frags))

    # Store fragments on node-0 (primary) and node-1 (replica)
    for f in frags:
        node0.store(f, is_primary=True)
        directory.record_fragment_location(f.content_hash, node0.node_id)

    for f in frags[:4]:
        node1.store(f, is_primary=False)
        directory.record_fragment_location(f.content_hash, node1.node_id)

    logger.info("Node-0 stores %s fragments", len(node0.fragments))
    logger.info("Node-1 stores %s fragments", len(node1.fragments))
    logger.info("Node-2 stores %s fragments (empty)", len(node2.fragments))

    # Transfer missing fragments from node-0 to node-2
    svc = TransferService()
    transferred = svc.sync_nodes(node0, node2)
    logger.info("Transferred %s fragments to node-2", len(transferred))

    # Reconstruct the prompt from node-2
    adapter = PrefillAdapter()
    from membrane.reconstruction_engine import ReconstructionConfig
    recon = ReconstructionEngine(
        node2.index_system,
        adapter,
        config=ReconstructionConfig(max_gap_tokens=50),
    )
    result = recon.rebuild_context(prompt_tokens, model_id="kimi-linear-1t")

    logger.info("Reconstruction coverage: %.2f%%", result.coverage_ratio * 100)
    logger.info("Prefill invoked: %s", result.prefill_invoked)
    logger.info("Missing segments: %s", len(result.missing_segments))

    # Check directory
    locations = directory.locate_fragment(frags[0].content_hash)
    logger.info("Fragment 0 located on nodes: %s", locations)

    # Optimal nodes for reconstruction
    optimal = directory.optimal_nodes_for_reconstruction(
        prompt_tokens, model_id="kimi-linear-1t", k=2
    )
    logger.info("Optimal nodes for reconstruction: %s", optimal)

    logger.info("=== Demo Complete ===")


if __name__ == "__main__":
    main()
