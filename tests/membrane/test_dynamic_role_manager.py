"""Tests for dynamic_role_manager module."""

import pytest

from membrane.dynamic_role_manager import DynamicRoleManager, NodeRole, SystemState
from membrane.membrane_node import MembraneNode


class TestDynamicRoleManager:
    """Test suite for DynamicRoleManager."""

    def test_high_memory_low_gpu_becomes_memory_host(self):
        mgr = DynamicRoleManager()
        node = MembraneNode("n", max_memory_bytes=100)
        from tests.membrane.test_cluster_replicator import make_fragment

        f = make_fragment("x", size=80)
        node.store(f, is_primary=True)
        state = SystemState(average_gpu_load=0.1)
        role = mgr.evaluate_role(node, state)
        assert role == NodeRole.MEMORY_HOST

    def test_low_memory_high_gpu_becomes_prefill(self):
        mgr = DynamicRoleManager()
        node = MembraneNode("n")
        state = SystemState(average_gpu_load=0.8)
        role = mgr.evaluate_role(node, state)
        assert role == NodeRole.PREFILL_WORKER

    def test_balanced_compute_demand_prefers_decode(self):
        mgr = DynamicRoleManager()
        node = MembraneNode("n")
        state = SystemState(total_compute_demand=0.8, total_memory_demand=0.2)
        role = mgr.evaluate_role(node, state)
        assert role == NodeRole.DECODE_WORKER

    def test_balanced_memory_demand_prefers_memory_host(self):
        mgr = DynamicRoleManager()
        node = MembraneNode("n")
        state = SystemState(total_compute_demand=0.2, total_memory_demand=0.8)
        role = mgr.evaluate_role(node, state)
        assert role == NodeRole.MEMORY_HOST

    def test_system_state_defaults(self):
        state = SystemState()
        assert state.total_compute_demand == 0.5
        assert state.total_memory_demand == 0.5
        assert state.average_gpu_load == 0.5

    def test_role_enum_values(self):
        assert NodeRole.MEMORY_HOST.value == "memory_host"
        assert NodeRole.PREFILL_WORKER.value == "prefill_worker"
        assert NodeRole.DECODE_WORKER.value == "decode_worker"
