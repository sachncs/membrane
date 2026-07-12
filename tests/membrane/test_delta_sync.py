"""Tests for delta_sync module."""

import pytest

from membrane.delta_sync import DeltaSync, SyncPlan
from membrane.fragment import Fragment
from membrane.membrane_node import MembraneNode
from membrane.structural_signature import StructuralSignature


def make_fragment(content_hash: str, size: int = 10, version_id: int = 1):
    return Fragment(
        content_hash=content_hash,
        embedding=(0.1,),
        structural_signature=StructuralSignature(model_id="m", layer_range=(0, 1), token_span=(0, 1)),
        size=size,
        ttl=3600.0,
        reuse_score=0.5,
        version_id=version_id,
    )


class TestDeltaSync:
    """Test suite for DeltaSync."""

    def test_build_plan_no_differences(self):
        source = MembraneNode("src")
        target = MembraneNode("tgt")
        f = make_fragment("h1")
        source.store(f, is_primary=True)
        target.store(f, is_primary=False)
        ds = DeltaSync()
        plan = ds.build_plan(source, target)
        assert plan.missing_hashes == []
        assert plan.outdated_hashes == []
        assert plan.estimated_bytes == 0

    def test_build_plan_detects_missing(self):
        source = MembraneNode("src")
        target = MembraneNode("tgt")
        source.store(make_fragment("h1", size=20), is_primary=True)
        ds = DeltaSync()
        plan = ds.build_plan(source, target)
        assert plan.missing_hashes == ["h1"]
        assert plan.estimated_bytes == 20

    def test_build_plan_detects_outdated(self):
        source = MembraneNode("src")
        target = MembraneNode("tgt")
        source.store(make_fragment("h1", version_id=2), is_primary=True)
        target.store(make_fragment("h1", version_id=1), is_primary=False)
        ds = DeltaSync()
        plan = ds.build_plan(source, target)
        assert plan.outdated_hashes == ["h1"]

    def test_execute_plan_transfers(self):
        source = MembraneNode("src")
        target = MembraneNode("tgt")
        source.store(make_fragment("h1", size=20), is_primary=True)
        ds = DeltaSync()
        plan = ds.build_plan(source, target)
        result = ds.execute_plan(plan, source, target)
        assert "h1" in result.transferred_hashes
        assert target.retrieve("h1") is not None
        assert result.bytes_transferred == 20

    def test_execute_plan_missing_on_source_fails(self):
        source = MembraneNode("src")
        target = MembraneNode("tgt")
        ds = DeltaSync()
        plan = SyncPlan(
            source_id="src",
            target_id="tgt",
            missing_hashes=["ghost"],
            outdated_hashes=[],
            estimated_bytes=0,
        )
        result = ds.execute_plan(plan, source, target)
        assert "ghost" in result.failed_hashes
        assert result.bytes_transferred == 0

    def test_sync_one_shot(self):
        source = MembraneNode("src")
        target = MembraneNode("tgt")
        source.store(make_fragment("h1"), is_primary=True)
        ds = DeltaSync()
        result = ds.sync(source, target)
        assert "h1" in result.transferred_hashes
        assert target.retrieve("h1") is not None

    def test_batch_sync(self):
        source = MembraneNode("src")
        t1 = MembraneNode("t1")
        t2 = MembraneNode("t2")
        source.store(make_fragment("h1"), is_primary=True)
        ds = DeltaSync()
        results = ds.batch_sync(source, [t1, t2])
        assert results["t1"].transferred_hashes == ["h1"]
        assert results["t2"].transferred_hashes == ["h1"]
        assert t1.retrieve("h1") is not None
        assert t2.retrieve("h1") is not None

    def test_plan_estimated_bytes(self):
        source = MembraneNode("src")
        target = MembraneNode("tgt")
        source.store(make_fragment("a", size=10), is_primary=True)
        source.store(make_fragment("b", size=20), is_primary=True)
        ds = DeltaSync()
        plan = ds.build_plan(source, target)
        assert plan.estimated_bytes == 30
