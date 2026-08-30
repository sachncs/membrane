"""Tests for the data-integrity gauges (Phase 3.2.3)."""

from __future__ import annotations

import pytest

from membrane.errors import CorruptPayloadError
from membrane.integrity import (
    CORRUPT_PAYLOAD_COUNTER,
    MERKLE_DRIFT_GAUGE,
    measure_merkle_drift,
    record_corrupt_from_exception,
    record_corrupt_payload,
    record_merkle_drift,
)
from membrane.merkle import MerkleTree
from membrane.metrics import MetricsCollector


class TestMeasureMerkleDrift:
    def test_identical_trees_have_zero_drift(self):
        a = MerkleTree.from_inventory([("h1", "node-a"), ("h2", "node-b")])
        b = MerkleTree.from_inventory([("h1", "node-a"), ("h2", "node-b")])
        drift = measure_merkle_drift(a, b)
        assert drift.drift_size == 0

    def test_different_inventory_has_drift(self):
        a = MerkleTree.from_inventory([("h1", "node-a"), ("h2", "node-b")])
        b = MerkleTree.from_inventory([("h1", "node-a"), ("h3", "node-c")])
        drift = measure_merkle_drift(a, b)
        assert drift.drift_size >= 1

    def test_drift_includes_roots(self):
        a = MerkleTree.from_inventory([("h1", "node-a")])
        b = MerkleTree.from_inventory([("h1", "node-b")])
        drift = measure_merkle_drift(a, b)
        assert drift.local_root == a.root.hex()
        assert drift.remote_root == b.root.hex()


class TestRecordMerkleDrift:
    def test_records_into_registry(self):
        registry = MetricsCollector()
        a = MerkleTree.from_inventory([("h1", "node-a"), ("h2", "node-b")])
        b = MerkleTree.from_inventory([("h1", "node-a")])
        drift = measure_merkle_drift(a, b)
        record_merkle_drift(registry, drift, "peer-1")
        gauge_name = f"{MERKLE_DRIFT_GAUGE}:peer-1"
        assert gauge_name in registry.gauges
        assert registry.gauges[gauge_name].value == float(drift.drift_size)


class TestRecordCorruptPayload:
    def test_increments_counter(self):
        registry = MetricsCollector()
        record_corrupt_payload(registry)
        record_corrupt_payload(registry)
        assert registry.counters[CORRUPT_PAYLOAD_COUNTER].value == 2.0

    def test_from_exception_does_not_count_non_corrupt(self):
        registry = MetricsCollector()
        record_corrupt_from_exception(registry, ValueError("not corrupt"))
        assert CORRUPT_PAYLOAD_COUNTER not in registry.counters

    def test_from_exception_counts_corrupt(self):
        registry = MetricsCollector()
        exc = CorruptPayloadError("oops")
        record_corrupt_from_exception(registry, exc)
        assert registry.counters[CORRUPT_PAYLOAD_COUNTER].value == 1.0

    def test_no_registry_is_no_op(self):
        record_corrupt_payload(None)
        record_corrupt_from_exception(None, CorruptPayloadError("oops"))
