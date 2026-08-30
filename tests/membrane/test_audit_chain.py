"""Tests for the tamper-evident audit log (Phase 3.2.8)."""

from __future__ import annotations

import json

import pytest

from membrane.audit import (
    AuditEntry,
    AuditLog,
    FileAuditStorage,
    verify_chain,
)


class TestAuditLog:
    def test_first_entry_has_empty_prev_hash(self):
        log = AuditLog()
        entry = log.record("alice", "fragment.store", payload={"hash": "h1"})
        assert entry.prev_hash == ""
        assert entry.entry_hash != ""
        assert entry.index == 0

    def test_chain_links_consecutive_entries(self):
        log = AuditLog()
        first = log.record("alice", "fragment.store", payload={"hash": "h1"})
        second = log.record("alice", "fragment.store", payload={"hash": "h2"})
        assert second.prev_hash == first.entry_hash
        assert first.entry_hash != second.entry_hash
        assert second.index == 1

    def test_chain_verifies_when_intact(self):
        log = AuditLog()
        for i in range(10):
            log.record("alice", "fragment.store", payload={"i": i})
        assert verify_chain(log.all()) is None

    def test_tampered_entry_detected(self):
        log = AuditLog()
        log.record("alice", "fragment.store", payload={"i": 0})
        log.record("alice", "fragment.store", payload={"i": 1})
        entries = log.all()
        # Tamper with the payload of the first entry; the hash
        # chain should fail verification at index 0.
        bad = AuditEntry(
            index=entries[0].index,
            timestamp=entries[0].timestamp,
            actor=entries[0].actor,
            action=entries[0].action,
            payload={"i": 99},
            prev_hash=entries[0].prev_hash,
            entry_hash=entries[0].entry_hash,
        )
        tampered = [bad, entries[1]]
        assert verify_chain(tampered) == 0

    def test_chain_breaks_when_prev_hash_lies(self):
        log = AuditLog()
        log.record("alice", "fragment.store", payload={"i": 0})
        log.record("alice", "fragment.store", payload={"i": 1})
        entries = log.all()
        # Replace the second entry's prev_hash with a bogus value.
        broken = [
            entries[0],
            AuditEntry(
                index=entries[1].index,
                timestamp=entries[1].timestamp,
                actor=entries[1].actor,
                action=entries[1].action,
                payload=entries[1].payload,
                prev_hash="0" * 64,
                entry_hash=entries[1].entry_hash,
            ),
        ]
        assert verify_chain(broken) == 1


class TestFileAuditStorage:
    def test_append_and_reload(self, tmp_path):
        path = tmp_path / "audit.jsonl"
        storage = FileAuditStorage(path=path)
        log = AuditLog(storage=storage)
        for i in range(3):
            log.record("alice", "fragment.store", payload={"i": i})
        loaded = storage.all()
        assert len(loaded) == 3
        assert verify_chain(loaded) is None
        assert [e.payload["i"] for e in loaded] == [0, 1, 2]

    def test_storage_round_trips_with_chain(self, tmp_path):
        path = tmp_path / "audit.jsonl"
        storage = FileAuditStorage(path=path)
        log = AuditLog(storage=storage)
        log.record("alice", "fragment.store", payload={"i": 0})
        log.record("bob", "fragment.evict", payload={"i": 1})
        loaded = storage.all()
        assert loaded[0].actor == "alice"
        assert loaded[1].actor == "bob"
        assert verify_chain(loaded) is None


class TestVerifyChainEmpty:
    def test_empty_chain_verifies(self):
        """An empty chain is trivially valid."""
        assert verify_chain([]) is None


class TestAuditEntryEquality:
    def test_entries_compare_by_hash(self):
        log = AuditLog()
        a = log.record("alice", "fragment.store", payload={"hash": "h1"})
        b = log.record("alice", "fragment.store", payload={"hash": "h1"})
        # Same content but different prev_hash / entry_hash.
        assert a != b
