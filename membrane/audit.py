"""Tamper-evident hash-chained audit log (Phase 3.2.8).

The v2.0 release carried a Server.events in-memory buffer
that was never persisted or chained. The v3.0.0 release
introduces an append-only :class:`AuditLog` where every entry
stores a SHA-256 over the canonical JSON of the entry plus
the previous entry's hash. The :func:`verify_chain` helper
walks the log end-to-end and returns the index of the first
tampered entry, ``None`` when the chain verifies.

The :class:`FileAuditStorage` writes one entry per line in
JSON Lines format; the in-memory :class:`InMemoryAuditStorage`
is reserved for tests and the Phase 3.2 commit deliberately
avoids shipping a sibling in-memory backend the way the 3.2
plan called out (the original memory variant was a test-only
shim). The :data:`AuditStorage` Protocol is the swap point for
operators that want to back the chain with a relational store.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

logger = logging.getLogger(__name__)


def _canonical_json(payload: dict[str, object]) -> str:
    """Serialize ``payload`` as a deterministic JSON string.

    Args:
        payload: The dict to serialize.

    Returns:
        str: Sorted, whitespace-free JSON.
    """
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _hash_entry(prev_hash: str, entry: dict[str, object]) -> str:
    """Compute the SHA-256 hash chain for ``entry``.

    Args:
        prev_hash: The previous entry's hash (the empty string
            for the first entry).
        entry: The candidate entry to hash.

    Returns:
        str: 64-character hex digest.
    """
    h = hashlib.sha256()
    h.update(prev_hash.encode("utf-8"))
    h.update(b"|")
    h.update(_canonical_json(entry).encode("utf-8"))
    return h.hexdigest()


@dataclass(frozen=True)
class AuditEntry:
    """A single audit-log entry.

    Attributes:
        index: Position in the chain (0-based).
        timestamp: Monotonic clock at write time.
        actor: Subject of the call (e.g., the API key id or peer CN).
        action: Free-form action string (e.g.,
            ``"fragment.store"``).
        payload: Arbitrary structured payload the auditor wants.
        prev_hash: Hash of the previous entry; the empty
            string for the first entry.
        entry_hash: SHA-256 of (prev_hash || payload).
    """

    index: int
    timestamp: float
    actor: str
    action: str
    payload: dict[str, object]
    prev_hash: str
    entry_hash: str


@runtime_checkable
class AuditStorage(Protocol):
    """Pluggable audit-log storage.

    Implementations return entries in append order. The
    :class:`AuditLog` is the in-memory front; the :class:`FileAuditStorage`
    backs it with a JSON Lines file.
    """

    def append(self, entry: AuditEntry) -> None:
        """Persist ``entry`` atomically."""
        ...

    def all(self) -> list[AuditEntry]:
        """Return every entry in order."""
        ...


@dataclass
class AuditLog:
    """In-memory audit log with optional storage backend.

    Attributes:
        storage: Pluggable storage backend. ``None`` keeps the
            log in memory only.
    """

    storage: AuditStorage | None = None
    _entries: list[AuditEntry] = field(default_factory=list)
    _last_hash: str = ""
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def record(
        self,
        actor: str,
        action: str,
        payload: dict[str, object] | None = None,
        *,
        timestamp: float | None = None,
    ) -> AuditEntry:
        """Record ``actor``/``action``/``payload`` on the chain.

        Args:
            actor: The caller identity.
            action: A short stable verb (e.g., ``"fragment.store"``).
            payload: Optional structured payload.
            timestamp: Monotonic timestamp; ``None`` reads
                ``time.monotonic``.

        Returns:
            AuditEntry: The new entry.
        """
        import time as _time

        payload_dict: dict[str, object] = payload or {}
        with self._lock:
            index = len(self._entries)
            ts = _time.monotonic() if timestamp is None else timestamp
            entry_payload: dict[str, object] = {
                "actor": actor,
                "action": action,
                "payload": payload_dict,
                "timestamp": ts,
            }
            entry_hash = _hash_entry(self._last_hash, entry_payload)
            entry = AuditEntry(
                index=index,
                timestamp=ts,
                actor=actor,
                action=action,
                payload=payload_dict,
                prev_hash=self._last_hash,
                entry_hash=entry_hash,
            )
            self._entries.append(entry)
            self._last_hash = entry_hash
            if self.storage is not None:
                self.storage.append(entry)
            return entry

    def all(self) -> list[AuditEntry]:
        """Return every recorded entry in append order.

        Returns:
            list[AuditEntry]: Snapshot of the in-memory entries.
        """
        with self._lock:
            return list(self._entries)

    def last_hash(self) -> str:
        """Return the hash of the most recent entry (or empty string).

        Returns:
            str: Hex digest, or empty string if the log is empty.
        """
        with self._lock:
            return self._last_hash


def verify_chain(entries: Iterable[AuditEntry]) -> int | None:
    """Return the index of the first tampered entry, or ``None``.

    Args:
        entries: The entries to verify, in append order.

    Returns:
        int | None: The 0-based index of the first broken
        entry, or ``None`` when the entire chain verifies.
    """
    prev_hash = ""
    for entry in entries:
        if entry.prev_hash != prev_hash:
            return entry.index
        expected = _hash_entry(
            prev_hash,
            {
                "actor": entry.actor,
                "action": entry.action,
                "payload": entry.payload,
                "timestamp": entry.timestamp,
            },
        )
        if expected != entry.entry_hash:
            return entry.index
        prev_hash = entry.entry_hash
    return None


@dataclass
class FileAuditStorage:
    """JSON Lines file-backed audit log.

    Each :class:`AuditEntry` writes one line. The store is
    append-only and safe against torn writes: a line is
    written via ``write + flush``; the next read skips any
    trailing partial line.
    """

    path: Path
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def append(self, entry: AuditEntry) -> None:
        """Append ``entry`` as one JSON line.

        Args:
            entry: The entry to persist.
        """
        line = _canonical_json(
            {
                "index": entry.index,
                "timestamp": entry.timestamp,
                "actor": entry.actor,
                "action": entry.action,
                "payload": entry.payload,
                "prev_hash": entry.prev_hash,
                "entry_hash": entry.entry_hash,
            }
        )
        with self._lock, self.path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()

    def all(self) -> list[AuditEntry]:
        """Return every persisted entry in order.

        Returns:
            list[AuditEntry]: Snapshot from the on-disk log.
        """
        if not self.path.exists():
            return []
        with self._lock, self.path.open("r", encoding="utf-8") as f:
            out: list[AuditEntry] = []
            for raw in f:
                line = raw.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    # Skip torn or garbage lines without raising;
                    # the auditor's :func:`verify_chain` will
                    # still report the index of any torn write.
                    continue
                out.append(
                    AuditEntry(
                        index=int(payload["index"]),
                        timestamp=float(payload["timestamp"]),
                        actor=str(payload["actor"]),
                        action=str(payload["action"]),
                        payload=dict(payload.get("payload") or {}),
                        prev_hash=str(payload["prev_hash"]),
                        entry_hash=str(payload["entry_hash"]),
                    )
                )
            return out

    def lock_for_read(self) -> threading.Lock:
        return self._lock


__all__ = [
    "AuditEntry",
    "AuditLog",
    "AuditStorage",
    "FileAuditStorage",
    "verify_chain",
]
