"""Cluster snapshot for durable restart recovery.

A cluster's state — membership view, shard ownership, server
counters — lives in memory while the process is running. On a
crash, a SIGTERM, or a routine restart, every in-memory structure
would otherwise be rebuilt from scratch.

This module writes a JSON snapshot to a configurable state_dir. The
snapshot carries a monotonic ``cluster_epoch`` so a node that was
partitioned for an extended outage refuses to rejoin with a stale
view of the world; ``cluster_epoch`` increments on every successful
restore so the next start writes a fresh value back.

Layout::

    {state_dir}/{node_id}.json

JSON contents::

    {
      "schema_version": 2,
      "cluster_epoch": 17,
      "captured_at": 1735600000.123,
      "membership": {
          "{peer_id}": {"host": ..., "port": ..., "cluster_epoch": ...}
      },
      "shards": {
          "primary_map": {"{hash}": "{node_id}", ...},
          "replica_map": {"{hash}": ["{node_id}", ...], ...}
      },
      "server": {"request_count": ..., "error_count": ...}
    }

The snapshot is a **secondary** source of truth. Redis (or any
durable KV store) carries the canonical state; the file is the
fallback when Redis is empty or unreachable.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


SNAPSHOT_SCHEMA_VERSION: int = 2


class Snapshot:
    """Durable cluster-state snapshot serialized as a JSON file.

    The class is mostly a namespace for ``save`` / ``load``; state
    payloads are plain ``dict`` objects so callers don't need a
    second serialization path.
    """

    def __init__(self, state_dir: str | os.PathLike[str]) -> None:
        """Initialize the snapshotter.

        Args:
            state_dir: Directory under which ``{node_id}.json``
                files are written. Created when missing.
        """
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)

    def path_for(self, node_id: str) -> Path:
        """Return the canonical snapshot path for ``node_id``."""
        return self.state_dir / f"{node_id}.json"

    def save(self, node_id: str, payload: dict[str, Any]) -> Path:
        """Atomically write ``payload`` as ``{node_id}.json``.

        The write goes through a ``.tmp`` file with explicit fsync
        followed by ``os.replace`` so a crash mid-write never
        leaves a half-written manifest under the live path.

        Args:
            node_id: This node's identifier.
            payload: Snapshot body. The caller is responsible for
                populating ``schema_version`` and
                ``cluster_epoch``.

        Returns:
            Path: The path actually written.
        """
        target = self.path_for(node_id)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload_with_stamp = dict(payload)
        payload_with_stamp.setdefault("captured_at", time.time())
        body = json.dumps(payload_with_stamp, sort_keys=True, default=str).encode("utf-8")
        # Write to a temp file in the same directory, fsync, replace.
        with tempfile.NamedTemporaryFile(
            delete=False,
            dir=str(self.state_dir),
            prefix=f".{node_id}.",
            suffix=".tmp",
        ) as tmp:
            tmp.write(body)
            tmp.flush()
            os.fsync(tmp.fileno())
            tmp_path = tmp.name
        os.replace(tmp_path, target)
        dir_fd = os.open(str(self.state_dir), os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
        logger.debug("Snapshot saved for node %s at %s", node_id, target)
        return target

    def load(self, node_id: str) -> dict[str, Any] | None:
        """Read ``{node_id}.json`` or return ``None`` when absent.

        Args:
            node_id: Node whose snapshot should be loaded.

        Returns:
            dict[str, Any] | None: Loaded payload, or ``None``
            when no snapshot has ever been written or when the
            file is corrupt.
        """
        path = self.path_for(node_id)
        if not path.exists():
            return None
        try:
            with path.open("rb") as fh:
                data = json.loads(fh.read().decode("utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Could not load snapshot for %s: %s", node_id, exc)
            return None
        schema = data.get("schema_version")
        if schema != SNAPSHOT_SCHEMA_VERSION:
            logger.warning(
                "Snapshot for %s has schema_version=%s (expected %s); discarding",
                node_id,
                schema,
                SNAPSHOT_SCHEMA_VERSION,
            )
            return None
        return data

    def latest(self) -> dict[str, Any] | None:
        """Return the most recently captured snapshot across all nodes.

        When the state directory holds snapshots from multiple
        nodes, this returns the one with the largest
        ``captured_at``. Used by recovery when the caller's
        ``node_id`` is not yet known.

        Returns:
            dict[str, Any] | None: The most recent snapshot, or
            ``None`` when the directory is empty.
        """
        latest_path: Path | None = None
        latest_payload: dict[str, Any] | None = None
        latest_time = -1.0
        for path in self.state_dir.glob("*.json"):
            try:
                with path.open("rb") as fh:
                    data = json.loads(fh.read().decode("utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            captured = float(data.get("captured_at", 0.0))
            if captured > latest_time:
                latest_time = captured
                latest_path = path
                latest_payload = data
        if latest_payload is None:
            return None
        latest_payload.setdefault("__path__", str(latest_path))
        return latest_payload

    def remove(self, node_id: str) -> bool:
        """Best-effort delete of ``{node_id}.json``.

        Args:
            node_id: Node whose snapshot should be purged.

        Returns:
            bool: ``True`` when a file was removed.
        """
        path = self.path_for(node_id)
        try:
            path.unlink()
            return True
        except FileNotFoundError:
            return False

    def __len__(self) -> int:
        """Count the snapshots currently held in the directory."""
        return sum(1 for _ in self.state_dir.glob("*.json"))


class ClusterEpochGuard:
    """Reject stale snapshots by comparing on a monotonic integer.

    Used during recovery: if the persisted ``cluster_epoch`` is more
    than ``1`` behind the live epoch observed on the network, the
    node refuses to restore the snapshot and instead boots with an
    empty in-memory view. This protects a node that lost a long
    partition from rebuilding an already-superseded shard map.
    """

    def __init__(self, current: int = 0) -> None:
        """Initialize with a current epoch.

        Args:
            current: The live epoch (default ``0``).
        """
        self.current = current

    def accept(self, persisted: int | None) -> bool:
        """Return whether ``persisted`` is fresh enough to apply.

        The guard accepts values in ``[current - 1, current + 1]``
        so a one-step lag is tolerated but a stale partition is
        rejected.

        Args:
            persisted: Value read from the snapshot, or
                ``None`` when missing.

        Returns:
            bool: ``True`` when the persisted value can be
            adopted.
        """
        if persisted is None:
            return False
        return abs(int(persisted) - self.current) <= 1

    def bump(self) -> int:
        """Increment the live epoch and return the new value.

        Returns:
            int: The new epoch value.
        """
        self.current += 1
        return self.current


__all__ = ["SNAPSHOT_SCHEMA_VERSION", "ClusterEpochGuard", "Snapshot"]
