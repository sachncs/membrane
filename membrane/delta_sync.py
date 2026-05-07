"""DeltaSync: inventory-based delta synchronization between nodes.

Builds on TransferService to provide efficient, version-aware batch
synchronization.  Computes the minimal set of fragments that need to move
between two nodes, schedules batch transfers, and tracks sync progress.
"""

import logging

logger = logging.getLogger(__name__)


from dataclasses import dataclass, field

from membrane.fragment import Fragment
from membrane.membrane_node import MembraneNode
from membrane.transfer_service import TransferService


@dataclass
class SyncPlan:
    """Planned synchronization between two nodes.

    Attributes:
        source_id: Source node identifier.
        target_id: Target node identifier.
        missing_hashes: Content hashes the target is missing or outdated.
        outdated_hashes: Content hashes the target has but with older version.
        estimated_bytes: Approximate total bytes to transfer.
    """

    source_id: str
    target_id: str
    missing_hashes: list[str] = field(default_factory=list)
    outdated_hashes: list[str] = field(default_factory=list)
    estimated_bytes: int = 0


@dataclass
class SyncResult:
    """Result of executing a sync plan.

    Attributes:
        transferred_hashes: Hashes successfully transferred.
        failed_hashes: Hashes that could not be transferred.
        bytes_transferred: Total bytes moved.
    """

    transferred_hashes: list[str] = field(default_factory=list)
    failed_hashes: list[str] = field(default_factory=list)
    bytes_transferred: int = 0


class DeltaSync:
    """Version-aware delta synchronization engine.

    Args:
        transfer_service: TransferService for moving fragments.
    """

    def __init__(self, transfer_service: TransferService | None = None) -> None:
        self.transfer_service = transfer_service or TransferService()

    def build_plan(
        self,
        source: MembraneNode,
        target: MembraneNode,
    ) -> SyncPlan:
        """Compute the minimal sync plan from *source* to *target*.

        Args:
            source: Source node (has the canonical copy).
            target: Target node (may be missing or outdated).

        Returns:
            SyncPlan describing required transfers.
        """
        source_digest = self.transfer_service.inventory_digest(source)
        target_digest = self.transfer_service.inventory_digest(target)

        missing: list[str] = []
        outdated: list[str] = []
        estimated_bytes = 0

        for h, source_version in source_digest.items():
            target_version = target_digest.get(h)
            if target_version is None:
                missing.append(h)
                frag = source.fragments.get(h)
                if frag is not None:
                    estimated_bytes += frag.size
            elif target_version < source_version:
                outdated.append(h)
                frag = source.fragments.get(h)
                if frag is not None:
                    estimated_bytes += frag.size

        return SyncPlan(
            source_id=source.node_id,
            target_id=target.node_id,
            missing_hashes=missing,
            outdated_hashes=outdated,
            estimated_bytes=estimated_bytes,
        )

    def execute_plan(
        self,
        plan: SyncPlan,
        source: MembraneNode,
        target: MembraneNode,
    ) -> SyncResult:
        """Execute a sync plan and return results.

        Args:
            plan: SyncPlan to execute.
            source: Source node.
            target: Target node.

        Returns:
            SyncResult with transfer outcomes.
        """
        result = SyncResult()

        all_hashes = plan.missing_hashes + plan.outdated_hashes
        for h in all_hashes:
            frag = source.fragments.get(h)
            if frag is None:
                result.failed_hashes.append(h)
                continue
            if self.transfer_service.transfer_fragment(source, target, h):
                result.transferred_hashes.append(h)
                result.bytes_transferred += frag.size
            else:
                result.failed_hashes.append(h)

        logger.info(
            "Sync %s -> %s: %s transferred, %s failed",
            plan.source_id,
            plan.target_id,
            len(result.transferred_hashes),
            len(result.failed_hashes),
        )
        return result

    def sync(
        self,
        source: MembraneNode,
        target: MembraneNode,
    ) -> SyncResult:
        """One-shot build + execute sync from *source* to *target*.

        Args:
            source: Source node.
            target: Target node.

        Returns:
            SyncResult.
        """
        plan = self.build_plan(source, target)
        return self.execute_plan(plan, source, target)

    def batch_sync(
        self,
        source: MembraneNode,
        targets: list[MembraneNode],
    ) -> dict[str, SyncResult]:
        """Sync *source* to all *targets* in parallel (sequentially here).

        Args:
            source: Source node.
            targets: Target nodes.

        Returns:
            Mapping target node_id -> SyncResult.
        """
        results: dict[str, SyncResult] = {}
        for target in targets:
            results[target.node_id] = self.sync(source, target)
        return results
