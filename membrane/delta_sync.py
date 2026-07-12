"""DeltaSync: inventory-based delta synchronization between nodes.

Builds on :class:`~membrane.transfer_service.TransferService` to
provide efficient, version-aware batch synchronization. Computes
the minimal set of fragments that need to move between two nodes,
schedules batch transfers, and tracks sync progress.

A sync plan distinguishes two kinds of divergence:

* **Missing** — the target has never seen the fragment.
* **Outdated** — the target holds an older ``version_id`` and
  needs to be refreshed.

Both classes contribute to the same estimated byte count, and
:meth:`DeltaSync.execute_plan` transfers them in a single pass.

Thread safety:
    The class itself is stateless; thread safety depends on the
    underlying :class:`TransferService` and the
    :class:`MembraneNode` references passed in.
"""

import logging

logger = logging.getLogger(__name__)


from dataclasses import dataclass, field

from membrane.membrane_node import MembraneNode
from membrane.transfer_service import TransferService


@dataclass
class SyncPlan:
    """Planned synchronization between two nodes.

    Attributes:
        source_id: Source node identifier.
        target_id: Target node identifier.
        missing_hashes: Content hashes the target is missing
            entirely.
        outdated_hashes: Content hashes the target has but with
            an older ``version_id`` than the source.
        estimated_bytes: Approximate total bytes that will be
            transferred if the plan is executed as-is.
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
        bytes_transferred: Total bytes moved across the wire.
    """

    transferred_hashes: list[str] = field(default_factory=list)
    failed_hashes: list[str] = field(default_factory=list)
    bytes_transferred: int = 0


class DeltaSync:
    """Version-aware delta synchronization engine.

    Args:
        transfer_service: :class:`TransferService` for moving
            fragments.
    """

    def __init__(self, transfer_service: TransferService | None = None) -> None:
        """Initialize the engine with an optional transfer service.

        Args:
            transfer_service: Service used to move fragments.
                A default :class:`TransferService` is created
                when ``None``.
        """
        self.transfer_service = transfer_service or TransferService()

    def build_plan(
        self,
        source: MembraneNode,
        target: MembraneNode,
    ) -> SyncPlan:
        """Compute the minimal sync plan from ``source`` to ``target``.

        Compares the per-node inventory digests (a mapping from
        ``content_hash`` to ``version_id``) returned by
        :meth:`TransferService.inventory_digest`. Every hash that
        is missing on the target, or older than the source, is
        added to the plan together with an estimated byte cost.

        Args:
            source: Source node (canonical copy).
            target: Target node (may be missing or outdated).

        Returns:
            SyncPlan: Description of the required transfers.
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

        Iterates over the union of ``missing_hashes`` and
        ``outdated_hashes`` and attempts to transfer each one.
        Hashes whose source fragment has been removed between
        plan construction and execution are recorded as failures
        rather than raising.

        Args:
            plan: SyncPlan to execute.
            source: Source node.
            target: Target node.

        Returns:
            SyncResult: Per-hash transfer outcomes and the total
            byte count.
        """
        result = SyncResult()

        all_hashes = plan.missing_hashes + plan.outdated_hashes
        for h in all_hashes:
            frag = source.fragments.get(h)
            if frag is None:
                # Plan was built against a stale snapshot; record
                # the hash as failed and continue.
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
        """One-shot build + execute sync from ``source`` to ``target``.

        Args:
            source: Source node.
            target: Target node.

        Returns:
            SyncResult: Result of executing the freshly built
            plan.
        """
        plan = self.build_plan(source, target)
        return self.execute_plan(plan, source, target)

    def batch_sync(
        self,
        source: MembraneNode,
        targets: list[MembraneNode],
    ) -> dict[str, SyncResult]:
        """Sync ``source`` to every ``target`` in sequence.

        The current implementation runs synchronously, one target
        at a time. A future revision could parallelize the
        per-target syncs using ``concurrent.futures``.

        Args:
            source: Source node.
            targets: Target nodes.

        Returns:
            dict[str, SyncResult]: Mapping from ``target.node_id``
            to the corresponding sync result.
        """
        results: dict[str, SyncResult] = {}
        for target in targets:
            results[target.node_id] = self.sync(source, target)
        return results
