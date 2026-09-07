"""Tier migration on eviction (Phase 3.5.6 follow-up).

The v3.0.0 release ships a :class:`TierMigration` helper that
demotes fragments from one tier to another on eviction.
Production deployments use this to flow cold fragments to a
warm tier (e.g. LMCache) when the local node's hot tier is
under memory pressure.

The :func:`on_evict` callback on :class:`Node` is invoked
when the eviction policy evicts a fragment; the helper
attaches the tier machinery to that callback so operators
get a one-line wiring point.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field

from membrane.tiers import TierPolicy, select_tier

logger = logging.getLogger(__name__)


@dataclass
class TierMigration:
    """Routes evicted fragments to a down-stream tier.

    Attributes:
        policy: The :class:`TierPolicy` driving the assignment.
        on_demote: Callable invoked with ``(fragment, tier_name)``
            for every demoted fragment. Production deployments
            attach a callable that moves the fragment to the
            down-stream tier (e.g. LMCache).
    """

    policy: TierPolicy
    on_demote: Callable[[object, str], None] | None = None
    demotions: int = field(default=0, init=False)
    _seen: set[str] = field(default_factory=set, init=False)

    def on_evict(self, fragment: object) -> str | None:
        """Demote ``fragment`` to its assigned tier.

        Args:
            fragment: The :class:`membrane.fragment.Fragment`
                that the eviction policy just removed.

        Returns:
            str | None: The tier name (when :attr:`on_demote` is
            configured), or ``None`` when no downstream callback
            is attached.
        """
        tier = select_tier(self.policy, fragment)
        if self.on_demote is None:
            return None
        # Avoid double-processing the same fragment within a
        # single eviction pass.
        ident = getattr(getattr(fragment, "identity", None), "payload_hash", None)
        if ident in self._seen:
            return tier
        if ident is not None:
            self._seen.add(ident)
        self.on_demote(fragment, tier)
        self.demotions += 1
        logger.debug("Demoted fragment %s to tier %s", ident, tier)
        return tier

    def reset_seen(self) -> None:
        """Reset the de-duplication set so a new eviction pass can
        re-process the same fragment."""
        self._seen.clear()


def install_default_demote(node: object, migration: TierMigration) -> None:
    """Install ``migration.on_evict`` as a node callback.

    Args:
        node: A :class:`membrane.node.Node` instance.
        migration: The :class:`TierMigration` to attach.
    """
    if not hasattr(node, "add_eviction_callback"):
        return  # backwards-compatible with test stubs
    node.add_eviction_callback(migration.on_evict)


__all__ = ["TierMigration", "install_default_demote"]
