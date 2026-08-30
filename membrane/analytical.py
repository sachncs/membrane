"""Analytical classes: routing/decision/prediction utilities.

This package-level consolidation hosts the routing, prediction,
and workload-analytics classes that used to live in their own
modules (``economic.py``, ``latency.py``, ``joint.py``,
``selector.py``, ``offload.py``, ``policy.py``, ``predict.py``,
``workload.py``, ``roles.py``, plus their config dataclasses
and the ``isolation.py`` cross-tenant policy).

The classes are intentionally grouped together so callers can
discover the full set of analytical helpers in one place. Each
class is implemented in its own concern; ``membrane.analytical``
exposes them as a flat surface.

Backwards compatibility:

The original single-class modules are re-exported by name. New
code should prefer importing from ``membrane.analytical``.
"""

import logging

logger = logging.getLogger(__name__)


from dataclasses import dataclass

from membrane.economic import Economic, EconomicRouterConfig
from membrane.fragment import Fragment
from membrane.fragment_kind import FragmentKind
from membrane.joint import Joint, PlacementDecision
from membrane.latency import Latency
from membrane.offload import Offload, OffloadConfig, OffloadResult
from membrane.policy import Promotion, PromotionConfig, PromotionResult
from membrane.predict import Predict
from membrane.roles import NodeRole, Roles, SystemState
from membrane.selector import Selector, SelectorConfig
from membrane.workload import Workload


@dataclass(frozen=True)
class Tenant:
    """Cross-tenant sharing policy.

    Attributes:
        allow_public_prefixes: Whether fragments produced for
            public/common prefixes (``FragmentKind.PREFIX``) may
            be shared across tenants.
        allow_tool_traces: Whether tool traces
            (``FragmentKind.TRACE``) may be shared.
        allow_artifacts: Whether retrieved-document artifacts
            (``FragmentKind.ARTIFACT``) may be shared.
        min_reuse_score_for_share: Minimum ``reuse_score``
            required before any cross-tenant sharing is allowed.
    """

    allow_public_prefixes: bool = True
    allow_tool_traces: bool = False
    allow_artifacts: bool = True
    min_reuse_score_for_share: float = 0.6


class Isolation:
    """Evaluates whether fragments can be shared across tenants."""

    def __init__(self, policy: Tenant | None = None) -> None:
        """Initialize with an optional tenant policy."""
        self.policy = policy or Tenant()

    def can_share(
        self,
        fragment: Fragment,
        tenant_a: str,
        tenant_b: str,
    ) -> bool:
        """Determine if ``fragment`` can be shared between two tenants.

        Same-tenant sharing is always permitted. Cross-tenant
        sharing requires:

        1. ``fragment.reuse_score >= policy.min_reuse_score_for_share``.
        2. The fragment's ``kind`` is not blocked by the
           policy (``PREFIX``, ``TRACE``, ``ARTIFACT`` switches).

        Args:
            fragment: Fragment to evaluate.
            tenant_a: First tenant identifier.
            tenant_b: Second tenant identifier.

        Returns:
            bool: True if sharing is permitted.
        """
        if tenant_a == tenant_b:
            return True
        if fragment.reuse_score < self.policy.min_reuse_score_for_share:
            return False

        kind = fragment.structural_signature.model_id
        return not (
            (kind == FragmentKind.PREFIX and not self.policy.allow_public_prefixes)
            or (kind == FragmentKind.TRACE and not self.policy.allow_tool_traces)
            or (kind == FragmentKind.ARTIFACT and not self.policy.allow_artifacts)
        )


__all__ = [
    "Economic",
    "EconomicRouterConfig",
    "Isolation",
    "Joint",
    "Latency",
    "NodeRole",
    "Offload",
    "OffloadConfig",
    "OffloadResult",
    "PlacementDecision",
    "Predict",
    "Promotion",
    "PromotionConfig",
    "PromotionResult",
    "Roles",
    "Selector",
    "SelectorConfig",
    "SystemState",
    "Tenant",
    "Workload",
]
