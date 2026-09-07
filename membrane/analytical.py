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


from collections import Counter
from dataclasses import dataclass

from membrane.economic import Economic, EconomicRouterConfig
from membrane.fragment import Fragment
from membrane.fragment_kind import FragmentKind
from membrane.joint import Joint, PlacementDecision
from membrane.latency import Latency
from membrane.model.profiler import kv_size
from membrane.node import Node
from membrane.offload import Offload, OffloadConfig, OffloadResult
from membrane.policy import Promotion, PromotionConfig, PromotionResult
from membrane.roles import NodeRole, Roles, SystemState
from membrane.selector import Selector, SelectorConfig


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

        kind = fragment.identity.model_id
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

class Predict:
    """Lightweight heuristic predictor for KV size, reuse probability, and optimal region.

    Attributes:
        kv_size_bias: Multiplicative bias applied to KV size
            estimates, useful for inflating or deflating the
            expected footprint to match a specific deployment.
    """

    def __init__(self, kv_size_bias: float = 1.0) -> None:
        """Initialize the predictor."""
        self.kv_size_bias = kv_size_bias

    def predict_kv_size(self, prompt_tokens: list[int]) -> float:
        """Predict KV cache size for a prompt."""
        return kv_size(len(prompt_tokens)) * self.kv_size_bias

    def predict_reuse_probability(
        self,
        content_hash: str,
        session_history: list[str],
    ) -> float:
        """Predict likelihood of reuse based on session history."""
        if not session_history:
            return 0.0
        recent = session_history[-10:]
        count = recent.count(content_hash)
        return min(1.0, count / len(recent))

    def predict_optimal_region(
        self,
        prompt_tokens: list[int],
        nodes: list[Node],
    ) -> str:
        """Predict the optimal node for a prompt based on heartbeat load."""
        if not nodes:
            return ""
        best = min(nodes, key=lambda node: node.heartbeat())
        return best.node_id


class Workload:
    """Analyzes access logs to detect repeated prefix patterns."""

    def __init__(self) -> None:
        """Initialize the analyzer."""

    def analyze_patterns(self, access_log: list[str]) -> dict[str, float]:
        """Compute a content_hash -> normalized frequency map."""
        if not access_log:
            return {}
        counts = Counter(access_log)
        total = len(access_log)
        return {h: count / total for h, count in counts.items()}

    def top_patterns(
        self,
        access_log: list[str],
        k: int = 5,
    ) -> list[tuple[str, float]]:
        """Return the top-k most frequent patterns."""
        frequencies = self.analyze_patterns(access_log)
        sorted_items = sorted(
            frequencies.items(), key=lambda item: item[1], reverse=True
        )
        return sorted_items[:k]

    def reuse_ratio(self, access_log: list[str]) -> float:
        """Fraction of accesses that repeat an earlier hash."""
        if not access_log:
            return 0.0
        unique = len(set(access_log))
        total = len(access_log)
        return (total - unique) / total if total > unique else 0.0

