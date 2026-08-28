"""TenantIsolation: cross-tenant sharing policy.

This module defines :class:`TenantIsolation` and its supporting
:class:`TenantPolicy` dataclass. Together they answer the question
"is it safe for tenant A's request to benefit from a fragment that
tenant B produced?" — a question that must be answered before any
cross-tenant deduplication, prefetch, or replication is allowed.

The policy is intentionally simple: it distinguishes three
"memory types" via the fragment's synthetic ``model_id`` (set by
the various ``materialize`` methods):

* ``"prefix"`` — public token sequences (e.g., system prompts).
* ``"tool"`` — tool invocation traces, which may contain
  user-specific output.
* ``"artifact"`` — retrieved documents, typically safe to share
  but policy-controlled.

In addition, every cross-tenant share must clear a minimum
``reuse_score`` threshold so that only confidently-reusable
fragments are exposed.

Thread safety:
    The class is **stateless beyond the policy**; instances can be
    safely shared across threads.
"""

import logging

logger = logging.getLogger(__name__)


from dataclasses import dataclass

from membrane.fragment import Fragment


@dataclass(frozen=True)
class TenantPolicy:
    """Policy governing what can be shared across tenants.

    Attributes:
        allow_public_prefixes: Whether fragments produced for
            public/common prefixes (``model_id == "prefix"``) may
            be shared across tenants.
        allow_tool_traces: Whether tool traces
            (``model_id == "tool"``) may be shared.
        allow_artifacts: Whether retrieved-document artifacts
            (``model_id == "artifact"``) may be shared.
        min_reuse_score_for_share: Minimum ``reuse_score``
            required before any cross-tenant sharing is allowed.
    """

    allow_public_prefixes: bool = True
    allow_tool_traces: bool = False
    allow_artifacts: bool = True
    min_reuse_score_for_share: float = 0.6


class TenantIsolation:
    """Evaluates whether fragments can be shared across tenants."""

    def __init__(self, policy: TenantPolicy | None = None) -> None:
        """Initialize with an optional tenant policy.

        Args:
            policy: Sharing policy rules. A default
                :class:`TenantPolicy` is used when ``None``.
        """
        self.policy = policy or TenantPolicy()

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
        2. The fragment's ``model_id`` is not blocked by the
           policy (``prefix``, ``tool``, ``artifact`` switches).

        Args:
            fragment: Fragment to evaluate.
            tenant_a: First tenant identifier.
            tenant_b: Second tenant identifier.

        Returns:
            bool: True if sharing is permitted.
        """
        if tenant_a == tenant_b:
            # Same-tenant sharing is always allowed.
            return True

        if fragment.reuse_score < self.policy.min_reuse_score_for_share:
            # Below the confidence threshold — refuse.
            return False

        # Each synthetic model_id has its own switch in the
        # policy. Unknown ids fall through to "allowed" because
        # they were not explicitly opted out of.
        model_id = fragment.structural_signature.model_id
        return not (
            (model_id == "prefix" and not self.policy.allow_public_prefixes)
            or (model_id == "tool" and not self.policy.allow_tool_traces)
            or (model_id == "artifact" and not self.policy.allow_artifacts)
        )
