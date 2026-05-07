"""TenantIsolation: cross-tenant sharing policy."""

import logging

logger = logging.getLogger(__name__)


from dataclasses import dataclass

from membrane.fragment import Fragment


@dataclass(frozen=True)
class TenantPolicy:
    """Policy governing what can be shared across tenants.

    Attributes:
        allow_public_prefixes: Whether public/common prefixes are shared.
        allow_tool_traces: Whether tool outputs are shared.
        allow_artifacts: Whether retrieved documents are shared.
        min_reuse_score_for_share: Minimum reuse score to allow cross-tenant.
    """

    allow_public_prefixes: bool = True
    allow_tool_traces: bool = False
    allow_artifacts: bool = True
    min_reuse_score_for_share: float = 0.6


class TenantIsolation:
    """Evaluates whether fragments can be shared across tenants."""

    def __init__(self, policy: TenantPolicy | None = None) -> None:
        """Initialize with optional tenant policy.

        Args:
            policy: Sharing policy rules.
        """
        """Initialize with optional tenant policy.

        Args:
            policy: Sharing policy rules.
        """
        self.policy = policy or TenantPolicy()

    def can_share(
        self,
        fragment: Fragment,
        tenant_a: str,
        tenant_b: str,
    ) -> bool:
        """Determine if a fragment can be shared between two tenants.

        Args:
            fragment: Fragment to evaluate.
            tenant_a: First tenant identifier.
            tenant_b: Second tenant identifier.

        Returns:
            True if sharing is permitted.
        """
        if tenant_a == tenant_b:
            return True

        if fragment.reuse_score < self.policy.min_reuse_score_for_share:
            return False

        model_id = fragment.structural_signature.model_id
        if model_id == "prefix" and not self.policy.allow_public_prefixes:
            return False
        if model_id == "tool" and not self.policy.allow_tool_traces:
            return False
        if model_id == "artifact" and not self.policy.allow_artifacts:
            return False

        return True
