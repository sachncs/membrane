"""Per-tenant access checks for node reads and writes.

An empty caller tenant means authentication is off and every check
passes. Otherwise the rules of :class:`~membrane.security.tenant.TenantAuthorizer`
apply: a caller reads and writes its own tenant, ``admin`` reaches all
tenants, and the system tenant is readable by everyone.
"""

from membrane.errors import TenantScopeError
from membrane.runtime.concurrency import share
from membrane.security.tenant import TenantAuthorizer

#: The scopes of a caller without any (the default argument of node reads
#: and writes). One shared object, switched to deferred reference counting,
#: so request threads do not contend on a default argument's count.
NO_SCOPES: frozenset[str] = frozenset()
share(NO_SCOPES)


class TenantGuard:
    """Applies the tenant rules to one request."""

    @staticmethod
    def check_write(fragment_tenant: str, caller_tenant: str, caller_scopes: frozenset[str]) -> None:
        """Refuse a write into another tenant.

        Args:
            fragment_tenant: Tenant of the fragment being written.
            caller_tenant: Caller's tenant (empty when authentication is off).
            caller_scopes: Caller's scopes.

        Raises:
            TenantScopeError: When the caller may not write ``fragment_tenant``.
        """
        if caller_tenant:
            TenantAuthorizer(caller_tenant=caller_tenant, scopes=caller_scopes).authorize_write(fragment_tenant)

    @staticmethod
    def can_read(fragment_tenant: str, caller_tenant: str, caller_scopes: frozenset[str]) -> bool:
        """Whether the caller may read a fragment of ``fragment_tenant``.

        Args:
            fragment_tenant: Tenant of the fragment.
            caller_tenant: Caller's tenant (empty when authentication is off).
            caller_scopes: Caller's scopes.

        Returns:
            bool: True when the read is allowed. A forbidden read looks like
            a miss to the caller.
        """
        if not caller_tenant:
            return True
        try:
            TenantAuthorizer(caller_tenant=caller_tenant, scopes=caller_scopes).authorize_read(fragment_tenant)
        except TenantScopeError:
            return False
        return True


__all__ = ["NO_SCOPES", "TenantGuard"]
