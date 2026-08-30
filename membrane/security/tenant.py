"""Tenant scope authorization (Phase 3.1.6).

The v2.0 release carried :class:`~membrane.analytical.Tenant` only
as a cross-tenant sharing-policy dataclass with no callers in
production. The v3.0.0 release promotes the tenant id to a
first-class field on :class:`~membrane.fragment.Fragment` and
enforces per-tenant access on every store / retrieve /
replicate op.

Rules:

* The system tenant ``"public"`` is readable by every caller.
* The system tenant is writable only by callers holding the
  ``"admin"`` scope (i.e., the cluster operator). Non-admin
  callers can still write to their own tenant or to any other
  tenant via the explicit ACL helpers below.
* A caller with the ``"admin"`` scope can read and write any
  tenant; this is the cluster operator escape hatch.
* A non-admin caller can read and write only their own tenant.

The :class:`TenantAuthorizer` dataclass encapsulates the per-op
check so the same logic runs in the HTTP op layer, the gRPC
layer, and the internal store path.
"""

from __future__ import annotations

from dataclasses import dataclass

from membrane.errors import TenantScopeError

SYSTEM_TENANT: str = "public"
"""The system tenant. Writable only by admin scope."""


def has_admin_scope(scopes: frozenset[str]) -> bool:
    """Return True if ``scopes`` grants the admin scope.

    Args:
        scopes: The caller's granted scopes.

    Returns:
        bool: ``True`` if ``"admin"`` is in scopes.
    """
    return "admin" in scopes


@dataclass(frozen=True)
class TenantAuthorizer:
    """Per-op tenant authorization policy.

    Attributes:
        caller_tenant: The tenant id the caller is associated
            with. Empty string means "unauthenticated".
        scopes: The caller's granted scopes.
        public_readable: When ``True`` (the default), every
            caller can read the system tenant.
        public_writable: When ``False`` (the default), only
            admin can write the system tenant.
    """

    caller_tenant: str
    scopes: frozenset[str]
    public_readable: bool = True
    public_writable: bool = False

    def is_admin(self) -> bool:
        """Return True when the caller has admin scope."""
        return has_admin_scope(self.scopes)

    def authorize_read(self, fragment_tenant: str) -> None:
        """Raise :class:`TenantScopeError` if the caller may not read the fragment.

        Args:
            fragment_tenant: The tenant id on the fragment.

        Raises:
            TenantScopeError: When the caller's tenant does
                not match the fragment's tenant and the caller
                is not admin.
        """
        if self.is_admin():
            return
        if can_read_tenant(
            self.caller_tenant,
            fragment_tenant,
            public_readable=self.public_readable,
        ):
            return
        raise TenantScopeError(
            f"caller tenant {self.caller_tenant!r} cannot read fragment in tenant {fragment_tenant!r}"
        )

    def authorize_write(self, fragment_tenant: str) -> None:
        """Raise :class:`TenantScopeError` if the caller may not write the fragment.

        Args:
            fragment_tenant: The tenant id the fragment will
                carry.

        Raises:
            TenantScopeError: When the caller's tenant does
                not match the fragment's tenant and the caller
                is not admin.
        """
        if self.is_admin():
            return
        if can_write_tenant(
            self.caller_tenant,
            fragment_tenant,
            public_writable=self.public_writable,
        ):
            return
        raise TenantScopeError(
            f"caller tenant {self.caller_tenant!r} cannot write fragment in tenant {fragment_tenant!r}"
        )


def can_read_tenant(
    caller_tenant: str,
    fragment_tenant: str,
    public_readable: bool = True,
) -> bool:
    """Return True if ``caller_tenant`` may read ``fragment_tenant``.

    Args:
        caller_tenant: The caller's tenant id; empty string
            means "no tenant".
        fragment_tenant: The fragment's tenant id.
        public_readable: When ``True`` the system tenant is
            readable by every caller.

    Returns:
        bool: True when the read is permitted.
    """
    if not fragment_tenant or not caller_tenant:
        return False
    return fragment_tenant == caller_tenant or (
        public_readable and fragment_tenant == SYSTEM_TENANT
    )


def can_write_tenant(
    caller_tenant: str,
    fragment_tenant: str,
    public_writable: bool = False,
) -> bool:
    """Return True if ``caller_tenant`` may write ``fragment_tenant``.

    Args:
        caller_tenant: The caller's tenant id.
        fragment_tenant: The fragment's tenant id.
        public_writable: When ``True`` (an explicit ACL grant)
            non-admin callers may write the system tenant.

    Returns:
        bool: True when the write is permitted.
    """
    if not fragment_tenant or not caller_tenant:
        return False
    return fragment_tenant == caller_tenant or (
        public_writable and fragment_tenant == SYSTEM_TENANT
    )


__all__ = [
    "SYSTEM_TENANT",
    "TenantAuthorizer",
    "can_read_tenant",
    "can_write_tenant",
    "has_admin_scope",
]
