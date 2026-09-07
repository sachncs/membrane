"""Tests for the per-tenant filter (Phase 3.1.6)."""

from __future__ import annotations

import pytest

from membrane.errors import TenantScopeError
from membrane.fragment import Fragment
from membrane.identity import PayloadIdentity
from membrane.node import Node
from membrane.security import (
    SYSTEM_TENANT,
    TenantAuthorizer,
    can_read_tenant,
    can_write_tenant,
)


def _identity() -> PayloadIdentity:
    return PayloadIdentity(
        payload_hash="h" * 64,
        model_id="m",
        model_revision="",
        tokenizer_name="m",
        tokenizer_revision="",
        layer_range=(0, 1),
        head_range=(-1, -1),
        token_span=(0, 7),
        dtype="float16",
        shape=(1, 1, 1, 8, 64),
    )


class TestCanReadTenant:
    def test_same_tenant(self):
        assert can_read_tenant("acme", "acme") is True

    def test_different_tenant_rejected_by_default(self):
        assert can_read_tenant("acme", "globex") is False

    @pytest.mark.parametrize("caller", ["acme", "globex"])
    def test_public_is_readable_to_everyone(self, caller):
        assert can_read_tenant(caller, SYSTEM_TENANT) is True

    def test_public_readable_can_be_disabled(self):
        assert can_read_tenant("acme", SYSTEM_TENANT, public_readable=False) is False

    def test_empty_tenant_args_rejected(self):
        assert can_read_tenant("", "acme") is False
        assert can_read_tenant("acme", "") is False


class TestCanWriteTenant:
    def test_same_tenant(self):
        assert can_write_tenant("acme", "acme") is True

    def test_different_tenant_rejected(self):
        assert can_write_tenant("acme", "globex") is False

    @pytest.mark.parametrize("caller", ["acme", "globex", ""])
    def test_public_is_not_writable_by_default(self, caller):
        assert can_write_tenant(caller, SYSTEM_TENANT) is False

    def test_public_can_be_writable(self):
        assert can_write_tenant("acme", SYSTEM_TENANT, public_writable=True) is True


class TestTenantAuthorizer:
    def test_admin_can_read_any_tenant(self):
        auth = TenantAuthorizer(caller_tenant="ops", scopes=frozenset({"admin"}))
        assert auth.is_admin() is True
        auth.authorize_read("acme")  # should not raise
        auth.authorize_read("globex")

    def test_admin_can_write_any_tenant(self):
        auth = TenantAuthorizer(caller_tenant="ops", scopes=frozenset({"admin"}))
        auth.authorize_write("acme")
        auth.authorize_write("globex")

    def test_non_admin_can_read_own_tenant(self):
        auth = TenantAuthorizer(caller_tenant="acme", scopes=frozenset({"read"}))
        auth.authorize_read("acme")

    def test_non_admin_cannot_read_other_tenant(self):
        auth = TenantAuthorizer(caller_tenant="acme", scopes=frozenset({"read"}))
        with pytest.raises(TenantScopeError, match="cannot read"):
            auth.authorize_read("globex")

    def test_non_admin_can_read_public(self):
        auth = TenantAuthorizer(caller_tenant="acme", scopes=frozenset({"read"}))
        auth.authorize_read(SYSTEM_TENANT)

    def test_non_admin_cannot_write_public(self):
        auth = TenantAuthorizer(caller_tenant="acme", scopes=frozenset({"write"}))
        with pytest.raises(TenantScopeError, match="cannot write"):
            auth.authorize_write(SYSTEM_TENANT)

    def test_non_admin_can_write_own_tenant(self):
        auth = TenantAuthorizer(caller_tenant="acme", scopes=frozenset({"write"}))
        auth.authorize_write("acme")

    def test_non_admin_cannot_write_other_tenant(self):
        auth = TenantAuthorizer(caller_tenant="acme", scopes=frozenset({"write"}))
        with pytest.raises(TenantScopeError, match="cannot write"):
            auth.authorize_write("globex")


class TestNodeStore:
    def _frag(self, tenant: str = "acme") -> Fragment:
        return Fragment(
            identity=_identity(),
            payload_ref=None,
            payload_size=10,
            ttl=60.0,
            reuse_score=0.5,
            version_id=1,
            tenant_id=tenant,
        )

    def test_store_same_tenant_succeeds(self):
        node = Node(node_id="n1", max_memory_bytes=1024)
        frag = self._frag(tenant="acme")
        assert node.store(frag, caller_tenant="acme", caller_scopes=frozenset({"write"})) is True

    def test_store_other_tenant_rejected(self):
        node = Node(node_id="n1", max_memory_bytes=1024)
        frag = self._frag(tenant="acme")
        with pytest.raises(TenantScopeError):
            node.store(frag, caller_tenant="globex", caller_scopes=frozenset({"write"}))

    def test_store_other_tenant_admin_allowed(self):
        node = Node(node_id="n1", max_memory_bytes=1024)
        frag = self._frag(tenant="acme")
        assert (
            node.store(
                frag, caller_tenant="ops", caller_scopes=frozenset({"admin"})
            )
            is True
        )

    def test_store_public_requires_admin(self):
        node = Node(node_id="n1", max_memory_bytes=1024)
        # A non-admin caller cannot write a fragment in the system tenant.
        frag = self._frag(tenant=SYSTEM_TENANT)
        with pytest.raises(TenantScopeError):
            node.store(
                frag, caller_tenant="acme", caller_scopes=frozenset({"write"})
            )

    def test_store_no_caller_bypasses(self):
        """A caller with no tenant id bypasses the check (single-node / test)."""
        node = Node(node_id="n1", max_memory_bytes=1024)
        frag = self._frag(tenant="acme")
        assert node.store(frag) is True


class TestNodeRetrieve:
    def _store(self, node: Node, tenant: str) -> Fragment:
        frag = Fragment(
            identity=_identity(),
            payload_ref=None,
            payload_size=10,
            ttl=60.0,
            reuse_score=0.5,
            version_id=1,
            tenant_id=tenant,
        )
        node.store(frag)
        return frag

    def test_retrieve_same_tenant(self):
        node = Node(node_id="n1", max_memory_bytes=1024)
        frag = self._store(node, tenant="acme")
        result = node.retrieve(
            frag.identity.payload_hash,
            caller_tenant="acme",
            caller_scopes=frozenset({"read"}),
        )
        assert result is not None

    def test_retrieve_other_tenant_returns_none(self):
        node = Node(node_id="n1", max_memory_bytes=1024)
        frag = self._store(node, tenant="acme")
        result = node.retrieve(
            frag.identity.payload_hash,
            caller_tenant="globex",
            caller_scopes=frozenset({"read"}),
        )
        assert result is None

    def test_retrieve_admin_can_read_any(self):
        node = Node(node_id="n1", max_memory_bytes=1024)
        frag = self._store(node, tenant="acme")
        result = node.retrieve(
            frag.identity.payload_hash,
            caller_tenant="ops",
            caller_scopes=frozenset({"admin"}),
        )
        assert result is not None

    def test_retrieve_public_for_everyone(self):
        node = Node(node_id="n1", max_memory_bytes=1024)
        frag = self._store(node, tenant=SYSTEM_TENANT)
        result = node.retrieve(
            frag.identity.payload_hash,
            caller_tenant="acme",
            caller_scopes=frozenset({"read"}),
        )
        assert result is not None

    def test_retrieve_no_caller_bypasses(self):
        node = Node(node_id="n1", max_memory_bytes=1024)
        frag = self._store(node, tenant="acme")
        result = node.retrieve(frag.identity.payload_hash)
        assert result is not None
