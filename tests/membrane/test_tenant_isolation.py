from tests.conftest import make_fragment

"""Tests for tenant_isolation module."""

import pytest

from membrane.analytical import Isolation, Tenant
from membrane.fragment import Fragment
from membrane.signature import Signature


class TestTenantIsolation:
    """Test suite for Isolation."""

    def test_same_tenant_always_true(self):
        ti = Isolation()
        frag = make_fragment()
        assert ti.can_share(frag, "t1", "t1")

    def test_low_reuse_score_blocks_share(self):
        ti = Isolation()
        frag = make_fragment(reuse_score=0.1)
        assert not ti.can_share(frag, "t1", "t2")

    def test_public_prefixes_allowed_by_default(self):
        ti = Isolation()
        frag = make_fragment(model_id="prefix", reuse_score=0.8)
        assert ti.can_share(frag, "t1", "t2")

    def test_public_prefixes_blocked_when_policy_false(self):
        policy = Tenant(allow_public_prefixes=False)
        ti = Isolation(policy=policy)
        frag = make_fragment(model_id="prefix", reuse_score=0.8)
        assert not ti.can_share(frag, "t1", "t2")

    def test_tool_traces_blocked_by_default(self):
        ti = Isolation()
        frag = make_fragment(model_id="tool", reuse_score=0.8)
        assert not ti.can_share(frag, "t1", "t2")

    def test_tool_traces_allowed_when_policy_true(self):
        policy = Tenant(allow_tool_traces=True)
        ti = Isolation(policy=policy)
        frag = make_fragment(model_id="tool", reuse_score=0.8)
        assert ti.can_share(frag, "t1", "t2")

    def test_artifacts_allowed_by_default(self):
        ti = Isolation()
        frag = make_fragment(model_id="artifact", reuse_score=0.8)
        assert ti.can_share(frag, "t1", "t2")

    def test_artifacts_blocked_when_policy_false(self):
        policy = Tenant(allow_artifacts=False)
        ti = Isolation(policy=policy)
        frag = make_fragment(model_id="artifact", reuse_score=0.8)
        assert not ti.can_share(frag, "t1", "t2")

    def test_default_policy_values(self):
        policy = Tenant()
        assert policy.allow_public_prefixes is True
        assert policy.allow_tool_traces is False
        assert policy.allow_artifacts is True
        assert policy.min_reuse_score_for_share == 0.6
