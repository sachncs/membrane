"""Tests for the /admin/* HTTP surface (Phase 3.2.6)."""

from __future__ import annotations

import pytest


class TestAdminRouterMounts:
    def test_admin_router_registers_endpoints(self):
        from fastapi import FastAPI

        from membrane.transport.admin import create_admin_router

        app = FastAPI()
        router = create_admin_router()
        app.include_router(router)
        # Inspect the source router's declared routes directly;
        # the FastAPI mount wraps the inner router in an
        # _IncludedRouter whose routes are not iterable as
        # ``route.path``.
        paths = {route.path for route in router.routes}
        assert "/admin/fragments/{content_hash}" in paths
        assert "/admin/placement" in paths
        assert "/admin/evict" in paths
        assert "/admin/repair" in paths
        assert "/admin/policy" in paths


class TestAdminRoutesAreAdminScoped:
    def test_scope_check_runs_for_admin_routes(self):
        from membrane.transport.authz import required_scope

        for path in (
            "/fragments/{content_hash}",
            "/placement",
            "/evict",
            "/repair",
            "/policy",
        ):
            for method in ("GET", "POST"):
                scope = required_scope(method, f"/admin{path}")
                assert scope == "admin", f"{method} /admin{path} should be admin-scoped"

    def test_read_only_key_is_rejected(self):
        from membrane.auth import AuthBackendError
        from membrane.auth.apikey import APIKeyAuthenticator
        from membrane.transport.authz import enforce_route_scope

        auth = APIKeyAuthenticator(keyfile_text="ro:reader1:read\n")
        with pytest.raises(AuthBackendError):
            enforce_route_scope(
                auth, "POST", "/admin/placement", headers={"authorization": "Bearer ro"}
            )
