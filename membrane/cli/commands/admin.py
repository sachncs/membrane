"""Admin CLI subcommands (Phase 3.2.7).

The v2.0 release exposed only ``serve``, ``dashboard``,
``cluster-status``, ``llm-status``, and ``config``. The
v3.0.0 release adds an ``admin`` subcommand that talks to the
``/admin/*`` HTTP surface (Phase 3.2.6):

* ``membrane admin inspect <hash>``
* ``membrane admin placement <hash> <node>``
* ``membrane admin evict <hash>``
* ``membrane admin repair <peer>``
* ``membrane admin policy [--min-reuse-score N] [--demand-threshold N]``

Each subcommand requires the ``admin`` scope (carried via the
``--api-key`` flag or the ``MEMBRANE_API_KEY`` env var).
"""

from __future__ import annotations

import json
import os
from typing import Any

import httpx
import typer

admin_app = typer.Typer(help="Admin operations against a running Membrane node.")


def _base_url(host: str, port: int) -> str:
    return f"http://{host}:{port}"


def _headers(api_key: str | None) -> dict[str, str]:
    if api_key:
        return {"authorization": f"Bearer {api_key}"}
    return {}


@admin_app.command("inspect")
def admin_inspect(
    content_hash: str = typer.Argument(...),
    host: str = typer.Option("localhost", "--host"),
    port: int = typer.Option(8080, "--port"),
    api_key: str | None = typer.Option(None, "--api-key"),
) -> None:
    """Inspect a fragment by content hash."""
    headers = _headers(api_key or os.environ.get("MEMBRANE_API_KEY"))
    url = f"{_base_url(host, port)}/admin/fragments/{content_hash}"
    resp = httpx.get(url, headers=headers, timeout=5.0)
    typer.echo(f"{resp.status_code} {resp.text}")


@admin_app.command("placement")
def admin_placement(
    content_hash: str = typer.Argument(...),
    primary_node_id: str = typer.Argument(...),
    host: str = typer.Option("localhost", "--host"),
    port: int = typer.Option(8080, "--port"),
    api_key: str | None = typer.Option(None, "--api-key"),
) -> None:
    """Override the primary node for a shard."""
    headers = _headers(api_key or os.environ.get("MEMBRANE_API_KEY"))
    url = f"{_base_url(host, port)}/admin/placement"
    body: dict[str, Any] = {
        "content_hash": content_hash,
        "primary_node_id": primary_node_id,
    }
    resp = httpx.post(url, json=body, headers=headers, timeout=5.0)
    typer.echo(f"{resp.status_code} {resp.text}")


@admin_app.command("evict")
def admin_evict(
    content_hash: str = typer.Argument(...),
    host: str = typer.Option("localhost", "--host"),
    port: int = typer.Option(8080, "--port"),
    api_key: str | None = typer.Option(None, "--api-key"),
) -> None:
    """Manually evict a fragment."""
    headers = _headers(api_key or os.environ.get("MEMBRANE_API_KEY"))
    url = f"{_base_url(host, port)}/admin/evict"
    body = {"content_hash": content_hash}
    resp = httpx.post(url, json=body, headers=headers, timeout=5.0)
    typer.echo(f"{resp.status_code} {resp.text}")


@admin_app.command("repair")
def admin_repair(
    peer_node_id: str = typer.Argument(...),
    host: str = typer.Option("localhost", "--host"),
    port: int = typer.Option(8080, "--port"),
    api_key: str | None = typer.Option(None, "--api-key"),
) -> None:
    """Trigger a repair for a peer."""
    headers = _headers(api_key or os.environ.get("MEMBRANE_API_KEY"))
    url = f"{_base_url(host, port)}/admin/repair"
    body = {"peer_node_id": peer_node_id}
    resp = httpx.post(url, json=body, headers=headers, timeout=5.0)
    typer.echo(f"{resp.status_code} {resp.text}")


@admin_app.command("policy")
def admin_policy(
    min_reuse_score: float | None = typer.Option(None, "--min-reuse-score"),
    demand_threshold: int | None = typer.Option(None, "--demand-threshold"),
    host: str = typer.Option("localhost", "--host"),
    port: int = typer.Option(8080, "--port"),
    api_key: str | None = typer.Option(None, "--api-key"),
) -> None:
    """Read or update the Promotion knobs."""
    headers = _headers(api_key or os.environ.get("MEMBRANE_API_KEY"))
    url = f"{_base_url(host, port)}/admin/policy"
    if min_reuse_score is None and demand_threshold is None:
        resp = httpx.get(url, headers=headers, timeout=5.0)
    else:
        body = {
            "min_reuse_score": min_reuse_score if min_reuse_score is not None else 0.0,
            "demand_threshold": demand_threshold if demand_threshold is not None else 0,
        }
        resp = httpx.post(url, json=body, headers=headers, timeout=5.0)
    typer.echo(json.dumps(resp.json(), indent=2))


__all__ = ["admin_app", "main"]


def main() -> Any:
    """Entry point for ``membrane admin``."""
    return admin_app()


if __name__ == "__main__":
    main()
