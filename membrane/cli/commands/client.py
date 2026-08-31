"""`membrane client` CLI subcommand (Phase 3.6.1 follow-up).

The Phase 3.6.1 commit shipped the typed ``MembraneClient``;
the v2.0 CLI never exposed a parity surface. This commit adds
``membrane client`` as a Typer subcommand for one-off
interactions with a running Membrane server:

* ``membrane client store`` -- POST a fragment.
* ``membrane client retrieve --hash <hash>`` -- GET a fragment.
* ``membrane client inventory`` -- GET the inventory digest.
* ``membrane client prefill --prompt-tokens 1 2 3`` -- run prefill.

Each subcommand takes a ``--base-url`` (default
``http://localhost:8080``) and an ``--api-key`` for bearer
auth. The CLI is intentionally thin: it does not maintain
state or perform retries; operators use the Python client
for advanced flows.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Sequence  # noqa: F401

import typer

from membrane.client import MembraneClient, MembraneClientError

client_app = typer.Typer(
    name="client",
    help="One-off interactions with a running Membrane server.",
    no_args_is_help=True,
)


def _build_client(base_url: str, api_key: str) -> MembraneClient:
    """Construct a MembraneClient from CLI flags.

    Args:
        base_url: Server URL.
        api_key: Optional bearer token.

    Returns:
        MembraneClient: A fresh client instance.
    """
    import httpx

    return MembraneClient(
        base_url=base_url,
        api_key=api_key,
        transport=httpx.Client(timeout=10.0),
    )


def _emit(payload: dict | list | None) -> None:
    """Pretty-print ``payload`` to stdout as JSON.

    Args:
        payload: The result to print; ``None`` prints an empty
            object.
    """
    if payload is None:
        payload = {}
    json.dump(payload, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")


@client_app.command("store")
def client_store(
    body: str = typer.Option(
        ...,
        "--body",
        help="Wire-format fragment dict as a JSON string.",
    ),
    base_url: str = typer.Option("http://localhost:8080", "--base-url"),
    api_key: str = typer.Option("", "--api-key"),
    is_primary: bool = typer.Option(False, "--primary/--no-primary"),
) -> None:
    """POST a fragment to ``/store``."""
    import httpx

    try:
        payload = json.loads(body)
    except json.JSONDecodeError as exc:
        typer.echo(f"error: invalid JSON: {exc}", err=True)
        raise typer.Exit(code=1) from None

    client = MembraneClient(
        base_url=base_url,
        api_key=api_key,
        transport=httpx.Client(timeout=10.0),
    )
    try:
        result = client.store(payload, is_primary=is_primary)
    except MembraneClientError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from None
    _emit(result)


@client_app.command("retrieve")
def client_retrieve(
    content_hash: str = typer.Option(..., "--hash", help="Content hash to retrieve."),
    base_url: str = typer.Option("http://localhost:8080", "--base-url"),
    api_key: str = typer.Option("", "--api-key"),
) -> None:
    """GET a fragment from ``/retrieve``."""
    import httpx

    client = MembraneClient(
        base_url=base_url,
        api_key=api_key,
        transport=httpx.Client(timeout=10.0),
    )
    try:
        result = client.retrieve(content_hash)
    except MembraneClientError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from None
    _emit(result)


@client_app.command("inventory")
def client_inventory(
    base_url: str = typer.Option("http://localhost:8080", "--base-url"),
    api_key: str = typer.Option("", "--api-key"),
) -> None:
    """GET the inventory digest from ``/inventory``."""
    import httpx

    client = MembraneClient(
        base_url=base_url,
        api_key=api_key,
        transport=httpx.Client(timeout=10.0),
    )
    try:
        result = client.inventory()
    except MembraneClientError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from None
    _emit(result)


@client_app.command("prefill")
def client_prefill(
    prompt_tokens: str = typer.Option(
        ...,
        "--prompt-tokens",
        help="Whitespace-separated token ids (e.g. '1 2 3 4 5').",
    ),
    model_id: str = typer.Option("default", "--model-id"),
    base_url: str = typer.Option("http://localhost:8080", "--base-url"),
    api_key: str = typer.Option("", "--api-key"),
) -> None:
    """POST a prefill request to ``/prefill``."""
    import httpx

    client = MembraneClient(
        base_url=base_url,
        api_key=api_key,
        transport=httpx.Client(timeout=30.0),
    )
    try:
        tokens = [int(t) for t in prompt_tokens.split()]
        result = client.prefill(tokens, model_id=model_id)
    except (ValueError, MembraneClientError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from None
    _emit(result)


def main() -> None:
    """Entry point registered as the ``membrane client`` subcommand."""
    client_app()


__all__ = ["client_app", "main"]
