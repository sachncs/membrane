"""``membrane keys``: create API keys for the keyfile.

``membrane keys generate --subject ingest --scope read --scope write``
writes two lines to stdout: the key, which goes to the client, and its
hashed keyfile line, which goes in ``--api-key-file``. The server never
stores the key itself.
"""

from pathlib import Path
from typing import Annotated

import typer
from rich.markup import escape

from membrane.auth.apikey import generate_key
from membrane.cli import output
from membrane.security.keyring import write_next_key

keys_app = typer.Typer(help="Create API keys for the --api-key-file keyfile.")

KNOWN_SCOPES = ("read", "write", "admin")


@keys_app.command("generate")
def generate(
    subject: Annotated[str, typer.Option("--subject", "-s", help="Caller identity (also its tenant)")],
    scope: Annotated[
        list[str] | None, typer.Option("--scope", help="Scope to grant: read, write, admin (repeatable)")
    ] = None,
    as_json: Annotated[bool, typer.Option("--json", help="Print {key, keyfile_line} as JSON")] = False,
) -> None:
    """Generate a random API key and its hashed keyfile line.

    Args:
        subject: Caller identity (also its tenant).
        scope: Scopes to grant; defaults to ``read``.
        as_json: Print ``{key, keyfile_line}`` as JSON.

    Raises:
        typer.Exit: With status 2 on an invalid subject or scope.
    """
    scopes = scope or ["read"]
    unknown = sorted(set(scopes) - set(KNOWN_SCOPES))
    if unknown or not subject or ":" in subject:
        problem = f"unknown scope(s) {', '.join(unknown)}" if unknown else "subject must be non-empty without ':'"
        output.error(f"[bold red]{escape(problem)}[/bold red]")
        raise typer.Exit(2)
    key, line = generate_key(subject, scopes)
    if as_json:
        output.result_json({"key": key, "keyfile_line": line})
    else:
        output.result(escape(key))
        output.result(escape(line))
    output.info("Give the first line to the client; append the second to the server's --api-key-file.")


@keys_app.command("rotate-data-key")
def rotate_data_key(
    directory: Annotated[str, typer.Argument(help="Data-key directory (the --data-key-file of the nodes)")],
) -> None:
    """Add the next data-key version to a key directory.

    Running nodes activate it within a minute and re-encrypt their blobs
    under it; afterwards older ``v<N>.key`` files can be deleted. To adopt
    versioned keys, move an existing key file to ``DIR/v1.key`` first.

    Args:
        directory: Data-key directory (the nodes' ``--data-key-file``).
    """
    path = write_next_key(Path(directory))
    output.result(escape(str(path)))
    output.info("Nodes using this directory switch to the new key within a minute.")


__all__ = ["KNOWN_SCOPES", "generate", "keys_app", "rotate_data_key"]
