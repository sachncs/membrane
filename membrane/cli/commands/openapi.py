"""``membrane openapi``: print or write the HTTP API's OpenAPI 3 schema."""

import json
from pathlib import Path
from typing import Annotated

import typer
from rich.markup import escape

from membrane.cli import output
from membrane.openapi import membrane_spec


def main(
    output_file: Annotated[str, typer.Option("--output", "-o", help="Write to this file instead of stdout")] = "",
) -> None:
    """Print (or write) the OpenAPI 3 schema of the Membrane HTTP API.

    Args:
        output_file: File to write; stdout when empty.
    """
    spec = membrane_spec()
    if output_file:
        Path(output_file).write_text(json.dumps(spec, indent=2) + "\n")
        output.info(f"wrote {escape(output_file)}")
    else:
        output.result_json(spec)


__all__ = ["main"]
