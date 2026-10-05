"""OpenAPI 3 schema of the Membrane HTTP API.

``membrane openapi`` prints it (or writes it with ``-o``) without a
running server; a server started with ``--enable-api-docs`` serves the
same document at ``/openapi.json`` behind the ``read`` scope.
"""

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


def generate_spec(app: Any) -> dict[str, Any]:
    """Read the OpenAPI 3 spec from a FastAPI app.

    Args:
        app: The FastAPI app.

    Returns:
        dict: The OpenAPI 3 spec as a JSON-serializable dict.
    """
    return app.openapi()


def membrane_spec() -> dict[str, Any]:
    """Return the schema of a fully configured Membrane app.

    Returns:
        dict[str, Any]: The OpenAPI 3 document.
    """
    from membrane.node import Node
    from membrane.transfer import TransferService
    from membrane.transport.fastapi import create_app
    from membrane.transport.limits import TransportLimits

    app = create_app(
        node=Node("openapi"),
        compute_backend=None,
        transfer_service=TransferService(),
        cluster_manager=None,
        limits=TransportLimits(enable_api_docs=True),
    )
    return generate_spec(app)


def write_spec(app: Any, path: str) -> None:
    """Write the OpenAPI spec to ``path`` as JSON.

    Args:
        app: The FastAPI app.
        path: Output file path.
    """
    spec = generate_spec(app)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(spec, f, indent=2)


__all__ = ["generate_spec", "membrane_spec", "write_spec"]
