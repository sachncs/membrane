"""OpenAPI spec generation (Phase 3.6.2).

The v3.0.0 release exposes a :func:`generate_spec` helper
that walks the FastAPI app and dumps the OpenAPI 3 JSON.
The :func:`write_spec` helper writes the spec to a file so
operators can publish it under ``docs/openapi.json``.
"""

from __future__ import annotations

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


def write_spec(app: Any, path: str) -> None:
    """Write the OpenAPI spec to ``path`` as JSON.

    Args:
        app: The FastAPI app.
        path: Output file path.
    """
    spec = generate_spec(app)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(spec, f, indent=2)


__all__ = ["generate_spec", "write_spec"]
