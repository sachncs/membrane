"""v2 → v5 JSON wire dict migration (Phase 3.0+ follow-up).

The 3.0.0 release drops the v2 JSON wire format
(``schema_version=4``, no ``tenant_id`` field). Operators that
upgraded their on-disk blobs via
``tools/upgrade_v2_to_v5.py`` (Phase 3.3.1) also need to
re-emit their persistence-layer JSON envelopes, which are
the wire format used by :class:`membrane.persistence.Memory`
and :class:`membrane.persistence.Redis` (and any custom
backend that round-trips through
:func:`membrane.serialization.from_dict`).

This tool rewrites a v4 dict to a v5 dict in memory and
optionally writes the result back. The transformation is
deterministic:

* ``schema_version`` bumps from 4 to 5.
* A new ``tenant_id`` field is added (default ``"public"``;
  overridden via ``--tenant``).
* Every other field passes through unchanged.

The v2 canonical frame and the v4 JSON wire dict can be
upgraded in the same window: the canonical frame tool
handles the on-disk bytes (Phase 3.3.1) and this tool handles
the in-memory wire dicts.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


V5_SCHEMA_VERSION: int = 5


def migrate_dict(d: dict[str, Any], *, default_tenant: str = "public") -> dict[str, Any]:
    """Migrate a single v4 wire dict to v5.

    Args:
        d: A v4 wire dict (must have ``schema_version: 4``).
        default_tenant: Tenant id to insert when the source
            dict has none.

    Returns:
        dict: A new dict with ``schema_version: 5`` + the
        added ``tenant_id`` field. The input dict is not
        mutated.
    """
    out = dict(d)
    out["schema_version"] = V5_SCHEMA_VERSION
    out.setdefault("tenant_id", default_tenant)
    # v4 emits ``payload_ref=None``; v5 prefers an explicit
    # empty string in that case. Leave the field alone; the
    # v5 reader treats ``None`` identically.
    return out


def migrate_file(path: Path, *, default_tenant: str = "public", write: bool = False) -> tuple[int, int]:
    """Migrate a single JSON file (one wire dict per line).

    Args:
        path: Filesystem path to the JSON wire file (one dict
            per line).
        default_tenant: Tenant id to insert when the source
            dict has none.
        write: When ``True``, rewrite the file in place.

    Returns:
        tuple[int, int]: ``(rewritten, skipped)`` counters.
    """
    if not path.exists():
        return (0, 0)
    rewritten = 0
    skipped = 0
    new_lines: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            new_lines.append(raw)
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            new_lines.append(raw)
            continue
        if not isinstance(payload, dict) or payload.get("schema_version") != 4:
            new_lines.append(raw)
            skipped += 1
            continue
        new_payload = migrate_dict(payload, default_tenant=default_tenant)
        new_lines.append(json.dumps(new_payload, sort_keys=True))
        rewritten += 1
    if write and rewritten:
        path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
    return (rewritten, skipped)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Args:
        argv: Optional argv override. ``None`` reads ``sys.argv``.

    Returns:
        int: Process exit code.
    """
    parser = argparse.ArgumentParser(
        prog="membrane-upgrade-v2-to-v5-json",
        description=(
            "Rewrite v4 JSON wire dicts to v5 (bumps schema_version, "
            "adds tenant_id)."
        ),
    )
    parser.add_argument("--path", required=True, help="JSON file or directory of files")
    parser.add_argument(
        "--tenant",
        default="public",
        help="Default tenant id when the source dict has none (default: public)",
    )
    parser.add_argument(
        "--write", action="store_true", help="Apply the migration (default: dry-run)"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Verbose logging"
    )
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    target = Path(args.path)
    if not target.exists():
        logger.error("Path does not exist: %s", target)
        return 1

    paths = (
        [target]
        if target.is_file()
        else sorted(target.rglob("*.jsonl")) + sorted(target.rglob("*.json"))
    )

    if not paths:
        logger.warning("No .jsonl / .json files under %s", target)
        return 0

    total_rewritten, total_skipped = 0, 0
    for path in paths:
        rewritten, skipped = migrate_file(
            path, default_tenant=args.tenant, write=args.write
        )
        if rewritten or skipped:
            logger.info(
                "%s %s: rewritten=%d skipped=%d",
                "wrote" if args.write else "would-write",
                path,
                rewritten,
                skipped,
            )
        total_rewritten += rewritten
        total_skipped += skipped

    logger.info(
        "Migration %s (rewritten=%d, skipped=%d)",
        "applied" if args.write else "dry-run",
        total_rewritten,
        total_skipped,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
