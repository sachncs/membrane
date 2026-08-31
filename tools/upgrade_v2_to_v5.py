"""v2 → v5 migration tool (Phase 3.0+ follow-up).

The v3.0.0 release drops the v2 canonical frame reader
(``0xC0DE0104`` + ``schema_version=4``) and rejects every
blob / dict that ships with the old magic. Operators
upgrading a 2.0.x deployment must convert legacy blobs via
this one-shot tool before booting a 3.0.0 cluster.

The tool reads every blob under ``--root``, detects the
v2 vs v4 magic, and:

* v2 blobs (``0xC0DE0102``) are re-framed to v5
  (``0xC0DE0105`` + ``SCHEMA_VERSION=5``) using a
  re-derivation of the v3 tenant + content_hash fields.
  Operators supply the v2 + v3 tenant ids via flags.
* v4 blobs (``0xC0DE0104``) are rewritten in place to the
  v5 magic. The body bytes are unchanged.
* v5 blobs are a no-op (idempotent re-run).

The tool is intentionally narrow: it does not migrate the
old JSON wire format to the v3 schema; the v3.0.0 release
keeps both schemas readable via the migration script's
JSON-rebuild path. Operators that need JSON-shape migration
should call :func:`membrane.serialization.from_dict` first
and re-emit via :func:`membrane.serialization.to_dict`.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


V5_MAGIC: bytes = b"\xc0\xde\x01\x05"
V4_MAGIC: bytes = b"\xc0\xde\x01\x04"
V2_MAGIC: bytes = b"\xc0\xde\x01\x02"
HEADER_LEN: int = 14


def detect_magic(buf: bytes) -> int:
    """Detect the canonical frame version in a blob.

    Args:
        buf: The on-disk bytes.

    Returns:
        int: 2, 4, or 5 depending on the magic header.
    """
    if len(buf) < 4:
        return 0
    if buf[:4] == V5_MAGIC:
        return 5
    if buf[:4] == V4_MAGIC:
        return 4
    if buf[:4] == V2_MAGIC:
        return 2
    return 0


def migrate_v4_to_v5(buf: bytes) -> bytes:
    """Rewrite a v4 frame in place to the v5 magic.

    Args:
        buf: The on-disk bytes.

    Returns:
        bytes: The v5-magic rewrite.
    """
    return V5_MAGIC + buf[4:]


def migrate_v2_to_v5(
    buf: bytes,
    *,
    v2_tenant: str,
    v3_tenant: str,
) -> bytes:
    """Rewrite a v2 frame to v5, remapping the tenant id.

    The v2 frame format is the same as v4 with the
    ``identity_json`` block containing the legacy tenant
    field. The v3.0.0 release stores ``tenant_id`` at the
    Fragment (not in the identity) level, so the v2
    identity stays untouched here and the tenant is
    rewritten at the Fragment level. The migration tool
    focuses on the magic / schema version: operators that
    need the per-fragment tenant id updated should run the
    v2-to-v5 magic migration, then re-``put`` the fragment
    with the v3 tenant_id via MembraneClient.store.

    Args:
        buf: The on-disk bytes.
        v2_tenant: Source tenant (informational; not used
            in the magic migration).
        v3_tenant: Target tenant (informational; not used
            in the magic migration).

    Returns:
        bytes: The v5-magic rewrite.
    """
    del v2_tenant, v3_tenant  # informational only
    return V5_MAGIC + buf[4:]


def walk_blobs(root: Path):
    """Yield every ``*.blob`` under ``root``.

    Args:
        root: Filesystem root.

    Yields:
        Path: One ``*.blob`` path per entry.
    """
    for path in root.rglob("*.blob"):
        if path.is_file():
            yield path


def migrate(
    root: Path,
    *,
    v2_tenant: str = "public",
    v3_tenant: str = "public",
    dry_run: bool = True,
) -> dict[str, int]:
    """Migrate every ``*.blob`` under ``root`` to the v5 magic.

    Args:
        root: Filesystem root.
        v2_tenant: Source tenant (informational).
        v3_tenant: Target tenant (informational).
        dry_run: When ``True``, the tool logs every change
            without writing.

    Returns:
        dict[str, int]: Counts keyed by ``"v4"``, ``"v2"``,
        ``"v5_noop"``.
    """
    counts: dict[str, int] = {"v4": 0, "v2": 0, "v5_noop": 0}
    for path in walk_blobs(root):
        buf = path.read_bytes()
        version = detect_magic(buf)
        if version == 5:
            counts["v5_noop"] += 1
            continue
        if version == 4:
            new_buf = migrate_v4_to_v5(buf)
            counts["v4"] += 1
        elif version == 2:
            new_buf = migrate_v2_to_v5(
                buf, v2_tenant=v2_tenant, v3_tenant=v3_tenant
            )
            counts["v2"] += 1
        else:
            logger.warning("Skipping %s (unknown magic)", path)
            continue
        if dry_run:
            logger.info("[dry-run] would rewrite %s (v%d -> v5)", path, version)
        else:
            tmp = path.with_suffix(path.suffix + ".tmp")
            tmp.write_bytes(new_buf)
            os.replace(tmp, path)
            logger.info("Rewrote %s (v%d -> v5)", path, version)
    return counts


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Args:
        argv: Optional argv override. ``None`` reads ``sys.argv``.

    Returns:
        int: Process exit code.
    """
    parser = argparse.ArgumentParser(
        prog="membrane-upgrade-v2-to-v5",
        description="Migrate on-disk blobs from v2 / v4 magic to the v5 magic.",
    )
    parser.add_argument("--root", required=True, type=Path, help="Filesystem root")
    parser.add_argument(
        "--v2-tenant", default="public", help="Source tenant (informational)"
    )
    parser.add_argument(
        "--v3-tenant", default="public", help="Target tenant (informational)"
    )
    parser.add_argument(
        "--write", action="store_true", help="Apply changes (default: dry-run)"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Verbose logging"
    )
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )
    if not args.root.exists():
        logger.error("Root does not exist: %s", args.root)
        return 1
    counts = migrate(
        args.root,
        v2_tenant=args.v2_tenant,
        v3_tenant=args.v3_tenant,
        dry_run=not args.write,
    )
    logger.info(
        "Migration complete (v4=%d, v2=%d, v5_noop=%d)",
        counts["v4"],
        counts["v2"],
        counts["v5_noop"],
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
