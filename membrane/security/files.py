"""Permission checks for secret files (API keyfiles, data keys).

A secret that every local user can read is not a secret. The server
refuses to start on such a file. Group read stays allowed, with a
warning, because Kubernetes mounts secrets group-readable (0440) when a
pod sets ``fsGroup``.
"""

import logging
import os
import stat
from pathlib import Path

logger = logging.getLogger(__name__)


class InsecureFileError(ValueError):
    """Raised when a secret file is accessible to other users."""


def require_private_file(path: Path | str, what: str) -> None:
    """Refuse a secret file that other users can read or anyone else can write.

    Symbolic links are followed (Kubernetes secret volumes are links
    into a ``..data`` directory). The check is skipped on platforms
    without POSIX permissions.

    Args:
        path: The secret file.
        what: Human-readable name used in messages (e.g. ``"API keyfile"``).

    Raises:
        InsecureFileError: When the file is world-accessible or
            group-writable.
    """
    if os.name != "posix":
        return
    mode = stat.S_IMODE(Path(path).stat().st_mode)
    if mode & (stat.S_IRWXO | stat.S_IWGRP):
        raise InsecureFileError(
            f"{what} file {str(path)!r} has mode {mode:04o}; other users can read or change it. Run: chmod 600 {path}"
        )
    if mode & stat.S_IRGRP:
        logger.warning("%s file %s is group-readable (mode %04o)", what, path, mode)


__all__ = ["InsecureFileError", "require_private_file"]
