"""Permission checks for secret files (API keyfiles, data keys).

A secret that every local user can read is not a secret. The server
refuses to start on such a file. Group read stays allowed because
Kubernetes mounts secrets group-readable (0440, owned by the pod's
``fsGroup``); it is logged when the group is not one of the process's
own.
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
    info = Path(path).stat()
    mode = stat.S_IMODE(info.st_mode)
    if mode & (stat.S_IRWXO | stat.S_IWGRP):
        raise InsecureFileError(
            f"{what} file {str(path)!r} has mode {mode:04o}; other users can read or change it. Run: chmod 600 {path}"
        )
    if mode & stat.S_IRGRP and info.st_gid != os.getegid() and info.st_gid not in os.getgroups():
        logger.warning("%s file %s is readable by group %s (mode %04o)", what, path, info.st_gid, mode)


__all__ = ["InsecureFileError", "require_private_file"]
