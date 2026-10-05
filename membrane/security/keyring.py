"""Data-key loading: a single key, or a directory of versioned keys.

``--data-key-file`` accepts:

* a file holding 32 raw bytes or 64 hex characters;
* ``secret://name``, resolved through the secret provider (hex);
* a directory of ``v1.key``, ``v2.key``, ... files. The highest version
  encrypts new data; every version still decrypts. Add a version with
  ``membrane keys rotate-data-key DIR``: a running node picks it up
  within a minute (:meth:`DirectoryKeyring.refresh`) and re-encrypts
  existing blobs under it, after which old versions can be deleted.
"""

import logging
import os
import re
import secrets
from pathlib import Path
from typing import override

from membrane.security.encryption import KeyProvider
from membrane.security.files import require_private_file
from membrane.security.key_rotation import RotatingKeyProvider

logger = logging.getLogger(__name__)

VERSION_FILE = re.compile(r"v(\d+)\.key")


def parse_key(raw: bytes, where: str) -> bytes:
    """Decode a 32-byte key stored raw or as 64 hex characters.

    Args:
        raw: File or secret contents.
        where: Source, for the error message.

    Returns:
        bytes: The 32-byte key.

    Raises:
        ValueError: When the contents are neither form.
    """
    if len(raw) == 32:
        return raw  # raw key bytes: never strip, they may look like whitespace
    try:
        key = bytes.fromhex(raw.decode("ascii").strip())
    except UnicodeDecodeError, ValueError:
        key = b""
    if len(key) != 32:
        raise ValueError(f"data key in {where} must be 32 bytes (or 64 hex characters)")
    return key


def versioned_keys(directory: Path) -> list[tuple[int, Path]]:
    """List the ``v<N>.key`` files of ``directory`` in version order.

    Args:
        directory: Key directory.

    Returns:
        list[tuple[int, Path]]: ``(version, path)`` pairs, ascending.
    """
    found = []
    for path in directory.iterdir():
        match = VERSION_FILE.fullmatch(path.name)
        if match:
            found.append((int(match.group(1)), path))
    return sorted(found)


class DirectoryKeyring(KeyProvider):
    """Versioned master keys read from a directory.

    Attributes:
        directory: The key directory.
    """

    def __init__(self, directory: Path) -> None:
        """Load every version in ``directory``.

        Args:
            directory: Directory of ``v<N>.key`` files.

        Raises:
            ValueError: When it holds no key or a key is malformed.
        """
        self.directory = directory
        versions = versioned_keys(directory)
        if not versions:
            raise ValueError(f"key directory {str(directory)!r} has no v<N>.key files")
        first_version, first_path = versions[0]
        self.__provider = RotatingKeyProvider(self.__read(first_path))
        self.__loaded = [first_version]
        self.refresh()

    @staticmethod
    def __read(path: Path) -> bytes:
        """Read one key file after the permission check.

        Args:
            path: A ``v<N>.key`` file.

        Returns:
            bytes: The 32-byte key.
        """
        require_private_file(path, "data key")
        return parse_key(path.read_bytes(), str(path))

    @property
    def active_version(self) -> int:
        """The version that encrypts new data."""
        return self.__loaded[-1]

    def refresh(self) -> list[int]:
        """Load versions added since the last call.

        Returns:
            list[int]: Newly activated versions (the last one is active).
        """
        added = []
        for version, path in versioned_keys(self.directory):
            if version <= self.__loaded[-1]:
                continue
            try:
                self.__provider.rotate(self.__read(path))
            except ValueError as exc:
                logger.error("ignoring data key %s: %s", path, exc)
                continue
            self.__loaded.append(version)
            added.append(version)
            logger.info("data key version %s is now active", version)
        return added

    def version_keys(self) -> tuple[bytes, ...]:
        """Return every loaded key, oldest first.

        Returns:
            tuple[bytes, ...]: The keys; the last one is active.
        """
        return self.__provider.version_keys()

    @override
    def master_key(self) -> bytes:
        """Return the active key.

        Returns:
            bytes: The 32-byte active key.
        """
        return self.__provider.master_key()


def write_next_key(directory: Path) -> Path:
    """Create the next ``v<N>.key`` (mode 0600) in ``directory``.

    Args:
        directory: Key directory; created when missing.

    Returns:
        Path: The new key file.
    """
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    existing = versioned_keys(directory)
    path = directory / f"v{existing[-1][0] + 1 if existing else 1}.key"
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as handle:
        handle.write(secrets.token_bytes(32))
    return path


__all__ = ["VERSION_FILE", "DirectoryKeyring", "parse_key", "versioned_keys", "write_next_key"]
