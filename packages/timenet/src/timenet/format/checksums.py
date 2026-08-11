"""File checksums recorded in the manifest and verified on read.

The reader must hash files the same way the writer did, or :meth:`~timenet.reader.TimeFReader.verify`
means nothing. The algorithm and the block size live here so both sides call the same code.
"""

import hashlib
from pathlib import Path
from typing import Protocol


CHECKSUM_PREFIX = "sha256:"
_BLOCK_BYTES = 1 << 20  # hash a block at a time, not all-in-memory


class _Readable(Protocol):
    """A binary stream that yields its bytes a block at a time.

    Both a ``BufferedReader`` (``path.open("rb")``) and a ``pyarrow`` ``NativeFile`` satisfy it, so the
    same hashing runs over a local ``Path`` and a file opened through a pyarrow filesystem.
    """

    def read(self, size: int = ..., /) -> bytes:
        """Read up to ``size`` bytes, or the rest of the stream."""
        ...


def file_checksum(path: Path) -> str:
    """Return a file's manifest checksum, hashing it a block at a time.

    Args:
        path: The file to hash.

    Returns:
        The checksum as ``"sha256:<hex>"``.
    """
    with path.open("rb") as handle:
        return stream_checksum(handle)


def stream_checksum(handle: _Readable) -> str:
    """Return an open binary stream's manifest checksum, hashing it a block at a time.

    The stream counterpart of :func:`file_checksum`, so a reader can hash a file it opened through a
    pyarrow filesystem (local now, an object store later) rather than a local ``Path``.

    Args:
        handle: An open binary stream positioned at the start.

    Returns:
        The checksum as ``"sha256:<hex>"``.
    """
    digest = hashlib.sha256()
    while block := handle.read(_BLOCK_BYTES):
        digest.update(block)
    return CHECKSUM_PREFIX + digest.hexdigest()
