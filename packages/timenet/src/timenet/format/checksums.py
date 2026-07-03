"""File checksums recorded in the manifest and verified on read.

Part of the format contract rather than the writer: the reader has to hash files exactly the way the
writer did for :meth:`~timenet.reader.TimeFReader.verify` to mean anything, so the algorithm and the
block size live in one place both sides call.
"""

import hashlib
from pathlib import Path


CHECKSUM_PREFIX = "sha256:"
_BLOCK_BYTES = 1 << 20  # hash a block at a time, not all-in-memory


def file_checksum(path: Path) -> str:
    """Return a file's manifest checksum, hashing it a block at a time.

    Args:
        path: The file to hash.

    Returns:
        The checksum as ``"sha256:<hex>"``.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(_BLOCK_BYTES):
            digest.update(block)
    return CHECKSUM_PREFIX + digest.hexdigest()
