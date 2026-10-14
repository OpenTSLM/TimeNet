"""Check immutable release identities and owned artifact bytes."""

from hashlib import sha256
from pathlib import Path

from timenet.errors import TimeFFormatError
from timenet.format.checksums import CHECKSUM_PREFIX, file_checksum
from timenet.manifest import Manifest


def check_manifest(content: bytes, expected: Manifest, checksum: str | None = None) -> Manifest:
    """Validate exact manifest bytes before accepting a staged or cached release.

    Returns:
        The validated artifact manifest.

    Raises:
        TimeFFormatError: If the artifact differs from the requested release or pin.
    """
    actual = Manifest.from_artifact(content)
    if actual.to_dict() != expected.to_dict():
        raise TimeFFormatError(
            f"manifest differs from requested release {expected.dataset_id}@{expected.metadata.dataset_version}"
        )
    if checksum is not None and CHECKSUM_PREFIX + sha256(content).hexdigest() != checksum:
        raise TimeFFormatError(
            f"payload owner manifest checksum mismatch for {expected.dataset_id}@{expected.metadata.dataset_version}"
        )
    return actual


def verify_owned(root: Path, manifest: Manifest) -> None:
    """Validate the size and checksum of every owned file in a local release.

    Raises:
        TimeFFormatError: If an artifact is missing or differs from its manifest.
    """
    for part in manifest.files.all_files():
        path = root / part.path
        if not path.is_file():
            raise TimeFFormatError(f"missing artifact {path}")
        if path.stat().st_size != part.size:
            raise TimeFFormatError(f"size mismatch for {path}")
        if file_checksum(path) != part.checksum:
            raise TimeFFormatError(f"checksum mismatch for {path}")
