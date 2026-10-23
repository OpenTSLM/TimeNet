"""Write-time collection of direct physical owners for shared payloads."""

from typing import TYPE_CHECKING

from timenet.errors import TimeFValidationError
from timenet.format.stored_values import PackedLocation, PayloadLocation
from timenet.types.reference_keys import OwnerKey
from timenet.types.references import DatasetRef
from timenet.values_backends import ValuesBackend
from timenet.values_backends.zarr.layout import validity_path


if TYPE_CHECKING:
    from timenet.manifest import Manifest
    from timenet.registry.version import DatasetVersion


class StorageSources:
    """Collect the exact releases that store a new catalog's shared payloads."""

    def __init__(self, target: DatasetRef | None = None) -> None:
        self.target = target
        self.sources: dict[OwnerKey, DatasetRef] = {}
        # Each owner's listed paths, gathered once, so checking a payload is one lookup per path.
        self._stored_paths: dict[OwnerKey, frozenset[str]] = {}

    def add(self, release: "DatasetVersion", location: PayloadLocation, *, nullable: bool = False) -> OwnerKey:
        """Check a reused payload's files and record the release that stores them.

        Args:
            release: The release whose catalog records ``location``. The location's owner names one
                of its payload sources, or is ``None`` when ``release`` stores the payload itself.
            location: Where the payload's bytes live.
            nullable: Whether the spec of a signal array is nullable. Zarr stores the array's
                validity in parallel arrays, which the owner must then list too.

        Returns:
            Its owner key in the child manifest: the exact release as ``org/name@version``.

        Raises:
            TimeFValidationError: If the storing release is the output release, or its manifest
                lacks a file of the payload.
        """
        manifest = release.resolve_payload_source(location.owner).manifest
        ref = DatasetRef(dataset_id=manifest.dataset_id, version=manifest.metadata.dataset_version)
        if ref == self.target:
            raise TimeFValidationError(
                f"shared payload owner {ref} is also the output release; use a different dataset ID or version"
            )
        # The exact release itself names the owner, so catalog rows stay readable without the manifest.
        key = OwnerKey(str(ref))
        stored = self._stored_paths.get(key)
        if stored is None:
            stored = self._stored_paths[key] = _stored_paths(manifest)
        with_validity = nullable and manifest.values_backend == ValuesBackend.ZARR
        for path in _payload_paths(location, with_validity=with_validity):
            if path not in stored:
                raise TimeFValidationError(f"payload {path!r} is absent from {ref}")
        self.sources[key] = ref
        return key


def _stored_paths(manifest: "Manifest") -> frozenset[str]:
    """Return every file a release lists, and every directory that holds one.

    A Zarr array is a directory of files, so a payload path can name a directory.

    Returns:
        The paths relative to the release root.
    """
    paths: set[str] = set()
    for part in manifest.files.all_files():
        path = part.path
        while path and path not in paths:
            paths.add(path)
            path = path.rpartition("/")[0]
    return frozenset(paths)


def _payload_paths(location: PayloadLocation, *, with_validity: bool) -> set[str]:
    """Return the payload files, or Zarr arrays, that hold a stored value.

    Args:
        location: Where the value's bytes live.
        with_validity: Whether a signal array's Zarr validity arrays hold part of it.

    Returns:
        The paths relative to the root of the release that stores the value.
    """
    if isinstance(location, PackedLocation):
        return {location.path}
    paths = {chunk.chunk_file for chunk in location.chunks}
    if with_validity:
        paths |= {validity_path(path) for path in paths}
    return paths
