"""Write-time collection of the payload sources a new release reads, pinned to their files."""

from typing import TYPE_CHECKING

from timenet.errors import TimeFFormatError, TimeFValidationError, TimeNetDatasetNotFoundError
from timenet.format.stored_values import PackedLocation, PayloadLocation
from timenet.manifest import FilePart, PayloadSource
from timenet.types.reference_keys import SourceKey
from timenet.types.references import DatasetRef
from timenet.values_backends import ValuesBackend
from timenet.values_backends.zarr.layout import validity_path


if TYPE_CHECKING:
    from timenet.manifest import Manifest
    from timenet.registry.base import BaseRegistry
    from timenet.registry.version import DatasetVersion


class StorageSources:
    """Collect the releases that store a new release's reused payloads, and pin the files it reads.

    With a target registry, each payload source is checked there at its first reuse. The registry
    must hold the source with the same records of the reused files as the release the values were
    read from, and the source must not read payloads of the new release itself.
    """

    def __init__(self, target: DatasetRef | None = None, registry: "BaseRegistry | None" = None) -> None:
        """Start an empty collection.

        Args:
            target: The release being written. A reused payload cannot be stored by it.
            registry: The registry the release is published to. ``None`` skips the registry check.
        """
        self.target = target
        self.registry = registry
        self._releases: dict[SourceKey, DatasetRef] = {}
        self._pins: dict[SourceKey, dict[str, FilePart]] = {}
        # Each source's own file records by path, and by every directory that holds one, gathered
        # once, so finding a payload's files is one lookup.
        self._listed: dict[SourceKey, dict[str, tuple[FilePart, ...]]] = {}
        # The target registry's file records of each source, fetched at the source's first reuse.
        self._published: dict[SourceKey, dict[str, FilePart]] = {}
        self._checked: set[tuple[SourceKey, str]] = set()

    def add(self, release: "DatasetVersion", location: PayloadLocation, *, nullable: bool = False) -> SourceKey:
        """Check a reused payload's files and pin them on the release that stores them.

        Args:
            release: The release whose catalog records ``location``. The location's source names one
                of its payload sources, or is ``None`` when ``release`` stores the payload itself.
            location: Where the payload's bytes live.
            nullable: Whether the spec of a signal array is nullable. Zarr stores the array's
                validity in parallel arrays, which the source must then list too.

        Returns:
            Its source key in the new release: the exact release as ``org/name@version``.

        Raises:
            TimeFValidationError: If the storing release is the output release, its manifest lacks a
                file of the payload, or the target registry holds a source that reads payloads of the
                output release.
            TimeNetDatasetNotFoundError: If the target registry does not hold the source.
            TimeFFormatError: If the target registry's source records a reused file differently.
        """  # noqa: DOC502 - the registry check raises in a helper
        manifest = release.resolve_payload_source(location.source).manifest
        ref = DatasetRef(dataset_id=manifest.dataset_id, version=manifest.metadata.dataset_version)
        if ref == self.target:
            raise TimeFValidationError(
                f"payload source {ref} is also the output release; use a different dataset ID or version"
            )
        # The exact release itself is the key, so catalog rows stay readable without the manifest.
        key = SourceKey(str(ref))
        listed = self._listed.get(key)
        if listed is None:
            listed = self._listed[key] = _listed_files(manifest)
        pins = self._pins.setdefault(key, {})
        with_validity = nullable and manifest.values_backend == ValuesBackend.ZARR
        for path in sorted(_payload_paths(location, with_validity=with_validity)):
            if (key, path) in self._checked:
                continue
            parts = listed.get(path)
            if not parts:
                raise TimeFValidationError(f"payload {path!r} is absent from {ref}")
            self._check_published(key, ref, parts)
            pins.update((part.path, part) for part in parts)
            self._checked.add((key, path))
        self._releases[key] = ref
        return key

    @property
    def sources(self) -> dict[SourceKey, PayloadSource]:
        """Return each payload source with the files the new release reads from it, for the manifest."""
        sources: dict[SourceKey, PayloadSource] = {}
        for key in sorted(self._releases):
            pins = self._pins[key]
            sources[key] = PayloadSource(release=self._releases[key], files=tuple(pins[path] for path in sorted(pins)))
        return sources

    def _check_published(self, key: SourceKey, ref: DatasetRef, parts: tuple[FilePart, ...]) -> None:
        """Check that the target registry holds a payload source with the same reused files.

        The errors match those of opening a release whose payload source is missing or differs.

        Raises:
            TimeNetDatasetNotFoundError: Naming the source and the registry, if the registry lacks it.
            TimeFFormatError: Naming the source and the first differing file, in path order, if the
                registry's source records a reused file differently.
            TimeFValidationError: If the registry's source reads payloads of the output release.
        """
        if self.registry is None:
            return
        published = self._published.get(key)
        if published is None:
            try:
                manifest = self.registry.get_manifest(ref.dataset_id, str(ref.version))
            except TimeNetDatasetNotFoundError as error:
                raise TimeNetDatasetNotFoundError(
                    f"payload source {ref} of {self.target} is not in {self.registry}; "
                    f"store {self.target} in the registry that holds {ref}, or copy {ref} there first"
                ) from error
            if self.target is not None and any(
                source.release == self.target for source in manifest.payload_sources.values()
            ):
                raise TimeFValidationError(
                    f"{self.target} would reuse payloads stored by {ref}, which reads payloads of {self.target} "
                    "itself; a release cannot reuse payloads from a release that depends on it"
                )
            published = self._published[key] = {part.path: part for part in manifest.files.all_files()}
        for part in sorted(parts, key=lambda item: item.path):
            stored = published.get(part.path)
            if stored != part:
                found = "does not list it" if stored is None else f"records {_describe(stored)}"
                raise TimeFFormatError(
                    f"payload source {ref} in {self.registry} differs from the release the reused values were "
                    f"read from: they read {part.path!r} as {_describe(part)}, but {ref} there {found}"
                )


def _describe(part: FilePart) -> str:
    """Describe a file's manifest record for an error message.

    Returns:
        Its size and checksum.
    """
    return f"{part.size} bytes with checksum {part.checksum}"


def _listed_files(manifest: "Manifest") -> dict[str, tuple[FilePart, ...]]:
    """Return a release's file records by path, and by every directory that holds files.

    A Zarr array is a directory of files, so a payload path can name a directory.

    Returns:
        The records of each file, and of every file under each directory, by release-relative path.
    """
    listed: dict[str, list[FilePart]] = {}
    for part in manifest.files.all_files():
        path = part.path
        while path:
            listed.setdefault(path, []).append(part)
            path = path.rpartition("/")[0]
    return {path: tuple(parts) for path, parts in listed.items()}


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
