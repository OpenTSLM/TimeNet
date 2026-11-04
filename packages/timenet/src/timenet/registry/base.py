"""The registry contract and its shared search implementation.

A registry serves compiled TimeF versions. It does not run connector code. Concrete backends
must implement the four data-access methods: :meth:`list_datasets`, :meth:`get_manifest`,
:meth:`open_file`, and :meth:`open_release`. :meth:`BaseRegistry.search` is shared. It filters
the output of :meth:`list_datasets` and uses :meth:`get_manifest` for the type filters.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path
import shutil
from typing import BinaryIO, ClassVar

from timenet._fs import staged_directory
from timenet.errors import TimeNetDatasetNotFoundError, TimeNetFileError, TimeNetRegistryError
from timenet.format.checksums import file_checksum
from timenet.format.constants import MANIFEST_FILE
from timenet.manifest import FilePart, Manifest
from timenet.registry._paths import safe_version_path
from timenet.registry.dependencies import validate_manifest_reference
from timenet.registry.shared import complete_copy
from timenet.registry.version import DatasetVersion
from timenet.sources import FoundInRegistry, NotFound, NotServed, ReleaseLookup
from timenet.types import Access, DatasetFilter, DatasetMetadata, ListedRelease, Version
from timenet.types.access import access_requirements, effective_access
from timenet.types.references import DatasetRef


# Reports download progress: called with each file (or chunk) of bytes as it lands. A ``download_release``
# implementation calls it so a caller can render a progress bar; ``None`` downloads silently.
ByteCountCallback = Callable[[int], None]


class BaseRegistry(ABC):
    """A registry of TimeF manifests, files, and searchable dataset metadata.

    :meth:`open_release` and :meth:`download_release` handle one release without its payload
    sources. :func:`~timenet.registry.copy_release` copies a release with its payload sources.
    """

    has_catalog: ClassVar[bool] = True
    """Whether :meth:`list_datasets` and :meth:`search` work. Without a catalog, both raise ``NotImplementedError``."""
    _read_only: bool = False

    def __str__(self) -> str:
        return type(self).__name__

    @property
    def read_only(self) -> bool:
        """Whether this registry refuses every write, set with ``read_only=True``."""
        return self._read_only

    @property
    @abstractmethod
    def location(self) -> str:
        """Where this registry is: a URL or a path, else the name of the registry."""

    def check_writable(self, write: str) -> None:
        """Refuse a write if this registry is read-only.

        Args:
            write: The write to refuse, for the message, such as ``"put_dataset()"``.

        Raises:
            TimeNetRegistryError: If this registry is read-only.
        """
        if self._read_only:
            raise TimeNetRegistryError(
                f"{self} has read_only=True, so it refuses {write}. Open it without read_only=True to write to it."
            )

    @property
    def requires_open_access(self) -> bool:
        """Whether stored or served releases must declare open access."""
        return True

    @abstractmethod
    def list_datasets(self) -> list[DatasetMetadata]:
        """Return the metadata of every dataset, using the latest version. The result is sorted by dataset id.

        Returns:
            One :class:`~timenet.types.DatasetMetadata` object for each dataset.
        """

    @abstractmethod
    def get_manifest(self, dataset_id: str, version: str | None = None) -> Manifest:
        """Return the manifest of a dataset.

        Args:
            dataset_id: The dataset id.
            version: The version string, or ``None`` for the latest version.

        Returns:
            The :class:`~timenet.manifest.Manifest` of the dataset.

        Raises:
            TimeNetDatasetNotFoundError: If the dataset id or version is unknown.
        """

    def find_manifest(self, release: DatasetRef) -> Manifest | None:
        """Return the manifest of a release, or ``None`` if this registry does not hold it.

        Args:
            release: The exact release.

        Returns:
            The manifest, or ``None``.

        Raises:
            TimeFFormatError: If the stored manifest names another release.
        """  # ruff: ignore[docstring-extraneous-exception] - raised by validate_manifest_reference
        try:
            manifest = self.get_manifest(release.dataset_id, str(release.version))
        except TimeNetDatasetNotFoundError:
            return None
        validate_manifest_reference(manifest, release)
        return manifest

    @abstractmethod
    def open_file(self, dataset_id: str, version: str, relpath: str) -> BinaryIO:
        """Open one file of a dataset version for binary reading.

        Args:
            dataset_id: The dataset id.
            version: The version string.
            relpath: The file path relative to the version directory.

        Returns:
            An open binary file object.

        Raises:
            TimeNetDatasetNotFoundError: If the dataset id or version is unknown.
        """

    @abstractmethod
    def open_release(self, reference: DatasetRef, *, manifest: Manifest | None = None) -> DatasetVersion:
        """Open an exact release without attaching its payload sources.

        Opening reads metadata only. Attach payload sources with
        :func:`~timenet.registry.dependencies.attach_payload_sources`.

        Args:
            reference: The dataset ID and exact version to open.
            manifest: A manifest already read from this registry, or ``None`` to fetch it.

        Returns:
            A handle to the release's manifest and files.

        Raises:
            TimeNetDatasetNotFoundError: If the registry lacks the release.
            TimeFFormatError: If the manifest declares a different release.
            TimeNetFileError: If a local file is missing or has an incorrect size.
        """

    def resolve(self, dataset_id: str, version: Version | None) -> ReleaseLookup:
        """Check whether the registry serves a dataset release.

        If :attr:`requires_open_access` is true, the registry serves a release only if its card and each card in its
        lineage declare open access.

        Args:
            dataset_id: The dataset ID.
            version: An exact version, or ``None`` for the latest version.

        Returns:
            ``FoundInRegistry`` if served, ``NotServed`` if access is restricted, or ``NotFound`` if absent.
        """
        try:
            manifest = self.get_manifest(dataset_id, None if version is None else str(version))
        except TimeNetDatasetNotFoundError:
            return NotFound()
        held = manifest.metadata.dataset_version
        access = effective_access(manifest.metadata, manifest.lineage)
        if access is Access.OPEN or not self.requires_open_access:
            return FoundInRegistry(held, self, manifest)
        return NotServed(held, access, access_requirements(manifest.metadata, manifest.lineage), manifest)

    def download_release(
        self,
        reference: DatasetRef,
        directory: str | Path,
        *,
        force: bool = False,
        manifest: Manifest | None = None,
        progress_cb: ByteCountCallback | None = None,
    ) -> None:
        """Download one release's files to a local directory.

        Reuse a complete copy unless ``force`` is set.
        Replace ``directory`` only after every downloaded file matches its manifest record.

        Args:
            reference: The exact release.
            directory: The local release directory, ``<root>/<dataset_id>/<version>``.
            force: Download again even if a complete copy exists.
            manifest: The release's manifest, or ``None`` to fetch it from this registry.
            progress_cb: Callback that receives each downloaded file's byte count, or ``None``.

        Raises:
            TimeNetFileError: If a downloaded file's size or checksum differs from its manifest record.
        """  # ruff: ignore[docstring-extraneous-exception] - raised by _check_staged_file
        dataset_id, version = reference.dataset_id, str(reference.version)
        if manifest is None:
            manifest = self.get_manifest(dataset_id, version)
        target = Path(directory)
        if complete_copy(target, manifest) and not force:
            return  # already downloaded and current
        with staged_directory(target) as staging:
            for part in manifest.files.all_files():
                self._stage_file(dataset_id, version, part.path, staging)
                self._check_staged_file(staging / part.path, part, reference)
                if progress_cb is not None:
                    progress_cb(part.size)
            (staging / MANIFEST_FILE).write_text(manifest.to_json())

    def _check_staged_file(self, path: Path, part: FilePart, release: DatasetRef) -> None:
        """Compare one downloaded file with its manifest record, first by size, then by checksum.

        Raises:
            TimeNetFileError: Naming the file and whichever of the two differs.
        """
        owner = f"download of {release} from {self}"
        size = path.stat().st_size
        if size != part.size:
            raise TimeNetFileError(release, owner, part.path, f"{part.size:,} bytes", f"{size:,} bytes")
        checksum = file_checksum(path)
        if checksum != part.checksum:
            raise TimeNetFileError(release, owner, part.path, part.checksum, checksum)

    def _stage_file(self, dataset_id: str, version: str, relpath: str, staging: Path) -> None:
        """Copy one version file from the registry into the ``staging`` directory.

        Args:
            dataset_id: The dataset id.
            version: The version string.
            relpath: The version-relative file path from the manifest.
            staging: The staging directory to write into.

        Raises:
            TimeFFormatError: If ``relpath`` would escape ``staging`` (an absolute or ``..`` path).
        """  # ruff: ignore[docstring-extraneous-exception] - raised by safe_version_path
        destination = safe_version_path(staging, relpath)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with self.open_file(dataset_id, version, relpath) as source, destination.open("wb") as sink:
            shutil.copyfileobj(source, sink)

    def search(self, filter: DatasetFilter) -> list[DatasetMetadata]:
        """Find datasets that match a filter.

        A ``task`` or ``time_series_spec`` filter reads the manifest of a dataset only if its card matches.

        Args:
            filter: The filters and the limit.

        Returns:
            The matching dataset metadata, sorted by dataset ID, with at most ``filter.limit`` entries.
        """
        # Sorted, so that the limit keeps the first matches by dataset ID, also for a catalog in another order.
        rows = sorted(self.list_datasets(), key=lambda row: row.dataset_id)
        return filter.select(ListedRelease(row, self) for row in rows)
