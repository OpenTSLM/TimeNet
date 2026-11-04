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
from typing import BinaryIO, TypeVar

from timenet._fs import staged_directory
from timenet.errors import TimeFValidationError, TimeNetDatasetNotFoundError, TimeNetFileError
from timenet.format.checksums import file_checksum
from timenet.format.constants import MANIFEST_FILE
from timenet.manifest import FilePart, Manifest
from timenet.registry._paths import safe_version_path, split_release_dir
from timenet.registry.dependencies import prepare_local_copy, validate_manifest_reference
from timenet.registry.shared import complete_copy
from timenet.registry.version import DatasetVersion
from timenet.sources import Found, NotFound, NotServed
from timenet.types import Access, DatasetMetadata, Domain, License, Task
from timenet.types.access import access_requirements, effective_access
from timenet.types.references import DatasetRef


T = TypeVar("T")

# Reports download progress: called with each file (or chunk) of bytes as it lands. A ``download_version``
# implementation calls it so a caller can render a progress bar; ``None`` downloads silently.
ProgressCallback = Callable[[int], None]


class BaseRegistry(ABC):
    """A searchable source of TimeF datasets, manifests, and files.

    :meth:`open_release` and :meth:`download_release` handle one release without its payload
    sources. :func:`~timenet.registry.copy_release` copies a release with its payload sources.
    """

    def __str__(self) -> str:
        return type(self).__name__

    @property
    @abstractmethod
    def local_root(self) -> Path:
        """The local registry or cache directory that holds prepared releases."""

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
        """Open one exact release without its payload sources.

        Opening reads only metadata. The handle has no payload sources,
        so a caller can open each of them from another source and attach it with
        :func:`~timenet.registry.dependencies.attach_payload_sources`.

        Args:
            reference: The exact release.
            manifest: The manifest of the release from this registry, if you already read it.

        Returns:
            A handle to the manifest and files of the release, without its payload sources.

        Raises:
            TimeNetDatasetNotFoundError: If the release is unknown.
            TimeFFormatError: If the manifest declares another release.
            TimeNetFileError: If a file of a local copy is missing or has another size than recorded.
        """

    def resolve(self, dataset_id: str, version: str | None) -> Found | NotFound | NotServed:
        """Check whether the registry serves a dataset release.

        A remote registry serves a release only if its card and each card in its lineage declare open access.

        Args:
            dataset_id: The dataset ID.
            version: An exact version, or ``None``, ``""``, or ``"latest"`` for the latest version.

        Returns:
            ``Found`` if served, ``NotServed`` if access is restricted, or ``NotFound`` if absent.
        """
        try:
            manifest = self.get_manifest(dataset_id, version)
        except TimeNetDatasetNotFoundError:
            return NotFound()
        held = manifest.metadata.dataset_version
        access = effective_access(manifest.metadata, manifest.lineage)
        if access is Access.OPEN:
            return Found(held, self, manifest)
        requirements = access_requirements(manifest.metadata, manifest.lineage)
        return NotServed(held, access, requirements[0].access_url if requirements else None, requirements, manifest)

    def download_version(  # ruff: ignore[too-many-arguments]
        self,
        dataset_id: str,
        version: str,
        dest_dir: str | Path,
        *,
        force: bool = False,
        manifest: Manifest | None = None,
        progress_cb: ProgressCallback | None = None,
    ) -> None:
        """Download complete local copies of a release and of each of its payload sources.

        Only the release's own payload sources are downloaded: it lists every release whose files it
        reads. Each is checked against the files the release pins.

        Args:
            dataset_id: The dataset ID.
            version: The exact release version.
            dest_dir: The local ``<registry>/<org>/<name>/<version>`` directory.
            force: Download the requested release again.
            manifest: An already-loaded manifest for the requested release.
            progress_cb: Report transferred bytes.

        Raises:
            TimeFValidationError: If the destination does not identify the requested release.
        """
        reference = DatasetRef.parse(f"{dataset_id}@{version}")
        root, destination_ref = split_release_dir(Path(dest_dir))
        if destination_ref != reference:
            raise TimeFValidationError(
                f"download destination {dest_dir} identifies {destination_ref}, expected {reference}"
            )
        prepare_local_copy(
            root,
            reference,
            fetch_from=self,
            manifest=manifest,
            force=force,
            progress_cb=progress_cb,
        )

    def download_release(
        self,
        reference: DatasetRef,
        directory: str | Path,
        *,
        force: bool = False,
        manifest: Manifest | None = None,
        progress_cb: ProgressCallback | None = None,
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

    def search(  # ruff: ignore[too-many-arguments]
        self,
        *,
        query: str | list[str] | None = None,
        domain: Domain | list[Domain] | None = None,
        task: type[Task] | list[type[Task]] | None = None,
        license: License | list[License] | None = None,
        time_series_spec: str | list[str] | None = None,
        dataset_id: str | list[str] | None = None,
        tag: str | list[str] | None = None,
        limit: int = 100,
    ) -> list[DatasetMetadata]:
        """Filter datasets by any combination of criteria.

        Each filter accepts a single value or a list of values. The method ignores filters set to
        ``None``. It combines all other filters with AND logic. The filters ``query``, ``domain``,
        ``task``, ``license``, and ``dataset_id`` match a dataset when any of their values match.
        The filters ``time_series_spec`` and ``tag`` match a dataset only when all of their values
        match. The type filters ``task`` and ``time_series_spec`` read the manifest schema of each
        dataset.

        Args:
            query: Free-text terms. The method matches each term as a case-insensitive substring
                in the dataset name, description, or tags.
            domain: Keep datasets that share any of these domains.
            task: Keep datasets whose schema includes any of these task classes.
            license: Keep datasets that have any of these licenses.
            time_series_spec: Keep datasets that declare all of these ``spec_type`` values.
            dataset_id: Keep only these ids.
            tag: Keep datasets that declare all of these tags.
            limit: The maximum number of results. The default is 100.

        Returns:
            The matching dataset metadata. The list has at most ``limit`` entries.

        Raises:
            ValueError: If ``limit`` is negative.
        """
        if limit < 0:
            raise ValueError(f"limit must be non-negative, got {limit}")
        queries = _as_list(query)
        domains = _as_list(domain)
        tasks = _as_list(task)
        licenses = _as_list(license)
        specs = _as_list(time_series_spec)
        ids = _as_list(dataset_id)
        tags = _as_list(tag)

        results: list[DatasetMetadata] = []
        for metadata in self.list_datasets():
            if len(results) >= limit:
                break
            if ids and metadata.dataset_id not in ids:
                continue
            if licenses and metadata.license not in licenses:
                continue
            if domains and not set(metadata.domains) & set(domains):
                continue
            if tags and not set(tags) <= set(metadata.tags):
                continue
            if queries and not _query_matches(metadata, queries):
                continue
            if (tasks or specs) and not self._schema_matches(metadata.dataset_id, tasks, specs):
                continue
            results.append(metadata)
        return results

    def _schema_matches(self, dataset_id: str, tasks: list[type[Task]], specs: list[str]) -> bool:
        """Return whether the manifest schema of a dataset satisfies the type filters."""
        schema = self.get_manifest(dataset_id).dataset_schema
        if tasks and not set(tasks) & set(schema.tasks):
            return False
        return not (specs and not set(specs) <= {spec.spec_type for spec in schema.time_series_specs})


def _as_list(value: T | list[T] | None) -> list[T]:
    """Normalize a filter value to a list.

    Args:
        value: ``None``, a single value, or a list of values.

    Returns:
        An empty list, the list itself, or a list with one value.
    """
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _query_matches(metadata: DatasetMetadata, terms: list[str]) -> bool:
    """Return whether any term is a case-insensitive substring of the dataset name, description, or tags."""
    haystack = " ".join([metadata.name, metadata.description, *metadata.tags]).lower()
    return any(term.lower() in haystack for term in terms)
