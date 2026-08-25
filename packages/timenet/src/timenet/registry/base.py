"""The registry contract and its shared search implementation.

A registry serves compiled TimeF versions. It does not run connector code. Concrete backends
must implement the four data-access methods: :meth:`list_datasets`, :meth:`get_manifest`,
:meth:`open_file`, and :meth:`open_version`. :meth:`BaseRegistry.search` is shared. It filters
the output of :meth:`list_datasets` and uses :meth:`get_manifest` for the type filters.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path
import shutil
from typing import BinaryIO, TypeVar
import uuid

from timenet.manifest import Manifest
from timenet.registry._paths import safe_version_path
from timenet.registry.version import DatasetVersion
from timenet.types import DatasetMetadata, Domain, License, Task


T = TypeVar("T")

# Reports download progress: called with each file (or chunk) of bytes as it lands. A ``download_version``
# implementation calls it so a caller can render a progress bar; ``None`` downloads silently.
ProgressCallback = Callable[[int], None]


class BaseRegistry(ABC):
    """A source of TimeF datasets: their manifests, files, and searchable metadata."""

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
    def open_version(self, dataset_id: str, version: str | None = None) -> DatasetVersion:
        """Open a committed dataset version as a random-access handle.

        The returned :class:`~timenet.registry.version.DatasetVersion` object bundles the parsed
        manifest with a handle to the files of the version. The handle is rooted in a filesystem.
        A reader built from this object does not re-open the registry. It also does not re-read
        ``manifest.json``. The filesystem of the handle MUST serve range reads. The reader uses it
        to read Parquet footers and value slices out of order.

        Args:
            dataset_id: The dataset id.
            version: The version string, or ``None`` for the latest version.

        Returns:
            A handle to the manifest and files of the committed version.

        Raises:
            TimeNetDatasetNotFoundError: If the dataset id or version is unknown.
        """

    def download_version(  # noqa: PLR0913
        self,
        dataset_id: str,
        version: str,
        dest_dir: str | Path,
        *,
        force: bool = False,
        manifest: Manifest | None = None,
        progress_cb: ProgressCallback | None = None,
    ) -> None:
        """Download a version's files into ``dest_dir``, swapping the directory in atomically.

        Fetches every file through :meth:`open_file` into a sibling ``<version>.tmp-*`` staging
        directory, with ``manifest.json`` last as the commit marker. It then renames the staging
        directory over ``dest_dir`` in one step. An interrupted download never leaves a half-written
        copy in place. Subclasses can override this method with a faster path, such as concurrent
        streaming from a remote service.

        Args:
            dataset_id: The dataset id.
            version: The version string.
            dest_dir: The target ``<...>/<id>/<version>`` directory.
            force: Re-download even if a copy already exists.
            manifest: The parsed manifest, passed to avoid re-fetching it. Fetched if ``None``.
            progress_cb: Called with each staged file's byte count, for a progress display.
        """
        if manifest is None:
            manifest = self.get_manifest(dataset_id, version)
        target = Path(dest_dir)
        if (target / "manifest.json").exists() and not force:
            return  # already downloaded and current
        parent = target.parent
        parent.mkdir(parents=True, exist_ok=True)
        # Each download stages into a unique dir and removes it in the finally below. A hard kill
        # (SIGKILL/power loss) can leave one behind, but do NOT sweep sibling <version>.tmp-* dirs here:
        # a concurrent download of the same version has a live staging dir with the same prefix, and
        # sweeping it would break that download mid-write. A rare orphaned dir is the lesser evil.
        staging = parent / f"{target.name}.tmp-{uuid.uuid4().hex}"
        try:
            for relpath in (*manifest.files.all_parts(), "manifest.json"):
                self._stage_file(dataset_id, version, relpath, staging)
                if progress_cb is not None and relpath != "manifest.json":
                    progress_cb((staging / relpath).stat().st_size)
            if target.exists():
                shutil.rmtree(target)
            staging.replace(target)
        finally:
            shutil.rmtree(staging, ignore_errors=True)

    def _stage_file(self, dataset_id: str, version: str, relpath: str, staging: Path) -> None:
        """Copy one version file from the registry into the ``staging`` directory.

        Args:
            dataset_id: The dataset id.
            version: The version string.
            relpath: The version-relative file path from the manifest.
            staging: The staging directory to write into.

        Raises:
            TimeFFormatError: If ``relpath`` would escape ``staging`` (an absolute or ``..`` path).
        """  # noqa: DOC502 - raised by safe_version_path
        destination = safe_version_path(staging, relpath)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with self.open_file(dataset_id, version, relpath) as source, destination.open("wb") as sink:
            shutil.copyfileobj(source, sink)

    def search(  # noqa: PLR0913
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
        schema = self.get_manifest(dataset_id).schema
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
