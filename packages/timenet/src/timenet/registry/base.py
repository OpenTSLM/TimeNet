"""The registry contract and its shared search implementation.

A registry serves compiled TimeF versions; it never runs connector code. Concrete backends implement
the four data-access methods (:meth:`list_datasets`, :meth:`get_manifest`, :meth:`open_file`,
:meth:`open_version`); :meth:`BaseRegistry.search` is shared, filtering :meth:`list_datasets` output
and consulting :meth:`get_manifest` for the type-filters.
"""

from abc import ABC, abstractmethod
from typing import BinaryIO, TypeVar

from timenet.manifest import Manifest
from timenet.registry.version import DatasetVersion
from timenet.types import DatasetMetadata, Domain, License, Task


T = TypeVar("T")


class BaseRegistry(ABC):
    """A source of TimeF datasets: their manifests, files, and searchable metadata."""

    @abstractmethod
    def list_datasets(self) -> list[DatasetMetadata]:
        """Return the metadata of every dataset (latest version), sorted by id.

        Returns:
            One :class:`~timenet.types.DatasetMetadata` per dataset.
        """

    @abstractmethod
    def get_manifest(self, dataset_id: str, version: str | None = None) -> Manifest:
        """Return a dataset's manifest.

        Args:
            dataset_id: The dataset id.
            version: The version string, or ``None`` for the latest.

        Returns:
            The dataset's :class:`~timenet.manifest.Manifest`.

        Raises:
            DatasetNotFoundError: If the dataset id or version is unknown.
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
            DatasetNotFoundError: If the dataset id or version is unknown.
        """

    @abstractmethod
    def open_version(self, dataset_id: str, version: str | None = None) -> DatasetVersion:
        """Open a committed dataset version as a random-access handle.

        The returned :class:`~timenet.registry.version.DatasetVersion` bundles the parsed manifest with a
        filesystem-rooted handle to the version's files, so a reader built from it never re-opens the
        registry nor re-reads ``manifest.json``. Its filesystem MUST serve range reads: the reader pulls
        Parquet footers and value slices out of order through it.

        Args:
            dataset_id: The dataset id.
            version: The version string, or ``None`` for the latest.

        Returns:
            A handle to the committed version's manifest and files.

        Raises:
            DatasetNotFoundError: If the dataset id or version is unknown.
        """

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

        Each filter accepts a scalar or a list; ``None`` filters are ignored and non-``None`` filters
        are ANDed. ``query``/``domain``/``task``/``license``/``dataset_id`` match any of their values;
        ``time_series_spec``/``tag`` require all of theirs. Type-filters (``task``,
        ``time_series_spec``) read each dataset's manifest schema.

        Args:
            query: Free-text terms matched (case-insensitive substring) against name/description/tags.
            domain: Keep datasets sharing any of these domains.
            task: Keep datasets whose schema includes any of these task classes.
            license: Keep datasets with any of these licenses.
            time_series_spec: Keep datasets declaring all of these ``spec_type`` values.
            dataset_id: Keep only these ids.
            tag: Keep datasets declaring all of these tags.
            limit: Maximum number of results (default 100).

        Returns:
            The matching dataset metadata, at most ``limit`` entries.

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
        """Return whether a dataset's manifest schema satisfies the type-filters."""
        schema = self.get_manifest(dataset_id).schema
        if tasks and not set(tasks) & set(schema.tasks):
            return False
        return not (specs and not set(specs) <= {spec.spec_type for spec in schema.time_series_specs})


def _as_list(value: T | list[T] | None) -> list[T]:
    """Normalize a scalar-or-list-or-None filter to a list.

    Args:
        value: ``None``, a single value, or a list of values.

    Returns:
        The empty list, the list itself, or a one-element list.
    """
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _query_matches(metadata: DatasetMetadata, terms: list[str]) -> bool:
    """Return whether any term is a case-insensitive substring of name/description/tags."""
    haystack = " ".join([metadata.name, metadata.description, *metadata.tags]).lower()
    return any(term.lower() in haystack for term in terms)
