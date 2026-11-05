"""Builder discovery: how the SDK finds something able to build a dataset the registry lacks.

The SDK cannot import a connectors package (the dependency runs the other way), so a build
backend registers itself under the ``timenet.builders`` entry-point group instead. This keeps the
SDK's whole knowledge of building to one protocol and one lookup, and lets a third-party connector
package plug in the same way the first-party one does.
"""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from importlib.metadata import entry_points
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from timenet.errors import TimeNetDatasetNotFoundError, TimeNetVersionUnavailableError
from timenet.progress import ProgressCallback
from timenet.types import DatasetMetadata, Version
from timenet.types.references import DatasetRef


if TYPE_CHECKING:
    from timenet.manifest import PayloadSource
    from timenet.registry import BaseRegistry, LocalRegistry


ENTRY_POINT_GROUP = "timenet.builders"


@dataclass(frozen=True)
class BuildOptions:
    """Build settings that do not change the files of the release.

    Attributes:
        keep_cache: Keep the raw download cache after the build.
        progress: The callback for build progress events, or ``None``.
        quiet: Suppress output from the backend and its child processes.
    """

    keep_cache: bool = False
    progress: ProgressCallback | None = None
    quiet: bool = False


class BuilderBackend(Protocol):
    """An interface for building dataset releases from connectors."""

    def dataset_ids(self) -> Iterable[str]:
        """List the datasets this backend can build.

        Implementations must answer without importing a connector.

        Returns:
            The dataset IDs.

        Raises:
            TimeNetInvalidCardError: If a connector card cannot be read.
        """
        ...

    def knows(self, dataset_id: str) -> bool:
        """Report whether this backend has a connector for the id.

        Implementations must answer without importing the connector: this runs before any
        connector environment exists.

        Args:
            dataset_id: The dataset id.

        Returns:
            Whether the backend can build it.
        """
        ...

    def declared_version(self, dataset_id: str) -> str | None:
        """Return the version the connector declares, without importing or building it.

        A caller checks a version pin against this before the build runs, so a pin the connector
        cannot satisfy fails fast instead of after a full build. Answer without importing the
        connector, like :meth:`knows`.

        Args:
            dataset_id: The dataset id.

        Returns:
            The declared version string, or ``None`` if the backend has no connector for the id.

        Raises:
            TimeNetInvalidCardError: If the card of the connector cannot be read.
        """
        ...

    def card(self, dataset_id: str) -> DatasetMetadata | None:
        """Return the card the connector declares, without importing or building it.

        A build planner reads the card for a parent's exact version, its own parents, and its access
        requirement before anything runs. Answer without importing the connector, like :meth:`knows`.

        Args:
            dataset_id: The dataset id.

        Returns:
            The validated card, or ``None`` if the backend has no connector for the id.

        Raises:
            TimeNetInvalidCardError: If the card cannot be read.
        """
        ...

    def build(  # ruff: ignore[too-many-arguments]
        self,
        dataset_id: str,
        root: Path,
        *,
        force: bool = False,
        build_parents: bool = False,
        fetch_from: "str | BaseRegistry | None" = None,
        pinned_by: "Mapping[DatasetRef, PayloadSource] | None" = None,
    ) -> Path:
        """Build the dataset into a registry directory.

        Missing parent releases are prepared before the build starts, or the build fails before
        any download. Each one is reused, copied from ``fetch_from``, or built when
        ``build_parents`` is set. A build with ``pinned_by`` publishes the release only if it
        reproduces each pinned file.

        Args:
            dataset_id: The dataset id.
            root: The output registry directory.
            force: Rebuild even if the version is already built.
            build_parents: Build missing parent releases from their installed connectors.
            fetch_from: The registry, or its URI, to copy missing parent releases from.
            pinned_by: The releases that already read files of this release, with their pins.

        Returns:
            The committed version directory.
        """
        ...

    def build_prepared(
        self,
        release: DatasetRef,
        store: "LocalRegistry",
        *,
        force: bool,
        pinned_by: "Mapping[DatasetRef, PayloadSource]",
        options: BuildOptions,
    ) -> None:
        """Build the requested dataset version into a local registry.

        The store must already contain all parents and their payload sources.
        Publish only if all pinned files keep the same path, size, and checksum.

        Args:
            release: The exact dataset ID and version to build.
            store: The local registry to build into.
            force: Rebuild even if ``store`` already holds the release.
            pinned_by: The releases that pin files in this release, with their file records.
            options: Build settings, including ``options.progress`` to receive build events.

        Raises:
            TimeNetDatasetNotFoundError: If this backend has no connector for the dataset ID.
            TimeNetVersionUnavailableError: If the connector declares a different version.
            TimeFFormatError: If the build omits or changes a pinned file.
            TimeNetBuildError: If the build fails.
        """
        ...


@dataclass(frozen=True)
class Claim:
    """A dataset's build backend and connector card.

    Attributes:
        backend: The installed backend that builds the dataset.
        card: The dataset's connector card.
    """

    backend: BuilderBackend
    card: DatasetMetadata

    @property
    def release(self) -> DatasetRef:
        """The release that the card declares."""
        return DatasetRef(dataset_id=self.card.dataset_id, version=self.card.dataset_version)


def claim(dataset_id: str) -> Claim | None:
    """Find the first registered backend that lists a dataset, and read its card.

    Args:
        dataset_id: The dataset ID.

    Returns:
        The claim, or ``None`` if no backend lists the dataset or the backend has no card for it.

    Raises:
        TimeNetInvalidCardError: If the dataset's card cannot be read.
    """  # ruff: ignore[docstring-extraneous-exception] - raised by the backend
    for entry in entry_points(group=ENTRY_POINT_GROUP):
        backend = entry.load()()
        if dataset_id in backend.dataset_ids():
            card = backend.card(dataset_id)
            return None if card is None else Claim(backend, card)
    return None


def declared(dataset_id: str, version: Version | str | None = None) -> Claim:
    """Find a dataset's build backend and check its version.

    Args:
        dataset_id: The dataset ID.
        version: The version to build, or ``None`` for the version that the card declares.

    Returns:
        The claim, whose ``release`` is the release to build.

    Raises:
        TimeNetDatasetNotFoundError: If no installed connector builds the dataset.
        TimeNetVersionUnavailableError: If the connector builds another version.
        TimeNetInvalidCardError: If the dataset's card cannot be read.
    """  # ruff: ignore[docstring-extraneous-exception] - the card error is raised by the backend
    found = claim(dataset_id)
    if found is None:
        raise TimeNetDatasetNotFoundError(f"no installed connector builds {dataset_id!r}")
    if version is not None:
        wanted = DatasetRef(dataset_id=dataset_id, version=version)
        if wanted.version != found.card.dataset_version:
            raise TimeNetVersionUnavailableError(wanted, found.card.dataset_version)
    return found


def find_card(dataset_id: str) -> DatasetMetadata | None:
    """Read a dataset's card from the first registered backend that lists it.

    Args:
        dataset_id: The dataset ID.

    Returns:
        The card, or ``None`` if no backend lists the dataset or the backend has no card.

    Raises:
        TimeNetInvalidCardError: If the card cannot be read.
    """  # ruff: ignore[docstring-extraneous-exception] - raised by the backend
    found = claim(dataset_id)
    return None if found is None else found.card
