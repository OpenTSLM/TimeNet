"""Builder discovery: how the SDK finds something able to build a dataset the registry lacks.

The SDK cannot import a connectors package (the dependency runs the other way), so a build
backend registers itself under the ``timenet.builders`` entry-point group instead. This keeps the
SDK's whole knowledge of building to one protocol and one lookup, and lets a third-party connector
package plug in the same way the first-party one does.
"""

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from importlib.metadata import EntryPoint, entry_points
from typing import TYPE_CHECKING, Literal, Protocol

from timenet.errors import TimeNetDatasetNotFoundError, TimeNetVersionUnavailableError
from timenet.progress import ProgressCallback
from timenet.types import DatasetMetadata, Version
from timenet.types.references import DatasetRef


if TYPE_CHECKING:
    from timenet.manifest import PayloadSource
    from timenet.registry import LocalRegistry


ENTRY_POINT_GROUP = "timenet.builders"


@dataclass(frozen=True)
class BuildOptions:
    """Build settings that do not change the files of the release.

    Attributes:
        keep_cache: Keep the raw download cache after the build.
        progress: The callback for build progress events, or ``None``.
        quiet: Suppress output from the backend and its child processes.
        isolation: Use a separate environment (``"on"``), this process (``"off"``), or ``TIMENET_ISOLATION`` (``None``).
    """

    keep_cache: bool = False
    progress: ProgressCallback | None = None
    quiet: bool = False
    isolation: Literal["on", "off"] | None = None


class BuilderBackend(Protocol):
    """Define the interface for reading connector cards and building dataset releases."""

    def dataset_ids(self) -> Iterable[str]:
        """List the datasets this backend can build.

        Implementations must answer without importing a connector.

        Returns:
            The dataset IDs.

        Raises:
            TimeNetInvalidCardError: If a connector card cannot be read.
        """
        ...

    def card(self, dataset_id: str) -> DatasetMetadata | None:
        """Return the card the connector declares, without importing or building it.

        A build planner reads the card for a parent's exact version, its own parents, and its access
        requirement before anything runs. Answer without importing the connector, like :meth:`dataset_ids`.

        Args:
            dataset_id: The dataset id.

        Returns:
            The validated card, or ``None`` if the backend has no connector for the id.

        Raises:
            TimeNetInvalidCardError: If the card cannot be read.
        """
        ...

    def build(
        self,
        release: DatasetRef,
        store: "LocalRegistry",
        *,
        force: bool,
        pinned_by: "Mapping[DatasetRef, PayloadSource]",
        options: BuildOptions,
    ) -> None:
        """Build the requested dataset version into a local registry.

        The local registry must already contain all parents and their payload sources.
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
class InstalledBuilder:
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


@dataclass(frozen=True)
class ConnectorClaim:
    """A dataset ID and the installed backend that owns it.

    Attributes:
        dataset_id: The dataset ID.
        backend: The first backend that lists the ID.
        entry: The entry point that registers the backend.
    """

    dataset_id: str
    backend: BuilderBackend
    entry: EntryPoint


def claims() -> Iterator[ConnectorClaim]:
    """Yield each dataset ID of the installed backends once, owned by the first backend that lists it.

    The walk loads each backend only when it reaches its entry point.

    Yields:
        One claim per dataset ID, in the order of the entry points.
    """
    seen: set[str] = set()
    for entry in entry_points(group=ENTRY_POINT_GROUP):
        backend = entry.load()()
        for dataset_id in backend.dataset_ids():
            if dataset_id not in seen:
                seen.add(dataset_id)
                yield ConnectorClaim(dataset_id, backend, entry)


def installed_builder(dataset_id: str) -> InstalledBuilder | None:
    """Find the first registered backend that lists a dataset, and read its card.

    Args:
        dataset_id: The dataset ID.

    Returns:
        The installed builder, or ``None`` if no backend lists the dataset or the backend has no card for it.

    Raises:
        TimeNetInvalidCardError: If the dataset's card cannot be read.
    """  # ruff: ignore[docstring-extraneous-exception] - raised by the backend
    claim = next((claim for claim in claims() if claim.dataset_id == dataset_id), None)
    if claim is None:
        return None
    card = claim.backend.card(dataset_id)
    return None if card is None else InstalledBuilder(claim.backend, card)


def declared(dataset_id: str, version: Version | str | None = None) -> InstalledBuilder:
    """Find a dataset's build backend and check its version.

    Args:
        dataset_id: The dataset ID.
        version: The version to build, or ``None`` for the version that the card declares.

    Returns:
        The installed builder, whose ``release`` is the release to build.

    Raises:
        TimeNetDatasetNotFoundError: If no installed connector builds the dataset.
        TimeNetVersionUnavailableError: If the connector builds another version.
        TimeNetInvalidCardError: If the dataset's card cannot be read.
    """  # ruff: ignore[docstring-extraneous-exception] - the card error is raised by the backend
    found = installed_builder(dataset_id)
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
    found = installed_builder(dataset_id)
    return None if found is None else found.card
