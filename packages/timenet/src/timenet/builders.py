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

from timenet.progress import ProgressCallback
from timenet.types import DatasetMetadata
from timenet.types.references import DatasetRef


if TYPE_CHECKING:
    from timenet.manifest import PayloadSource
    from timenet.registry import BaseRegistry, LocalRegistry


ENTRY_POINT_GROUP = "timenet.builders"


@dataclass(frozen=True)
class BuildOptions:
    """Options of a build that do not change the files of the release.

    Attributes:
        keep_cache: Keep the raw download cache after the build.
        progress: Receives the progress events of the build, or ``None``.
        quiet: Ask the backend to print nothing of its own. An isolated child otherwise prints its
            build lines and downloads to the inherited stderr.
    """

    keep_cache: bool = False
    progress: ProgressCallback | None = None
    quiet: bool = False


class BuilderBackend(Protocol):
    """Something that can build a dataset release from its connector.

    :meth:`build_prepared` builds exactly one release, into a store that already holds the parents
    of that release and their payload sources. It does not plan, copy or build other releases.
    """

    def dataset_ids(self) -> Iterable[str]:
        """Return the IDs of the datasets that this backend can build.

        Implementations must answer without importing a connector: this runs before any connector
        environment exists.

        Returns:
            The dataset IDs.

        Raises:
            TimeNetInvalidCardError: If the card of a connector cannot be read.
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
        """Build exactly this release into the store.

        Each parent of the release, and each payload source of such a parent, is already in
        ``store`` when this method runs. If ``pinned_by`` is not empty, other releases already
        read files of this release. The build then publishes the release only if it writes each
        pinned file again with the same path, size and checksum. Otherwise it publishes nothing
        and raises an error.

        Args:
            release: The exact dataset ID and version to build.
            store: The local registry to build into.
            force: Build the release again, even if ``store`` already holds it.
            pinned_by: The releases that already read files of this release, with their pins.
            options: The options of the build that do not change its files. The backend sends the
                progress events of the build to ``options.progress``. If ``options.quiet`` is set,
                the backend prints nothing of its own, also not in an isolated child.

        Raises:
            TimeNetDatasetNotFoundError: If this backend does not know the dataset ID of ``release``.
            TimeNetVersionUnavailableError: If the connector declares another version than
                ``release``. It is also a ``TimeFValidationError``.
            TimeFFormatError: If the build does not reproduce a pinned file. Nothing is published
                then.
            TimeNetBuildError: If the build runs but fails.
        """
        ...


def find_builder(dataset_id: str) -> BuilderBackend | None:
    """Return the first registered backend whose dataset IDs hold this dataset ID.

    Args:
        dataset_id: The dataset id.

    Returns:
        The backend, or ``None`` when no installed package claims the id.

    Raises:
        TimeNetInvalidCardError: If a backend cannot list its dataset IDs, because the card of a
            connector cannot be read.
    """  # ruff: ignore[docstring-extraneous-exception] - raised by the backend
    for entry in entry_points(group=ENTRY_POINT_GROUP):
        backend = entry.load()()
        if dataset_id in backend.dataset_ids():
            return backend
    return None


def find_card(dataset_id: str) -> DatasetMetadata | None:
    """Return the card of the installed connector that claims this dataset.

    Args:
        dataset_id: The dataset id.

    Returns:
        The card, or ``None`` when no installed package claims the dataset.

    Raises:
        TimeNetInvalidCardError: If the card of the connector cannot be read.
    """  # ruff: ignore[docstring-extraneous-exception] - raised by the backend
    builder = find_builder(dataset_id)
    return None if builder is None else builder.card(dataset_id)


def exact_builder(reference: DatasetRef) -> BuilderBackend | None:
    """Return the backend that can build exactly this release.

    A connector builds only the dataset version that its card declares. So a pinned release can only
    be built by a connector whose card declares that exact version.

    Args:
        reference: The dataset ID and the version.

    Returns:
        The backend, or ``None`` if no installed backend claims the dataset or its card declares
        another version.

    Raises:
        TimeNetInvalidCardError: If the card of the connector cannot be read.
    """  # ruff: ignore[docstring-extraneous-exception] - raised by the backend
    builder = find_builder(reference.dataset_id)
    card = None if builder is None else builder.card(reference.dataset_id)
    return builder if card is not None and card.dataset_version == reference.version else None
