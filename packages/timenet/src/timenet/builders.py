"""Builder discovery: how the SDK finds something able to build a dataset the registry lacks.

The SDK cannot import a connectors package (the dependency runs the other way), so a build
backend registers itself under the ``timenet.builders`` entry-point group instead. This keeps the
SDK's whole knowledge of building to one protocol and one lookup, and lets a third-party connector
package plug in the same way the first-party one does.
"""

from importlib.metadata import entry_points
from pathlib import Path
from typing import Protocol

from timenet.errors import TimeFValidationError
from timenet.types.references import DatasetRef


ENTRY_POINT_GROUP = "timenet.builders"


class BuilderBackend(Protocol):
    """Something that can build a dataset from its connector."""

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
            The declared version string, or ``None`` if it cannot be read.
        """
        ...

    def build(
        self,
        dataset_id: str,
        root: Path,
        *,
        force: bool = False,
        build_parents: bool = False,
        fetch_parents: str | None = None,
    ) -> Path:
        """Build the dataset into a registry directory.

        Missing parent releases are prepared before the build starts, or the build fails before
        any download.

        Args:
            dataset_id: The dataset id.
            root: The output registry directory.
            force: Rebuild even if the version is already built.
            build_parents: Build missing parent releases from their installed connectors.
            fetch_parents: Registry URI to fetch missing parent releases from.

        Returns:
            The committed version directory.
        """
        ...


def find_builder(dataset_id: str) -> BuilderBackend | None:
    """Return the first registered backend that claims this dataset id.

    Args:
        dataset_id: The dataset id.

    Returns:
        The backend, or ``None`` when no installed package claims the id.
    """
    for entry in entry_points(group=ENTRY_POINT_GROUP):
        backend = entry.load()()
        if backend.knows(dataset_id):
            return backend
    return None


def build_exact(reference: DatasetRef, root: Path, *, fetch_parents: str | None = None) -> None:
    """Build a missing payload source from a connector that declares the exact release.

    The dependency resolver checks the resulting release for completeness. Existing committed
    releases are never forced over; an incomplete release must be restored or explicitly rebuilt.

    Args:
        reference: The required dataset ID and version.
        root: The local registry directory to build into.
        fetch_parents: Registry URI to fetch missing build inputs from.

    Raises:
        TimeFValidationError: If no installed connector declares the required version.
    """
    builder = find_builder(reference.dataset_id)
    if builder is None or builder.declared_version(reference.dataset_id) != str(reference.version):
        raise TimeFValidationError(f"no installed connector builds exact input {reference}")
    builder.build(reference.dataset_id, root, build_parents=True, fetch_parents=fetch_parents)
