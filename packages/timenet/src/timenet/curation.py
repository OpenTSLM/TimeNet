"""Curator discovery: how the SDK finds something able to build a dataset the registry lacks.

The SDK cannot import a connectors package (the dependency runs the other way), so a curation
backend registers itself under the ``timenet.curators`` entry-point group instead. This keeps the
SDK's whole knowledge of curation to one protocol and one lookup, and lets a third-party connector
package plug in the same way the first-party one does.
"""

from importlib.metadata import entry_points
from pathlib import Path
from typing import Protocol


ENTRY_POINT_GROUP = "timenet.curators"


class CuratorBackend(Protocol):
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

    def build(self, dataset_id: str, root: Path, *, force: bool = False) -> Path:
        """Build the dataset into a registry directory.

        Args:
            dataset_id: The dataset id.
            root: The output registry directory.
            force: Rebuild even if the version is already curated.

        Returns:
            The committed version directory.
        """
        ...


def find_curator(dataset_id: str) -> CuratorBackend | None:
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
