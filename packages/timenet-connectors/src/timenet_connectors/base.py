"""Base connector contract for TimeNet dataset integrations."""

from abc import ABC, abstractmethod


class BaseConnector(ABC):
    """Abstract base for dataset connectors.

    A connector holds the dataset-specific logic to fetch a dataset's raw
    artifacts and convert them into TimeNet's shared TimeF format. Each
    integrated dataset implements one concrete subclass.

    This is a placeholder; the full contract (metadata, download, convert,
    store) lands with the first real connector.
    """

    @abstractmethod
    def download(self) -> None:
        """Fetch the dataset's raw artifacts from their source."""

    @abstractmethod
    def convert(self) -> None:
        """Convert the downloaded raw artifacts into the TimeF format."""
