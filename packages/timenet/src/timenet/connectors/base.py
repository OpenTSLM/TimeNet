"""The :class:`BaseConnector` contract every dataset integration implements.

A connector fetches raw data and converts it into a :class:`~timenet.dataset.TimeFDataset`. It has no
knowledge of the registry, engine, or any other connector. The engine drives it
``download -> convert -> derive_schema -> store``; the consumer SDK never runs connector code.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar

from timenet.dataset import TimeFDataset
from timenet.types import DatasetMetadata


class BaseConnector[TRaw](ABC):
    """Abstract base for dataset connectors. One concrete subclass per dataset.

    Subclasses set the ``METADATA`` class attribute and implement the two abstract stages, kept
    distinct: ``download`` is I/O-only and ``convert`` is CPU-only. Connectors take no constructor
    arguments.
    """

    METADATA: ClassVar[DatasetMetadata]
    """The dataset's descriptive identity, set by each concrete connector."""

    def metadata(self) -> DatasetMetadata:
        """Return the dataset's descriptive identity.

        Returns:
            The connector's static :class:`~timenet.types.DatasetMetadata`; its ``dataset_id`` must
            match the id the connector is registered/curated under.
        """
        return self.METADATA

    @abstractmethod
    def download(self, cache_dir: Path) -> list[TRaw]:
        """Fetch or discover raw source files and return lightweight references to them.

        I/O only: no parsing, no array work. Must be idempotent for a given ``cache_dir``.

        Args:
            cache_dir: Directory to write downloaded files into (created by the engine).

        Returns:
            Raw references passed directly to :meth:`convert`.
        """

    @abstractmethod
    def convert(self, raw_refs: list[TRaw]) -> TimeFDataset:
        """Parse raw references and populate a :class:`~timenet.dataset.TimeFDataset`.

        CPU-bound: no network I/O. Time-series values are attached as lazy loaders, not materialized.

        Args:
            raw_refs: The references returned by :meth:`download`.

        Returns:
            The populated dataset.
        """
