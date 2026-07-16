"""The :class:`BaseConnector` contract every dataset integration implements.

A connector fetches raw data and converts it into a :class:`~timenet.dataset.TimeFDataset`. It has no
knowledge of the registry, engine, or any other connector. The engine drives it
``download -> convert -> derive_schema -> store``; the consumer SDK never runs connector code.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path
from typing import ClassVar, Generic, TypeVar

from timenet.dataset import TimeFDataset
from timenet.types import DatasetMetadata
from timenet.writer import TimeFWriter, WriteProgressEvent


TRaw = TypeVar("TRaw")


class BaseConnector(ABC, Generic[TRaw]):
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

    def store(
        self,
        dataset: TimeFDataset,
        root: Path,
        *,
        progress_cb: Callable[[WriteProgressEvent], None] | None = None,
    ) -> Path:
        """Serialize a populated dataset to the TimeF format under ``root``.

        Derives the schema first if the dataset has none, then streams it through a
        :class:`~timenet.writer.TimeFWriter`. Most connectors do not override this.

        Args:
            dataset: The populated dataset from :meth:`convert`.
            root: Parent directory; the version directory is created beneath it.
            progress_cb: Optional writer progress callback.

        Returns:
            The committed version directory.
        """
        if dataset.schema is None:
            dataset.derive_schema()
        with TimeFWriter(root, dataset, progress_cb=progress_cb) as writer:
            writer.write()
        return root / dataset.metadata.dataset_id / str(dataset.metadata.dataset_version)
