"""The :class:`BaseConnector` contract every dataset integration implements.

A connector fetches raw data and converts it into a :class:`~timenet.dataset.TimeFDataset`. It has no
knowledge of the registry, engine, or any other connector. The engine drives it
``download -> convert -> derive_schema -> store``; the consumer SDK never runs connector code.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable
import inspect
from pathlib import Path
from typing import ClassVar, Generic, TypeVar

from timenet.dataset import TimeFDataset
from timenet.types import DatasetMetadata
from timenet.writer import TimeFWriter, WriteProgressEvent


TRaw = TypeVar("TRaw")


class BaseConnector(ABC, Generic[TRaw]):
    """Abstract base for dataset connectors. One concrete subclass per dataset.

    A connector lives in its own folder and declares its descriptive identity in a ``dataset.yaml``
    card beside it (read by :meth:`metadata`); set :attr:`CARD` to point elsewhere. Subclasses
    implement ``convert`` (CPU-only) and, for networked sources, ``download`` (I/O-only); a synthetic
    connector that generates everything in ``convert`` can skip ``download``, which defaults to
    returning no references. Connectors take no constructor arguments.
    """

    CARD: ClassVar[str | Path | None] = None
    """Optional explicit path to the dataset card YAML. When ``None`` (the default), the card is read
    from ``dataset.yaml`` in the connector's own folder."""

    @classmethod
    def _card_path(cls) -> Path:
        """Resolve the dataset card path by convention.

        Returns:
            :attr:`CARD` if set, otherwise ``dataset.yaml`` in the directory of the connector's module.
        """
        if cls.CARD is not None:
            return Path(cls.CARD)
        return Path(inspect.getfile(cls)).with_name("dataset.yaml")

    def metadata(self) -> DatasetMetadata:
        """Return the dataset's descriptive identity, loaded and validated from its card YAML.

        Reads the card by convention (``dataset.yaml`` beside the connector, unless :attr:`CARD`
        overrides it). Its ``dataset_id`` must match the id the connector is registered/curated under.

        Returns:
            The dataset's :class:`~timenet.types.DatasetMetadata`.
        """
        return DatasetMetadata.from_yaml(self._card_path())

    def download(self, cache_dir: Path) -> list[TRaw]:  # noqa: ARG002 (default no-op: synthetic connectors)
        """Fetch or discover raw source files and return lightweight references to them.

        I/O only: no parsing, no array work. Must be idempotent for a given ``cache_dir``. Defaults to
        returning no references, so a synthetic connector that builds everything in :meth:`convert`
        need not override it.

        Args:
            cache_dir: Directory to write downloaded files into (created by the engine).

        Returns:
            Raw references passed directly to :meth:`convert` (empty by default).
        """
        return []

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
