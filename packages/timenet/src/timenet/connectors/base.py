"""The :class:`BaseConnector` contract every dataset integration implements.

A connector fetches raw data and converts it into a :class:`~timenet.dataset.TimeFDataset`. It has no
knowledge of the registry, engine, or any other connector. The engine drives it
``download -> convert``, then stores the result itself; the consumer SDK never runs connector code.
"""

from abc import ABC, abstractmethod
import inspect
from pathlib import Path
from typing import ClassVar, Generic, TypeVar

from timenet.dataset import TimeFDataset
from timenet.types import DatasetMetadata


TRaw = TypeVar("TRaw")


class BaseConnector(ABC, Generic[TRaw]):
    """Abstract base for dataset connectors. One concrete subclass per dataset.

    A connector declares its descriptive identity in a dataset card YAML beside its module (read by
    :meth:`metadata`); set :attr:`CARD` to point elsewhere. Subclasses implement the two abstract
    stages, kept distinct: ``download`` is I/O-only and ``convert`` is CPU-only. Connectors take no
    constructor arguments.
    """

    CARD: ClassVar[str | Path | None] = None
    """Explicit path to the dataset card YAML. When ``None``, the card is read from ``dataset.yaml`` in
    the connector's own folder (:meth:`_card_path`)."""

    @classmethod
    def _card_path(cls) -> Path:
        """Resolve the dataset card path.

        Returns:
            :attr:`CARD` if set, otherwise ``dataset.yaml`` beside the connector's module.
        """
        if cls.CARD is not None:
            return Path(cls.CARD)
        return Path(inspect.getfile(cls)).with_name("dataset.yaml")

    def metadata(self) -> DatasetMetadata:
        """Return the dataset's descriptive identity, loaded and validated from its card YAML.

        The card is the single source of truth for identity; the connector never restates it in code.
        Its ``dataset_id`` must match the id the connector is registered/curated under.

        Returns:
            The dataset's :class:`~timenet.types.DatasetMetadata`.
        """
        return DatasetMetadata.from_yaml(self._card_path())

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
