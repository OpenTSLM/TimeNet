"""The :class:`BaseConnector` contract every dataset integration implements.

A connector fetches raw data and converts it into a :class:`~timenet.dataset.TimeFDataset`. It has no
knowledge of the registry, engine, or any other connector. The engine drives it
``download -> convert``, then stores the result itself. The consumer SDK never imports connector
code: it builds only through the ``timenet.curators`` entry point.
"""

from abc import ABC, abstractmethod
import asyncio
import inspect
from pathlib import Path
from typing import ClassVar, Generic, TypeVar

from timenet.dataset import TimeFDataset
from timenet.types import DatasetMetadata


TRaw = TypeVar("TRaw")


class BaseConnector(ABC, Generic[TRaw]):
    """Abstract base for dataset connectors. One concrete subclass per dataset.

    A connector lives in its own folder and declares its descriptive identity in a ``dataset.yaml``
    card beside it (read by :meth:`metadata`). Set :attr:`CARD` to point elsewhere. Subclasses
    implement the two abstract stages. ``download`` is I/O-only and ``convert`` is CPU-only.
    Connectors take no constructor arguments.
    """

    __test__ = False  # a connector named Test* (for example, the test_mean dataset) is not a pytest test class

    CARD: ClassVar[str | Path | None] = None
    """Optional explicit path to the dataset card YAML. When ``None`` (the default), the connector
    reads the card from ``dataset.yaml`` in its own folder."""

    def __init__(self) -> None:
        cls = type(self)
        # download() is concrete because it bridges to download_async. A subclass that overrides
        # neither would instantiate and only fail deep in the engine. Catch it at construction instead.
        if cls.download is BaseConnector.download and cls.download_async is BaseConnector.download_async:
            raise TypeError(f"{cls.__name__} must implement download() or download_async()")

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
        overrides it). Its ``dataset_id`` must match the id used to register or build the connector.

        Returns:
            The dataset's :class:`~timenet.types.DatasetMetadata`.
        """
        return DatasetMetadata.from_yaml(self._card_path())

    def download(self, cache_dir: Path) -> list[TRaw]:
        """Fetch or discover raw source files and return lightweight references to them.

        I/O only: no parsing, no array work. Must be idempotent for a given ``cache_dir``. Override
        this for a synchronous connector. For an I/O-bound one, override :meth:`download_async`
        instead and leave this default, which drives it to completion, since the engine calls
        connectors synchronously.

        Args:
            cache_dir: Directory to write downloaded files into (created by the engine).

        Returns:
            Raw references passed directly to :meth:`convert`.
        """
        return asyncio.run(self.download_async(cache_dir))

    async def download_async(self, cache_dir: Path) -> list[TRaw]:
        """Async variant of :meth:`download` for connectors whose downloads are I/O-bound.

        Override this to fetch artifacts concurrently, for example with the connector HTTP download
        helpers. The default :meth:`download` runs it for you. Implement exactly one of the two.

        Args:
            cache_dir: Directory to write downloaded files into (created by the engine).

        Returns:
            Raw references passed directly to :meth:`convert`.

        Raises:
            NotImplementedError: If a subclass overrides neither :meth:`download` nor
                :meth:`download_async`.
        """
        raise NotImplementedError(
            "a connector must implement download() (synchronous) or download_async() (asynchronous)"
        )

    @abstractmethod
    def convert(self, raw_refs: list[TRaw]) -> TimeFDataset:
        """Parse raw references and populate a :class:`~timenet.dataset.TimeFDataset`.

        CPU-bound: no network I/O. Attach time-series values as lazy loaders instead of materializing
        them.

        Args:
            raw_refs: The references returned by :meth:`download`.

        Returns:
            The populated dataset.
        """
