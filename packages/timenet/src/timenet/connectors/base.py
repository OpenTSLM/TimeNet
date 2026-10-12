"""The :class:`BaseConnector` contract every dataset integration implements.

A connector downloads raw data and builds a :class:`~timenet.dataset.TimeFDataset`.
The engine calls ``convert``, or ``compose`` for cards with parents, then stores the dataset.
The consumer SDK builds through the ``timenet.builders`` entry point.
"""

from abc import ABC
import asyncio
import inspect
from pathlib import Path
from typing import ClassVar, Generic, TypeVar

from timenet.composition import BuildContext
from timenet.dataset import TimeFDataset
from timenet.types import DatasetMetadata


TRaw = TypeVar("TRaw")


class BaseConnector(ABC, Generic[TRaw]):
    """Abstract base for dataset connectors. One concrete subclass per dataset.

    Each connector has a ``dataset.yaml`` card beside it. :attr:`CARD` can override the card path.
    Subclasses implement ``download`` and ``convert``, or ``compose`` for cards with parents.
    Connectors take no constructor arguments.
    """

    __test__ = False  # a connector named Test* (for example, the test_mean dataset) is not a pytest test class

    CARD: ClassVar[str | Path | None] = None
    """Optional explicit path to the dataset card YAML. When ``None`` (the default), the connector
    reads the card from ``dataset.yaml`` in its own folder."""

    values_backend: str = "parquet"
    """Default storage backend for this connector's values plane."""

    def __init__(self) -> None:
        cls = type(self)
        # download() is concrete because it bridges to download_async. A subclass that overrides
        # neither would instantiate and only fail deep in the engine. Catch it at construction instead.
        if cls.download is BaseConnector.download and cls.download_async is BaseConnector.download_async:
            raise TypeError(f"{cls.__name__} must implement download() or download_async()")
        if cls.convert is BaseConnector.convert and cls.compose is BaseConnector.compose:
            raise TypeError(f"{cls.__name__} must implement convert() or compose()")

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

    def convert(self, raw_refs: list[TRaw]) -> TimeFDataset:
        """Parse raw references and populate a :class:`~timenet.dataset.TimeFDataset`.

        Use lazy Signal loaders without network I/O. Cards with parents use :meth:`compose` instead.

        Args:
            raw_refs: The references returned by :meth:`download`.

        Returns:
            The populated dataset.

        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement convert(), or compose() when its card declares parents"
        )

    def compose(self, raw_refs: list[TRaw], context: BuildContext) -> TimeFDataset:
        """Build a dataset from raw references and its declared parents.

        Like :meth:`convert`, this method runs without network I/O.

        Args:
            raw_refs: The references returned by :meth:`download`.
            context: Open parent views keyed by their full dataset IDs.

        Returns:
            The populated dataset with parent records imported by reference.

        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """
        raise NotImplementedError(f"{type(self).__name__} declares parents in its card, so it must implement compose()")
