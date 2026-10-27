"""A dataset handle: a reader's whole-dataset entry point that loads every value."""

from typing import TYPE_CHECKING

from timenet.dataset import TimeFDataset
from timenet.reader.load_plan import LoadPlan
from timenet.types import DatasetMetadata


if TYPE_CHECKING:
    from timenet.reader.reader import TimeFReader


class DatasetHandle:
    """Hold an open reader; creating one reads neither the catalog nor any payload."""

    def __init__(self, reader: "TimeFReader") -> None:
        self.reader = reader

    @property
    def metadata(self) -> DatasetMetadata:
        """Return manifest metadata without opening the catalog."""
        return self.reader.metadata

    def read(self) -> TimeFDataset:
        """Return the whole dataset with every stored value loaded."""
        return self.reader.read(load=LoadPlan.ALL_EAGER)

    def close(self) -> None:
        """Release reader connections and cached payload groups."""
        self.reader.close()

    def __enter__(self) -> "DatasetHandle":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
