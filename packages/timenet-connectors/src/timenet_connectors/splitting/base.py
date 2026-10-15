"""Common splitter contract and named partitions of unchanged source samples."""

from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping, Sequence
from typing import Generic, TypeVar

from timenet.types.splits import Split


T = TypeVar("T")


class DatasetSplits(Mapping[Split, tuple[T, ...]]):
    """Named partitions, accessed with ``splits[Split.TRAIN]``.

    Iteration yields partition names, not samples. An absent partition raises
    ``KeyError``; a configured but empty partition contains an empty tuple.
    Samples retain their original identity and order within each partition.
    """

    def __init__(self, partitions: Mapping[Split, Sequence[T]]) -> None:
        self._partitions = {split: tuple(samples) for split, samples in partitions.items()}

    def __getitem__(self, split: Split) -> tuple[T, ...]:
        return self._partitions[split]

    def __iter__(self) -> Iterator[Split]:
        return iter(self._partitions)

    def __len__(self) -> int:
        return len(self._partitions)

    @property
    def counts(self) -> Mapping[Split, int]:
        """Actual sample counts, including configured partitions with no samples."""
        return {split: len(samples) for split, samples in self.items()}


class Splitter(ABC, Generic[T]):
    """Assign samples to partitions without modifying the samples themselves."""

    @abstractmethod
    def split(self, samples: Sequence[T]) -> DatasetSplits[T]:
        """Partition source samples for a connector to turn into split-tagged tasks.

        Returns:
            Named partitions containing every input occurrence exactly once.
        """
