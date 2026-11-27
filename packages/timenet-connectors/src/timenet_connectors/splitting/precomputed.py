"""A splitter that reads a split that a script computed once and wrote to a file beside its connector."""

from collections import Counter
from collections.abc import Callable, Sequence
import lzma
from pathlib import Path
from typing import Self

from pydantic import BaseModel, ConfigDict, NonNegativeInt, ValidationError, model_validator

from timenet.errors import TimeNetConnectorError
from timenet.types.splits import Split
from timenet_connectors.splitting.base import DatasetSplits, Splitter, T


SampleKey = NonNegativeInt | str
"""The two ways a split file can store a sample key: the integer index of the sample in the samples,
or a more generic string key."""


class _SplitFile(BaseModel):
    """The content of a split file.

    It names the samples of each split and the digest of the source that the split was made from.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_sha256: str
    """SHA-256 digest of the source file that the split was made from."""
    train_keys: tuple[SampleKey, ...]
    validation_keys: tuple[SampleKey, ...]
    test_keys: tuple[SampleKey, ...]

    @model_validator(mode="after")
    def _name_each_key_once(self) -> Self:
        counts = Counter(key for keys in self.keys_by_split().values() for key in keys)
        repeated = sorted((key for key, count in counts.items() if count > 1), key=str)
        if repeated:
            raise ValueError(f"{repeated} are named more than once")
        return self

    @classmethod
    def from_file(cls, path: Path, *, source_sha256: str) -> Self:
        """Read a split file, and check that the split was made from the pinned source.

        The file is JSON, compressed with xz when its name ends in ``.xz``.

        Args:
            path: The split file.
            source_sha256: The pinned digest of the source file.

        Returns:
            The content of the file.

        Raises:
            TimeNetConnectorError: If the file does not hold a valid split, or the split was made from
                another source.
        """
        try:
            content = path.read_bytes()
            if path.suffix == ".xz":
                content = lzma.decompress(content)
            split_file = cls.model_validate_json(content)
        except (lzma.LZMAError, ValidationError) as exc:
            raise TimeNetConnectorError(f"{path.name} does not hold a valid split: {exc}") from exc
        if split_file.source_sha256 != source_sha256:
            raise TimeNetConnectorError(
                f"the split was made from the source {split_file.source_sha256}, not from the pinned {source_sha256}"
            )
        return split_file

    def keys_by_split(self) -> dict[Split, tuple[SampleKey, ...]]:
        """Return the keys of each split.

        Returns:
            The train, validation and test keys.
        """
        return {Split.TRAIN: self.train_keys, Split.VALIDATION: self.validation_keys, Split.TEST: self.test_keys}


class PrecomputedSplitter(Splitter[T]):
    """A Splitter that assigns samples to a split based on a precomputed mapping stored in a file.

    The split files are created by the ``generate_splits.py`` script beside each connector that uses this
    splitter. The script ran once, and its output file is committed next to the connector.

    We recommend using this when the source data stores no split of its own. By precomputing and storing
    the split instead of computing it at build time, we make sure a changing library dependency cannot
    change the split without a warning.
    """

    def __init__(self, split_file: _SplitFile, *, key: Callable[[T], SampleKey] | None = None) -> None:
        """Create a splitter from the content of a split file.

        Args:
            split_file: The content of the split file.
            key: Optional mapping from a sample to its key. Without it, samples are keyed by position.
        """
        self._split_of = {
            sample_key: split for split, keys in split_file.keys_by_split().items() for sample_key in keys
        }
        self._key = key

    @classmethod
    def from_file(cls, path: Path, *, source_sha256: str, key: Callable[[T], SampleKey] | None = None) -> Self:
        """Read a split file, and create a splitter from it.

        Args:
            path: The split file.
            source_sha256: The pinned digest of the source file.
            key: Optional mapping from a sample to its key. Without it, samples are keyed by position.

        Returns:
            The splitter.
        """
        return cls(_SplitFile.from_file(path, source_sha256=source_sha256), key=key)

    def split(self, samples: Sequence[T]) -> DatasetSplits[T]:
        """Put each sample in its split, keeping the order of the samples within each split.

        The split must name each key of the samples, and no other key. Otherwise the method raises a
        :class:`~timenet.errors.TimeNetConnectorError`.

        Returns:
            The train, validation and test samples.
        """
        # Map samples to keys.
        keys: Sequence[SampleKey] = range(len(samples)) if self._key is None else [self._key(s) for s in samples]
        # Validate sample set against split set.
        self._check_that_the_split_names_exactly(keys)
        # Build the partitions.
        partitions: dict[Split, list[T]] = {Split.TRAIN: [], Split.VALIDATION: [], Split.TEST: []}
        for sample, sample_key in zip(samples, keys, strict=True):
            partitions[self._split_of[sample_key]].append(sample)
        return DatasetSplits(partitions)

    def _check_that_the_split_names_exactly(self, keys: Sequence[SampleKey]) -> None:
        """Check that the split names each key of the samples, and no other key.

        Raises:
            TimeNetConnectorError: If the split names a key that no sample has, or a sample has a key
                that the split does not name.
        """
        unknown = sorted(set(self._split_of) - set(keys), key=str)
        if unknown:
            raise TimeNetConnectorError(f"the split names {unknown} that the source does not have")
        unassigned = sorted(set(keys) - set(self._split_of), key=str)
        if unassigned:
            raise TimeNetConnectorError(f"the split does not name {unassigned}, which the source has")
