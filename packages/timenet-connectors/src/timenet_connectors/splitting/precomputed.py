"""A splitter that reads a split that a script computed once and wrote to a file beside its connector."""

from collections import Counter
from collections.abc import Callable, Iterable, Sequence
import lzma
from pathlib import Path
from typing import Self

from pydantic import BaseModel, ConfigDict, NonNegativeInt, ValidationError

from timenet.errors import TimeNetConnectorError
from timenet.types.splits import Split
from timenet_connectors.splitting.base import DatasetSplits, Splitter, T


SampleKey = NonNegativeInt | str
"""How a split file names a sample: by its position in the samples, or by a key such as a source id."""


class _SplitFile(BaseModel):
    """The content of a split file."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_sha256: str
    """SHA-256 digest of the source file that the split was made from."""
    validation: tuple[SampleKey, ...]
    test: tuple[SampleKey, ...]


class PrecomputedSplitter(Splitter[T]):
    """Put each sample in the split that a split file names for it, and every other sample in train.

    Use it when the source has no split of its own and the split depends on the version of a library,
    such as a seeded shuffle, so that making it again at build time could change it. A script makes
    the split once and writes the file. The file is JSON, compressed with xz when its name ends in
    ``.xz``. It names the validation and the test samples, either by their position in the samples or
    by the key that ``key`` returns for each sample.
    """

    def __init__(
        self,
        *,
        validation: Iterable[SampleKey],
        test: Iterable[SampleKey],
        key: Callable[[T], SampleKey] | None = None,
    ) -> None:
        """Create a splitter from the validation and test keys.

        Args:
            validation: The keys of the validation samples.
            test: The keys of the test samples.
            key: Return the key of a sample. Without it, the key of a sample is its position.

        Raises:
            TimeNetConnectorError: If a key is in both splits or twice in one split.
        """
        validation, test = tuple(validation), tuple(test)
        repeated = sorted((k for k, count in Counter((*validation, *test)).items() if count > 1), key=str)
        if repeated:
            raise TimeNetConnectorError(f"{repeated} are named more than once")
        self._split_of = dict.fromkeys(validation, Split.VALIDATION) | dict.fromkeys(test, Split.TEST)
        self._key = key

    @classmethod
    def from_file(cls, path: Path, *, source_sha256: str, key: Callable[[T], SampleKey] | None = None) -> Self:
        """Read the split from a file, and check that it was made from the pinned source.

        Args:
            path: The split file.
            source_sha256: The pinned digest of the source file.
            key: Return the key of a sample. Without it, the key of a sample is its position.

        Returns:
            The splitter.

        Raises:
            TimeNetConnectorError: If the file does not hold a valid split, or the split was made from
                another source.
        """
        try:
            content = path.read_bytes()
            if path.suffix == ".xz":
                content = lzma.decompress(content)
            split_file = _SplitFile.model_validate_json(content)
            splitter = cls(validation=split_file.validation, test=split_file.test, key=key)
        except (lzma.LZMAError, ValidationError, TimeNetConnectorError) as exc:
            raise TimeNetConnectorError(f"{path.name} does not hold a valid split: {exc}") from exc
        if split_file.source_sha256 != source_sha256:
            raise TimeNetConnectorError(
                f"the split was made from the source {split_file.source_sha256}, not from the pinned {source_sha256}"
            )
        return splitter

    def split(self, samples: Sequence[T]) -> DatasetSplits[T]:
        """Put each sample in its split, keeping the order of the samples within each split.

        Returns:
            The train, validation and test samples.

        Raises:
            TimeNetConnectorError: If the split names a key that no sample has.
        """
        keys: Sequence[SampleKey] = range(len(samples)) if self._key is None else [self._key(s) for s in samples]
        unknown = sorted(set(self._split_of) - set(keys), key=str)
        if unknown:
            raise TimeNetConnectorError(f"the split names {unknown} that the source does not have")
        partitions: dict[Split, list[T]] = {Split.TRAIN: [], Split.VALIDATION: [], Split.TEST: []}
        for sample, sample_key in zip(samples, keys, strict=True):
            partitions[self._split_of.get(sample_key, Split.TRAIN)].append(sample)
        return DatasetSplits(partitions)
