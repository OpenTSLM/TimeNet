"""A split that a script computed once and wrote to a file beside its connector."""

from collections import Counter
from collections.abc import Sequence
import lzma
from pathlib import Path
from typing import Generic, Self, TypeVar

from pydantic import BaseModel, ConfigDict, ValidationError, model_validator

from timenet.errors import TimeNetConnectorError
from timenet.types.splits import Split


K = TypeVar("K", int, str)
"""How a split file names a sample: a row position or a source id."""


class PrecomputedSplits(BaseModel, Generic[K]):
    """The validation and test samples of a split that a script computed once and wrote to a file.

    Use it when the source has no split of its own and the published split depends on a library that
    the connector should not pin, such as a seeded shuffle. The script runs the published code once
    and writes this file. The file names each sample by a key: a row position or a source id. Every
    sample that the file does not name is in the train split. The file is JSON, compressed with xz
    when its name ends in ``.xz``.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_sha256: str
    """SHA-256 digest of the source file that the split was made from."""
    validation: tuple[K, ...]
    test: tuple[K, ...]

    @model_validator(mode="after")
    def _each_key_in_one_split(self) -> Self:
        repeated = sorted(key for key, count in Counter(self.named_keys).items() if count > 1)
        if repeated:
            raise ValueError(f"{repeated} are named more than once")
        return self

    @property
    def named_keys(self) -> tuple[K, ...]:
        """The validation and test keys together."""
        return (*self.validation, *self.test)

    @classmethod
    def read_from_file(cls, path: Path) -> Self:
        """Read the split from a JSON file, or from an xz-compressed one if the name ends in ``.xz``.

        Returns:
            The split.

        Raises:
            TimeNetConnectorError: If the file does not hold a valid split.
        """
        try:
            content = path.read_bytes()
            if path.suffix == ".xz":
                content = lzma.decompress(content)
            return cls.model_validate_json(content)
        except (lzma.LZMAError, ValidationError) as exc:
            raise TimeNetConnectorError(f"{path.name} does not hold a valid split: {exc}") from exc

    def assign_splits(self, keys: Sequence[K], *, source_sha256: str) -> list[Split]:
        """Assign a split to each sample of the pinned source.

        Args:
            keys: The key of each sample of the source, in the order of the samples.
            source_sha256: The pinned digest of the source file.

        Returns:
            The split of each sample, in the order of ``keys``.

        Raises:
            TimeNetConnectorError: If the split was made from another source, or names a key that the
                source does not have.
        """
        if self.source_sha256 != source_sha256:
            raise TimeNetConnectorError(
                f"the split was made from the source {self.source_sha256}, not from the pinned {source_sha256}"
            )
        unknown = sorted(set(self.named_keys) - set(keys))
        if unknown:
            raise TimeNetConnectorError(f"the split names {unknown} that the source does not have")
        split_of = dict.fromkeys(self.validation, Split.VALIDATION) | dict.fromkeys(self.test, Split.TEST)
        return [split_of.get(key, Split.TRAIN) for key in keys]
