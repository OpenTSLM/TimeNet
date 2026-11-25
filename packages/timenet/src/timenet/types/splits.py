"""The partition a task belongs to, so a dataset can hand out its training and test tasks."""

from collections.abc import Iterable
from enum import StrEnum, unique
from typing import Self

from timenet.errors import TimeFValidationError


@unique
class Split(StrEnum):
    """Which partition of a dataset a task belongs to."""

    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"

    @classmethod
    def from_name(cls, name: str) -> Self:
        """Parse a canonical partition name without accepting source-specific aliases.

        Connectors map source filenames to canonical names before calling this method.

        Returns:
            The partition named by ``name``.

        Raises:
            TimeFValidationError: If the name is not train, validation, or test.
        """
        try:
            return cls(name)
        except ValueError as exc:
            raise TimeFValidationError(
                f"unknown split {name!r}; expected one of {[member.value for member in cls]}"
            ) from exc


def parse_splits(values: Split | str | Iterable[Split | str]) -> frozenset[Split]:
    """Normalize one or more split names, ignoring repeated names.

    Returns:
        The selected partitions.

    Raises:
        TimeFValidationError: If given no splits or an unknown name.
    """
    if isinstance(values, str):
        values = (values,)
    selected = frozenset(Split.from_name(value) for value in values)
    if not selected:
        raise TimeFValidationError("select at least one split; use get_all() for every task")
    return selected
