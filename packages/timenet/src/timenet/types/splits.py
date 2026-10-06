"""The partition a task belongs to, so a dataset can hand out its training and test tasks."""

from collections.abc import Iterable
from enum import StrEnum, unique

from timenet.errors import TimeFValidationError


@unique
class Split(StrEnum):
    """Which partition of a dataset a task belongs to."""

    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


def parse_split(value: Split | str) -> Split:
    """Coerce a split name to :class:`Split`.

    Args:
        value: The member itself, or its value such as ``"train"``.

    Returns:
        The matching member.

    Raises:
        TimeFValidationError: If ``value`` names no split.
    """
    try:
        return Split(value)
    except ValueError as exc:
        raise TimeFValidationError(
            f"unknown split {value!r}; expected one of {[member.value for member in Split]}"
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
    selected = frozenset(parse_split(value) for value in values)
    if not selected:
        raise TimeFValidationError("select at least one split; use get_all() for every task")
    return selected
