"""The partition a task belongs to, so a dataset can hand out its training and test tasks."""

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
