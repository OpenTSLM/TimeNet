"""Load plans, which say which stored values a read loads.

The step that loads them lives in :mod:`timenet.reader.load_plan`. The plans live here, apart from
the reader, so that the dataset layer can take one too. :mod:`timenet.reader` exports them.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from enum import StrEnum, unique
from typing import ClassVar

from timenet.errors import TimeFValidationError


@unique
class Loading(StrEnum):
    """How a read delivers one kind of stored value on the objects it builds."""

    LAZY = "lazy"
    """Read from storage when accessed."""
    EAGER = "eager"
    """Loaded during the read and kept in memory in its value handle, so later reads don't touch storage."""


@dataclass(frozen=True, slots=True, kw_only=True)
class LoadPlan:
    """Specify, for each kind of stored value, whether a read loads it or leaves it lazy.

    A read applies its plan to the objects it builds: a dataset's records, tasks, and annotations; a
    record's signals, time offsets, and annotations; a task's text, annotations, and the records,
    signals, and annotations it references. It loads nothing on objects passed to it, such as the
    records given to ``read_tasks``, and nothing on tasks it builds only as another task's
    ``from_tasks``. A plan never widens what a read reaches.
    """

    signal_values: Loading = Loading.LAZY
    """Signal arrays."""
    time_offsets: Loading = Loading.LAZY
    """Time offsets of irregular axes, one array per axis that its signals share."""
    text: Loading = Loading.LAZY
    """Packed text: prompts, rationales, text targets, and annotation strings and string lists."""

    ALL_LAZY: ClassVar[LoadPlan]
    """Load nothing. Every read's default."""
    ALL_EAGER: ClassVar[LoadPlan]
    """Load every kind, so the objects a read builds need no storage afterwards."""

    def __post_init__(self) -> None:
        for field in fields(self):
            value = getattr(self, field.name)
            if not isinstance(value, Loading):
                raise TimeFValidationError(f"LoadPlan.{field.name} takes Loading.LAZY or Loading.EAGER, got {value!r}")


LoadPlan.ALL_LAZY = LoadPlan()
LoadPlan.ALL_EAGER = LoadPlan(signal_values=Loading.EAGER, time_offsets=Loading.EAGER, text=Loading.EAGER)
