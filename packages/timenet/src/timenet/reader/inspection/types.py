"""Small, storage-independent result types for bounded reader inspection."""

from dataclasses import dataclass
from typing import Generic, TypeVar


T = TypeVar("T")


@dataclass(frozen=True)
class InspectionPage(Generic[T]):
    """A keyset-paginated inspection result."""

    items: tuple[T, ...]
    next_after: str | None


@dataclass(frozen=True)
class RecordSummary:
    """The inexpensive fields shown in a record list."""

    record_id: str
    start_us: int | None
    end_us: int | None


@dataclass(frozen=True)
class SourceSummary:
    """The inexpensive fields shown in a source tree."""

    source_id: str
    parent_id: str | None
    name: str


@dataclass(frozen=True)
class SignalSummary:
    """The inexpensive fields shown in a signal list."""

    signal_id: str
    name: str
    axis_kind: str
    n_values: int
    dtype: str
    unit: str | None
    value_shape: tuple[int, ...]
