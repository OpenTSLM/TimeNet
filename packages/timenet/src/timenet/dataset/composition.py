"""In-memory declarations for records imported from parent datasets."""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass
from datetime import datetime
from enum import Enum
from fractions import Fraction
from functools import partial
from typing import TYPE_CHECKING, cast

from pydantic import BaseModel
from typing_extensions import TypeAliasType


if TYPE_CHECKING:
    from timenet.dataset.record import Record


_LEAVES = (str, int, float, bool, Fraction, Enum, datetime, type(None))
"""Immutable types retained in a snapshot."""

Snapshot = TypeAliasType(
    "Snapshot",
    str | int | float | bool | Fraction | Enum | datetime | Sequence["Snapshot"] | Mapping[str, "Snapshot"] | None,
)
"""snapshot turns a live, mutable object tree into plain nested dicts, tuples, and scalars, so two
versions of a record can be compared with == without comparing the objects themselves."""

RecordState = TypeAliasType("RecordState", Mapping[str, Snapshot])
"""Describes the entire current state of a Record."""


@dataclass(frozen=True)
class RecordImport:
    """A record reused from a parent without copying its hierarchy or values."""

    parent_dataset_id: str
    """The dataset ID of the parent that owns the record."""
    inherited_annotation_ids: frozenset[str]
    """Occurrence IDs that the records contain. These belong to the parent, annotations added
    later belong to this dataset. Stored here because the parent is not available at write time."""
    state: Callable[[], RecordState]
    """Returns the snapshot taken at import time. A dataset read back from disk loads it from the
    parent on first use instead."""

    def changed(self, record: "Record") -> bool:
        """Return whether the record differs from its snapshot.

        Returns:
            ``True`` if a parent-owned field changed. New tasks and annotations do not count.
        """
        return inherited_state(record, self.inherited_annotation_ids) != self.state()


def captured_state(record: "Record", annotation_ids: frozenset[str]) -> Callable[[], RecordState]:
    """Snapshot parent-owned fields now and return a loader that yields that snapshot.

    Returns:
        A picklable zero-argument callable.
    """
    return partial(_constant, inherited_state(record, annotation_ids))


def _constant(value: RecordState) -> RecordState:
    return value


def inherited_state(
    record: "Record",
    annotation_ids: frozenset[str],
    *,
    memo: dict[int, tuple[object, Snapshot]] | None = None,
    callable_ids: Mapping[int, int] | None = None,
) -> RecordState:
    """Snapshot parent-owned fields without loading Signal values.

    Task ids and child annotation overlays are excluded. Loaders are compared by identity.

    Args:
        record: The imported record.
        annotation_ids: Occurrence ids of parent-owned record annotations.
        memo: Shared snapshots of objects visited during this read.
        callable_ids: Loader identities to report in place of the record's own, so a fresh
            hydration of a parent record compares equal to the consumer's copy.

    Returns:
        Comparable parent-owned fields.

    Raises:
        TypeError: If a field contains an unsupported type.
    """  # noqa: DOC502 - raised by _snapshot
    snapshots = {} if memo is None else memo
    state: dict[str, Snapshot] = {}
    for field in fields(record):
        if field.name == "task_ids":
            continue
        value = getattr(record, field.name)
        if field.name == "annotations":
            value = tuple(a for a in record.annotations if a.occurrence_id in annotation_ids)
        state[field.name] = _snapshot(value, record.id, snapshots, callable_ids)
    return state


def _snapshot(
    value: object,
    record_id: str,
    memo: dict[int, tuple[object, Snapshot]],
    callable_ids: Mapping[int, int] | None,
) -> Snapshot:
    """Copy a value tree without calling loaders.

    Args:
        value: A field or nested value to copy.
        record_id: Record identifier used in error messages.
        memo: Snapshots already copied during this read, shared across recursive calls.
        callable_ids: Replacement loader identities for comparison with a fresh parent read.

    Returns:
        Nested mappings, tuples, and scalars suitable for equality comparison.

    Raises:
        TypeError: If a field contains an unsupported type.
    """
    if isinstance(value, _LEAVES):
        return value

    identity = id(value)
    if callable(value):
        return identity if callable_ids is None else callable_ids.get(identity, identity)
    if identity in memo:
        return memo[identity][1]

    copied: Snapshot
    if isinstance(value, BaseModel):
        copied = value.model_dump(mode="json")
    elif is_dataclass(value) and not isinstance(value, type):
        copied = {
            field.name: _snapshot(getattr(value, field.name), record_id, memo, callable_ids) for field in fields(value)
        }
    elif isinstance(value, Mapping):
        # Metadata mappings are JSON objects, so their keys are strings.
        copied = {
            key: _snapshot(item, record_id, memo, callable_ids)
            for key, item in cast("Mapping[str, object]", value).items()
        }
    elif isinstance(value, (tuple, list)):
        copied = tuple(_snapshot(item, record_id, memo, callable_ids) for item in value)
    else:
        raise TypeError(f"cannot snapshot a {type(value).__name__} on imported record {record_id!r}")

    # Retain the input so temporary tuples cannot reuse a cached identity.
    memo[identity] = (value, copied)
    return copied
