"""In-memory declarations for records imported from parent datasets."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, fields, is_dataclass
from datetime import datetime
from enum import Enum
from fractions import Fraction
from functools import partial
from typing import TYPE_CHECKING

from pydantic import BaseModel


if TYPE_CHECKING:
    from timenet.dataset.record import Record


_LEAVES = (str, int, float, bool, Fraction, Enum, datetime, type(None))
"""Immutable types retained in a snapshot."""


@dataclass(frozen=True)
class RecordImport:
    """A record reused from a parent without copying its hierarchy or values."""

    parent_dataset_id: str
    """The dataset ID of the parent that owns the record."""
    inherited_annotation_ids: frozenset[str]
    """Occurrence ids the record carried when imported; they stay in the parent layer."""
    state: Callable[[], object]
    """Return the detached snapshot of parent-owned fields, excluding values and child-owned
    additions. A build-time import captures it when the record is imported; a read-back import
    hydrates the originals from the parents on the first validation."""


def captured_state(record: "Record", annotation_ids: frozenset[str]) -> Callable[[], object]:
    """Snapshot parent-owned fields now and return a loader that yields that snapshot.

    Returns:
        A picklable zero-argument callable.
    """
    return partial(_constant, inherited_state(record, annotation_ids))


def _constant(value: object) -> object:
    return value


def inherited_state(
    record: "Record",
    annotation_ids: frozenset[str],
    *,
    memo: dict[int, tuple[object, object]] | None = None,
    callable_ids: Mapping[int, int] | None = None,
) -> object:
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
    """  # noqa: DOC502 - raised by the nested snapshot helper
    snapshots = {} if memo is None else memo

    def snapshot(value: object) -> object:
        if isinstance(value, _LEAVES):
            return value
        if callable(value):
            return id(value) if callable_ids is None else callable_ids.get(id(value), id(value))
        identity = id(value)
        if identity in snapshots:
            return snapshots[identity][1]
        if isinstance(value, BaseModel):
            copied = value.model_dump(mode="json")
        elif is_dataclass(value) and not isinstance(value, type):
            copied = {field.name: snapshot(getattr(value, field.name)) for field in fields(value)}
        elif isinstance(value, Mapping):
            copied = {key: snapshot(item) for key, item in value.items()}
        elif isinstance(value, (tuple, list)):
            copied = tuple(snapshot(item) for item in value)
        else:
            raise TypeError(f"cannot snapshot a {type(value).__name__} on imported record {record.id!r}")
        # Retain the input so temporary tuples cannot reuse a cached identity.
        snapshots[identity] = (value, copied)
        return copied

    return {
        **{
            field.name: snapshot(getattr(record, field.name))
            for field in fields(record)
            if field.name not in {"annotations", "task_ids"}
        },
        "annotations": snapshot(tuple(a for a in record.annotations if a.occurrence_id in annotation_ids)),
    }
