"""In-memory declarations for records imported from parent datasets."""

from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass
from datetime import datetime
from enum import Enum
from fractions import Fraction
from typing import TYPE_CHECKING

from pydantic import BaseModel


if TYPE_CHECKING:
    from timenet.dataset.record import Record


_LEAVES = (str, int, float, bool, Fraction, Enum, datetime, type(None))
"""Immutable types retained in a snapshot."""


@dataclass(frozen=True)
class RecordImport:
    """A record reused from a parent without copying its hierarchy or values."""

    parent_alias: str
    """The card alias of the parent that owns the record."""
    inherited_annotation_ids: frozenset[str]
    """Occurrence ids the record carried when imported; they stay in the parent layer."""
    state: object
    """Detached structural snapshot, excluding values and child-owned additions."""


def inherited_state(record: "Record", annotation_ids: frozenset[str]) -> object:
    """Snapshot parent-owned fields without loading Signal values.

    Task ids and child annotation overlays are excluded. Loaders are compared by identity.

    Args:
        record: The imported record.
        annotation_ids: Occurrence ids of parent-owned record annotations.

    Returns:
        Comparable parent-owned fields.

    Raises:
        TypeError: If a field contains an unsupported type.
    """  # noqa: DOC502 - raised by the nested snapshot helper

    def snapshot(value: object) -> object:
        if callable(value):
            return id(value)
        if isinstance(value, BaseModel):
            return value.model_dump(mode="json")
        if is_dataclass(value) and not isinstance(value, type):
            return {field.name: snapshot(getattr(value, field.name)) for field in fields(value)}
        if isinstance(value, Mapping):
            return {key: snapshot(item) for key, item in value.items()}
        if isinstance(value, (tuple, list)):
            return tuple(snapshot(item) for item in value)
        if isinstance(value, _LEAVES):
            return value
        raise TypeError(f"cannot snapshot a {type(value).__name__} on imported record {record.id!r}")

    return {
        **{
            field.name: snapshot(getattr(record, field.name))
            for field in fields(record)
            if field.name not in {"annotations", "task_ids"}
        },
        "annotations": snapshot(tuple(a for a in record.annotations if a.occurrence_id in annotation_ids)),
    }
