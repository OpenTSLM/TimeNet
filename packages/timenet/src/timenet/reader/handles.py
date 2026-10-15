"""Explicit lazy dataset and entity access, separate from materialized reads."""

from builtins import object as _object
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pyarrow as pa

from timenet.dataset import Record, Signal, TimeFDataset
from timenet.errors import TimeFValidationError
from timenet.format.external import ValueHandle
from timenet.types import Annotation, Task


if TYPE_CHECKING:
    from timenet.reader.reader import TimeFReader


@dataclass(frozen=True)
class MemoryArray:
    """Retain an eagerly materialized Arrow array."""

    array: pa.Array
    source: Any = None

    def __call__(self) -> pa.Array:
        """Return the retained array without storage access."""
        return self.array


def _set_field(instance: Any, name: str, value: Any) -> None:
    """Populate a validated frozen object during eager materialization."""
    _object.__setattr__(instance, name, value)  # noqa: PLC2801


def materialize(value: Any, seen: set[int] | None = None) -> Any:  # noqa: PLR0912 - graph node kinds
    """Materialize a selected object graph without repeatedly loading shared objects.

    Returns:
        The native value or populated object graph.
    """
    if isinstance(value, ValueHandle):
        return value.read()
    if isinstance(value, tuple):
        return tuple(materialize(item, seen) for item in value)
    if seen is None:
        seen = set()
    if id(value) in seen:
        return value
    seen.add(id(value))
    if isinstance(value, Annotation):
        _set_field(value, "value", materialize(value.value, seen))
    elif isinstance(value, Signal):
        _set_field(value, "loader", MemoryArray(value.to_arrow(), value.loader))
        if value.time_offsets_loader is not None:
            _set_field(value, "time_offsets_loader", MemoryArray(value.time_offsets_loader(), value.time_offsets_loader))
        materialize(value.annotations, seen)
    elif isinstance(value, Record):
        for source in value.walk_sources():
            materialize(source.annotations, seen)
            materialize(source.signals, seen)
        materialize(value.annotations, seen)
    elif isinstance(value, Task):
        for field in (
            "inputs",
            "targets",
            "prompt",
            "rationale",
            "annotations",
            "input_annotations",
            "target_annotations",
            "from_tasks",
        ):
            _set_field(value, field, materialize(getattr(value, field), seen))
        if hasattr(value, "candidate_records"):
            materialize(value.candidate_records, seen)
    elif isinstance(value, TimeFDataset):
        materialize(value.records, seen)
        materialize(value.tasks, seen)
        materialize(value.annotations, seen)
    return value


@dataclass(frozen=True)
class FieldHandle:
    """Defer reading one entity field until explicitly requested."""

    entity: "EntityHandle"
    field: str

    def read(self) -> Any:
        """Return a materialized native field value."""
        obj = self.entity._object()
        if self.field == "values" and isinstance(obj, Signal):
            return obj.to_arrow()
        if self.field == "offsets" and isinstance(obj, Signal):
            return obj.time_offsets_us()
        return materialize(getattr(obj, self.field))

    def to_arrow(self) -> pa.Array:
        """Return the field as a materialized Arrow array."""
        return self.read()

    def to_numpy(self) -> Any:
        """Return the field as a materialized NumPy array."""
        return self.to_arrow().to_numpy(zero_copy_only=False)


@dataclass(frozen=True)
class EntityHandle:
    """Identify an object without constructing it or reading its payloads."""

    dataset: "DatasetHandle"
    kind: str
    id: str
    namespace: str | None = None

    def _object(self) -> Any:
        """Return the structural object with deferred external values.

        Raises:
            TimeFValidationError: If the object or its namespace does not exist.
        """
        identity = self.id
        reader = self.dataset.reader
        control = reader._control_reader()
        if self.namespace is not None:
            row = control.connection.execute(
                "SELECT object_id FROM imported_objects JOIN namespaces USING (namespace_key) "
                "WHERE (namespace = ? OR alias = ?) AND object_kind = ? AND source_id = ?",
                [self.namespace, self.namespace, self.kind, identity],
            ).fetchone()
            if row is None:
                raise TimeFValidationError(f"no {self.kind} {identity!r} in namespace {self.namespace!r}")
            identity = row[0]
        catalog = reader._read_catalog()
        if self.kind == "Record":
            objects = catalog.records
        elif self.kind == "Source":
            objects = (source for record in catalog.records for source in record.walk_sources())
        elif self.kind == "Signal":
            objects = (signal for record in catalog.records for signal in record.signals)
        elif self.kind == "Task":
            objects = catalog.tasks
        elif self.kind == "Annotation":
            objects = (
                *catalog.annotations,
                *(ann for record in catalog.records for ann in record.walk_annotations()),
                *(ann for task in catalog.tasks for ann in task.annotations),
            )
            for obj in objects:
                if obj.occurrence_id == identity:
                    return obj
            return self._missing(identity)
        else:
            raise TimeFValidationError(f"unknown object kind {self.kind!r}")
        for obj in objects:
            if obj.id == identity:
                return obj
        return self._missing(identity)

    def _missing(self, identity: str) -> Any:
        """Raise a diagnostic for an absent requested object.

        Raises:
            TimeFValidationError: Always.
        """
        raise TimeFValidationError(f"no {self.kind} {identity!r} in {self.dataset.metadata.dataset_id}")

    def get_handle(self, field: str) -> FieldHandle:
        """Return a lazy handle to a field such as values, prompt, targets, or value."""
        return FieldHandle(self, field)

    def read(self) -> Any:
        """Return the eagerly materialized object."""
        return materialize(self._object())

    def to_arrow(self) -> pa.Array:
        """Return the materialized values of a Signal."""
        return self._object().to_arrow()

    def to_numpy(self) -> Any:
        """Return the materialized NumPy values of a Signal."""
        return self._object().to_numpy()

    def read_steps(self, start: int, stop: int) -> pa.Array:
        """Return only the requested steps of a Signal."""
        return self._object().read_steps(start, stop)


class DatasetHandle:
    """A lazy catalog entry point; construction never reads payload files."""

    def __init__(self, reader: "TimeFReader") -> None:
        self.reader = reader

    @property
    def metadata(self) -> Any:
        """Return manifest metadata without opening the catalog."""
        return self.reader.metadata

    def get_handle(self, kind: str, id: str, *, namespace: str | None = None) -> EntityHandle:
        """Return a deferred entity lookup, optionally qualified by a parent namespace."""
        return EntityHandle(self, kind, id, namespace)

    def read(self) -> TimeFDataset:
        """Return the fully materialized dataset."""
        return self.reader.read()

    def close(self) -> None:
        """Release reader connections and cached payload groups."""
        self.reader.close()

    def __enter__(self) -> "DatasetHandle":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
