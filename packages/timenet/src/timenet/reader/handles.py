"""Explicit lazy dataset and entity access, separate from materialized reads."""

from builtins import object as _object
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pyarrow as pa

from timenet.dataset import Record, Signal, Source, TimeFDataset
from timenet.errors import TimeFValidationError
from timenet.format.stored_values import ValueHandle
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


def _materialize_annotations(annotations: tuple[Annotation, ...]) -> None:
    """Resolve only the annotations whose values are stored externally."""
    for annotation in annotations:
        if isinstance(annotation.value, ValueHandle):
            _set_field(annotation, "value", annotation.value.read())


def _materialize_signal(signal: Signal) -> None:
    """Retain signal arrays without revisiting their native metadata."""
    if not isinstance(signal.loader, MemoryArray):
        _set_field(signal, "loader", MemoryArray(signal.to_arrow(), signal.loader))
    loader = signal.time_offsets_loader
    if loader is not None and not isinstance(loader, MemoryArray):
        _set_field(signal, "time_offsets_loader", MemoryArray(loader(), loader))
    _materialize_annotations(signal.annotations)


def _materialize_dataset(dataset: TimeFDataset) -> TimeFDataset:
    """Materialize the complete catalog with one pass over each entity collection.

    Returns:
        The dataset with native values retained in memory.
    """
    for record in dataset.records:
        for source in record.walk_sources():
            for signal in source.signals:
                _materialize_signal(signal)
            _materialize_annotations(source.annotations)
        _materialize_annotations(record.annotations)
    for task in dataset.tasks:
        for field in ("prompt", "rationale"):
            value = getattr(task, field)
            if isinstance(value, ValueHandle):
                setattr(task, field, value.read())
        if task.targets and any(isinstance(target, ValueHandle) for target in task.targets):
            task.targets = tuple(
                target.read() if isinstance(target, ValueHandle) else target for target in task.targets
            )
        _materialize_annotations(task.annotations)
        _materialize_annotations(task.input_annotations)
        _materialize_annotations(task.target_annotations)
    _materialize_annotations(dataset.annotations)
    return dataset


def materialize(value: Any, seen: set[int] | None = None) -> Any:  # noqa: PLR0911, PLR0912 - graph node kinds
    """Materialize a selected object graph without repeatedly loading shared objects.

    Returns:
        The native value or populated object graph.
    """
    if type(value) is TimeFDataset:
        return _materialize_dataset(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, ValueHandle):
        return value.read()
    if seen is None:
        seen = set()
    if isinstance(value, tuple):
        if any(isinstance(item, ValueHandle) for item in value):
            return tuple(materialize(item, seen) for item in value)
        for item in value:
            materialize(item, seen)
        return value
    if id(value) in seen:
        return value
    seen.add(id(value))
    if type(value) is Annotation:
        if isinstance(value.value, ValueHandle):
            _set_field(value, "value", value.value.read())
    elif type(value) is Signal:
        _materialize_signal(value)
    elif type(value) is Source:
        materialize(value.signals, seen)
        materialize(value.sources, seen)
        _materialize_annotations(value.annotations)
    elif type(value) is Record:
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
            original = getattr(value, field)
            resolved = materialize(original, seen)
            if resolved is not original:
                _set_field(value, field, resolved)
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
        if self.kind == "Task":
            return control.read_task(identity)
        if self.kind in {"Record", "Source", "Signal"}:
            if self.kind == "Record":
                record_id = identity
            else:
                query = (
                    "SELECT record_id FROM records JOIN sources USING (record_key) WHERE source_id = ?"
                    if self.kind == "Source"
                    else "SELECT record_id FROM records JOIN sources USING (record_key) "
                    "JOIN signals USING (source_key) WHERE signal_id = ?"
                )
                row = control.connection.execute(query, [identity]).fetchone()
                if row is None:
                    return self._missing(identity)
                record_id = row[0]
            record = control.read_records([record_id])[0]
            if self.kind == "Record":
                return record
            objects = record.walk_sources() if self.kind == "Source" else record.signals
            for obj in objects:
                if obj.id == identity:
                    return obj
        elif self.kind == "Annotation":
            return control.read_annotation(identity)
        else:
            raise TimeFValidationError(f"unknown object kind {self.kind!r}")
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
