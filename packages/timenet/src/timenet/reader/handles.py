"""Explicit lazy dataset and entity access, separate from materialized reads."""

from builtins import object as _object
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pyarrow as pa

from timenet.dataset import Record, Signal, Source, TimeFDataset
from timenet.errors import TimeFValidationError
from timenet.format.signal_loader import MemoryArray
from timenet.format.stored_values import ValueHandle
from timenet.types import Annotation, ObjectKind, Task


if TYPE_CHECKING:
    from timenet.format.control_reader import DuckDBControlReader
    from timenet.reader.reader import TimeFReader


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


def _materialize_source(root: Source) -> None:
    """Load a source subtree without replacing its objects."""
    for source in root.walk_sources():
        for signal in source.signals:
            _materialize_signal(signal)
        _materialize_annotations(source.annotations)


def _materialize_record(record: Record) -> None:
    """Load values on a caller-supplied Record while preserving its object references."""
    for source in record.sources:
        _materialize_source(source)
    _materialize_annotations(record.annotations)


def _read_field_value(value: Any, control: "DuckDBControlReader") -> Any:
    """Read one selected field without loading the rest of its owning object.

    Returns:
        Native values, with referenced objects materialized.
    """
    if isinstance(value, ValueHandle):
        return value.read()
    if isinstance(value, tuple):
        if value and all(isinstance(item, Task) for item in value):
            return control.read_tasks_by_id((task.id for task in value), eager=True)
        return tuple(_read_field_value(item, control) for item in value)
    if isinstance(value, Task):
        return control.read_task(value.id, eager=True)
    if isinstance(value, Record):
        _materialize_record(value)
    elif isinstance(value, Source):
        _materialize_source(value)
    elif isinstance(value, Signal):
        _materialize_signal(value)
    elif isinstance(value, Annotation):
        _materialize_annotations((value,))
    return value


@dataclass(frozen=True)
class FieldHandle:
    """Defer reading one entity field until explicitly requested."""

    entity: "EntityHandle"
    field: str

    def read(self) -> Any:
        """Return a materialized native field value."""
        obj = self.entity._read_entity()
        if self.field == "values" and isinstance(obj, Signal):
            return obj.to_arrow()
        if self.field == "offsets" and isinstance(obj, Signal):
            return obj.time_offsets_us()
        return _read_field_value(getattr(obj, self.field), self.entity.dataset.reader._control_reader())

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
    kind: ObjectKind
    id: str
    namespace: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.kind, ObjectKind):
            raise TimeFValidationError(f"entity handle kind must be an ObjectKind, got {self.kind!r}")

    def _read_entity(self, *, eager: bool = False) -> Any:
        """Read one entity, optionally loading its stored values.

        Args:
            eager: Load values on the requested entity and its referenced objects.

        Returns:
            The requested entity.

        Raises:
            TimeFValidationError: If the object or its namespace does not exist.
        """
        control = self.dataset.reader._control_reader()
        identity = self._resolve_local_id(control)
        if self.kind == ObjectKind.RECORD:
            return control.read_records([identity], eager=eager)[0]
        if self.kind == ObjectKind.TASK:
            return control.read_task(identity, eager=eager)
        if self.kind == ObjectKind.ANNOTATION:
            return control.read_annotation(identity, eager=eager)
        if self.kind in {ObjectKind.SOURCE, ObjectKind.SIGNAL}:
            member = self._read_record_member(control, identity)
            return _read_field_value(member, control) if eager else member
        raise TimeFValidationError(f"unknown object kind {self.kind!r}")

    def _resolve_local_id(self, control: "DuckDBControlReader") -> str:
        """Translate a parent ID and namespace or alias into this catalog's object ID.

        Returns:
            The local ID, or the handle's ID when no namespace was given.

        Raises:
            TimeFValidationError: If the requested object is absent from the namespace.
        """
        if self.namespace is None:
            return self.id
        row = control.connection.execute(
            "SELECT object_id FROM imported_objects JOIN namespaces USING (namespace_key) "
            "WHERE (namespace = ? OR alias = ?) AND object_kind = ? AND source_id = ?",
            [self.namespace, self.namespace, self.kind.value, self.id],
        ).fetchone()
        if row is None:
            raise TimeFValidationError(f"no {self.kind} {self.id!r} in namespace {self.namespace!r}")
        return row[0]

    def _read_record_member(self, control: "DuckDBControlReader", identity: str) -> Source | Signal:
        """Find a Source or Signal in its Record without loading payload values.

        Args:
            control: The dataset's open catalog reader.
            identity: The Source or Signal ID in this catalog.

        Returns:
            The requested Source or Signal, with its values still lazy.
        """
        if self.kind == ObjectKind.SOURCE:
            query = "SELECT record_id FROM records JOIN sources USING (record_key) WHERE source_id = ?"
        else:
            query = (
                "SELECT record_id FROM records JOIN sources USING (record_key) "
                "JOIN signals USING (source_key) WHERE signal_id = ?"
            )
        row = control.connection.execute(query, [identity]).fetchone()
        if row is None:
            return self._missing(identity)
        record = control.read_records([row[0]])[0]
        members = record.walk_sources() if self.kind == ObjectKind.SOURCE else record.signals
        for member in members:
            if member.id == identity:
                return member
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
        return self._read_entity(eager=True)

    def to_arrow(self) -> pa.Array:
        """Return the materialized values of a Signal."""
        return self._read_entity().to_arrow()

    def to_numpy(self) -> Any:
        """Return the materialized NumPy values of a Signal."""
        return self._read_entity().to_numpy()

    def read_steps(self, start: int, stop: int) -> pa.Array:
        """Return only the requested steps of a Signal."""
        return self._read_entity().read_steps(start, stop)


class DatasetHandle:
    """A lazy catalog entry point; construction never reads payload files."""

    def __init__(self, reader: "TimeFReader") -> None:
        self.reader = reader

    @property
    def metadata(self) -> Any:
        """Return manifest metadata without opening the catalog."""
        return self.reader.metadata

    def get_handle(self, kind: ObjectKind, id: str, *, namespace: str | None = None) -> EntityHandle:
        """Return a deferred entity lookup, optionally qualified by a parent namespace.

        Args:
            kind: The object kind, such as ``ObjectKind.SIGNAL`` or ``ObjectKind.TASK``.
            id: The object's string ID, or its original parent ID when a namespace is given.
            namespace: The parent namespace or alias, or ``None`` for a local object ID.

        Returns:
            A handle that reads the object when requested.
        """
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
