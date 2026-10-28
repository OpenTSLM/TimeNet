"""The single description of the control database's tables.

The DDL, the writer's Arrow batch schemas, and the audit's key and relationship checks are all
derived from :data:`CONTROL_TABLES`, so a schema change is made in one place.

This is version 1 of the control schema. Every table uses ids as keys. Records, Signals, and Tasks
use the public ids that the caller sees. All other objects use the writer's storage ids. Every
foreign key stores the id that it refers to.

Fixed-shape fields are typed columns, so SQL can filter them. Only the free-form ``metadata``
dictionaries are JSON. The database has no persisted constraints. The writer keeps
:data:`Table.unique` and :data:`Table.foreign_keys` while it streams rows, and
:func:`timenet.format.control_audit.audit_control_database` checks them again.
"""

from dataclasses import dataclass
from enum import Enum

import pyarrow as pa

from timenet.types.object_kind import ObjectKind


class ColumnType(Enum):
    """The DuckDB column types the control schema uses, each with the Arrow type that feeds it.

    The writer builds every batch with the table's Arrow schema, so a value of the wrong Python type
    fails in Arrow, naming the column, instead of being cast by DuckDB.
    """

    BIGINT = ("BIGINT", pa.int64())
    DOUBLE = ("DOUBLE", pa.float64())
    BOOLEAN = ("BOOLEAN", pa.bool_())
    VARCHAR = ("VARCHAR", pa.string())
    JSON = ("JSON", pa.string())
    VARCHAR_LIST = ("VARCHAR[]", pa.list_(pa.string()))
    BIGINT_LIST = ("BIGINT[]", pa.list_(pa.int64()))

    def __init__(self, sql: str, arrow: pa.DataType) -> None:
        self.sql = sql
        """The type name in ``CREATE TABLE``."""
        self.arrow = arrow
        """The Arrow type of the column in a writer batch."""


@dataclass(frozen=True)
class Column:
    """One column of a control table."""

    name: str
    type: ColumnType
    nullable: bool = True

    def ddl(self) -> str:
        """Return the column's ``CREATE TABLE`` fragment.

        Returns:
            The column definition without a trailing comma.
        """
        parts = [self.name, self.type.sql]
        if not self.nullable:
            parts.append("NOT NULL")
        return " ".join(parts)


@dataclass(frozen=True)
class ForeignKey:
    """A column whose non-NULL values must exist in another table's column."""

    column: str
    table: str
    references: str
    many: bool = False
    """Whether ``column`` is a list column whose every element must exist."""


@dataclass(frozen=True)
class Table:
    """One control table: its columns in storage order, and the invariants it holds."""

    name: str
    columns: tuple[Column, ...]
    unique: tuple[tuple[str, ...], ...] = ()
    """Column sets that identify one row."""
    foreign_keys: tuple[ForeignKey, ...] = ()

    @property
    def column_names(self) -> tuple[str, ...]:
        """Return the column names in storage order."""
        return tuple(column.name for column in self.columns)

    @property
    def arrow_schema(self) -> pa.Schema:
        """Return the schema of a writer batch for this table, one Arrow field per column."""
        return pa.schema([pa.field(column.name, column.type.arrow) for column in self.columns])

    def ddl(self) -> str:
        """Return the table's ``CREATE TABLE`` statement.

        Returns:
            The statement, with a trailing semicolon.
        """
        body = ",\n    ".join(column.ddl() for column in self.columns)
        return f"CREATE TABLE {self.name} (\n    {body}\n);"


def _key(name: str) -> Column:
    return Column(name, ColumnType.BIGINT, nullable=False)


def _required(name: str, column_type: ColumnType = ColumnType.VARCHAR) -> Column:
    return Column(name, column_type, nullable=False)


ANNOTATED_OWNER_TABLES: dict[ObjectKind, tuple[str, str] | None] = {
    ObjectKind.DATASET: None,
    ObjectKind.RECORD: ("records", "record_id"),
    ObjectKind.SOURCE: ("sources", "source_id"),
    ObjectKind.SIGNAL: ("signals", "signal_id"),
    ObjectKind.TASK: ("tasks", "task_id"),
}
"""The owner table and key column of each annotated object type. The Dataset has no owner row."""

ANNOTATED_OBJECT_TYPES: frozenset[ObjectKind] = frozenset(ANNOTATED_OWNER_TABLES)
"""The values ``annotation_occurrences.object_type`` can hold."""

LAST_ID_PREFIX = "last_id."
"""Prefix of the ``control_metadata`` keys that store the last id used per entity kind."""

CONTROL_METADATA = Table(
    "control_metadata",
    (_required("key"), _required("value")),
    unique=(("key",),),
)

DATASETS = Table(
    "datasets",
    (_required("dataset_id"),),
    unique=(("dataset_id",),),
)
"""Exactly one row, the dataset the file describes."""

RECORDS = Table(
    "records",
    (
        _required("record_id", ColumnType.BIGINT),
        _key("clock_id"),
        Column("time_span_start_us", ColumnType.BIGINT),
        Column("time_span_end_us", ColumnType.BIGINT),
        _required("metadata", ColumnType.JSON),
    ),
    unique=(("record_id",),),
    foreign_keys=(ForeignKey("clock_id", "clocks", "clock_id"),),
)

CLOCKS = Table(
    "clocks",
    (_key("clock_id"), Column("start_time_us", ColumnType.BIGINT)),
    unique=(("clock_id",),),
)

SOURCES = Table(
    "sources",
    (
        _required("source_id", ColumnType.BIGINT),
        _key("record_id"),
        Column("parent_source_id", ColumnType.BIGINT),
        _required("name"),
        _required("metadata", ColumnType.JSON),
    ),
    unique=(("source_id",),),
    foreign_keys=(
        ForeignKey("record_id", "records", "record_id"),
        ForeignKey("parent_source_id", "sources", "source_id"),
    ),
)

AXES = Table(
    "axes",
    (
        _required("axis_id", ColumnType.BIGINT),
        _required("axis_type"),
        Column("period_numerator_us", ColumnType.BIGINT),
        Column("period_denominator", ColumnType.BIGINT),
        Column("start_index", ColumnType.BIGINT),
        Column("offset_us", ColumnType.BIGINT),
        Column("first_us", ColumnType.BIGINT),
        Column("last_us", ColumnType.BIGINT),
    ),
    unique=(("axis_id",),),
)

SIGNALS = Table(
    "signals",
    (
        _required("signal_id", ColumnType.BIGINT),
        _key("source_id"),
        _required("name"),
        _key("axis_id"),
        _required("spec_type"),
        _required("spec_name"),
        Column("unit", ColumnType.VARCHAR),
        _required("dtype"),
        _required("categories", ColumnType.VARCHAR_LIST),
        _required("value_shape", ColumnType.BIGINT_LIST),
        _required("dimension_names", ColumnType.VARCHAR_LIST),
        _required("nullable", ColumnType.BOOLEAN),
        _key("n_values"),
        _required("metadata", ColumnType.JSON),
        _required("modality"),
    ),
    unique=(("signal_id",),),
    foreign_keys=(
        ForeignKey("source_id", "sources", "source_id"),
        ForeignKey("axis_id", "axes", "axis_id"),
    ),
)

SIGNAL_CHUNKS = Table(
    "signal_chunks",
    (
        _key("signal_id"),
        _key("chunk_index"),
        _required("value_path"),
        Column("source_key", ColumnType.VARCHAR),
        _key("chunk_major_index"),
        Column("chunk_minor_index", ColumnType.BIGINT),
        _key("n_values"),
    ),
    unique=(("signal_id", "chunk_index"),),
    foreign_keys=(ForeignKey("signal_id", "signals", "signal_id"),),
)

ANNOTATION_CONTENTS = Table(
    "annotation_contents",
    (
        _required("content_id", ColumnType.BIGINT),
        _required("name"),
        Column("value_kind", ColumnType.VARCHAR),
        Column("text_value", ColumnType.VARCHAR),
        Column("integer_value", ColumnType.BIGINT),
        Column("float_value", ColumnType.DOUBLE),
        Column("boolean_value", ColumnType.BOOLEAN),
        Column("text_list_value", ColumnType.VARCHAR_LIST),
        Column("unit", ColumnType.VARCHAR),
        Column("description", ColumnType.VARCHAR),
        _required("metadata", ColumnType.JSON),
    ),
    unique=(("content_id",),),
)

ANNOTATION_OCCURRENCES = Table(
    "annotation_occurrences",
    (
        _required("occurrence_id", ColumnType.BIGINT),
        _required("content_id", ColumnType.BIGINT),
        _required("object_type"),
        Column("object_id", ColumnType.BIGINT),
        _required("span_type"),
        Column("start_us", ColumnType.BIGINT),
        Column("end_us", ColumnType.BIGINT),
        Column("provenance", ColumnType.JSON),
        Column("confidence", ColumnType.DOUBLE),
        _required("metadata", ColumnType.JSON),
    ),
    unique=(("occurrence_id",),),
    foreign_keys=(ForeignKey("content_id", "annotation_contents", "content_id"),),
)
"""The owner is ``(object_type, object_id)``. ``object_id`` is the public id of a Record, Signal,
or Task, or the storage id of a Source. Ids are unique only within one kind, so the type is part of
the reference. ``object_id`` is NULL exactly when the owner is the Dataset."""

TASKS = Table(
    "tasks",
    (
        _required("task_id", ColumnType.BIGINT),
        _required("task_type"),
        Column("prompt", ColumnType.VARCHAR),
        _required("input_modalities", ColumnType.VARCHAR_LIST),
        Column("split", ColumnType.VARCHAR),
        Column("scope_type", ColumnType.VARCHAR),
        Column("scope_start", ColumnType.BIGINT),
        Column("scope_end", ColumnType.BIGINT),
        Column("scope_signal_ids", ColumnType.BIGINT_LIST),
        _required("has_inline_targets", ColumnType.BOOLEAN),
        Column("target_schema", ColumnType.VARCHAR),
        Column("unit", ColumnType.VARCHAR),
        Column("target_name", ColumnType.VARCHAR),
        Column("mode", ColumnType.VARCHAR),
        Column("rationale", ColumnType.VARCHAR),
        _required("metadata", ColumnType.JSON),
    ),
    unique=(("task_id",),),
    foreign_keys=(ForeignKey("scope_signal_ids", "signals", "signal_id", many=True),),
)

TASK_TARGETS = Table(
    "task_targets",
    (
        _key("task_id"),
        _key("position"),
        _required("target_kind"),
        Column("text_value", ColumnType.VARCHAR),
        Column("integer_value", ColumnType.BIGINT),
        Column("float_value", ColumnType.DOUBLE),
        Column("boolean_value", ColumnType.BOOLEAN),
        Column("record_id", ColumnType.BIGINT),
        Column("signal_id", ColumnType.BIGINT),
        Column("span_start", ColumnType.BIGINT),
        Column("span_end", ColumnType.BIGINT),
        Column("span_signal_ids", ColumnType.BIGINT_LIST),
    ),
    unique=(("task_id", "position"),),
    foreign_keys=(
        ForeignKey("task_id", "tasks", "task_id"),
        ForeignKey("record_id", "records", "record_id"),
        ForeignKey("signal_id", "signals", "signal_id"),
        ForeignKey("span_signal_ids", "signals", "signal_id", many=True),
    ),
)
"""The value columns match :class:`timenet.format.task_target_row.TargetValueRow`. A span names its
series in ``span_signal_ids``: one Signal for a step span, and any number for a time span."""

TASK_RECORD_REFS = Table(
    "task_record_refs",
    (_key("task_id"), _required("field"), _key("position"), _key("record_id")),
    unique=(("task_id", "field", "position"),),
    foreign_keys=(
        ForeignKey("task_id", "tasks", "task_id"),
        ForeignKey("record_id", "records", "record_id"),
    ),
)

TASK_ANNOTATION_REFS = Table(
    "task_annotation_refs",
    (_key("task_id"), _required("field"), _key("position"), _key("occurrence_id")),
    unique=(("task_id", "field", "position"),),
    foreign_keys=(
        ForeignKey("task_id", "tasks", "task_id"),
        ForeignKey("occurrence_id", "annotation_occurrences", "occurrence_id"),
    ),
)

TASK_DEPENDENCIES = Table(
    "task_dependencies",
    (_key("task_id"), _key("position"), _key("parent_task_id")),
    unique=(("task_id", "position"),),
    foreign_keys=(
        ForeignKey("task_id", "tasks", "task_id"),
        ForeignKey("parent_task_id", "tasks", "task_id"),
    ),
)

EXTERNAL_OWNER_TABLES: dict[ObjectKind, tuple[str, str]] = {
    ObjectKind.TASK: ("tasks", "task_id"),
    ObjectKind.ANNOTATION_CONTENT: ("annotation_contents", "content_id"),
    ObjectKind.AXIS: ("axes", "axis_id"),
}
"""The owner table and key column of each object kind whose values can be stored externally."""

EXTERNAL_VALUES = Table(
    "external_values",
    (
        _required("object_kind"),
        _required("object_id", ColumnType.BIGINT),
        _required("field"),
        _key("position"),
        Column("source_key", ColumnType.VARCHAR),
        _required("value_path"),
        _key("row_index"),
        _required("value_kind"),
        _key("n_values"),
        Column("value_digest", ColumnType.VARCHAR),
    ),
    unique=(("object_kind", "object_id", "field", "position"),),
)
"""The owner is ``(object_kind, object_id)``: a Task by its public id, or an annotation content row
or an irregular axis by the storage id that this write gave its row. Ids are unique only within one
kind, so the kind is part of the reference. Signals that share an axis share its row, so the axis's
time offsets are stored once."""

IMPORTS = Table(
    "imports",
    (_required("alias"), _required("parent_ref")),
    unique=(("alias",), ("parent_ref",)),
)
"""Catalog lookup projection of the aliases and exact releases in ``Manifest.parent_cards``."""

IMPORTED_OBJECTS = Table(
    "imported_objects",
    (
        _required("alias"),
        _required("object_kind"),
        _required("parent_id", ColumnType.BIGINT),
        _required("object_id", ColumnType.BIGINT),
    ),
    unique=(("alias", "object_kind", "parent_id"),),
    foreign_keys=(ForeignKey("alias", "imports", "alias"),),
)
"""Local object IDs indexed by parent alias, object kind, and original parent ID."""

CONTROL_TABLES: tuple[Table, ...] = (
    CONTROL_METADATA,
    IMPORTS,
    EXTERNAL_VALUES,
    IMPORTED_OBJECTS,
    DATASETS,
    RECORDS,
    CLOCKS,
    SOURCES,
    AXES,
    SIGNALS,
    SIGNAL_CHUNKS,
    ANNOTATION_CONTENTS,
    ANNOTATION_OCCURRENCES,
    TASKS,
    TASK_TARGETS,
    TASK_RECORD_REFS,
    TASK_ANNOTATION_REFS,
    TASK_DEPENDENCIES,
)
"""Every control table, in creation order."""

TABLES: dict[str, Table] = {table.name: table for table in CONTROL_TABLES}
"""The control tables by name."""


def schema_ddl() -> str:
    """Return the statements that create every table of the control schema.

    Returns:
        DDL for :func:`timenet.format.duckdb.create_control_schema`.
    """
    return "\n\n".join(table.ddl() for table in CONTROL_TABLES) + "\n"
