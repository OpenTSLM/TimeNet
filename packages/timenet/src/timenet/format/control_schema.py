"""The single description of the control database's tables.

The DDL, the writer's Arrow batch schemas, and the audit's key and relationship checks are all
derived from :data:`CONTROL_TABLES`, so a schema change is made in one place.

Version 1 of the control schema. Every table carries a ``BIGINT`` key drawn from a sequence and
stores public ids once, on the row that owns them. Fixed-shape fields are typed columns so they can
be filtered in SQL; only the free-form ``metadata`` dictionaries are JSON. The database carries no
persisted constraints: :data:`Table.unique` and :data:`Table.foreign_keys` are what the writer keeps
while streaming and what :func:`timenet.format.control_audit.audit_control_database` re-checks.
"""

from dataclasses import dataclass
from enum import Enum

import pyarrow as pa


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

    def ddl(self, key_sequence: str | None = None) -> str:
        """Return the column's ``CREATE TABLE`` fragment.

        Args:
            key_sequence: The sequence whose ``nextval`` is this column's default, for a key column.

        Returns:
            The column definition without a trailing comma.
        """
        parts = [self.name, self.type.sql]
        if not self.nullable:
            parts.append("NOT NULL")
        if key_sequence is not None:
            parts.append(f"DEFAULT nextval('{key_sequence}')")
        return " ".join(parts)


@dataclass(frozen=True)
class ForeignKey:
    """A column whose non-NULL values must exist in another table's column."""

    column: str
    table: str
    references: str


@dataclass(frozen=True)
class Table:
    """One control table: its columns in storage order, and the invariants it holds."""

    name: str
    columns: tuple[Column, ...]
    unique: tuple[tuple[str, ...], ...] = ()
    """Column sets that identify one row."""
    foreign_keys: tuple[ForeignKey, ...] = ()
    key_sequence: str | None = None
    """The sequence that fills the first column when it is a generated key."""

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
        lines = [
            column.ddl(self.key_sequence if index == 0 and self.key_sequence is not None else None)
            for index, column in enumerate(self.columns)
        ]
        body = ",\n    ".join(lines)
        return f"CREATE TABLE {self.name} (\n    {body}\n);"


def _key(name: str) -> Column:
    return Column(name, ColumnType.BIGINT, nullable=False)


def _required(name: str, column_type: ColumnType = ColumnType.VARCHAR) -> Column:
    return Column(name, column_type, nullable=False)


OBJECT_KEYS = "object_key_sequence"
"""Datasets, records, sources, signals, and tasks share one key space, so an annotation occurrence
can name any of them with one ``object_key``."""
AXIS_KEYS = "axis_key_sequence"
CONTENT_KEYS = "content_key_sequence"
OCCURRENCE_KEYS = "occurrence_key_sequence"

SEQUENCES: tuple[str, ...] = (OBJECT_KEYS, AXIS_KEYS, CONTENT_KEYS, OCCURRENCE_KEYS)

CONTROL_METADATA = Table(
    "control_metadata",
    (_required("key"), _required("value")),
    unique=(("key",),),
)

DATASETS = Table(
    "datasets",
    (_key("dataset_key"), _required("dataset_id")),
    unique=(("dataset_key",), ("dataset_id",)),
    key_sequence=OBJECT_KEYS,
)
"""Exactly one row, the dataset the file describes. The row gives dataset-level annotation
occurrences an ``object_key`` in the shared object key space, like every other annotated object."""

RECORDS = Table(
    "records",
    (
        _key("record_key"),
        _required("record_id"),
        Column("start_time_us", ColumnType.BIGINT),
        Column("time_span_start_us", ColumnType.BIGINT),
        Column("time_span_end_us", ColumnType.BIGINT),
        _required("metadata", ColumnType.JSON),
    ),
    unique=(("record_key",), ("record_id",)),
    key_sequence=OBJECT_KEYS,
)

SOURCES = Table(
    "sources",
    (
        _key("source_key"),
        _required("source_id"),
        _key("record_key"),
        Column("parent_source_key", ColumnType.BIGINT),
        _required("name"),
        _required("metadata", ColumnType.JSON),
    ),
    unique=(("source_key",), ("source_id",)),
    foreign_keys=(
        ForeignKey("record_key", "records", "record_key"),
        ForeignKey("parent_source_key", "sources", "source_key"),
    ),
    key_sequence=OBJECT_KEYS,
)

AXES = Table(
    "axes",
    (
        _key("axis_key"),
        _required("axis_id"),
        _required("axis_type"),
        Column("period_numerator_us", ColumnType.BIGINT),
        Column("period_denominator", ColumnType.BIGINT),
        Column("origin_us", ColumnType.BIGINT),
        Column("first_us", ColumnType.BIGINT),
        Column("last_us", ColumnType.BIGINT),
    ),
    unique=(("axis_key",), ("axis_id",)),
    key_sequence=AXIS_KEYS,
)

AXIS_OFFSETS = Table(
    "axis_offsets",
    (_key("axis_key"), _key("position"), _key("offset_us")),
    unique=(("axis_key", "position"),),
    foreign_keys=(ForeignKey("axis_key", "axes", "axis_key"),),
)

SIGNALS = Table(
    "signals",
    (
        _key("signal_key"),
        _required("signal_id"),
        _key("source_key"),
        _required("name"),
        _key("axis_key"),
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
    ),
    unique=(("signal_key",), ("signal_id",)),
    foreign_keys=(
        ForeignKey("source_key", "sources", "source_key"),
        ForeignKey("axis_key", "axes", "axis_key"),
    ),
    key_sequence=OBJECT_KEYS,
)

SIGNAL_CHUNKS = Table(
    "signal_chunks",
    (
        _key("signal_key"),
        _key("chunk_index"),
        _required("value_path"),
        _key("chunk_major_index"),
        Column("chunk_minor_index", ColumnType.BIGINT),
        _key("n_values"),
    ),
    unique=(("signal_key", "chunk_index"),),
    foreign_keys=(ForeignKey("signal_key", "signals", "signal_key"),),
)

ANNOTATION_CONTENTS = Table(
    "annotation_contents",
    (
        _key("content_key"),
        _required("content_id"),
        _required("name"),
        Column("value_kind", ColumnType.VARCHAR),
        Column("text_value", ColumnType.VARCHAR),
        Column("integer_value", ColumnType.BIGINT),
        Column("float_value", ColumnType.DOUBLE),
        Column("boolean_value", ColumnType.BOOLEAN),
        Column("text_list_value", ColumnType.VARCHAR_LIST),
        Column("unit", ColumnType.VARCHAR),
        _required("metadata", ColumnType.JSON),
    ),
    unique=(("content_key",), ("content_id",)),
    key_sequence=CONTENT_KEYS,
)

ANNOTATION_OCCURRENCES = Table(
    "annotation_occurrences",
    (
        _key("occurrence_key"),
        _required("occurrence_id"),
        _key("content_key"),
        _required("object_type"),
        _key("object_key"),
        _required("span_type"),
        Column("start_us", ColumnType.BIGINT),
        Column("end_us", ColumnType.BIGINT),
        Column("signal_keys", ColumnType.BIGINT_LIST),
        Column("provenance", ColumnType.JSON),
        Column("confidence", ColumnType.DOUBLE),
        _required("metadata", ColumnType.JSON),
    ),
    unique=(("occurrence_key",), ("occurrence_id",)),
    foreign_keys=(ForeignKey("content_key", "annotation_contents", "content_key"),),
    key_sequence=OCCURRENCE_KEYS,
)

TASKS = Table(
    "tasks",
    (
        _key("task_key"),
        _required("task_id"),
        _required("task_type"),
        Column("prompt", ColumnType.VARCHAR),
        Column("scope_type", ColumnType.VARCHAR),
        Column("scope_start", ColumnType.BIGINT),
        Column("scope_end", ColumnType.BIGINT),
        Column("scope_signal_keys", ColumnType.BIGINT_LIST),
        _required("has_inline_targets", ColumnType.BOOLEAN),
        Column("target_schema", ColumnType.VARCHAR),
        Column("unit", ColumnType.VARCHAR),
        Column("target_name", ColumnType.VARCHAR),
        Column("mode", ColumnType.VARCHAR),
        Column("rationale", ColumnType.VARCHAR),
        _required("metadata", ColumnType.JSON),
    ),
    unique=(("task_key",), ("task_id",)),
    key_sequence=OBJECT_KEYS,
)

TASK_TARGETS = Table(
    "task_targets",
    (
        _key("task_key"),
        _key("position"),
        _required("target_kind"),
        Column("text_value", ColumnType.VARCHAR),
        Column("integer_value", ColumnType.BIGINT),
        Column("float_value", ColumnType.DOUBLE),
        Column("boolean_value", ColumnType.BOOLEAN),
        Column("record_key", ColumnType.BIGINT),
        Column("signal_key", ColumnType.BIGINT),
        Column("span_start", ColumnType.BIGINT),
        Column("span_end", ColumnType.BIGINT),
        Column("signal_keys", ColumnType.BIGINT_LIST),
    ),
    unique=(("task_key", "position"),),
    foreign_keys=(
        ForeignKey("task_key", "tasks", "task_key"),
        ForeignKey("record_key", "records", "record_key"),
        ForeignKey("signal_key", "signals", "signal_key"),
    ),
)

TASK_RECORD_REFS = Table(
    "task_record_refs",
    (_key("task_key"), _required("field"), _key("position"), _key("record_key")),
    unique=(("task_key", "field", "position"),),
    foreign_keys=(
        ForeignKey("task_key", "tasks", "task_key"),
        ForeignKey("record_key", "records", "record_key"),
    ),
)

TASK_ANNOTATION_REFS = Table(
    "task_annotation_refs",
    (_key("task_key"), _required("field"), _key("position"), _key("occurrence_key")),
    unique=(("task_key", "field", "position"),),
    foreign_keys=(
        ForeignKey("task_key", "tasks", "task_key"),
        ForeignKey("occurrence_key", "annotation_occurrences", "occurrence_key"),
    ),
)

TASK_DEPENDENCIES = Table(
    "task_dependencies",
    (_key("task_key"), _key("position"), _key("parent_task_key")),
    unique=(("task_key", "position"),),
    foreign_keys=(
        ForeignKey("task_key", "tasks", "task_key"),
        ForeignKey("parent_task_key", "tasks", "task_key"),
    ),
)

CONTROL_TABLES: tuple[Table, ...] = (
    CONTROL_METADATA,
    DATASETS,
    RECORDS,
    SOURCES,
    AXES,
    AXIS_OFFSETS,
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
    """Return the statements that create every sequence and table of the control schema.

    Returns:
        DDL for :func:`timenet.format.duckdb.create_control_schema`.
    """
    sequences = [f"CREATE SEQUENCE {name} START 1;" for name in SEQUENCES]
    return "\n\n".join([*sequences, *(table.ddl() for table in CONTROL_TABLES)]) + "\n"
