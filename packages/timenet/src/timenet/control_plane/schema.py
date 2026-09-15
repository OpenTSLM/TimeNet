"""The control-plane schema, as DuckDB DDL, plus the checks a build must pass before it publishes.

The tables hold the same model the Parquet control tables held: a flat :class:`~timenet.dataset.Record`
with its list of :class:`~timenet.dataset.TimeSeries`, the annotations a record carries, the ones
registered for tasks to reference, and the ten fields every :class:`~timenet.types.Task` shares plus
the payload its own class declares. Nothing about the authoring model changes; only where it is
stored does.

**Every join runs on a dense integer key, and the caller's name is kept once.** A record, a series,
an annotation and a task each carry a surrogate id assigned in walk order, and every link row names
that. The id the caller chose survives as ``external_id`` on the entity that owns it, which is what
the reader hands back and what stays stable across a rebuild. Surrogate ids are assigned per build,
so nothing outside the file should quote one.

**An annotation is attached through one table per target kind.** A polymorphic
``(object_type, object_id)`` table was measured at 55.7 MB against 50.2 MB for one table per kind on
a 3.06M-attachment corpus, and the query that gathers everything for one object ran 25.50 ms against
21.24 ms. Every target column is ``NOT NULL``, there is no discriminator to store or branch on, and
each validation is a bare anti-join. There are three kinds because main's model has three: a record
carries annotations, a task references them as input or as its answer, and the dataset holds the
registered ones that no record carries.

**The shipped database declares no primary or foreign keys.** Measured on ECG-QA (1.35M control
rows): plain tables are 19.7 MB, the same content with keys and indexes is 189.0 MB, and the write
takes 3.9 s against 11.9 s. A version is written once by one process and is immutable afterwards, so
a constraint carried in the file would re-check, for the rest of the dataset's life, something that
cannot change. :data:`VALIDATIONS` checks each invariant once, as a bulk anti-join, inside the load
transaction.

**A chunk locator names an artifact and two integers, not a Parquet row group.** The columns are
``(artifact_id, chunk_major_idx, chunk_minor_idx)`` and the backend that wrote the artifact says
what they mean: Parquet reads them as a row group and a row inside it, Zarr reads the major index as
an element offset and leaves the minor index null. ``values_artifacts`` lists every file the values
plane wrote and which backend wrote it, so a chunk row can be checked against a declared artifact at
write time.
"""

from dataclasses import dataclass, fields
from enum import StrEnum, unique
from typing import Final

from timenet.errors import TimeFValidationError
from timenet.types import TASKS, TaskType


SCHEMA_VERSION: Final = 1
"""The control-plane schema version, recorded in ``meta`` so a reader can refuse a newer file."""

ID_TYPE: Final = "UINTEGER"
"""The DuckDB type of every surrogate id column.

Width costs less than it first appears. At ``BLOCK_SIZE`` 16 KiB, DuckDB's minimum, it refuses to
bitpack these columns, so 64-bit looks like a flat 2x. Raise the block size and the premium
disappears: on a SLIP-shape control plane, ``UINTEGER`` against ``UBIGINT`` is 147.6 MB against
228.4 MB at 16 KiB, but 48.6 MB against 48.6 MB at 64 KiB. 64-bit still costs 9 to 12% on the hot
joins, which is why the declared type stays 32-bit. Exhausting it means 4.29 billion records in one
version, which needs a sharded control plane long before it needs a wider id.
"""

MAX_ID: Final = 2**32 - 1
"""The largest surrogate id an :data:`ID_TYPE` column holds. The writer refuses to assign one past it."""

BLOCK_SIZE: Final = 65_536
"""The database block size, set when the file is created and never changeable afterwards.

DuckDB claims at least one block per table, so a large block sets a floor of a few megabytes on a
database holding a few hundred rows. At 16 KiB it refuses to bitpack the id columns, so a large
corpus pays roughly double for them. Measured on a SLIP-shape control plane:

===========  =========  =========  =========
records        16 KiB     64 KiB    256 KiB
===========  =========  =========  =========
        100    0.47 MB    0.73 MB    2.11 MB
      1,000    0.65 MB    0.80 MB    2.11 MB
      5,000    1.47 MB    1.13 MB    2.37 MB
     20,000    4.73 MB    3.16 MB    3.94 MB
    100,000   21.97 MB   13.25 MB   14.43 MB
    400,000   87.31 MB   54.34 MB   54.80 MB
===========  =========  =========  =========

64 KiB wins everywhere above a few thousand records and 256 KiB never wins. Below the crossover
16 KiB is better by 0.26 MB, which is not worth choosing a block size per corpus.
"""


_DDL_TEMPLATE: Final = """
CREATE TABLE meta (
    key   VARCHAR NOT NULL,
    value VARCHAR NOT NULL
);

CREATE TABLE specs (
    spec_id  {id} NOT NULL,
    spec_type VARCHAR NOT NULL
);

CREATE TABLE axes (
    axis_id             {id} NOT NULL,
    axis_type           VARCHAR NOT NULL,
    period_numerator_us BIGINT,
    period_denominator  BIGINT,
    start_index         BIGINT,
    first_us            BIGINT,
    last_us             BIGINT
);

CREATE TABLE records (
    record_id         {id} NOT NULL,
    external_id        VARCHAR NOT NULL,
    start_time_us      BIGINT,
    time_span_start_us BIGINT,
    time_span_end_us   BIGINT,
    subject_ids        VARCHAR[] NOT NULL,
    task_ids           VARCHAR[] NOT NULL
);

CREATE TABLE time_series (
    time_series_id {id} NOT NULL,
    external_id     VARCHAR NOT NULL,
    signal          VARCHAR NOT NULL,
    source_id       VARCHAR,
    spec_id        {id} NOT NULL,
    axis_id        {id} NOT NULL,
    n_values        BIGINT NOT NULL
);

CREATE TABLE record_time_series (
    record_id     {id} NOT NULL,
    time_series_id {id} NOT NULL,
    position       INTEGER NOT NULL
);

CREATE TABLE annotations (
    annotation_id       {id} NOT NULL,
    external_id          VARCHAR NOT NULL,
    key                  VARCHAR NOT NULL,
    value                VARCHAR,
    source               VARCHAR,
    span_start_us        BIGINT,
    span_end_us          BIGINT,
    span_time_series_ids VARCHAR[]
);

CREATE TABLE dataset_annotations (
    attachment_id {id} NOT NULL,
    annotation_id {id} NOT NULL,
    position       INTEGER NOT NULL
);

CREATE TABLE record_annotations (
    attachment_id {id} NOT NULL,
    annotation_id {id} NOT NULL,
    record_id     {id} NOT NULL,
    position       INTEGER NOT NULL
);

CREATE TABLE task_annotations (
    attachment_id {id} NOT NULL,
    annotation_id {id} NOT NULL,
    task_id       {id} NOT NULL,
    role           VARCHAR NOT NULL CHECK (role IN ('input', 'target')),
    position       INTEGER NOT NULL
);

CREATE TABLE tasks (
    task_id    {id} NOT NULL,
    external_id VARCHAR NOT NULL,
    task_type   VARCHAR NOT NULL,
    prompt      VARCHAR,
    rationale   VARCHAR
);

CREATE TABLE task_items (
    task_id    {id} NOT NULL,
    role        VARCHAR NOT NULL CHECK (role IN ('record', 'from_task')),
    position    INTEGER NOT NULL,
    external_id VARCHAR NOT NULL
);

CREATE TABLE task_fields (
    task_id     {id} NOT NULL,
    field        VARCHAR NOT NULL,
    text_value   VARCHAR,
    double_value DOUBLE
);

CREATE TABLE task_refs (
    task_id    {id} NOT NULL,
    field       VARCHAR NOT NULL,
    position    INTEGER NOT NULL,
    ref_kind    VARCHAR NOT NULL CHECK (ref_kind IN ('record', 'time_series')),
    external_id VARCHAR NOT NULL
);

CREATE TABLE task_spans (
    task_id        {id} NOT NULL,
    field           VARCHAR NOT NULL,
    position        INTEGER NOT NULL,
    frame           VARCHAR NOT NULL CHECK (frame IN ('seconds', 'steps')),
    start_us        BIGINT NOT NULL,
    end_us          BIGINT,
    time_series_ids VARCHAR[]
);

CREATE TABLE values_artifacts (
    artifact_id {id} NOT NULL,
    chunk_file   VARCHAR NOT NULL,
    backend      VARCHAR NOT NULL CHECK (backend IN ('parquet', 'zarr'))
);

CREATE TABLE signal_chunks (
    time_series_id {id} NOT NULL,
    chunk_idx       INTEGER NOT NULL,
    artifact_id    {id} NOT NULL,
    chunk_major_idx BIGINT NOT NULL,
    chunk_minor_idx BIGINT,
    n_values        INTEGER NOT NULL
);
"""
"""The table definitions, with ``{id}`` standing in for the surrogate id width."""

DDL: Final = _DDL_TEMPLATE.format(id=ID_TYPE)
"""Every table, in an order that lets each reference name a table that already exists."""

ANNOTATION_TABLES: Final = {
    "dataset": ("dataset_annotations", None),
    "record": ("record_annotations", "record_id"),
    "task": ("task_annotations", "task_id"),
}
"""Which table holds each kind of attachment, and the column naming its target.

The dataset's table has no target column: a control database holds one dataset, so the table itself
says which object the row is about.
"""

KEYED_TABLES: Final = {
    "records": "record_id",
    "time_series": "time_series_id",
    "specs": "spec_id",
    "axes": "axis_id",
    "annotations": "annotation_id",
    "tasks": "task_id",
    "values_artifacts": "artifact_id",
    **{table: "attachment_id" for table, _ in ANNOTATION_TABLES.values()},
}
"""Each entity table and its surrogate id column."""

EXTERNAL_IDS: Final = ("records", "time_series", "annotations", "tasks")
"""Each table that keeps the caller's own id, in its ``external_id`` column."""

TABLES: Final = (
    "meta",
    "specs",
    "axes",
    "records",
    "time_series",
    "record_time_series",
    "annotations",
    *(table for table, _ in ANNOTATION_TABLES.values()),
    "tasks",
    "task_items",
    "task_fields",
    "task_refs",
    "task_spans",
    "values_artifacts",
    "signal_chunks",
)
"""Every table the control plane defines, for a reader that wants to check what it opened."""


@unique
class PayloadKind(StrEnum):
    """How one task payload field is stored, which says which table holds its value."""

    TEXT = "text"
    """A string, in ``task_fields.text_value``. A ``StrEnum`` payload stores as its value."""
    NUMBER = "number"
    """A float, in ``task_fields.double_value``."""
    RECORD_REF = "record_ref"
    """One record id or a tuple of them, in ``task_refs`` with ``ref_kind = 'record'``."""
    TIME_SERIES_REF = "time_series_ref"
    """One series id or a tuple of them, in ``task_refs`` with ``ref_kind = 'time_series'``."""
    SPAN = "span"
    """One span or a tuple of them, in ``task_spans``."""


@dataclass(frozen=True)
class PayloadField:
    """One field of a task's type-specific payload, and where the control plane keeps it."""

    name: str
    """The dataclass field's name, stored verbatim in the ``field`` column."""
    kind: PayloadKind
    """Which table holds the value."""
    is_list: bool = False
    """Whether the field holds a tuple rather than a single value. Only a ref or a span field may.

    A list field distinguishes ``None`` from ``()``, and both mean something: a localization target
    of ``None`` says the answer is stored by reference, ``()`` says the task looked and found
    nothing. Zero element rows cannot tell them apart, so every payload field that is not ``None``
    gets one ``task_fields`` row whatever its kind, and that row alone says the field is set.
    """

    @property
    def stores_elements(self) -> bool:
        """Whether the field's value lives in ``task_refs`` or ``task_spans`` rather than in ``task_fields``."""
        return self.kind is not PayloadKind.TEXT and self.kind is not PayloadKind.NUMBER


TASK_PAYLOAD: Final[dict[TaskType, tuple[PayloadField, ...]]] = {
    TaskType.CLASSIFICATION: (
        PayloadField("target", PayloadKind.TEXT),
        PayloadField("target_schema", PayloadKind.TEXT),
    ),
    TaskType.ANSWER: (PayloadField("target", PayloadKind.TEXT),),
    TaskType.SCALAR_PREDICTION: (
        PayloadField("target", PayloadKind.NUMBER),
        PayloadField("unit", PayloadKind.TEXT),
        PayloadField("target_name", PayloadKind.TEXT),
    ),
    TaskType.TEMPORAL_LOCALIZATION: (
        PayloadField("target", PayloadKind.SPAN, is_list=True),
        PayloadField("mode", PayloadKind.TEXT),
    ),
    TaskType.FORECASTING: (
        PayloadField("context_record_ids", PayloadKind.RECORD_REF, is_list=True),
        PayloadField("target_record_id", PayloadKind.RECORD_REF),
        PayloadField("target_span", PayloadKind.SPAN),
    ),
    TaskType.TS_EDITING: (
        PayloadField("source_record_id", PayloadKind.RECORD_REF),
        PayloadField("target_record_id", PayloadKind.RECORD_REF),
    ),
    TaskType.TS_GENERATION: (PayloadField("target_record_id", PayloadKind.RECORD_REF),),
    TaskType.TS_CORRESPONDENCE: (
        PayloadField("candidate_record_ids", PayloadKind.RECORD_REF, is_list=True),
        PayloadField("target", PayloadKind.RECORD_REF, is_list=True),
        PayloadField("target_time_series_ids", PayloadKind.TIME_SERIES_REF, is_list=True),
    ),
}
"""The payload fields of each task type, beyond the frame every task shares.

The frame (``id``, ``record_ids``, ``from_task_ids``, ``prompt``, ``scope``, the annotation id
tuples and ``rationale``) is handled by the writer and the reader directly, so it is not listed
here. :func:`task_payload` checks this declaration against the live dataclass, so a field added to a
task type fails loudly rather than being dropped on write.
"""

_TASK_FRAME: Final = frozenset(
    {
        "id",
        "record_ids",
        "prompt",
        "scope",
        "input_annotation_ids",
        "target_annotation_ids",
        "rationale",
        "from_tasks",
    }
)
"""The base :class:`~timenet.types.Task` fields, which every task type stores the same way."""


def task_payload(task_type: TaskType) -> tuple[PayloadField, ...]:
    """Return one task type's payload declaration, checked against its dataclass.

    Args:
        task_type: The task type whose payload to describe.

    Returns:
        The payload fields, in declaration order.

    Raises:
        TimeFValidationError: If the task dataclass and this declaration have drifted apart, or a
            scalar-valued field is declared as a list.
    """
    declared = TASK_PAYLOAD[task_type]
    cls = TASKS[task_type]
    expected = {field.name for field in fields(cls)} - _TASK_FRAME
    if cls.answer_is_record:
        # The answer is a produced series, found through a payload record id, so `target` stays None.
        expected.discard("target")
    actual = {field.name for field in declared}
    if expected != actual:
        raise TimeFValidationError(
            f"{cls.__name__} payload fields {sorted(expected)} do not match the control-plane "
            f"declaration {sorted(actual)}"
        )
    scalar_lists = sorted(field.name for field in declared if field.is_list and not field.stores_elements)
    if scalar_lists:
        raise TimeFValidationError(
            f"{cls.__name__} declares {scalar_lists} as list-valued {PayloadKind.TEXT.value} or "
            f"{PayloadKind.NUMBER.value} fields, which task_fields stores one value per row"
        )
    return declared


def _dense_ids(table: str, column: str) -> tuple[str, str]:
    """Return the check that one table's ids are 0, 1, 2, with no gap and no repeat.

    Density is not cosmetic. A reader that partitions a corpus across workers with
    ``id % num_workers = worker_index`` covers every row exactly once only while the ids run without
    gaps, and a gap would show up as a worker silently seeing fewer records.

    Args:
        table: The table to check.
        column: Its surrogate id column.

    Returns:
        A description and the query, which must return no rows.
    """
    return (
        f"{table}.{column} is not dense",
        # Counting distinct ids catches a repeat; comparing the row count against the largest id
        # catches a gap. Both are needed: one repeat plus one gap leaves the row count unchanged.
        f"SELECT count(*) FROM {table} HAVING count(*) <> count(DISTINCT {column}) "  # noqa: S608
        f"OR count(*) - 1 <> max({column}) OR min({column}) <> 0",
    )


def _unique_external_id(table: str) -> tuple[str, str]:
    """Return the check that one table's external ids are unique.

    Args:
        table: The table to check.

    Returns:
        A description and the query, which must return no rows.
    """
    return (
        f"duplicate {table}.external_id",
        f"SELECT external_id FROM {table} GROUP BY external_id HAVING count(*) > 1",  # noqa: S608
    )


def _attachment_targets() -> tuple[tuple[str, str], ...]:
    """Return the reference checks for every attachment table.

    Each is a bare anti-join with no ``WHERE``, because a table-per-kind layout makes every target
    column ``NOT NULL``.

    Returns:
        One description and query per check.
    """
    entity_of = {"record": "records", "task": "tasks"}
    checks: list[tuple[str, str]] = []
    for kind, (table, column) in ANNOTATION_TABLES.items():
        checks.append(
            (
                f"{table} names an annotation that does not exist",
                f"SELECT a.attachment_id FROM {table} a "  # noqa: S608
                f"ANTI JOIN annotations n ON n.annotation_id = a.annotation_id",
            )
        )
        if column is not None:
            checks.append(
                (
                    f"{table} names a {kind} that does not exist",
                    f"SELECT a.attachment_id FROM {table} a "  # noqa: S608
                    f"ANTI JOIN {entity_of[kind]} e ON e.{column} = a.{column}",
                )
            )
    return tuple(checks)


def _task_payload_owners() -> tuple[tuple[str, str], ...]:
    """Return the check that every payload row belongs to a task that exists.

    Returns:
        One description and query per payload table.
    """
    return tuple(
        (
            f"{table} names a task that does not exist",
            f"SELECT p.task_id FROM {table} p ANTI JOIN tasks t ON t.task_id = p.task_id",  # noqa: S608
        )
        for table in ("task_items", "task_fields", "task_refs", "task_spans")
    )


VALIDATIONS: Final = (
    *(_dense_ids(table, column) for table, column in KEYED_TABLES.items()),
    *(_unique_external_id(table) for table in EXTERNAL_IDS),
    (
        "duplicate (record_id, time_series_id) link",
        "SELECT record_id, time_series_id FROM record_time_series "
        "GROUP BY record_id, time_series_id HAVING count(*) > 1",
    ),
    (
        "duplicate (time_series_id, chunk_idx)",
        "SELECT time_series_id, chunk_idx FROM signal_chunks GROUP BY time_series_id, chunk_idx HAVING count(*) > 1",
    ),
    (
        "duplicate values artifact",
        "SELECT chunk_file FROM values_artifacts GROUP BY chunk_file HAVING count(*) > 1",
    ),
    (
        "link names a record that does not exist",
        "SELECT l.record_id FROM record_time_series l ANTI JOIN records r ON r.record_id = l.record_id",
    ),
    (
        "link names a series that does not exist",
        "SELECT l.time_series_id FROM record_time_series l "
        "ANTI JOIN time_series s ON s.time_series_id = l.time_series_id",
    ),
    (
        "series names a spec that does not exist",
        "SELECT s.time_series_id FROM time_series s ANTI JOIN specs sp ON sp.spec_id = s.spec_id",
    ),
    (
        "series names an axis that does not exist",
        "SELECT s.time_series_id FROM time_series s ANTI JOIN axes a ON a.axis_id = s.axis_id",
    ),
    (
        "chunk names a series that does not exist",
        "SELECT c.time_series_id FROM signal_chunks c ANTI JOIN time_series s ON s.time_series_id = c.time_series_id",
    ),
    (
        "chunk names an artifact the values plane did not declare",
        "SELECT c.artifact_id FROM signal_chunks c ANTI JOIN values_artifacts a ON a.artifact_id = c.artifact_id",
    ),
    # The backend-neutral locator cannot say in its column types that a Parquet chunk needs both
    # indexes and a Zarr chunk needs only one. The artifact's backend says it instead, once the rows
    # are in.
    (
        "parquet chunk without a row offset",
        "SELECT a.chunk_file FROM signal_chunks c JOIN values_artifacts a ON a.artifact_id = c.artifact_id "
        "WHERE a.backend = 'parquet' AND c.chunk_minor_idx IS NULL",
    ),
    (
        "zarr chunk with a row offset",
        "SELECT a.chunk_file FROM signal_chunks c JOIN values_artifacts a ON a.artifact_id = c.artifact_id "
        "WHERE a.backend = 'zarr' AND c.chunk_minor_idx IS NOT NULL",
    ),
    (
        "series length disagrees with its chunks",
        "SELECT s.time_series_id FROM time_series s JOIN ("
        "SELECT time_series_id, sum(n_values) AS total FROM signal_chunks GROUP BY time_series_id"
        ") c ON c.time_series_id = s.time_series_id WHERE c.total <> s.n_values",
    ),
    (
        "series has no chunk in the values plane",
        "SELECT s.time_series_id FROM time_series s ANTI JOIN signal_chunks c ON c.time_series_id = s.time_series_id",
    ),
    *_attachment_targets(),
    *_task_payload_owners(),
    (
        "task item names a record that does not exist",
        "SELECT i.task_id FROM task_items i ANTI JOIN records r ON r.external_id = i.external_id "
        "WHERE i.role = 'record'",
    ),
    (
        "task payload names a record that does not exist",
        "SELECT p.task_id FROM task_refs p ANTI JOIN records r ON r.external_id = p.external_id "
        "WHERE p.ref_kind = 'record'",
    ),
    (
        "task payload names a series that does not exist",
        "SELECT p.task_id FROM task_refs p ANTI JOIN time_series s ON s.external_id = p.external_id "
        "WHERE p.ref_kind = 'time_series'",
    ),
    (
        "an interval span ends before it starts",
        "SELECT task_id FROM task_spans WHERE end_us IS NOT NULL AND end_us <= start_us "
        "UNION ALL "
        "SELECT annotation_id FROM annotations WHERE span_end_us IS NOT NULL AND span_end_us <= span_start_us",
    ),
    (
        "annotation span bound without a start",
        "SELECT annotation_id FROM annotations WHERE span_start_us IS NULL AND span_end_us IS NOT NULL",
    ),
)
"""Every invariant the dropped key constraints used to enforce, as a query that must return no rows.

The writer runs these against the loaded database before it commits, which is the only moment they
can be violated: a version is written once by one process and is immutable afterwards. Each check is
one bulk anti-join rather than a lookup per row.

``task_items`` with ``role = 'from_task'`` is deliberately absent. A streamed task skips the
cross-task checks that :meth:`~timenet.dataset.TimeFDataset.add_task` runs, so a dangling derivation
can reach the writer, and the reader is what reports it.
"""
