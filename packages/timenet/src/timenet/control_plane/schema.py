"""The control-plane schema, as DuckDB DDL.

This departs from the proposal's draft in three places, all of them about how relationships are
mapped. Each change is only safe because a dataset version is immutable once written.

**Reverse lookups are indexes, not tables.** The draft carries ``record_tasks`` and
``annotation_objects_index``, which hold no payload their forward tables do not already hold, and it
has the writer record row counts and checksums so a reader can detect a reverse index that drifted.
A database maintains that kind of structure itself. Both are plain indexes here, which removes two
tables, the code that writes them, and the consistency check that guards them.

**Annotation targets are typed foreign keys, not a polymorphic id.** The draft points an occurrence
at its object with ``object_type`` plus ``object_id``. A foreign key references one table, so that
pair cannot have one: nothing stops an occurrence naming a signal that does not exist. Here each
occurrence carries one nullable foreign key per target kind and a constraint that exactly one is
set. The database now rejects a dangling annotation, and a join to the target needs no string
comparison. ``object_type`` stays as the stored discriminator, because the format dispatches on a
declared tag rather than inferring the case from which columns came back null.

**An occurrence records the record it sits in.** ``scope_record_id`` is denormalized onto every
occurrence whose target is a record, a source, or a signal. The renderer's hot query, every
annotation anywhere in one record, is then a single indexed equality instead of a recursive walk of
the source tree unioned with two more lookups.

**A signal belongs to many sources, not one.** The draft gives ``signals`` a ``source_id``, so one
series belongs to exactly one source in one record. Real data disagrees: in ARFBench 7,013 of 9,187
series are referenced by more than one record, up to ten each, because several questions ask about
the same underlying metric. Forcing a tree there would either duplicate the series or invent a
distinct id per use, and both throw away the fact that it is the same series. ``source_signals`` is
therefore a link table, which is the same shape the draft already chose for annotations: the payload
is stored once and the links say where it is used.

**A source stores its path, not just its parent.** ``parent_source_id`` stays for the edge itself,
and ``path`` records the whole ancestry as zero-padded positions (``0000.0001``). A subtree is then
a prefix match and depth-first display order is ``ORDER BY path``, neither of which needs recursion.
A materialized path is a liability when a tree gets re-parented; these trees are written once and
never edited in place, so it is free.
"""

from typing import Final


SCHEMA_VERSION: Final = 1
"""The control-plane schema version, recorded in ``meta`` so a reader can refuse a newer file."""

PATH_DIGITS: Final = 4
"""Digits per level in a source's materialized path. Lexical order matches declaration order only
while a level's position fits this width, the same constraint the part-number templates carry."""


def source_path(parent_path: str | None, position: int) -> str:
    """Return a source's materialized path.

    Args:
        parent_path: The parent source's path, or ``None`` for a root source.
        position: The source's position among its siblings.

    Returns:
        The path, as dot-separated zero-padded positions.

    Raises:
        ValueError: If ``position`` needs more than :data:`PATH_DIGITS` digits.
    """
    if not 0 <= position < 10**PATH_DIGITS:
        raise ValueError(f"source position {position} does not fit {PATH_DIGITS} digits")
    segment = f"{position:0{PATH_DIGITS}d}"
    return segment if parent_path is None else f"{parent_path}.{segment}"


DDL: Final = """
CREATE TABLE meta (
    key   VARCHAR PRIMARY KEY,
    value VARCHAR NOT NULL
);

CREATE TABLE datasets (
    dataset_id VARCHAR PRIMARY KEY,
    metadata   VARCHAR
);

CREATE TABLE records (
    record_id     VARCHAR PRIMARY KEY,
    start_time_us BIGINT,
    metadata      VARCHAR
);

CREATE TABLE axes (
    axis_id             VARCHAR PRIMARY KEY,
    axis_type           VARCHAR NOT NULL,
    period_numerator_us BIGINT,
    period_denominator  BIGINT,
    start_index         BIGINT,
    first_us            BIGINT,
    last_us             BIGINT
);

CREATE TABLE specs (
    spec_id  VARCHAR PRIMARY KEY,
    name     VARCHAR NOT NULL,
    unit     VARCHAR NOT NULL,
    dtype    VARCHAR NOT NULL,
    nullable BOOLEAN NOT NULL
);

CREATE TABLE sources (
    source_id        VARCHAR PRIMARY KEY,
    record_id        VARCHAR NOT NULL REFERENCES records(record_id),
    parent_source_id VARCHAR REFERENCES sources(source_id),
    path             VARCHAR NOT NULL,
    depth            INTEGER NOT NULL,
    name             VARCHAR NOT NULL,
    position         INTEGER NOT NULL,
    metadata         VARCHAR
);

CREATE TABLE signals (
    signal_id VARCHAR PRIMARY KEY,
    name      VARCHAR NOT NULL,
    axis_id   VARCHAR NOT NULL REFERENCES axes(axis_id),
    spec_id   VARCHAR NOT NULL REFERENCES specs(spec_id),
    n_values  BIGINT  NOT NULL,
    metadata  VARCHAR
);

CREATE TABLE source_signals (
    source_id VARCHAR NOT NULL REFERENCES sources(source_id),
    signal_id VARCHAR NOT NULL REFERENCES signals(signal_id),
    position  INTEGER NOT NULL,
    PRIMARY KEY (source_id, signal_id)
);

CREATE TABLE signal_chunks (
    signal_id  VARCHAR NOT NULL REFERENCES signals(signal_id),
    chunk_idx  INTEGER NOT NULL,
    chunk_file VARCHAR NOT NULL,
    row_group  INTEGER NOT NULL,
    row_offset BIGINT  NOT NULL,
    n_values   BIGINT  NOT NULL,
    PRIMARY KEY (signal_id, chunk_idx)
);

CREATE TABLE tasks (
    task_id  VARCHAR PRIMARY KEY,
    prompt   VARCHAR NOT NULL,
    metadata VARCHAR
);

CREATE TABLE task_items (
    task_id    VARCHAR NOT NULL REFERENCES tasks(task_id),
    role       VARCHAR NOT NULL CHECK (role IN ('input', 'target')),
    position   INTEGER NOT NULL,
    item_type  VARCHAR NOT NULL CHECK (item_type IN ('text', 'record')),
    text_value VARCHAR,
    record_id  VARCHAR REFERENCES records(record_id),
    PRIMARY KEY (task_id, role, position)
);

CREATE TABLE annotation_contents (
    content_id VARCHAR PRIMARY KEY,
    name       VARCHAR NOT NULL,
    value      VARCHAR NOT NULL,
    unit       VARCHAR,
    metadata   VARCHAR
);

CREATE TABLE annotation_occurrences (
    occurrence_id   BIGINT  PRIMARY KEY,
    content_id      VARCHAR NOT NULL REFERENCES annotation_contents(content_id),
    object_type     VARCHAR NOT NULL CHECK (object_type IN ('dataset', 'task', 'record', 'source', 'signal')),
    on_dataset_id   VARCHAR REFERENCES datasets(dataset_id),
    on_task_id      VARCHAR REFERENCES tasks(task_id),
    on_record_id    VARCHAR REFERENCES records(record_id),
    on_source_id    VARCHAR REFERENCES sources(source_id),
    on_signal_id    VARCHAR REFERENCES signals(signal_id),
    scope_record_id VARCHAR REFERENCES records(record_id),
    span_type       VARCHAR NOT NULL CHECK (span_type IN ('static', 'point', 'interval')),
    start_us        BIGINT,
    end_us          BIGINT,
    provenance      VARCHAR,
    confidence      DOUBLE,
    metadata        VARCHAR,
    CHECK (
        CAST(on_dataset_id IS NOT NULL AS INTEGER)
      + CAST(on_task_id    IS NOT NULL AS INTEGER)
      + CAST(on_record_id  IS NOT NULL AS INTEGER)
      + CAST(on_source_id  IS NOT NULL AS INTEGER)
      + CAST(on_signal_id  IS NOT NULL AS INTEGER) = 1
    )
);
"""
"""Every table, in an order that lets each foreign key reference a table that already exists."""

INDEXES: Final = """
CREATE INDEX sources_by_record      ON sources (record_id);
CREATE INDEX sources_by_parent      ON sources (parent_source_id);
CREATE INDEX sources_by_path        ON sources (path);
CREATE INDEX source_signals_by_source ON source_signals (source_id);
CREATE INDEX source_signals_by_signal ON source_signals (signal_id);
CREATE INDEX task_items_by_record   ON task_items (record_id);
CREATE INDEX occurrences_by_scope   ON annotation_occurrences (scope_record_id);
CREATE INDEX occurrences_by_content ON annotation_occurrences (content_id);
CREATE INDEX occurrences_by_signal  ON annotation_occurrences (on_signal_id);
CREATE INDEX occurrences_by_source  ON annotation_occurrences (on_source_id);
"""
"""The indexes built after the rows are in.

``task_items_by_record`` replaces the draft's ``record_tasks`` table and ``occurrences_by_content``
replaces its ``annotation_objects_index``. Building them after the load is faster than maintaining
them row by row during it, and nothing reads the database before it is committed.
"""

TABLES: Final = (
    "meta",
    "datasets",
    "records",
    "axes",
    "specs",
    "sources",
    "signals",
    "source_signals",
    "signal_chunks",
    "tasks",
    "task_items",
    "annotation_contents",
    "annotation_occurrences",
)
"""Every table the control plane defines, for a reader that wants to check what it opened."""

TARGET_COLUMNS: Final = {
    "dataset": "on_dataset_id",
    "task": "on_task_id",
    "record": "on_record_id",
    "source": "on_source_id",
    "signal": "on_signal_id",
}
"""Which foreign key column holds the target, for each kind of annotated object."""
