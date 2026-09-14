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

**A chunk locator names an artifact and two integers, not a Parquet row group.** ``signal_chunks``
used to carry ``(chunk_file, row_group, row_offset)``, which is Parquet vocabulary. A Zarr store has
no row groups, so that schema could only ever describe one values backend. The columns are now
``(chunk_file, chunk_major_idx, chunk_minor_idx)`` and the backend that wrote the artifact says what
they mean: Parquet reads them as a row group and a row inside it, Zarr reads the major index as an
element offset along the array and leaves the minor index null. Both are still one addressed read.

``values_artifacts`` is the other half. It lists every artifact the values plane wrote and which
backend wrote it, so a chunk row can be checked against a declared artifact at write time and a
reader can pick the right backend without a per-chunk tag. It is also why the backend is a property
of the artifact rather than of the whole version: the reader resolves it once per artifact, which is
already the granularity it opens files at.
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
    key   VARCHAR NOT NULL,
    value VARCHAR NOT NULL
);

CREATE TABLE datasets (
    dataset_id VARCHAR NOT NULL,
    metadata   VARCHAR
);

CREATE TABLE records (
    record_id     VARCHAR NOT NULL,
    start_time_us BIGINT,
    metadata      VARCHAR
);

CREATE TABLE axes (
    axis_id             VARCHAR NOT NULL,
    axis_type           VARCHAR NOT NULL,
    period_numerator_us BIGINT,
    period_denominator  BIGINT,
    start_index         BIGINT,
    first_us            BIGINT,
    last_us             BIGINT
);

CREATE TABLE specs (
    spec_id  VARCHAR NOT NULL,
    name     VARCHAR NOT NULL,
    unit     VARCHAR NOT NULL,
    dtype    VARCHAR NOT NULL,
    nullable BOOLEAN NOT NULL
);

CREATE TABLE sources (
    source_id        VARCHAR NOT NULL,
    record_id        VARCHAR NOT NULL,
    parent_source_id VARCHAR,
    path             VARCHAR NOT NULL,
    depth            INTEGER NOT NULL,
    name             VARCHAR NOT NULL,
    position         INTEGER NOT NULL,
    metadata         VARCHAR
);

CREATE TABLE signals (
    signal_id VARCHAR NOT NULL,
    name      VARCHAR NOT NULL,
    axis_id   VARCHAR NOT NULL,
    spec_id   VARCHAR NOT NULL,
    n_values  BIGINT  NOT NULL,
    metadata  VARCHAR
);

CREATE TABLE source_signals (
    source_id VARCHAR NOT NULL,
    signal_id VARCHAR NOT NULL,
    position  INTEGER NOT NULL
);

CREATE TABLE values_artifacts (
    chunk_file VARCHAR NOT NULL,
    backend    VARCHAR NOT NULL CHECK (backend IN ('parquet', 'zarr'))
);

CREATE TABLE signal_chunks (
    signal_id       VARCHAR NOT NULL,
    chunk_idx       INTEGER NOT NULL,
    chunk_file      VARCHAR NOT NULL,
    chunk_major_idx BIGINT  NOT NULL,
    chunk_minor_idx BIGINT,
    n_values        BIGINT  NOT NULL
);

CREATE TABLE tasks (
    task_id  VARCHAR NOT NULL,
    prompt   VARCHAR NOT NULL,
    metadata VARCHAR
);

CREATE TABLE task_items (
    task_id    VARCHAR NOT NULL,
    role       VARCHAR NOT NULL CHECK (role IN ('input', 'target')),
    position   INTEGER NOT NULL,
    item_type  VARCHAR NOT NULL CHECK (item_type IN ('text', 'record')),
    text_value VARCHAR,
    record_id  VARCHAR
);

CREATE TABLE annotations (
    annotation_id VARCHAR NOT NULL,
    name       VARCHAR NOT NULL,
    value      VARCHAR NOT NULL,
    unit       VARCHAR,
    metadata   VARCHAR
);

CREATE TABLE entities_to_annotations (
    occurrence_id   BIGINT  NOT NULL,
    annotation_id      VARCHAR NOT NULL,
    object_type     VARCHAR NOT NULL CHECK (object_type IN ('dataset', 'task', 'record', 'source', 'signal')),
    on_dataset_id   VARCHAR,
    on_task_id      VARCHAR,
    on_record_id    VARCHAR,
    on_source_id    VARCHAR,
    on_signal_id    VARCHAR,
    scope_record_id VARCHAR,
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

INDEXES: Final = ""
"""No persistent indexes are shipped.

Measured on ECG-QA (1.35M control rows): plain tables are 19.4 MB, the same content with primary
keys, foreign keys and ten indexes is 189.0 MB. The indexes are the whole difference, and they do
not pay for themselves. Point lookups run in 0.5-5 ms either way, and the one query that joins
annotations to tasks across the whole dataset is *ten times faster* without them (41 ms indexed,
4 ms not), because the planner hash-joins columns instead of walking ART nodes.

Add an index here only when a measured query needs one, and record the measurement alongside it.
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
    "values_artifacts",
    "signal_chunks",
    "tasks",
    "task_items",
    "annotations",
    "entities_to_annotations",
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


#: Every invariant the dropped PRIMARY KEY and FOREIGN KEY constraints used to enforce, as a query
#: that must return no rows. The writer runs these against the finished database before it publishes
#: the version, which is the only moment they can be violated: a version is written once by one
#: process and is immutable afterwards, so a constraint carried in the shipped file would re-check,
#: for the rest of the dataset's life, something that cannot change. Each check is one bulk
#: anti-join rather than a lookup per row.
VALIDATIONS: Final = (
    ("duplicate record_id", "SELECT record_id FROM records GROUP BY record_id HAVING count(*) > 1"),
    ("duplicate source_id", "SELECT source_id FROM sources GROUP BY source_id HAVING count(*) > 1"),
    ("duplicate signal_id", "SELECT signal_id FROM signals GROUP BY signal_id HAVING count(*) > 1"),
    ("duplicate task_id", "SELECT task_id FROM tasks GROUP BY task_id HAVING count(*) > 1"),
    ("duplicate axis_id", "SELECT axis_id FROM axes GROUP BY axis_id HAVING count(*) > 1"),
    ("duplicate spec_id", "SELECT spec_id FROM specs GROUP BY spec_id HAVING count(*) > 1"),
    (
        "duplicate annotation_id",
        "SELECT annotation_id FROM annotations GROUP BY annotation_id HAVING count(*) > 1",
    ),
    (
        "duplicate occurrence_id",
        "SELECT occurrence_id FROM entities_to_annotations GROUP BY occurrence_id HAVING count(*) > 1",
    ),
    (
        "duplicate (source_id, signal_id) link",
        "SELECT source_id, signal_id FROM source_signals GROUP BY source_id, signal_id HAVING count(*) > 1",
    ),
    (
        "duplicate (signal_id, chunk_idx)",
        "SELECT signal_id, chunk_idx FROM signal_chunks GROUP BY signal_id, chunk_idx HAVING count(*) > 1",
    ),
    (
        "source names a record that does not exist",
        "SELECT s.source_id FROM sources s ANTI JOIN records r ON r.record_id = s.record_id",
    ),
    (
        "source names a parent that does not exist",
        "SELECT s.source_id FROM sources s ANTI JOIN sources p ON p.source_id = s.parent_source_id "
        "WHERE s.parent_source_id IS NOT NULL",
    ),
    (
        "source sits in a different record than its parent",
        "SELECT s.source_id FROM sources s JOIN sources p ON p.source_id = s.parent_source_id "
        "WHERE p.record_id <> s.record_id",
    ),
    (
        "source path disagrees with its parent's",
        "SELECT s.source_id FROM sources s JOIN sources p ON p.source_id = s.parent_source_id "
        "WHERE NOT starts_with(s.path, p.path || '.') OR s.depth <> p.depth + 1",
    ),
    (
        "signal names an axis that does not exist",
        "SELECT g.signal_id FROM signals g ANTI JOIN axes a ON a.axis_id = g.axis_id",
    ),
    (
        "signal names a spec that does not exist",
        "SELECT g.signal_id FROM signals g ANTI JOIN specs sp ON sp.spec_id = g.spec_id",
    ),
    (
        "link names a source that does not exist",
        "SELECT ss.signal_id FROM source_signals ss ANTI JOIN sources s ON s.source_id = ss.source_id",
    ),
    (
        "link names a signal that does not exist",
        "SELECT ss.signal_id FROM source_signals ss ANTI JOIN signals g ON g.signal_id = ss.signal_id",
    ),
    (
        "chunk names a signal that does not exist",
        "SELECT c.signal_id FROM signal_chunks c ANTI JOIN signals g ON g.signal_id = c.signal_id",
    ),
    (
        "duplicate values artifact",
        "SELECT chunk_file FROM values_artifacts GROUP BY chunk_file HAVING count(*) > 1",
    ),
    (
        "chunk names an artifact the values plane did not declare",
        "SELECT c.chunk_file FROM signal_chunks c ANTI JOIN values_artifacts a ON a.chunk_file = c.chunk_file",
    ),
    # The neutral locator cannot say in its column types that a Parquet chunk needs both indexes and
    # a Zarr chunk needs only one. The artifact's backend says it instead, once the rows are in.
    (
        "parquet chunk without a row offset",
        "SELECT c.chunk_file FROM signal_chunks c JOIN values_artifacts a ON a.chunk_file = c.chunk_file "
        "WHERE a.backend = 'parquet' AND c.chunk_minor_idx IS NULL",
    ),
    (
        "zarr chunk with a row offset",
        "SELECT c.chunk_file FROM signal_chunks c JOIN values_artifacts a ON a.chunk_file = c.chunk_file "
        "WHERE a.backend = 'zarr' AND c.chunk_minor_idx IS NOT NULL",
    ),
    (
        "task item names a task that does not exist",
        "SELECT ti.task_id FROM task_items ti ANTI JOIN tasks t ON t.task_id = ti.task_id",
    ),
    (
        "task item names a record that does not exist",
        "SELECT ti.task_id FROM task_items ti ANTI JOIN records r ON r.record_id = ti.record_id "
        "WHERE ti.record_id IS NOT NULL",
    ),
    (
        "occurrence names content that does not exist",
        "SELECT o.occurrence_id FROM entities_to_annotations o ANTI JOIN annotations c ON c.annotation_id = o.annotation_id",
    ),
    (
        "occurrence names a record that does not exist",
        "SELECT o.occurrence_id FROM entities_to_annotations o ANTI JOIN records r ON r.record_id = o.on_record_id "
        "WHERE o.on_record_id IS NOT NULL",
    ),
    (
        "occurrence names a source that does not exist",
        "SELECT o.occurrence_id FROM entities_to_annotations o ANTI JOIN sources s ON s.source_id = o.on_source_id "
        "WHERE o.on_source_id IS NOT NULL",
    ),
    (
        "occurrence names a signal that does not exist",
        "SELECT o.occurrence_id FROM entities_to_annotations o ANTI JOIN signals g ON g.signal_id = o.on_signal_id "
        "WHERE o.on_signal_id IS NOT NULL",
    ),
    (
        "occurrence names a task that does not exist",
        "SELECT o.occurrence_id FROM entities_to_annotations o ANTI JOIN tasks t ON t.task_id = o.on_task_id "
        "WHERE o.on_task_id IS NOT NULL",
    ),
    (
        "occurrence scope names a record that does not exist",
        "SELECT o.occurrence_id FROM entities_to_annotations o ANTI JOIN records r ON r.record_id = o.scope_record_id "
        "WHERE o.scope_record_id IS NOT NULL",
    ),
    (
        "signal length disagrees with its chunks",
        "SELECT g.signal_id FROM signals g JOIN (SELECT signal_id, sum(n_values) AS total FROM signal_chunks "
        "GROUP BY signal_id) c ON c.signal_id = g.signal_id WHERE c.total <> g.n_values",
    ),
)
