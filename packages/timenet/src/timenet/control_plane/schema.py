"""The control-plane schema, as DuckDB DDL.

This departs from the proposal's draft in several places, all of them about how relationships are
mapped. Each change is only safe because a dataset version is immutable once written.

**Every join runs on a dense integer key, and a name is kept once.** The draft joins on the string
ids the caller chose, so a 27-character id is repeated in every link row and compared character by
character on every join. Each entity here carries an ``INTEGER`` id assigned by a counter during the
walk, and every link and locator names that. The caller's own id survives as ``external_id`` on the
entity that owns it, which keeps a record addressable by the name it has outside the dataset.
Measured on SLIP: the sources lookup falls from 51.78 ms to 0.74 ms, the signals join from 358.88 ms
to 3.02 ms, and the whole control plane from 731 MB to 313 MB. Keeping the external names costs
about 43 MB of that. Integer ids are assigned in walk order, so they are *not* stable across
rebuilds and nothing outside the file should quote one; ``external_id`` is what survives a rebuild.

``axes`` and ``annotations`` have no ``external_id``, because they never had a name worth keeping:
both used to carry an id derived by hashing their own columns, which said nothing the row did not
already say. The writer dedupes them in memory and the derived id is gone.

**There is no metadata column.** Every entity used to carry a JSON blob. Across three real corpora
and 9.5 million rows it was non-null exactly zero times, no connector set it, and nothing read it.
It was also the wrong mechanism: an annotation is a queryable ``(name, value, unit)`` that is stored
once however many entities carry it, while a blob repeats in full on every row and cannot be
searched without parsing all of them. Anything a builder would put in a blob belongs in an
annotation, so the format offers one way to do it rather than two.

**There is no datasets table.** A control database holds exactly one dataset. Its id and metadata
live in ``meta`` and in the manifest, so a one-row table restating them, and a target column that
was null on all 3.09M attachment rows, bought nothing. A dataset-level annotation is a row in
``dataset_annotations`` and needs no target at all.

**An annotation is attached through one table per target kind.** The draft points an attachment at
its object with ``object_type`` plus ``object_id``; an earlier version here used five nullable typed
columns and a check that exactly one was set. Both lose to a narrow table per kind, measured on a
SLIP-sized 3.06M-attachment corpus: 50.2 MB against 55.7 MB polymorphic and 56.4 MB for the five
typed columns, and the query that gathers everything applying to one signal in context (the dataset,
its record, its source, and the signal) runs in 21.24 ms against 25.50 ms and 41.40 ms. Every target
column is ``NOT NULL``, there is no discriminator column to store or branch on, each validation is a
bare anti-join, and a signal lookup never touches the 2.13M attachment rows belonging to other kinds.

**A source attachment records the record it sits in.** ``scope_record_id`` is denormalized onto
``source_annotations`` so the renderer's hot query, every annotation anywhere in one record, is an
equality rather than a walk of the source tree. ``record_annotations`` needs no such column because
it already carries the record. ``signal_annotations`` deliberately has none: a series shared by
several records belongs to no single one, so those are reached through ``source_signals``.

**A signal belongs to many sources, not one.** The draft gives ``signals`` a ``source_id``, so one
series belongs to exactly one source in one record. Real data disagrees: in ARFBench 7,013 of 9,187
series are referenced by more than one record, up to ten each, because several questions ask about
the same underlying metric. ``source_signals`` is therefore a link table: the payload is stored once
and the links say where it is used.

**A source stores its path, not just its parent.** ``parent_source_id`` stays for the edge itself,
and ``path`` records the whole ancestry as zero-padded positions (``0000.0001``). A subtree is then
a prefix match and depth-first display order is ``ORDER BY path``, neither of which needs recursion.
A materialized path is a liability when a tree gets re-parented; these trees are written once and
never edited in place, so it is free.

**A chunk locator names an artifact and two integers, not a Parquet row group.** ``signal_chunks``
used to carry ``(chunk_file, row_group, row_offset)``, which is Parquet vocabulary. A Zarr store has
no row groups, so that schema could only ever describe one values backend. The columns are now
``(artifact_id, chunk_major_idx, chunk_minor_idx)`` and the backend that wrote the artifact says
what they mean: Parquet reads them as a row group and a row inside it, Zarr reads the major index as
an element offset along the array and leaves the minor index null. Both are still one addressed read.

``values_artifacts`` is the other half. It lists every artifact the values plane wrote and which
backend wrote it, so a chunk row can be checked against a declared artifact at write time and a
reader can pick the right backend without a per-chunk tag.

``chunk_major_idx`` stays ``BIGINT`` because a Zarr element offset runs the length of a whole
concatenated array, which passes two billion long before the dataset is unreasonable.
``chunk_minor_idx`` and ``n_values`` are ``INTEGER``: both are bounded by the writer's byte budgets,
1 MiB per chunk and 4 MiB per row group, so neither can reach two billion unless a caller configures
a chunk larger than that, and then the insert raises rather than wrapping. Width is worth this care.
Measured at SLIP scale, declaring the id columns ``BIGINT`` instead of ``INTEGER`` grew the file from
94.0 MB to 162.6 MB and made the signals join 19% slower; DuckDB does not compress the extra width
away.
"""

from typing import Final


SCHEMA_VERSION: Final = 2
"""The control-plane schema version, recorded in ``meta`` so a reader can refuse a newer file."""

ID_TYPE: Final = "UINTEGER"
"""The DuckDB type of every surrogate id column.

Measured on DuckDB 1.5.5 at SLIP attachment scale, 3.09M rows across three id columns:

===============  =========================  =======  ==========  =======
type             ceiling                    file     lookup      write
===============  =========================  =======  ==========  =======
``INTEGER``      2.15e9                     37.3 MB    7.96 ms   0.29 s
``UINTEGER``     4.29e9                     37.3 MB    7.20 ms   0.26 s
``BIGINT``       9.22e18                    74.6 MB   12.29 ms   0.46 s
``HUGEINT``      1.70e38                   149.2 MB   11.07 ms   1.73 s
``BIGNUM``       unbounded                  41.6 MB   97.81 ms   6.73 s
===============  =========================  =======  ==========  =======

``UINTEGER`` is free next to ``INTEGER``: identical storage, identical speed, double the ceiling.
Going 64-bit is a flat 2.00x on every id column, and no compression setting avoids it; DuckDB stores
these columns uncompressed and ignores ``force_compression`` on them, even for the dense counters.
``BIGNUM`` removes the ceiling for only 1.12x storage, but it is a variable-length byte string
underneath, so it reads 12x slower and writes 23x slower. That is the wrong trade for a join key:
moving these columns off strings is what bought the 29-46x lookup speedup in the first place.
"""

MAX_ID: Final = 2**32 - 1
"""The largest surrogate id a :data:`ID_TYPE` column holds.

The writer refuses to assign an id past this rather than letting one wrap. For scale, 4.29 billion
signals at even 100 values each is a values plane past a terabyte, and the largest corpus built so
far has 1.66 million.
"""

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


_DDL_TEMPLATE: Final = """
CREATE TABLE meta (
    key   VARCHAR NOT NULL,
    value VARCHAR NOT NULL
);

CREATE TABLE records (
    record_id    {id} NOT NULL,
    external_id   VARCHAR,
    start_time_us BIGINT
);

CREATE TABLE axes (
    axis_id            {id} NOT NULL,
    axis_type           VARCHAR NOT NULL,
    period_numerator_us BIGINT,
    period_denominator  BIGINT,
    start_index         BIGINT,
    first_us            BIGINT,
    last_us             BIGINT
);

CREATE TABLE specs (
    spec_id  {id} NOT NULL,
    spec_type VARCHAR NOT NULL,
    name      VARCHAR NOT NULL,
    unit      VARCHAR NOT NULL,
    dtype     VARCHAR NOT NULL,
    nullable  BOOLEAN NOT NULL
);

CREATE TABLE sources (
    source_id       {id} NOT NULL,
    external_id      VARCHAR,
    record_id       {id} NOT NULL,
    parent_source_id {id},
    path             VARCHAR NOT NULL,
    depth            INTEGER NOT NULL,
    name             VARCHAR NOT NULL,
    position         INTEGER NOT NULL
);

CREATE TABLE signals (
    signal_id  {id} NOT NULL,
    external_id VARCHAR,
    name        VARCHAR NOT NULL,
    axis_id    {id} NOT NULL,
    spec_id    {id} NOT NULL,
    n_values    BIGINT  NOT NULL
);

CREATE TABLE source_signals (
    source_id {id} NOT NULL,
    signal_id {id} NOT NULL,
    position  INTEGER NOT NULL
);

CREATE TABLE values_artifacts (
    artifact_id {id} NOT NULL,
    chunk_file  VARCHAR NOT NULL,
    backend     VARCHAR NOT NULL CHECK (backend IN ('parquet', 'zarr'))
);

CREATE TABLE signal_chunks (
    signal_id      {id} NOT NULL,
    chunk_idx       INTEGER NOT NULL,
    artifact_id    {id} NOT NULL,
    chunk_major_idx BIGINT  NOT NULL,
    chunk_minor_idx INTEGER,
    n_values        INTEGER NOT NULL
);

CREATE TABLE tasks (
    task_id    {id} NOT NULL,
    external_id VARCHAR,
    prompt      VARCHAR NOT NULL
);

CREATE TABLE task_items (
    task_id   {id} NOT NULL,
    role       VARCHAR NOT NULL CHECK (role IN ('input', 'target')),
    position   INTEGER NOT NULL,
    item_type  VARCHAR NOT NULL CHECK (item_type IN ('text', 'record')),
    text_value VARCHAR,
    record_id {id}
);

CREATE TABLE annotations (
    annotation_id {id} NOT NULL,
    name          VARCHAR NOT NULL,
    value         VARCHAR NOT NULL,
    unit          VARCHAR
);

CREATE TABLE dataset_annotations (
    attachment_id {id} NOT NULL,
    annotation_id {id} NOT NULL,
    span_type     VARCHAR NOT NULL CHECK (span_type IN ('static', 'point', 'interval')),
    start_us      BIGINT,
    end_us        BIGINT,
    provenance    VARCHAR,
    confidence    DOUBLE
);

CREATE TABLE task_annotations (
    attachment_id {id} NOT NULL,
    annotation_id {id} NOT NULL,
    task_id      {id} NOT NULL,
    span_type     VARCHAR NOT NULL CHECK (span_type IN ('static', 'point', 'interval')),
    start_us      BIGINT,
    end_us        BIGINT,
    provenance    VARCHAR,
    confidence    DOUBLE
);

CREATE TABLE record_annotations (
    attachment_id {id} NOT NULL,
    annotation_id {id} NOT NULL,
    record_id    {id} NOT NULL,
    span_type     VARCHAR NOT NULL CHECK (span_type IN ('static', 'point', 'interval')),
    start_us      BIGINT,
    end_us        BIGINT,
    provenance    VARCHAR,
    confidence    DOUBLE
);

CREATE TABLE source_annotations (
    attachment_id  {id} NOT NULL,
    annotation_id  {id} NOT NULL,
    source_id      {id} NOT NULL,
    scope_record_id {id} NOT NULL,
    span_type       VARCHAR NOT NULL CHECK (span_type IN ('static', 'point', 'interval')),
    start_us        BIGINT,
    end_us          BIGINT,
    provenance      VARCHAR,
    confidence      DOUBLE
);

CREATE TABLE signal_annotations (
    attachment_id {id} NOT NULL,
    annotation_id {id} NOT NULL,
    signal_id    {id} NOT NULL,
    span_type     VARCHAR NOT NULL CHECK (span_type IN ('static', 'point', 'interval')),
    start_us      BIGINT,
    end_us        BIGINT,
    provenance    VARCHAR,
    confidence    DOUBLE
);
"""
"""The table definitions, with ``{id}`` standing in for the surrogate id width."""

DDL: Final = _DDL_TEMPLATE.format(id=ID_TYPE)
"""Every table, in an order that lets each reference name a table that already exists."""

INDEXES: Final = ""
"""No persistent indexes are shipped.

Measured on ECG-QA (1.35M control rows): plain tables are 19.4 MB, the same content with primary
keys, foreign keys and ten indexes is 189.0 MB. The indexes are the whole difference, and they do
not pay for themselves. Point lookups run in 0.5-5 ms either way, and the one query that joins
annotations to tasks across the whole dataset is *ten times faster* without them (41 ms indexed,
4 ms not), because the planner hash-joins columns instead of walking ART nodes.

Add an index here only when a measured query needs one, and record the measurement alongside it.
"""

ANNOTATION_TABLES: Final = {
    "dataset": ("dataset_annotations", None),
    "task": ("task_annotations", "task_id"),
    "record": ("record_annotations", "record_id"),
    "source": ("source_annotations", "source_id"),
    "signal": ("signal_annotations", "signal_id"),
}
"""Which table holds each kind of attachment, and the column naming its target.

The dataset's table has no target column: a control database holds one dataset, so the table itself
says which object the row is about.
"""

TABLES: Final = (
    "meta",
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
    *(table for table, _ in ANNOTATION_TABLES.values()),
)
"""Every table the control plane defines, for a reader that wants to check what it opened."""

KEYED_TABLES: Final = {
    "records": "record_id",
    "axes": "axis_id",
    "specs": "spec_id",
    "sources": "source_id",
    "signals": "signal_id",
    "values_artifacts": "artifact_id",
    "tasks": "task_id",
    "annotations": "annotation_id",
    **{table: "attachment_id" for table, _ in ANNOTATION_TABLES.values()},
}
"""Each entity table and its surrogate id column."""

EXTERNAL_IDS: Final = ("records", "sources", "signals", "tasks")
"""Each table that keeps the caller's own id, in its ``external_id`` column."""


def _dense_ids(table: str, column: str) -> tuple[str, str]:
    """Return the check that one table's ids are 0, 1, 2, ... with no gap and no repeat.

    Density is not cosmetic. The reader partitions a corpus across DataLoader workers with
    ``id % num_workers = worker_index``, which only covers every row exactly once while the ids run
    without gaps, and a gap would otherwise show up as a worker silently seeing fewer records.

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
        # It is count - 1 rather than max + 1 so the check cannot overflow the id column at exactly
        # the ceiling it exists to police.
        f"SELECT count(*) FROM {table} HAVING count(*) <> count(DISTINCT {column}) "  # noqa: S608
        f"OR count(*) - 1 <> max({column}) OR min({column}) <> 0",
    )


def _unique_external_id(table: str) -> tuple[str, str]:
    """Return the check that one table's external ids are unique among the rows that have one.

    Args:
        table: The table to check.

    Returns:
        A description and the query, which must return no rows.
    """
    return (
        f"duplicate {table}.external_id",
        f"SELECT external_id FROM {table} WHERE external_id IS NOT NULL "  # noqa: S608
        f"GROUP BY external_id HAVING count(*) > 1",
    )


def _attachment_targets() -> tuple[tuple[str, str], ...]:
    """Return the reference checks for every attachment table.

    Each is a bare anti-join with no ``WHERE``, because a table-per-kind layout makes every target
    column ``NOT NULL``.

    Returns:
        One description and query per check.
    """
    entity_of = {"task": "tasks", "record": "records", "source": "sources", "signal": "signals"}
    checks: list[tuple[str, str]] = []
    for kind, (table, column) in ANNOTATION_TABLES.items():
        checks.append(
            (
                f"{table} names an annotation that does not exist",
                f"SELECT a.attachment_id FROM {table} a "  # noqa: S608
                f"ANTI JOIN annotations n ON n.annotation_id = a.annotation_id",
            )
        )
        if column is None:
            continue
        entity = entity_of[kind]
        checks.append(
            (
                f"{table} names a {kind} that does not exist",
                f"SELECT a.attachment_id FROM {table} a "  # noqa: S608
                f"ANTI JOIN {entity} e ON e.{column} = a.{column}",
            )
        )
    checks.append(
        (
            "source_annotations scope names a record that does not exist",
            "SELECT a.attachment_id FROM source_annotations a ANTI JOIN records r ON r.record_id = a.scope_record_id",
        )
    )
    checks.append(
        (
            "source_annotations scope disagrees with its source's record",
            "SELECT a.attachment_id FROM source_annotations a JOIN sources s ON s.source_id = a.source_id "
            "WHERE s.record_id <> a.scope_record_id",
        )
    )
    return tuple(checks)


#: Every invariant the dropped PRIMARY KEY and FOREIGN KEY constraints used to enforce, as a query
#: that must return no rows. The writer runs these against the finished database before it publishes
#: the version, which is the only moment they can be violated: a version is written once by one
#: process and is immutable afterwards, so a constraint carried in the shipped file would re-check,
#: for the rest of the dataset's life, something that cannot change. Each check is one bulk
#: anti-join rather than a lookup per row.
VALIDATIONS: Final = (
    *(_dense_ids(table, column) for table, column in KEYED_TABLES.items()),
    *(_unique_external_id(table) for table in EXTERNAL_IDS),
    (
        "duplicate (source_id, signal_id) link",
        "SELECT source_id, signal_id FROM source_signals GROUP BY source_id, signal_id HAVING count(*) > 1",
    ),
    (
        "duplicate (signal_id, chunk_idx)",
        "SELECT signal_id, chunk_idx FROM signal_chunks GROUP BY signal_id, chunk_idx HAVING count(*) > 1",
    ),
    (
        "duplicate values artifact",
        "SELECT chunk_file FROM values_artifacts GROUP BY chunk_file HAVING count(*) > 1",
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
        "chunk names an artifact the values plane did not declare",
        "SELECT c.artifact_id FROM signal_chunks c ANTI JOIN values_artifacts a ON a.artifact_id = c.artifact_id",
    ),
    # The neutral locator cannot say in its column types that a Parquet chunk needs both indexes and
    # a Zarr chunk needs only one. The artifact's backend says it instead, once the rows are in.
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
        "task item names a task that does not exist",
        "SELECT ti.task_id FROM task_items ti ANTI JOIN tasks t ON t.task_id = ti.task_id",
    ),
    # A record item is resolved from its caller-supplied external id by a join against records when
    # the load finishes, so an id naming no record arrives here as a record item with no record.
    (
        "task item names a record that does not exist",
        "SELECT ti.task_id FROM task_items ti WHERE ti.item_type = 'record' AND ti.record_id IS NULL",
    ),
    (
        "task item names a record id that does not exist",
        "SELECT ti.task_id FROM task_items ti ANTI JOIN records r ON r.record_id = ti.record_id "
        "WHERE ti.record_id IS NOT NULL",
    ),
    (
        "text item carries no text",
        "SELECT ti.task_id FROM task_items ti WHERE ti.item_type = 'text' AND ti.text_value IS NULL",
    ),
    *_attachment_targets(),
    (
        "an interval annotation ends before it starts",
        " UNION ALL ".join(
            f"SELECT attachment_id FROM {table} WHERE span_type = 'interval' "  # noqa: S608
            f"AND (start_us IS NULL OR end_us IS NULL OR end_us < start_us)"
            for table, _ in ANNOTATION_TABLES.values()
        ),
    ),
    (
        "signal length disagrees with its chunks",
        "SELECT g.signal_id FROM signals g JOIN (SELECT signal_id, sum(n_values) AS total FROM signal_chunks "
        "GROUP BY signal_id) c ON c.signal_id = g.signal_id WHERE c.total <> g.n_values",
    ),
)
