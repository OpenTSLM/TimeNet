"""What the control database is: its tables, as DuckDB DDL, and the widths and versions they use.

The tables hold the same model the Parquet control plane held. A flat
:class:`~timenet.dataset.Record` carries its list of :class:`~timenet.dataset.TimeSeries` and its
annotations. The dataset also holds the annotations registered for tasks to reference. Every
:class:`~timenet.types.Task` shares ten fields and adds the payload its own type declares. The
authoring model does not change. Only the place it is stored does.

**Every join runs on a dense integer key, and the caller's id is stored once.** A record, a series,
an annotation and a task each carry a surrogate id, assigned in walk order. Every link row names
that id. The caller's id survives as ``external_id`` on the entity that owns it. That is the id the
reader hands back, and the one that stays stable across a rebuild. The writer assigns surrogate ids
per build, so nothing outside the file refers to one.

**A type declaration is stored once, with its columns spelled out.** ``specs`` carries the whole
:class:`~timenet.types.TimeSeriesSpec`, and ``annotation_descriptors`` the whole
:class:`~timenet.types.AnnotationDescriptor`. The database therefore says what dtype, unit and value
shape a signal carries, without the manifest beside it. A client can attach the file over HTTP and
answer that from SQL. A million series reference the declaration by id, so the unit and the value
type are stored once rather than per row. The manifest keeps the same block as a projection, so a
registry can filter datasets without downloading the database. The reader still types its rows
against the database.

**An annotation is attached through one table per target kind.** On a 3.06M-attachment corpus, a
polymorphic ``(object_type, object_id)`` table measured 55.7 MB against 50.2 MB for one table per
kind. The query that gathers everything for one entity ran 25.50 ms against 21.24 ms. Every target
column is ``NOT NULL``. There is no discriminator to store or branch on, and each check is a bare
anti-join.

There are three kinds because main's model has three. A record carries annotations. A task
references them as input or as its answer. The dataset holds the registered ones that no record
carries.

**A chunk locator names an artifact and two integers, not a Parquet row group.** The columns are
``(artifact_id, chunk_major_idx, chunk_minor_idx)``, and the backend that wrote the artifact says
what they mean. Parquet reads them as a row group and a row inside it. Zarr reads the major index as
an element offset and leaves the minor index null. ``values_artifacts`` lists every file the values
plane wrote and the backend that wrote it. The writer therefore checks each chunk row against a
declared artifact.

**There is no axis-offsets table, and that is deliberate.** An irregular axis stores one microsecond
offset per value. The offsets are as long as the values themselves, and a read of the values reads
them too. They stay in the values plane, in a column beside the values of the same chunk, as they do
on main. One chunk locator finds both, and one read returns both. The control plane keeps only the
endpoints (``axes.first_us`` and ``axes.last_us``) that a query filters on.

A table here moves a per-value column into the database, which every structural query then steps
over. It also splits the read of one series across two planes. A reader that looks for the
proposal's ``axis_offsets`` table finds the offsets in the values plane.

Nothing below declares a primary or a foreign key. That is a measured trade, not an oversight.
:mod:`timenet.control_plane.checks` holds the checks that stand in for them, and the numbers that
decided it.
"""

from typing import Final


SCHEMA_VERSION: Final = 1
"""The control-plane schema version, recorded in ``meta`` so a reader can refuse a newer file."""

ID_TYPE: Final = "UINTEGER"
"""The DuckDB type of every surrogate id column.

Width costs less than it first appears. At ``BLOCK_SIZE`` 16 KiB, DuckDB's minimum, it refuses to
bitpack these columns, so 64-bit looks like a flat 2x. A larger block size removes that premium. On
a SLIP-shape control plane, ``UINTEGER`` against ``UBIGINT`` is 147.6 MB against 228.4 MB at 16 KiB,
but 48.6 MB against 48.6 MB at 64 KiB.

64-bit still costs 9 to 12% on the hot joins, which is why the declared type stays 32-bit. To
exhaust it needs 4.29 billion records in one version. A corpus that large needs a sharded control
plane long before it needs a wider id.
"""

MAX_ID: Final = 2**32 - 1
"""The largest surrogate id an :data:`ID_TYPE` column holds. The writer refuses to assign one past it."""

BLOCK_SIZE: Final = 65_536
"""The database block size, set when the file is created and fixed afterwards.

DuckDB claims at least one block per table. A large block therefore sets a floor of a few megabytes
on a database of a few hundred rows. At 16 KiB it refuses to bitpack the id columns, so a large
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

64 KiB wins for every corpus of more than a few thousand records, and 256 KiB never wins. Under the
crossover 16 KiB is better by 0.26 MB. That difference does not justify a block size per corpus.
"""


_DDL_TEMPLATE: Final = """
CREATE TABLE meta (
    key   VARCHAR NOT NULL,
    value VARCHAR NOT NULL
);

CREATE TABLE specs (
    spec_id              {id} NOT NULL,
    spec_type             VARCHAR NOT NULL,
    name                  VARCHAR NOT NULL,
    unit_value            VARCHAR NOT NULL,
    data_source_type      VARCHAR,
    data_source_name      VARCHAR,
    data_source_provider  VARCHAR,
    dtype                 VARCHAR NOT NULL,
    categories            VARCHAR[] NOT NULL,
    value_shape           BIGINT[] NOT NULL,
    dimension_names       VARCHAR[] NOT NULL,
    nullable              BOOLEAN NOT NULL
);

CREATE TABLE annotation_descriptors (
    descriptor_id  {id} NOT NULL,
    key             VARCHAR NOT NULL,
    annotation_type VARCHAR NOT NULL,
    value_type      VARCHAR,
    unit            VARCHAR,
    description     VARCHAR
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
    subject_ids        VARCHAR[] NOT NULL
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

CREATE TABLE record_tasks (
    record_id {id} NOT NULL,
    task_id   {id}
);

CREATE TABLE task_items (
    task_id   {id} NOT NULL,
    role       VARCHAR NOT NULL CHECK (role IN ('input', 'target')),
    position   INTEGER NOT NULL,
    item_type  VARCHAR NOT NULL CHECK (item_type IN ('text', 'record')),
    text_value VARCHAR,
    record_id {id}
);

CREATE TABLE task_from_tasks (
    task_id    {id} NOT NULL,
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
    task_id  {id} NOT NULL,
    field     VARCHAR NOT NULL,
    position  INTEGER NOT NULL,
    ref_kind  VARCHAR NOT NULL CHECK (ref_kind IN ('record', 'time_series')),
    ref_id   {id}
);

CREATE TABLE task_spans (
    task_id        {id} NOT NULL,
    field           VARCHAR NOT NULL,
    position        INTEGER NOT NULL,
    frame           VARCHAR NOT NULL CHECK (frame IN ('seconds', 'steps')),
    start_at        BIGINT NOT NULL,
    end_at          BIGINT,
    time_series_ids VARCHAR[]
);

CREATE TABLE values_artifacts (
    artifact_id {id} NOT NULL,
    chunk_file   VARCHAR NOT NULL,
    backend      VARCHAR NOT NULL CHECK (backend IN ('parquet', 'zarr'))
);

CREATE TABLE time_series_chunks (
    time_series_id {id} NOT NULL,
    chunk_idx       INTEGER NOT NULL,
    artifact_id    {id} NOT NULL,
    chunk_major_idx BIGINT NOT NULL,
    chunk_minor_idx BIGINT,
    n_values        INTEGER NOT NULL
);
"""
"""The table definitions. ``{id}`` stands for the surrogate id width."""

DDL: Final = _DDL_TEMPLATE.format(id=ID_TYPE)
"""Every table, in an order that lets each reference name a table that already exists."""

ANNOTATION_TABLES: Final = {
    "dataset": ("dataset_annotations", None),
    "record": ("record_annotations", "record_id"),
    "task": ("task_annotations", "task_id"),
}
"""Which table holds each kind of attachment, and the column that names its target.

The dataset's table has no target column. A control database holds one dataset, so the table itself
says which entity the row is about.
"""

KEYED_TABLES: Final = {
    "records": "record_id",
    "time_series": "time_series_id",
    "specs": "spec_id",
    "annotation_descriptors": "descriptor_id",
    "axes": "axis_id",
    "annotations": "annotation_id",
    "tasks": "task_id",
    "values_artifacts": "artifact_id",
    **{table: "attachment_id" for table, _ in ANNOTATION_TABLES.values()},
}
"""Each entity table and its surrogate id column."""

EXTERNAL_IDS: Final = ("records", "time_series", "annotations", "tasks")
"""Each table that keeps the caller's id, in its ``external_id`` column."""

TABLES: Final = (
    "meta",
    "specs",
    "annotation_descriptors",
    "axes",
    "records",
    "time_series",
    "record_time_series",
    "annotations",
    *(table for table, _ in ANNOTATION_TABLES.values()),
    "tasks",
    "record_tasks",
    "task_items",
    "task_from_tasks",
    "task_fields",
    "task_refs",
    "task_spans",
    "values_artifacts",
    "time_series_chunks",
)
"""Every table the control plane defines, for a reader that wants to check what it opened."""
