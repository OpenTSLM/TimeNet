"""A DuckDB control plane for TimeF: the hierarchy in a database, the values in a plane beside it.

The control plane holds the structure of a dataset: records, their recursive source trees, the
signals those sources produce, the tasks that refer to records, and the annotations attached to any
of them. It is small, deeply cross-referenced, and read by point lookups, joins, and tree walks.
That is a relational workload, so it lives in an embedded database.

The values plane holds the waveforms. It is bulk numeric data read by random access at a known
offset. Nothing about that is relational, so it stays in files: Parquet shards with per-column
encodings chosen from measured cardinality, or a Zarr store. ``signal_chunks`` bridges the two, and
its locator is written so that either backend can be addressed without a scan.

Both planes ship inside one version directory, described by one ``manifest.json``, fetched by the
registry machinery that already exists.
"""

from timenet.control_plane.model import (
    Annotation,
    DeclarativeDataset,
    Record,
    RecordRef,
    Signal,
    Source,
    Task,
)
from timenet.control_plane.reader import (
    RecordView,
    ResolvedAnnotation,
    SignalView,
    SourceView,
    TaskView,
    TimeFReader,
    render_task,
)
from timenet.control_plane.writer import TimeFWriter


__all__ = [
    "Annotation",
    "DeclarativeDataset",
    "Record",
    "RecordRef",
    "RecordView",
    "ResolvedAnnotation",
    "Signal",
    "SignalView",
    "Source",
    "SourceView",
    "Task",
    "TaskView",
    "TimeFReader",
    "TimeFWriter",
    "render_task",
]
