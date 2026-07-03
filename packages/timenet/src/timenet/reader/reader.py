"""``TimeFReader``: deserialize a TimeF version directory back into a :class:`TimeFDataset`.

Driven entirely by ``manifest.json`` — the reader never runs connector code. It loads the manifest,
tasks, annotations, and the time-series index eagerly, but keeps per-series values and sample
construction lazy. Types are reconstructed from the manifest's flat descriptors (no runtime class
synthesis), so read-back objects pickle and match the originals field-for-field.
"""

from collections import OrderedDict
from collections.abc import Iterator
from dataclasses import dataclass
import json
from pathlib import Path
import types as _types

import pyarrow as pa
import pyarrow.parquet as pq

from timenet.dataset import Sample, TimeFDataset, TimeSeries
from timenet.errors import TimeFFormatError
from timenet.format.checksums import file_checksum
from timenet.format.constants import MANIFEST_FILE
from timenet.format.schemas import TASK_COMMON_NAMES, task_schema
from timenet.manifest import Manifest
from timenet.types import (
    ANNOTATION_BASES,
    TASKS,
    Annotation,
    AnnotationType,
    DatasetMetadata,
    DatasetSchema,
    Task,
    TaskType,
    View,
)


_ROW_GROUP_CACHE_SIZE = 16  # decoded row-group value columns kept, so a shared row group decodes once


class TimeFReader:
    """Reads a committed TimeF version directory. Use as a context manager to close shard handles."""

    def __init__(self, root: Path) -> None:
        """Open a dataset version directory and load its manifest, tasks, annotations, and index.

        A malformed or unsupported-version manifest raises ``InvalidManifestError`` (a
        ``TimeFFormatError``) while parsing.

        Args:
            root: The version directory written by :class:`~timenet.writer.TimeFWriter`.

        Raises:
            FileNotFoundError: If ``root``, its ``manifest.json``, or any file the manifest lists is
                missing.
            TimeFFormatError: If a listed parquet file is unreadable or disagrees with the manifest
                (an unknown task partition, a column the schema does not declare, a dangling
                reference). Corrupt bytes on disk are a format failure, not a caller error, so they
                do not surface as the raw ``ValueError`` / ``KeyError`` the parsing happens to raise.
        """
        self._root = Path(root)
        if not self._root.exists():
            raise FileNotFoundError(f"dataset directory does not exist: {self._root}")
        manifest_path = self._root / MANIFEST_FILE
        if not manifest_path.exists():
            raise FileNotFoundError(f"no manifest at {manifest_path}")
        self._manifest = Manifest.from_json(manifest_path.read_text())
        self._check_files_exist()

        self._spec_by_type = {spec.spec_type: spec for spec in self._manifest.schema.time_series_specs}
        self._annotation_descriptors = {d.key: d for d in self._manifest.schema.annotations}
        # The loaders parse on-disk parquet against the manifest's schema. Anything they raise means
        # the artifact is corrupt or disagrees with its manifest, which is a TimeFFormatError; letting
        # a bare ValueError from TaskType()/pyarrow escape would contradict this method's contract.
        try:
            self._tasks = self._load_tasks()
            self._annotations = self._load_annotations()
            self._index = self._load_index()
        except TimeFFormatError:
            raise
        except (ValueError, KeyError, TypeError, AttributeError, pa.ArrowInvalid) as exc:
            raise TimeFFormatError(f"corrupt or inconsistent TimeF artifact at {self._root}: {exc}") from exc
        self._shard_cache: dict[Path, pq.ParquetFile] = {}
        self._row_group_cache: OrderedDict[tuple[str, int], pa.ChunkedArray] = OrderedDict()

    # ---- pickling ------------------------------------------------------------------------------

    def __getstate__(self) -> dict:
        """Drop the open shard handles and cached row groups so the reader (and its loaders) pickle.

        The lazy loaders returned by :meth:`read` reference this reader, so a read-back dataset is
        only picklable (e.g. for a multi-worker torch ``DataLoader``) if the reader is. The handle and
        row-group caches are per-process scratch, rebuilt lazily after unpickling.

        Returns:
            The reader's state without its file caches.
        """
        state = self.__dict__.copy()
        state["_shard_cache"] = {}
        state["_row_group_cache"] = OrderedDict()
        return state

    # ---- context manager -----------------------------------------------------------------------

    def __enter__(self) -> "TimeFReader":
        """Return this reader."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: _types.TracebackType | None,
    ) -> None:
        """Close all cached shard handles."""
        self.close()

    def close(self) -> None:
        """Close every cached shard file handle and drop cached row groups."""
        for handle in self._shard_cache.values():
            handle.close()
        self._shard_cache.clear()
        self._row_group_cache.clear()

    # ---- public API ----------------------------------------------------------------------------

    def verify(self) -> None:
        """Check every file the manifest lists against its recorded ``sha256:`` checksum.

        Not done on open: hashing every shard would read the whole dataset and defeat the lazy read
        path this class exists to provide. Call it explicitly when integrity matters more than
        latency (after a download, before a long training run, in a fsck-style command).

        Raises:
            TimeFFormatError: If a listed file is missing, or its contents do not match the manifest.
        """
        for rel, expected in sorted(self._manifest.checksums.items()):
            path = self._root / rel
            if not path.exists():
                raise TimeFFormatError(f"manifest lists a missing file: {rel}")
            actual = file_checksum(path)
            if actual != expected:
                raise TimeFFormatError(f"checksum mismatch for {rel}: manifest says {expected}, file is {actual}")

    @property
    def metadata(self) -> DatasetMetadata:
        """The dataset's descriptive identity."""
        return self._manifest.metadata

    @property
    def schema(self) -> DatasetSchema:
        """The dataset's type declaration, reconstructed from the manifest."""
        return self._manifest.schema

    @property
    def tasks(self) -> tuple[Task, ...]:
        """All tasks, with ``from_tasks`` resolved."""
        return self._tasks

    def read(self) -> TimeFDataset:
        """Materialize the full dataset.

        Returns:
            A :class:`TimeFDataset` with lazy per-series loaders and the reconstructed schema/tasks.
        """
        return TimeFDataset.from_parts(
            metadata=self._manifest.metadata,
            samples=list(self.iter_samples()),
            tasks=self._tasks,
            schema=self._manifest.schema,
        )

    def iter_samples(self) -> Iterator[Sample]:
        """Yield each sample lazily without materializing a :class:`TimeFDataset`.

        Yields:
            Each reconstructed :class:`Sample`.
        """
        for row in pq.read_table(self._root / self._manifest.files.samples).to_pylist():
            yield self._build_sample(row)

    # ---- loading -------------------------------------------------------------------------------

    def _check_files_exist(self) -> None:
        files = self._manifest.files
        listed = [files.samples, files.annotations, files.time_series_index, *files.tasks, *files.time_series]
        for rel in listed:
            if not (self._root / rel).exists():
                raise FileNotFoundError(f"manifest lists a missing file: {rel}")

    def _load_tasks(self) -> tuple[Task, ...]:
        by_id: dict[str, Task] = {}
        pending: dict[str, tuple[str, ...]] = {}
        for rel in self._manifest.files.tasks:
            task_type = TaskType(Path(rel).parent.name.split("=", 1)[1])
            cls = TASKS[task_type]
            payload_cols = [name for name in task_schema(task_type).names if name not in TASK_COMMON_NAMES]
            for row in pq.read_table(self._root / rel).to_pylist():
                payload = {name: _as_tuple_if_list(row[name]) for name in payload_cols}
                task = cls(id=row["id"], sample_ids=tuple(row["sample_ids"]), **payload)  # ty: ignore[invalid-argument-type]
                by_id[task.id] = task
                pending[task.id] = tuple(row["from_task_ids"])
        for task_id, from_ids in pending.items():
            resolved = []
            for from_id in from_ids:
                if from_id not in by_id:
                    raise ValueError(f"task {task_id!r} references unknown from_task_id {from_id!r}")
                resolved.append(by_id[from_id])
            by_id[task_id].from_tasks = tuple(resolved)
        return tuple(by_id.values())

    def _load_annotations(self) -> dict[str, Annotation]:
        annotations: dict[str, Annotation] = {}
        for row in pq.read_table(self._root / self._manifest.files.annotations).to_pylist():
            key = row["key"]
            if key not in self._annotation_descriptors:
                raise ValueError(f"annotation row references unknown key {key!r} (not in schema)")
            descriptor = self._annotation_descriptors[key]
            base = ANNOTATION_BASES[AnnotationType(row["annotation_type"])]
            value = None if row["value"] is None else json.loads(row["value"])
            fields: dict = {
                "id": row["id"],
                "key": key,
                "value": value,
                "unit": descriptor.unit,
                "description": descriptor.description,
            }
            if row["start_time_s"] is not None:
                fields["start_time_s"] = row["start_time_s"]
            if row["end_time_s"] is not None:
                fields["end_time_s"] = row["end_time_s"]
            if row["time_series_ids"] is not None:
                fields["time_series_ids"] = tuple(row["time_series_ids"])
            annotations[row["id"]] = base(**fields)
        return annotations

    def _load_index(self) -> dict[tuple[str, str], list[dict]]:
        index: dict[tuple[str, str], list[dict]] = {}
        for row in pq.read_table(self._root / self._manifest.files.time_series_index).to_pylist():
            index.setdefault((row["sample_id"], row["time_series_id"]), []).append(row)
        for rows in index.values():
            rows.sort(key=lambda r: r["chunk_idx"])
        return index

    # ---- sample construction -------------------------------------------------------------------

    def _build_sample(self, row: dict) -> Sample:
        series = tuple(self._build_series(row["sample_id"], struct) for struct in row["time_series"])
        annotations = tuple(self._resolve_annotation(row["sample_id"], aid) for aid in row["annotation_ids"])
        return Sample(
            sample_id=row["sample_id"],
            time_series=series,
            view=View(row["view"]),
            subject_ids=tuple(row["subject_ids"]),
            task_ids=tuple(row["task_ids"]),
            annotations=annotations,
        )

    def _build_series(self, sample_id: str, struct: dict) -> TimeSeries:
        spec_type = struct["spec_type"]
        if spec_type not in self._spec_by_type:
            raise ValueError(f"sample {sample_id!r} references unknown spec_type {spec_type!r}")
        return TimeSeries(
            spec=self._spec_by_type[spec_type],
            channel=struct["channel"],
            sampling_rate_hz=struct["sampling_rate_hz"],
            loader=_SeriesLoader(self, sample_id, struct["time_series_id"]),
            source_id=struct["source_id"],
            time_series_id=struct["time_series_id"],
            t_start_s=struct["t_start_s"],
            t_end_s=struct["t_end_s"],
        )

    def _resolve_annotation(self, sample_id: str, annotation_id: str) -> Annotation:
        if annotation_id not in self._annotations:
            raise ValueError(f"sample {sample_id!r} references unknown annotation {annotation_id!r}")
        return self._annotations[annotation_id]

    def _load_values(self, sample_id: str, time_series_id: str) -> pa.Array:
        """Read and concatenate a series' chunk values across its shards.

        Args:
            sample_id: The owning sample's id.
            time_series_id: The series id to read.

        Returns:
            The series' 1-D float32 values.

        Raises:
            ValueError: If the series has no index entry or a chunk cannot be read.
        """
        rows = self._index.get((sample_id, time_series_id))
        if not rows:
            raise ValueError(f"no index entry for sample {sample_id!r} series {time_series_id!r}")
        try:
            chunks = [
                self._row_group_values(row["shard_path"], row["row_group"])[row["row_offset"]].values for row in rows
            ]
        except (OSError, IndexError, ValueError) as exc:
            raise ValueError(f"failed to read series {time_series_id!r} for sample {sample_id!r}: {exc}") from exc
        return pa.concat_arrays([chunk.cast(pa.float32()) for chunk in chunks])

    def _row_group_values(self, rel_path: str, row_group: int) -> pa.ChunkedArray:
        """Return a shard row group's ``values`` column, decoding each row group at most once.

        Chunks of different series can share a row group; without this cache every per-series read
        would re-decode the whole column, making value materialization quadratic in chunks per group.

        Args:
            rel_path: The shard's path relative to the version directory.
            row_group: The row-group index within the shard.

        Returns:
            The decoded ``values`` column of the row group.
        """
        key = (rel_path, row_group)
        cached = self._row_group_cache.get(key)
        if cached is not None:
            self._row_group_cache.move_to_end(key)
            return cached
        values = self._shard(rel_path).read_row_group(row_group, columns=["values"]).column("values")
        self._row_group_cache[key] = values
        while len(self._row_group_cache) > _ROW_GROUP_CACHE_SIZE:
            self._row_group_cache.popitem(last=False)
        return values

    def _shard(self, rel_path: str) -> pq.ParquetFile:
        path = self._root / rel_path
        if path not in self._shard_cache:
            self._shard_cache[path] = pq.ParquetFile(path)
        return self._shard_cache[path]


@dataclass(frozen=True)
class _SeriesLoader:
    """A picklable lazy loader for one series' values (replaces a per-series closure).

    A nested closure can't be pickled, which would make every read-back dataset unpicklable and break
    a multi-worker torch ``DataLoader``. This holds the reader and the series' identity instead, and
    reads on call.
    """

    reader: "TimeFReader"
    sample_id: str
    time_series_id: str

    def __call__(self) -> pa.Array:
        """Read the series' values.

        Returns:
            The series' 1-D float32 Arrow array.
        """
        return self.reader._load_values(self.sample_id, self.time_series_id)


def _as_tuple_if_list(value: object) -> object:
    """Convert list payloads (and nested lists, e.g. windows) to tuples; pass scalars through.

    Args:
        value: A cell value read from a task partition.

    Returns:
        The value with any list (recursively) converted to a tuple.
    """
    if isinstance(value, list):
        return tuple(_as_tuple_if_list(item) for item in value)
    return value
