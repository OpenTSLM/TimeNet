"""``TimeFReader``: deserialize a TimeF version directory back into a :class:`TimeFDataset`.

Driven entirely by ``manifest.json`` — the reader never runs connector code. It loads the manifest,
tasks, annotations, and the time-series index eagerly, but keeps per-series values and sample
construction lazy. Types are reconstructed from the manifest's flat descriptors (no runtime class
synthesis), so read-back objects pickle and match the originals field-for-field.
"""

from collections.abc import Iterator
from dataclasses import dataclass
import json
from pathlib import Path
import types as _types
from typing import cast

import pyarrow as pa
import pyarrow.parquet as pq

from timenet.dataset import Sample, TimeFDataset, TimeSeries
from timenet.manifest import Manifest
from timenet.reader.values import ValuesReader, make_values_reader
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
from timenet.types.ids import id_from_bytes
from timenet.writer.constants import MANIFEST_FILE
from timenet.writer.schemas import TASK_COMMON_NAMES, TASK_PAYLOAD_ID_COLUMNS, task_schema


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
        """
        self._root = Path(root)
        if not self._root.exists():
            raise FileNotFoundError(f"dataset directory does not exist: {self._root}")
        manifest_path = self._root / MANIFEST_FILE
        if not manifest_path.exists():
            raise FileNotFoundError(f"no manifest at {manifest_path}")
        self._manifest = Manifest.from_json(manifest_path.read_text())
        self._check_files_exist()

        self._uuid16 = {name for name, enc in self._manifest.id_encoding.items() if enc == "uuid16"}
        self._spec_by_type = {spec.spec_type: spec for spec in self._manifest.schema.time_series_specs}
        self._annotation_descriptors = {d.key: d for d in self._manifest.schema.annotations}
        self._tasks = self._load_tasks()
        self._annotations = self._load_annotations()
        self._index = self._load_index()
        self._values: ValuesReader | None = None  # built lazily; dropped on pickle, rebuilt per process

    # ---- pickling ------------------------------------------------------------------------------

    def __getstate__(self) -> dict:
        """Drop the values backend (open handles/caches) so the reader (and its loaders) pickle.

        The lazy loaders returned by :meth:`read` reference this reader, so a read-back dataset is
        only picklable (e.g. for a multi-worker torch ``DataLoader``) if the reader is. The values
        backend holds per-process scratch (file handles, decode caches), rebuilt lazily after unpickling.

        Returns:
            The reader's state without its values backend.
        """
        state = self.__dict__.copy()
        state["_values"] = None
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
        """Close the values backend (its open handles and caches), if one was built."""
        if self._values is not None:
            self._values.close()
            self._values = None

    # ---- public API ----------------------------------------------------------------------------

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

    @property
    def values_backend(self) -> str:
        """The manifest's values-plane backend tag (``"parquet"`` or ``"zarr"``)."""
        return self._manifest.values_backend

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
        for row in self._read_rows(self._manifest.files.samples):
            yield self._build_sample(row)

    # ---- loading -------------------------------------------------------------------------------

    def _read_rows(self, parts: tuple[str, ...]) -> Iterator[dict]:
        """Yield every row across a multi-part artifact, in part order.

        Args:
            parts: The artifact's relative part paths from the manifest.

        Yields:
            Each row as a dict, concatenated across parts.
        """
        for rel in parts:
            yield from pq.read_table(self._root / rel).to_pylist()

    def _check_files_exist(self) -> None:
        for rel in self._manifest.files.all_parts():
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
                payload = {name: self._dec_payload(name, _as_tuple_if_list(row[name])) for name in payload_cols}
                task = cls(
                    id=self._dec("task_id", row["id"]),
                    sample_ids=tuple(self._dec_list("sample_id", row["sample_ids"])),
                    **payload,  # ty: ignore[invalid-argument-type]
                )
                by_id[task.id] = task
                pending[task.id] = tuple(self._dec_list("task_id", row["from_task_ids"]))
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
        for row in self._read_rows(self._manifest.files.annotations):
            key = row["key"]
            if key not in self._annotation_descriptors:
                raise ValueError(f"annotation row references unknown key {key!r} (not in schema)")
            descriptor = self._annotation_descriptors[key]
            base = ANNOTATION_BASES[AnnotationType(row["annotation_type"])]
            value = None if row["value"] is None else json.loads(row["value"])
            annotation_id = self._dec("annotation_id", row["id"])
            fields: dict = {
                "id": annotation_id,
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
                fields["time_series_ids"] = tuple(self._dec_list("time_series_id", row["time_series_ids"]))
            annotations[annotation_id] = base(**fields)
        return annotations

    def _load_index(self) -> dict[tuple[str, str], list[dict]]:
        index: dict[tuple[str, str], list[dict]] = {}
        for row in self._read_rows(self._manifest.files.time_series_index):
            key = (self._dec("sample_id", row["sample_id"]), self._dec("time_series_id", row["time_series_id"]))
            index.setdefault(key, []).append(row)
        for rows in index.values():
            rows.sort(key=lambda r: r["chunk_idx"])
        return index

    # ---- sample construction -------------------------------------------------------------------

    def _build_sample(self, row: dict) -> Sample:
        sample_id = self._dec("sample_id", row["sample_id"])
        series = tuple(self._build_series(sample_id, struct) for struct in row["time_series"])
        annotations = tuple(
            self._resolve_annotation(sample_id, aid) for aid in self._dec_list("annotation_id", row["annotation_ids"])
        )
        return Sample(
            sample_id=sample_id,
            time_series=series,
            view=View(row["view"]),
            subject_ids=tuple(self._dec_list("subject_id", row["subject_ids"])),
            task_ids=tuple(self._dec_list("task_id", row["task_ids"])),
            annotations=annotations,
        )

    def _build_series(self, sample_id: str, struct: dict) -> TimeSeries:
        spec_type = struct["spec_type"]
        if spec_type not in self._spec_by_type:
            raise ValueError(f"sample {sample_id!r} references unknown spec_type {spec_type!r}")
        time_series_id = self._dec("time_series_id", struct["time_series_id"])
        return TimeSeries(
            spec=self._spec_by_type[spec_type],
            channel=struct["channel"],
            sampling_rate_hz=struct["sampling_rate_hz"],
            loader=_SeriesLoader(self, sample_id, time_series_id),
            source_id=self._dec_opt("source_id", struct["source_id"]),
            time_series_id=time_series_id,
            t_start_s=struct["t_start_s"],
            t_end_s=struct["t_end_s"],
        )

    def _resolve_annotation(self, sample_id: str, annotation_id: str) -> Annotation:
        if annotation_id not in self._annotations:
            raise ValueError(f"sample {sample_id!r} references unknown annotation {annotation_id!r}")
        return self._annotations[annotation_id]

    # ---- id decoding ---------------------------------------------------------------------------

    def _dec(self, logical: str, value: object) -> str:
        """Decode one required id, turning 16 raw bytes back into a canonical string for uuid16 columns.

        Args:
            logical: The logical id the column holds (e.g. ``"sample_id"``).
            value: The raw cell value (bytes for a uuid16 column, string otherwise).

        Returns:
            The canonical id string.
        """
        if logical in self._uuid16:
            return id_from_bytes(cast(bytes, value))
        return cast(str, value)

    def _dec_opt(self, logical: str, value: object) -> str | None:
        """Decode an optional id (e.g. ``source_id``), passing ``None`` through.

        Args:
            logical: The logical id the column holds.
            value: The raw cell value, or ``None``.

        Returns:
            The canonical id string, or ``None``.
        """
        return None if value is None else self._dec(logical, value)

    def _dec_list(self, logical: str, values: object) -> list[str]:
        """Decode a list of id values element-wise via :meth:`_dec`.

        Args:
            logical: The logical id the column holds.
            values: The raw cell values.

        Returns:
            The decoded id strings.
        """
        return [self._dec(logical, value) for value in cast(list, values)]

    def _dec_payload(self, name: str, value: object) -> object:
        """Decode a task payload cell if it holds ids, leaving non-id payload untouched.

        Args:
            name: The payload column name.
            value: The (already tuple-normalized) cell value.

        Returns:
            The decoded value.
        """
        logical = TASK_PAYLOAD_ID_COLUMNS.get(name)
        if logical is None or logical not in self._uuid16 or value is None:
            return value
        if isinstance(value, tuple):
            return tuple(self._dec(logical, item) for item in value)
        return self._dec(logical, value)

    def _load_values(self, sample_id: str, time_series_id: str) -> pa.Array:
        """Read and concatenate a series' chunk values through the values backend.

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
        if self._values is None:
            self._values = make_values_reader(self._manifest.values_backend)
        try:
            spec_type = rows[0]["spec_type"]
            return self._values.load(self._root, rows, self._spec_by_type[spec_type])
        except (KeyError, OSError, IndexError, ValueError) as exc:
            raise ValueError(f"failed to read series {time_series_id!r} for sample {sample_id!r}: {exc}") from exc


@dataclass(frozen=True)
class _SeriesLoader:
    """A picklable lazy loader for one series' values (replaces a per-series closure).

    A nested closure can't be pickled, which would make every read-back dataset unpicklable and break
    a multi-worker torch ``DataLoader``. This holds the reader and the series' identity instead, and
    reads on call.
    """

    reader: "TimeFReader"
    """The reader that reads and decodes the series' values."""
    sample_id: str
    """The owning sample's id."""
    time_series_id: str
    """The id of the series to read."""

    def __call__(self) -> pa.Array:
        """Read the series' values.

        Returns:
            The series' 1-D float32 Arrow array.
        """
        return self.reader._load_values(self.sample_id, self.time_series_id)

    def read_steps(self, start: int, stop: int) -> pa.Array:
        """Read a temporal subsection through the selected values backend.

        Returns:
            The requested steps in their canonical Arrow representation.

        Raises:
            ValueError: If this series has no index entry.
        """
        rows = self.reader._index.get((self.sample_id, self.time_series_id))
        if not rows:
            raise ValueError(f"no index entry for sample {self.sample_id!r} series {self.time_series_id!r}")
        if self.reader._values is None:
            self.reader._values = make_values_reader(self.reader._manifest.values_backend)
        spec = self.reader._spec_by_type[rows[0]["spec_type"]]
        return self.reader._values.load_range(self.reader._root, rows, start, stop, spec)


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
