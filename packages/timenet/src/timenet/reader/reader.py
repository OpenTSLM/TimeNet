"""``TimeFReader``: deserialize a TimeF version directory back into a :class:`TimeFDataset`.

Driven entirely by ``manifest.json`` — the reader never runs connector code. It loads the manifest,
tasks, annotations, and the time-series index eagerly, but keeps per-series values and sample
construction lazy. Types are reconstructed from the manifest's flat descriptors (no runtime class
synthesis), so read-back objects pickle and match the originals field-for-field.
"""

from collections.abc import Iterator
from dataclasses import dataclass
from fractions import Fraction
import json
from pathlib import Path
import types as _types

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from timenet.dataset import Sample, TimeFDataset, TimeSeries
from timenet.dataset.axis import AxisType, IrregularAxis, OrdinalAxis, RegularAxis, TimeAxis
from timenet.dataset.sample import check_span_within_window
from timenet.errors import TimeFFormatError, TimeFValidationError
from timenet.format.checksums import file_checksum
from timenet.format.constants import MANIFEST_FILE
from timenet.format.schemas import TASK_COMMON_NAMES, IdCodec, task_schema
from timenet.manifest import Manifest
from timenet.types import (
    TASKS,
    Annotation,
    DatasetMetadata,
    DatasetSchema,
    Task,
    TaskType,
    annotation_type_of,
    value_type_of,
)
from timenet.values_backends.reader import BaseValuesReader, make_values_reader


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
            TimeFFormatError: If a listed control-plane file is unreadable or disagrees with the manifest
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

        self._codec = IdCodec.from_encoding(self._manifest.id_encoding)
        self._spec_by_type = {spec.spec_type: spec for spec in self._manifest.schema.time_series_specs}
        self._annotation_descriptors = {d.key: d for d in self._manifest.schema.annotations}
        # The eager loaders parse on-disk control tables against the manifest's schema. Anything they raise means
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
        self._values: BaseValuesReader | None = None  # built lazily; dropped on pickle, rebuilt per process

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
                payload = {
                    name: self._codec.decode_payload(cls.refs, name, _as_tuple_if_list(row[name]))
                    for name in payload_cols
                }
                task = cls(
                    id=self._codec.decode("task_id", row["id"]),
                    sample_ids=tuple(self._codec.decode_list("sample_id", row["sample_ids"])),
                    prompt=row["prompt"],
                    scope=self._codec.decode_span(row["scope"]),
                    input_annotation_ids=tuple(self._codec.decode_list("annotation_id", row["input_annotation_ids"])),
                    target_annotation_ids=tuple(self._codec.decode_list("annotation_id", row["target_annotation_ids"])),
                    rationale=row["rationale"],
                    **payload,  # ty: ignore[invalid-argument-type]
                )
                by_id[task.id] = task
                pending[task.id] = tuple(self._codec.decode_list("task_id", row["from_task_ids"]))
        for task_id, from_ids in pending.items():
            resolved = []
            for from_id in from_ids:
                if from_id not in by_id:
                    raise TimeFFormatError(f"task {task_id!r} references unknown from_task_id {from_id!r}")
                resolved.append(by_id[from_id])
            by_id[task_id].from_tasks = tuple(resolved)
        return tuple(by_id.values())

    def _load_annotations(self) -> dict[str, Annotation]:
        annotations: dict[str, Annotation] = {}
        for row in self._read_rows(self._manifest.files.annotations):
            key = row["key"]
            if key not in self._annotation_descriptors:
                raise TimeFFormatError(f"annotation row references unknown key {key!r} (not in schema)")
            descriptor = self._annotation_descriptors[key]
            value = None if row["value"] is None else json.loads(row["value"])
            annotation_id = self._codec.decode("annotation_id", row["id"])
            fields: dict = {
                "id": annotation_id,
                "key": key,
                "value": value,
                "unit": descriptor.unit,
                "description": descriptor.description,
            }
            span = self._codec.decode_span(row["span"])
            if span is not None:
                fields["span"] = span
            annotation = Annotation(**fields)
            # The descriptor is what a registry query filters on, so a decoded annotation whose shape
            # or value type disagrees with it would answer those queries wrongly. That is corruption.
            derived_type = annotation_type_of(annotation)
            if derived_type != descriptor.annotation_type:
                raise TimeFFormatError(
                    f"annotation {key!r} decodes to shape {derived_type.value!r} but its descriptor "
                    f"says {descriptor.annotation_type.value!r}"
                )
            derived_value_type = value_type_of(annotation.value)
            if derived_value_type != descriptor.value_type:
                raise TimeFFormatError(
                    f"annotation {key!r} decodes to value type {derived_value_type!r} but its "
                    f"descriptor says {descriptor.value_type!r}"
                )
            annotations[annotation_id] = annotation
        return annotations

    def _load_index(self) -> dict[tuple[str, str], list[dict]]:
        index: dict[tuple[str, str], list[dict]] = {}
        for row in self._read_rows(self._manifest.files.time_series_index):
            key = (
                self._codec.decode("sample_id", row["sample_id"]),
                self._codec.decode("time_series_id", row["time_series_id"]),
            )
            index.setdefault(key, []).append(row)
        for rows in index.values():
            rows.sort(key=lambda r: r["chunk_idx"])
        return index

    # ---- sample construction -------------------------------------------------------------------

    def _build_sample(self, row: dict) -> Sample:
        sample_id = self._codec.decode("sample_id", row["sample_id"])
        series = tuple(self._build_series(sample_id, struct) for struct in row["time_series"])
        annotations = tuple(
            self._resolve_annotation(sample_id, aid)
            for aid in self._codec.decode_list("annotation_id", row["annotation_ids"])
        )
        # A span is stored on the annotation but resolved against the sample; a stored span that no
        # longer fits the series it lands on is a corrupt artifact, not a caller mistake.
        for annotation in annotations:
            if annotation.span is not None:
                try:
                    check_span_within_window(f"annotation {annotation.key!r}", annotation.span, series, sample_id)
                except TimeFValidationError as exc:
                    raise TimeFFormatError(str(exc)) from exc
        return Sample(
            sample_id=sample_id,
            time_series=series,
            subject_ids=tuple(self._codec.decode_list("subject_id", row["subject_ids"])),
            task_ids=tuple(self._codec.decode_list("task_id", row["task_ids"])),
            annotations=annotations,
            start_time=row.get("start_time_us"),
        )

    def _build_series(self, sample_id: str, struct: dict) -> TimeSeries:
        spec_type = struct["spec_type"]
        if spec_type not in self._spec_by_type:
            raise TimeFFormatError(f"sample {sample_id!r} references unknown spec_type {spec_type!r}")
        time_series_id = self._codec.decode("time_series_id", struct["time_series_id"])
        axis = self._axis(struct, sample_id)
        n_values = struct["n_values"]
        time_offsets_loader = None
        if isinstance(axis, IrregularAxis):
            time_offsets_loader = _TimeOffsetsLoader(
                self, sample_id, time_series_id, axis.first_us, axis.last_us, n_values
            )
        try:
            return TimeSeries(
                spec=self._spec_by_type[spec_type],
                channel=struct["channel"],
                time_axis=axis,
                loader=_SeriesLoader(self, sample_id, time_series_id),
                time_offsets_loader=time_offsets_loader,
                source_id=self._codec.decode_opt("source_id", struct["source_id"]),
                time_series_id=time_series_id,
                n_values=n_values,
            )
        except TimeFValidationError as exc:
            # A series rebuilt from a corrupt struct (bad n_values, a time offsets/axis mismatch) is a
            # format failure, not a caller mistake, even though TimeSeries raises the same type for both.
            raise TimeFFormatError(f"sample {sample_id!r} has an unbuildable series {time_series_id!r}: {exc}") from exc

    @staticmethod
    def _axis(struct: dict, sample_id: str) -> TimeAxis:
        """Rebuild a series' time axis from the stored discriminator.

        The tag is read before any shape-specific column, so a corrupt row raises here rather than
        producing an axis inferred from which columns happen to be null.

        Args:
            struct: The stored time-series struct.
            sample_id: The owning sample, for the error message.

        Returns:
            The axis.

        Raises:
            TimeFFormatError: If the tag is missing, unknown, or disagrees with the columns beside it.
        """
        kind = struct["axis_type"]
        if kind == AxisType.ORDINAL:
            # An ordinal series has no cadence and no per-value time offsets, so every shape column must
            # be null. A populated one means the tag and the columns disagree, the corruption this
            # method catches.
            shape_cols = (
                "period_numerator_us",
                "period_denominator",
                "start_index",
                "first_time_offset_us",
                "last_time_offset_us",
            )
            if any(struct[c] is not None for c in shape_cols):
                raise TimeFFormatError(
                    f"sample {sample_id!r} has a series tagged {kind!r} but carries regular- or "
                    f"irregular-axis columns; an ordinal series has neither"
                )
            return OrdinalAxis()
        if kind == AxisType.REGULAR:
            numerator, denominator = struct["period_numerator_us"], struct["period_denominator"]
            start_index = struct["start_index"]
            if numerator is None or denominator is None or start_index is None:
                raise TimeFFormatError(
                    f"sample {sample_id!r} has a series tagged {kind!r} with no period or start "
                    f"index; a regular axis needs both. pyarrow only enforces non-null when the file "
                    f"is written, so this is the check that catches a corrupt row"
                )
            try:
                return RegularAxis(period_us=Fraction(numerator, denominator), start_index=start_index)
            except (ZeroDivisionError, TypeError, TimeFValidationError) as exc:
                raise TimeFFormatError(
                    f"sample {sample_id!r} has a series with an unbuildable regular axis "
                    f"(period {numerator}/{denominator}, start_index {start_index}): {exc}"
                ) from exc
        if kind == AxisType.IRREGULAR:
            first, last = struct["first_time_offset_us"], struct["last_time_offset_us"]
            if first is None or last is None:
                raise TimeFFormatError(
                    f"sample {sample_id!r} has a series tagged {kind!r} with no endpoints; an "
                    f"irregular axis needs both. pyarrow only enforces non-null when the file is "
                    f"written, so this check catches a corrupt row"
                )
            try:
                return IrregularAxis(first_us=first, last_us=last)
            except TimeFValidationError as exc:
                raise TimeFFormatError(
                    f"sample {sample_id!r} has a series with unbuildable irregular endpoints ({first}, {last}): {exc}"
                ) from exc
        raise TimeFFormatError(
            f"sample {sample_id!r} has a series with axis_type {kind!r}; expected one of {[t.value for t in AxisType]}"
        )

    def _load_time_offsets(self, sample_id: str, time_series_id: str) -> pa.Array:
        """Read an irregular series' per-value time offsets through the values backend.

        Args:
            sample_id: The owning sample's id.
            time_series_id: The series id to read.

        Returns:
            One int64 microsecond time offset per value.

        Raises:
            TimeFFormatError: If the series has no index entry or its time offsets cannot be read.
        """
        rows = self._index.get((sample_id, time_series_id))
        if not rows:
            raise TimeFFormatError(f"no index entry for sample {sample_id!r} series {time_series_id!r}")
        if self._values is None:
            self._values = make_values_reader(self._manifest.values_backend)
        try:
            return self._values.load_time_offsets(self._root, rows)
        except (KeyError, OSError, IndexError, ValueError) as exc:
            raise TimeFFormatError(
                f"failed to read time offsets for series {time_series_id!r} for sample {sample_id!r}: {exc}"
            ) from exc

    def _resolve_annotation(self, sample_id: str, annotation_id: str) -> Annotation:
        if annotation_id not in self._annotations:
            raise TimeFFormatError(f"sample {sample_id!r} references unknown annotation {annotation_id!r}")
        return self._annotations[annotation_id]

    # ---- id decoding ---------------------------------------------------------------------------

    def _load_values(self, sample_id: str, time_series_id: str) -> pa.Array:
        """Read and concatenate a series' chunk values through the values backend.

        Args:
            sample_id: The owning sample's id.
            time_series_id: The series id to read.

        Returns:
            The series values in the spec's canonical Arrow representation.

        Raises:
            TimeFFormatError: If the series has no index entry or a chunk cannot be read.
        """
        rows = self._index.get((sample_id, time_series_id))
        if not rows:
            raise TimeFFormatError(f"no index entry for sample {sample_id!r} series {time_series_id!r}")
        if self._values is None:
            self._values = make_values_reader(self._manifest.values_backend)
        try:
            spec_type = rows[0]["spec_type"]
            return self._values.load(self._root, rows, self._spec_by_type[spec_type])
        except (KeyError, OSError, IndexError, ValueError) as exc:
            raise TimeFFormatError(f"failed to read series {time_series_id!r} for sample {sample_id!r}: {exc}") from exc


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
            The series values in the spec's canonical Arrow representation.
        """
        return self.reader._load_values(self.sample_id, self.time_series_id)

    def read_steps(self, start: int, stop: int) -> pa.Array:
        """Read a temporal subsection through the selected values backend.

        Returns:
            The requested steps in their canonical Arrow representation.

        Raises:
            TimeFFormatError: If this series has no index entry.
        """
        rows = self.reader._index.get((self.sample_id, self.time_series_id))
        if not rows:
            raise TimeFFormatError(f"no index entry for sample {self.sample_id!r} series {self.time_series_id!r}")
        if self.reader._values is None:
            self.reader._values = make_values_reader(self.reader._manifest.values_backend)
        spec = self.reader._spec_by_type[rows[0]["spec_type"]]
        return self.reader._values.load_range(self.reader._root, rows, start, stop, spec)


def _as_tuple_if_list(value: object) -> object:
    """Convert list payloads (and nested lists) to tuples; pass scalars and structs through.

    Span structs stay dicts for :meth:`~timenet.format.schemas.IdCodec.decode_span` to rebuild; only the
    list nesting around them is normalized, since tasks store tuples.


    Args:
        value: A cell value read from a task partition.

    Returns:
        The value with any list (recursively) converted to a tuple.
    """
    if isinstance(value, list):
        return tuple(_as_tuple_if_list(item) for item in value)
    return value


@dataclass(frozen=True)
class _TimeOffsetsLoader:
    """A picklable lazy loader for an irregular series' per-value time offsets.

    A nested closure can't be pickled, so this is a class rather than a lambda: a multi-worker torch
    DataLoader sends the series to its workers, and a closure would fail there. It mirrors
    :class:`_SeriesLoader` because time offsets and values are separate columns and each is read on its
    own.
    """

    reader: "TimeFReader"
    """The reader that reads and decodes the series' time offsets."""
    sample_id: str
    """The owning sample's id."""
    time_series_id: str
    """The id of the series to read."""
    first_us: int
    """The axis' first time offset, checked against the stored stream."""
    last_us: int
    """The axis' last time offset, checked against the stored stream."""
    n_values: int
    """The declared value count, checked against the stored stream's length."""

    def __call__(self) -> pa.Array:
        """Read the series' time offsets, checking them against the axis and value count.

        The writer verifies ordering, count, and endpoints, but nothing re-checks them on read, so a
        corrupt shard could otherwise hand back a decreasing, wrong-length, or off-endpoint stream.

        Returns:
            One int64 microsecond time offset per value.

        Raises:
            TimeFFormatError: If the stored time offsets disagree with the axis endpoints or value count,
                or are not non-decreasing.
        """
        time_offsets = self.reader._load_time_offsets(self.sample_id, self.time_series_id)
        values = time_offsets.to_numpy(zero_copy_only=False)
        where = f"series {self.time_series_id!r} on sample {self.sample_id!r}"
        if len(values) != self.n_values:
            raise TimeFFormatError(f"{where} stores {len(values)} time offsets but declares n_values={self.n_values}")
        if len(values) and (int(values[0]) != self.first_us or int(values[-1]) != self.last_us):
            raise TimeFFormatError(
                f"{where} has time offsets [{values[0]}, {values[-1]}] disagreeing with its axis "
                f"endpoints ({self.first_us}, {self.last_us})"
            )
        if np.any(np.diff(values) < 0):
            raise TimeFFormatError(f"{where} has non-decreasing time offsets that decrease on disk")
        return time_offsets
