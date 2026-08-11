"""Arrow schemas for the TimeF on-disk files, parameterized by id storage type.

The column layout is fixed. Only the id columns vary. Every id column holds one of the six logical ids
in :data:`LOGICAL_IDS`. Each is stored as ``pa.string()``, or as ``pa.binary(16)`` when every value is a
canonical UUID. ``pa.binary(16)`` uses 16 raw bytes instead of a 36-character string. The writer picks
the type per logical id and records the choice in the manifest ``id_encoding``. The reader decodes
``binary(16)`` back to the canonical string, so callers always see string ids.
"""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, fields
from typing import cast

import pyarrow as pa

from timenet.errors import TimeFFormatError, TimeFValidationError
from timenet.types import (
    TASKS,
    Span,
    StepInterval,
    StepPoint,
    StepSpan,
    TaskRefs,
    TaskType,
    TimeInterval,
    TimePoint,
    TimeSpan,
)
from timenet.types.ids import id_from_bytes, id_to_bytes


#: The logical ids that cross-reference TimeF entities. Every id column holds exactly one of these.
LOGICAL_IDS: tuple[str, ...] = (
    "sample_id",
    "time_series_id",
    "annotation_id",
    "task_id",
    "source_id",
    "subject_id",
)

#: The compact on-disk form for a canonical-UUID id column: 16 raw bytes.
UUID16 = pa.binary(16)

IdTypes = dict[str, pa.DataType]


def default_id_types() -> IdTypes:
    """Return the all-``string`` id types (the fallback when a manifest carries no ``id_encoding``).

    Returns:
        A mapping from every logical id to ``pa.string()``.
    """
    return {name: pa.string() for name in LOGICAL_IDS}


def id_types_from_encoding(id_encoding: dict[str, str]) -> IdTypes:
    """Resolve a manifest ``id_encoding`` map to Arrow id types.

    Args:
        id_encoding: A mapping from logical id to ``"uuid16"`` or ``"string"`` (missing keys default to
            ``"string"``).

    Returns:
        A mapping from every logical id to its Arrow type.
    """
    return {name: (UUID16 if id_encoding.get(name) == "uuid16" else pa.string()) for name in LOGICAL_IDS}


def time_series_struct(id_types: IdTypes) -> pa.DataType:
    """Return the per-sample nested time-series struct type.

    Args:
        id_types: The resolved id storage types.

    Returns:
        The struct type used inside ``samples.time_series``.
    """
    return pa.struct(
        [
            ("spec_type", pa.string()),
            ("channel", pa.string()),
            ("source_id", id_types["source_id"]),
            ("time_series_id", id_types["time_series_id"]),
            ("axis_type", pa.string()),  # dispatched on before any shape-specific column is read
            ("period_numerator_us", pa.int64()),  # non-null iff regular
            ("period_denominator", pa.int64()),  # non-null iff regular
            ("start_index", pa.int64()),  # non-null iff regular
            ("first_time_offset_us", pa.int64()),  # non-null iff irregular
            ("last_time_offset_us", pa.int64()),  # non-null iff irregular
            ("n_values", pa.int64()),
        ]
    )


def samples_schema(id_types: IdTypes) -> pa.Schema:
    """Return the samples table schema.

    Args:
        id_types: The resolved id storage types.

    Returns:
        The Arrow schema.
    """
    return pa.schema(
        [
            ("sample_id", id_types["sample_id"]),
            ("start_time_us", pa.int64()),
            ("time_span", span_struct(id_types)),  # null unless the sample declares an explicit session span
            ("subject_ids", pa.list_(id_types["subject_id"])),
            ("time_series", pa.list_(time_series_struct(id_types))),
            ("task_ids", pa.list_(id_types["task_id"])),
            ("annotation_ids", pa.list_(id_types["annotation_id"])),
        ]
    )


def annotations_schema(id_types: IdTypes) -> pa.Schema:
    """Return the annotations table schema.

    Args:
        id_types: The resolved id storage types.

    Returns:
        The Arrow schema.
    """
    return pa.schema(
        [
            ("id", id_types["annotation_id"]),
            ("key", pa.string()),
            ("value", pa.string()),  # JSON-encoded scalar/list, null for a pure marker
            ("span", span_struct(id_types)),  # null for a static annotation
            ("sample_ids", pa.list_(id_types["sample_id"])),
        ]
    )


def shard_schema(id_types: IdTypes) -> pa.Schema:
    """Return the waveform shard schema.

    Args:
        id_types: The resolved id storage types.

    Returns:
        The Arrow schema.
    """
    return pa.schema(
        [
            ("time_series_id", id_types["time_series_id"]),
            ("spec_type", pa.string()),
            ("channel", pa.string()),
            ("chunk_idx", pa.int32()),
            ("n_values", pa.int32()),
            ("values", pa.list_(pa.float32())),
            ("time_offsets_us", pa.list_(pa.int64())),
        ]
    )


def index_schema(id_types: IdTypes) -> pa.Schema:
    """Return the time-series index table schema.

    Args:
        id_types: The resolved id storage types.

    Returns:
        The Arrow schema.
    """
    return pa.schema(
        [
            ("sample_id", id_types["sample_id"]),
            ("time_series_id", id_types["time_series_id"]),
            ("spec_type", pa.string()),
            ("channel", pa.string()),
            ("chunk_idx", pa.int32()),
            # Backend-neutral chunk locator (a ChunkDataIndex flattened). For Parquet, these are the
            # shard path, row group, and row offset. Other backends assign their own coordinate meanings.
            ("chunk_file", pa.string()),
            ("chunk_major_idx", pa.int64()),
            ("chunk_minor_idx", pa.int64()),
            ("n_values", pa.int32()),
        ]
    )


def span_struct(id_types: IdTypes) -> pa.DataType:
    """Return the struct type a :class:`~timenet.types.Span` is stored as.

    Args:
        id_types: The resolved id storage types.

    Returns:
        The struct type used for a task's ``scope`` and for localization target spans.
    """
    return pa.struct(
        [
            ("start_us", pa.int64()),  # microseconds in the seconds frame, a step ordinal in steps
            ("end_us", pa.int64()),  # null => the span is a point
            ("time_series_ids", pa.list_(id_types["time_series_id"])),  # a step span stores its one id here
            ("frame", pa.string()),  # "seconds" or "steps"; null on a legacy partition => seconds
        ]
    )


def _task_common(id_types: IdTypes) -> list[tuple[str, pa.DataType]]:
    return [
        ("id", id_types["task_id"]),
        ("sample_ids", pa.list_(id_types["sample_id"])),
        ("from_task_ids", pa.list_(id_types["task_id"])),
        ("prompt", pa.string()),
        ("scope", span_struct(id_types)),
        ("input_annotation_ids", pa.list_(id_types["annotation_id"])),
        ("target_annotation_ids", pa.list_(id_types["annotation_id"])),
        ("rationale", pa.string()),
    ]


# Names of the columns every task partition shares, regardless of task type. The writer's row builder
# and the reader's payload split both read this one source, so the two copies stay in lockstep.
TASK_COMMON_NAMES: tuple[str, ...] = (
    "id",
    "sample_ids",
    "from_task_ids",
    "prompt",
    "scope",
    "input_annotation_ids",
    "target_annotation_ids",
    "rationale",
)


def _task_payload(id_types: IdTypes) -> dict[TaskType, list[tuple[str, pa.DataType]]]:
    return {
        TaskType.CLASSIFICATION: [("target", pa.string()), ("target_schema", pa.string())],
        TaskType.ANSWER: [("target", pa.string())],
        TaskType.SCALAR_PREDICTION: [
            ("target", pa.float64()),
            ("unit", pa.string()),
            ("target_name", pa.string()),
        ],
        TaskType.TEMPORAL_LOCALIZATION: [
            ("target", pa.list_(span_struct(id_types))),
            ("mode", pa.string()),
        ],
        TaskType.FORECASTING: [
            ("context_sample_ids", pa.list_(id_types["sample_id"])),
            ("target_sample_id", id_types["sample_id"]),
            ("target_span", span_struct(id_types)),
        ],
        TaskType.TS_EDITING: [
            ("source_sample_id", id_types["sample_id"]),
            ("target_sample_id", id_types["sample_id"]),
        ],
        TaskType.TS_GENERATION: [("target_sample_id", id_types["sample_id"])],
        TaskType.TS_CORRESPONDENCE: [
            ("candidate_sample_ids", pa.list_(id_types["sample_id"])),
            ("target", pa.list_(id_types["sample_id"])),
        ],
    }


def task_schema(task_type: TaskType, id_types: IdTypes | None = None) -> pa.Schema:
    """Return the Arrow schema for one task partition.

    Args:
        task_type: The task type whose partition schema to build.
        id_types: The resolved id storage types (defaults to all-``string``; the reader passes none
            because it only reads the column names).

    Returns:
        The Arrow schema (common columns plus the type's payload columns).

    Raises:
        TimeFValidationError: If the task dataclass payload and its Arrow schema have drifted.
    """
    resolved = id_types if id_types is not None else default_id_types()
    schema = pa.schema(_task_common(resolved) + _task_payload(resolved)[task_type])
    cls = TASKS[task_type]
    non_payload = {*TASK_COMMON_NAMES, "from_tasks"}
    expected = {field.name for field in fields(cls)} - non_payload
    if cls.answer_is_sample:
        expected.discard("target")
    actual = set(schema.names) - set(TASK_COMMON_NAMES)
    if expected != actual:
        raise TimeFValidationError(
            f"{cls.__name__} payload fields {sorted(expected)} do not match its Arrow schema fields {sorted(actual)}"
        )
    return schema


@dataclass(frozen=True)
class IdCodec:
    """Convert ids between their in-memory strings and their on-disk form.

    This is the one place that holds the logical-id-per-column mapping and the ``bytes <-> str``
    conversion. The writer and the reader both build a codec (:meth:`from_uuid16` /
    :meth:`from_encoding`) and call it, so the two halves of the format contract cannot drift apart.
    """

    uuid16: frozenset[str]
    """The logical ids whose columns are stored as ``binary(16)``. The rest are ``pa.string()``."""

    @classmethod
    def from_uuid16(cls, uuid16: Iterable[str]) -> "IdCodec":
        """Build a codec from the set of logical ids stored as ``binary(16)`` (the writer's entry point).

        Args:
            uuid16: The logical ids whose columns are ``binary(16)``.

        Returns:
            The codec for those columns.
        """
        return cls(frozenset(uuid16))

    @classmethod
    def from_encoding(cls, id_encoding: Mapping[str, str]) -> "IdCodec":
        """Build a codec from a manifest ``id_encoding`` map (the reader's entry point).

        Args:
            id_encoding: Mapping from logical id to ``"uuid16"`` / ``"string"``.

        Returns:
            The codec for those columns.
        """
        return cls(frozenset(name for name, enc in id_encoding.items() if enc == "uuid16"))

    def encode(self, logical: str, value: object) -> object:
        """Encode one id to 16 raw bytes for a ``uuid16`` column, else pass it through unchanged.

        A reference id outside the entity id space (for example, a ``ForecastingTask.target_sample_id``
        that names no sample) raises a contextual :class:`TimeFValidationError` naming the column and
        value. This makes the writer fail legibly even if referential validation is bypassed.

        Args:
            logical: The logical id the column holds (for example, ``"sample_id"``).
            value: The id string, or ``None``.

        Returns:
            The 16-byte form for a ``uuid16`` column, else ``value`` unchanged.

        Raises:
            TimeFValidationError: If a ``uuid16`` column holds a non-canonical-UUID id.
        """
        if value is None or logical not in self.uuid16:
            return value
        try:
            return id_to_bytes(cast(str, value))
        except (ValueError, AttributeError, TypeError) as exc:
            raise TimeFValidationError(
                f"id {value!r} in the {logical!r} column is not a canonical UUID, but that column is "
                f"stored as uuid16 (binary(16)); a reference likely points outside the entity id space"
            ) from exc

    def encode_list(self, logical: str, values: Iterable[object]) -> list:
        """Encode a list of ids element-wise via :meth:`encode`.

        Args:
            logical: The logical id the column holds.
            values: The id strings.

        Returns:
            The encoded list.
        """
        return [self.encode(logical, value) for value in values]

    def encode_span(self, span: Span | None) -> dict | None:
        """Encode a span to its struct row, encoding the series ids it is scoped to.

        Args:
            span: The span, or ``None`` for a whole-sample scope.

        Returns:
            The struct row, or ``None``.

        Raises:
            TimeFValidationError: If ``span`` is not a concrete time or step span.
        """
        if span is None:
            return None
        if isinstance(span, StepSpan):  # one id, stored as a single-element list
            series_ids = self.encode_list("time_series_id", (span.time_series_id,))
            start = span.start
        elif isinstance(span, TimeSpan):
            series_ids = (
                None if span.time_series_ids is None else self.encode_list("time_series_id", span.time_series_ids)
            )
            start = span.start_us
        else:
            raise TimeFValidationError(f"cannot encode {span!r}: not a concrete span")
        return {
            "start_us": start,
            "end_us": None if span.is_point else span.exclusive_end,
            "time_series_ids": series_ids,
            "frame": span.frame,
        }

    def encode_payload(self, refs: TaskRefs, name: str, value: object) -> object:
        """Encode a task payload cell holding ids or spans, leaving plain payload untouched.

        Args:
            refs: The owning task class's reference declaration.
            name: The payload column name.
            value: The (already list-normalized) cell value.

        Returns:
            The encoded value.
        """
        if value is None:
            return value
        if name in refs.span_fields:
            if isinstance(value, Span):
                return self.encode_span(value)
            return [self.encode_span(span) for span in cast("list[Span]", value)]
        if name not in refs.sample_id_fields or "sample_id" not in self.uuid16:
            return value
        return self.encode_list("sample_id", value) if isinstance(value, list) else self.encode("sample_id", value)

    def decode(self, logical: str, value: object) -> str:
        """Decode one required id, turning 16 raw bytes back into a canonical string for ``uuid16``.

        Args:
            logical: The logical id the column holds.
            value: The raw cell value (bytes for a ``uuid16`` column, string otherwise).

        Returns:
            The canonical id string.
        """
        if logical in self.uuid16:
            return id_from_bytes(cast(bytes, value))
        return cast(str, value)

    def decode_opt(self, logical: str, value: object) -> str | None:
        """Decode an optional id (for example, ``source_id``), passing ``None`` through.

        Args:
            logical: The logical id the column holds.
            value: The raw cell value, or ``None``.

        Returns:
            The canonical id string, or ``None``.
        """
        return None if value is None else self.decode(logical, value)

    def decode_list(self, logical: str, values: object) -> list[str]:
        """Decode a list of id values element-wise via :meth:`decode`.

        Args:
            logical: The logical id the column holds.
            values: The raw cell values.

        Returns:
            The decoded id strings.
        """
        return [self.decode(logical, value) for value in cast(list, values)]

    def decode_span(self, row: object) -> Span | None:
        """Rebuild a span from its struct row, decoding the series ids it is scoped to.

        Args:
            row: The struct row read from a task partition, or ``None``.

        Returns:
            The span, or ``None``.

        Raises:
            TimeFFormatError: If the frame is neither ``"seconds"`` nor ``"steps"``, or a step span's
                stored id list is empty.
        """
        if row is None:
            return None
        struct = cast("dict", row)
        series_ids = struct["time_series_ids"]
        start, end = struct["start_us"], struct["end_us"]
        frame = struct.get("frame") or "seconds"  # a legacy partition has no frame column
        # A round-tripped span must come back as the leaf type its frame and bounds describe, or it
        # stops comparing equal to the one that was written.
        if frame == "steps":
            if not series_ids:
                raise TimeFFormatError("a step span stores the one series it counts on, but its id list is empty")
            time_series_id = self.decode_list("time_series_id", series_ids)[0]  # one id, stored as a list
            if end is None:
                return StepPoint(time_series_id=time_series_id, start=start)
            return StepInterval(time_series_id=time_series_id, start=start, stop=end)
        if frame != "seconds":
            raise TimeFFormatError(f"unknown span frame {frame!r}; expected 'seconds' or 'steps'")
        time_series_ids = None if series_ids is None else tuple(self.decode_list("time_series_id", series_ids))
        if end is None:
            return TimePoint(start_us=start, time_series_ids=time_series_ids)
        return TimeInterval(start_us=start, end_us=end, time_series_ids=time_series_ids)

    def decode_payload(self, refs: TaskRefs, name: str, value: object) -> object:
        """Decode a task payload cell holding ids or spans, leaving plain payload untouched.

        Args:
            refs: The owning task class's reference declaration.
            name: The payload column name.
            value: The (already tuple-normalized) cell value.

        Returns:
            The decoded value.
        """
        if value is None:
            return value
        if name in refs.span_fields:
            if isinstance(value, tuple):
                return tuple(self.decode_span(row) for row in value)
            return self.decode_span(value)
        if name not in refs.sample_id_fields or "sample_id" not in self.uuid16:
            return value
        if isinstance(value, tuple):
            return tuple(self.decode("sample_id", item) for item in value)
        return self.decode("sample_id", value)
