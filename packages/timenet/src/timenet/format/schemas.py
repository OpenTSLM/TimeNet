"""Arrow schemas for the TimeF on-disk files, parameterized by id storage type.

The column layout is pinned; only the id columns vary. Every id column holds one of the six logical ids
in :data:`LOGICAL_IDS`, and each is stored either as ``pa.string()`` or, when every value is a canonical
UUID, as ``pa.binary(16)`` (16 raw bytes instead of a 36-char string). The writer picks the type per
logical id, records the choice in the manifest's ``id_encoding``, and the reader decodes ``binary(16)``
back to the canonical string, so callers always see string ids.
"""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, fields
from typing import cast

import pyarrow as pa

from timenet.errors import TimeFValidationError
from timenet.types import TASKS, Span, TaskRefs, TaskType
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
            ("sampling_rate_hz", pa.float64()),
            ("t_start_s", pa.float64()),
            ("t_end_s", pa.float64()),
        ]
    )


def samples_schema(id_types: IdTypes) -> pa.Schema:
    """Return the ``samples.parquet`` schema.

    Args:
        id_types: The resolved id storage types.

    Returns:
        The Arrow schema.
    """
    return pa.schema(
        [
            ("sample_id", id_types["sample_id"]),
            ("view", pa.string()),
            ("start_time_us", pa.int64()),
            ("subject_ids", pa.list_(id_types["subject_id"])),
            ("source_ids", pa.list_(id_types["source_id"])),
            ("time_series", pa.list_(time_series_struct(id_types))),
            ("task_ids", pa.list_(id_types["task_id"])),
            ("annotation_ids", pa.list_(id_types["annotation_id"])),
        ]
    )


def annotations_schema(id_types: IdTypes) -> pa.Schema:
    """Return the ``annotations.parquet`` schema.

    Args:
        id_types: The resolved id storage types.

    Returns:
        The Arrow schema.
    """
    return pa.schema(
        [
            ("id", id_types["annotation_id"]),
            ("key", pa.string()),
            ("annotation_type", pa.string()),
            ("value", pa.string()),  # JSON-encoded scalar/list, null for a pure marker
            ("start_time_s", pa.float64()),
            ("end_time_s", pa.float64()),
            ("time_series_ids", pa.list_(id_types["time_series_id"])),
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
        ]
    )


def index_schema(id_types: IdTypes) -> pa.Schema:
    """Return the ``time_series_index.parquet`` schema.

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
            # shard path, row group, and row offset; other backends assign their own coordinate meanings.
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
            ("start_s", pa.float64()),
            ("end_s", pa.float64()),  # null => the span is a point at start_s
            ("time_series_ids", pa.list_(id_types["time_series_id"])),
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


# Names of the columns every task partition shares, regardless of task type. The single source the
# writer's row builder and the reader's payload split both read, so the copies stay in lockstep.
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
    """The single place that converts ids between their in-memory strings and their on-disk form.

    The writer and reader previously each carried their own copy of the per-column encode/decode logic,
    threaded around as a bare ``uuid16`` set with the same ``"sample_id" in u`` membership check repeated
    at every call site. Centralizing it means the logical-id-per-column mapping and the ``bytes <-> str``
    conversion live in one spot that both sides build (:meth:`from_uuid16` / :meth:`from_encoding`) and
    call, so the two halves of the format contract cannot drift apart.
    """

    uuid16: frozenset[str]
    """The logical ids whose columns are stored as ``binary(16)``; the rest are ``pa.string()``."""

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

        A reference id outside the entity id space (e.g. a ``ForecastingTask.target_sample_id`` naming
        no sample) used to surface as a bare ``ValueError: badly formed hexadecimal UUID string`` from
        inside ``uuid``. It now raises a contextual :class:`TimeFValidationError` naming the column and
        value, so even if referential validation is bypassed the writer fails legibly.

        Args:
            logical: The logical id the column holds (e.g. ``"sample_id"``).
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
        """
        if span is None:
            return None
        return {
            "start_s": span.start_s,
            "end_s": span.end_s,
            "time_series_ids": (
                None if span.time_series_ids is None else self.encode_list("time_series_id", span.time_series_ids)
            ),
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
        """Decode an optional id (e.g. ``source_id``), passing ``None`` through.

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
        """
        if row is None:
            return None
        struct = cast("dict", row)
        series_ids = struct["time_series_ids"]
        return Span(
            start_s=struct["start_s"],
            end_s=struct["end_s"],
            time_series_ids=None if series_ids is None else tuple(self.decode_list("time_series_id", series_ids)),
        )

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
