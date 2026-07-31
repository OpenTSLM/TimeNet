"""Arrow schemas for the TimeF on-disk files, parameterized by id storage type.

The column layout is pinned; only the id columns vary. Every id column holds one of the six logical ids
in :data:`LOGICAL_IDS`, and each is stored either as ``pa.string()`` or, when every value is a canonical
UUID, as ``pa.binary(16)`` (16 raw bytes instead of a 36-char string). The writer picks the type per
logical id, records the choice in the manifest's ``id_encoding``, and the reader decodes ``binary(16)``
back to the canonical string, so callers always see string ids.
"""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import cast

import pyarrow as pa

from timenet.errors import TimeFValidationError
from timenet.types import TaskType
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
            ("t0_unix_ns", pa.int64()),
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
            ("t_start_s", pa.float64()),
            ("n_values", pa.int32()),
            ("sampling_rate_hz", pa.float64()),
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
            ("t_start_s", pa.float64()),
            ("t_end_s", pa.float64()),
            ("n_values", pa.int32()),
        ]
    )


def _task_common(id_types: IdTypes) -> list[tuple[str, pa.DataType]]:
    return [
        ("id", id_types["task_id"]),
        ("sample_ids", pa.list_(id_types["sample_id"])),
        ("from_task_ids", pa.list_(id_types["task_id"])),
    ]


# Names of the columns every task partition shares, regardless of task type. The single source the
# writer's row builder and the reader's payload split both read, so the copies stay in lockstep.
TASK_COMMON_NAMES: tuple[str, ...] = ("id", "sample_ids", "from_task_ids")

#: Task payload columns that hold ids, and the logical id each holds. The common id/sample_ids/
#: from_task_ids columns are handled separately; this covers the type-specific payload ids.
TASK_PAYLOAD_ID_COLUMNS: dict[str, str] = {
    "time_series_ids": "time_series_id",
    "context_sample_ids": "sample_id",
    "target_sample_id": "sample_id",
}


def _task_payload(id_types: IdTypes) -> dict[TaskType, list[tuple[str, pa.DataType]]]:
    return {
        TaskType.CLASSIFICATION: [("target", pa.string()), ("target_schema", pa.string())],
        TaskType.LABELING: [
            ("target", pa.string()),
            ("target_schema", pa.string()),
            ("time_series_ids", pa.list_(id_types["time_series_id"])),
            ("windows_s", pa.list_(pa.list_(pa.float64()))),
        ],
        TaskType.CAPTIONING: [("target", pa.string())],
        TaskType.QUESTION_AND_ANSWER: [("question", pa.string()), ("target", pa.string())],
        TaskType.FORECASTING: [
            ("context_sample_ids", pa.list_(id_types["sample_id"])),
            ("target_sample_id", id_types["sample_id"]),
        ],
        TaskType.REASONING: [("question", pa.string()), ("rationale", pa.string()), ("target", pa.string())],
    }


def task_schema(task_type: TaskType, id_types: IdTypes | None = None) -> pa.Schema:
    """Return the Arrow schema for one task partition.

    Args:
        task_type: The task type whose partition schema to build.
        id_types: The resolved id storage types (defaults to all-``string``; the reader passes none
            because it only reads the column names).

    Returns:
        The Arrow schema (common columns plus the type's payload columns).
    """
    resolved = id_types if id_types is not None else default_id_types()
    return pa.schema(_task_common(resolved) + _task_payload(resolved)[task_type])


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

    def encode_payload(self, name: str, value: object) -> object:
        """Encode a task payload cell if it holds ids, leaving non-id payload untouched.

        Args:
            name: The payload column name.
            value: The (already list-normalized) cell value.

        Returns:
            The encoded value.
        """
        logical = TASK_PAYLOAD_ID_COLUMNS.get(name)
        if logical is None or logical not in self.uuid16 or value is None:
            return value
        return self.encode_list(logical, value) if isinstance(value, list) else self.encode(logical, value)

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

    def decode_payload(self, name: str, value: object) -> object:
        """Decode a task payload cell if it holds ids, leaving non-id payload untouched.

        Args:
            name: The payload column name.
            value: The (already tuple-normalized) cell value.

        Returns:
            The decoded value.
        """
        logical = TASK_PAYLOAD_ID_COLUMNS.get(name)
        if logical is None or logical not in self.uuid16 or value is None:
            return value
        if isinstance(value, tuple):
            return tuple(self.decode(logical, item) for item in value)
        return self.decode(logical, value)
