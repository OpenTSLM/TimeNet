"""Arrow schemas for the TimeF on-disk files, parameterized by id storage type.

The column layout is pinned; only the id columns vary. Every id column holds one of the six logical ids
in :data:`LOGICAL_IDS`, and each is stored either as ``pa.string()`` or, when every value is a canonical
UUID, as ``pa.binary(16)`` (16 raw bytes instead of a 36-char string). The writer picks the type per
logical id, records the choice in the manifest's ``id_encoding``, and the reader decodes ``binary(16)``
back to the canonical string, so callers always see string ids.
"""

import pyarrow as pa

from timenet.types import TaskType


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
            ("shard_path", pa.string()),
            ("row_group", pa.int32()),
            ("row_offset", pa.int32()),
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
        TaskType.CLASSIFICATION: [("label", pa.string()), ("label_schema", pa.string())],
        TaskType.LABELING: [
            ("label", pa.string()),
            ("label_schema", pa.string()),
            ("time_series_ids", pa.list_(id_types["time_series_id"])),
            ("windows_s", pa.list_(pa.list_(pa.float64()))),
        ],
        TaskType.CAPTIONING: [("answer", pa.string())],
        TaskType.QUESTION_AND_ANSWER: [("question", pa.string()), ("answer", pa.string())],
        TaskType.FORECASTING: [
            ("context_sample_ids", pa.list_(id_types["sample_id"])),
            ("target_sample_id", id_types["sample_id"]),
        ],
        TaskType.REASONING: [("question", pa.string()), ("rationale", pa.string()), ("answer", pa.string())],
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
