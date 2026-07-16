"""Pinned Arrow schemas for the TimeF on-disk files.

Fixing the schemas as constants keeps the byte-level contract explicit and stable across writer runs, so
the reader can rely on exact column names and types.
"""

import pyarrow as pa

from timenet.types import TaskType


TIME_SERIES_STRUCT = pa.struct(
    [
        ("spec_type", pa.string()),
        ("channel", pa.string()),
        ("source_id", pa.string()),
        ("time_series_id", pa.string()),
        ("sampling_rate_hz", pa.float64()),
        ("t_start_s", pa.float64()),
        ("t_end_s", pa.float64()),
    ]
)

SAMPLES_SCHEMA = pa.schema(
    [
        ("sample_id", pa.string()),
        ("view", pa.string()),
        ("subject_ids", pa.list_(pa.string())),
        ("source_ids", pa.list_(pa.string())),
        ("time_series", pa.list_(TIME_SERIES_STRUCT)),
        ("task_ids", pa.list_(pa.string())),
        ("annotation_ids", pa.list_(pa.string())),
    ]
)

ANNOTATIONS_SCHEMA = pa.schema(
    [
        ("id", pa.string()),
        ("key", pa.string()),
        ("annotation_type", pa.string()),
        ("value", pa.string()),  # JSON-encoded scalar/list, null for a pure marker
        ("start_time_s", pa.float64()),
        ("end_time_s", pa.float64()),
        ("time_series_ids", pa.list_(pa.string())),
        ("sample_ids", pa.list_(pa.string())),
    ]
)

SHARD_SCHEMA = pa.schema(
    [
        ("time_series_id", pa.string()),
        ("spec_type", pa.string()),
        ("channel", pa.string()),
        ("chunk_idx", pa.int32()),
        ("t_start_s", pa.float64()),
        ("n_values", pa.int32()),
        ("sampling_rate_hz", pa.float64()),
        ("values", pa.list_(pa.float32())),
    ]
)

INDEX_SCHEMA = pa.schema(
    [
        ("sample_id", pa.string()),
        ("time_series_id", pa.string()),
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

_TASK_COMMON = [
    ("id", pa.string()),
    ("sample_ids", pa.list_(pa.string())),
    ("from_task_ids", pa.list_(pa.string())),
]

# Names of the columns every task partition shares, regardless of task type. The single source the
# writer's row builder and the reader's payload split both read, so the copies stay in lockstep.
TASK_COMMON_NAMES: tuple[str, ...] = tuple(name for name, _ in _TASK_COMMON)

_TASK_PAYLOAD: dict[TaskType, list[tuple[str, pa.DataType]]] = {
    TaskType.CLASSIFICATION: [("label", pa.string()), ("label_schema", pa.string())],
    TaskType.LABELING: [
        ("label", pa.string()),
        ("label_schema", pa.string()),
        ("time_series_ids", pa.list_(pa.string())),
        ("windows_s", pa.list_(pa.list_(pa.float64()))),
    ],
    TaskType.CAPTIONING: [("answer", pa.string())],
    TaskType.QUESTION_AND_ANSWER: [("question", pa.string()), ("answer", pa.string())],
    TaskType.FORECASTING: [("context_sample_ids", pa.list_(pa.string())), ("target_sample_id", pa.string())],
    TaskType.REASONING: [("question", pa.string()), ("rationale", pa.string()), ("answer", pa.string())],
}


def task_schema(task_type: TaskType) -> pa.Schema:
    """Return the Arrow schema for one task partition.

    Args:
        task_type: The task type whose partition schema to build.

    Returns:
        The Arrow schema (common columns plus the type's payload columns).
    """
    return pa.schema(_TASK_COMMON + _TASK_PAYLOAD[task_type])
