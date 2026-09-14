"""Shared write, read, fingerprint, and timing operations for the regression suite."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, is_dataclass
from enum import Enum
import hashlib
import importlib
import inspect
import json
from pathlib import Path
import shutil
import time
from typing import Any

import numpy as np

from benchmarks.end_to_end.corpus import build_corpus
from timenet.control_plane import TimeFReader, TimeFWriter


@dataclass(frozen=True)
class MatrixCase:
    """One corpus/backend/chunking combination."""

    name: str
    profile: str
    values_backend: str
    chunk_max_bytes: int
    row_group_target_bytes: int
    shard_target_bytes: int


MATRIX = (
    MatrixCase("parquet-small-chunks", "portable", "parquet", 64 * 1024, 512 * 1024, 2 * 1024 * 1024),
    MatrixCase("parquet-large-chunks", "portable", "parquet", 1024 * 1024, 4 * 1024 * 1024, 16 * 1024 * 1024),
    MatrixCase("parquet-rich", "rich", "parquet", 256 * 1024, 1024 * 1024, 4 * 1024 * 1024),
)
"""The matrix, over two chunking regimes and the richer value shapes.

The Zarr cases are absent because this branch has one values backend. Restoring them needs a Zarr
backend and a locator in ``signal_chunks`` that can address a Zarr chunk, which the current
``(chunk_file, row_group, row_offset)`` cannot.
"""


def _canonical(value: Any) -> Any:  # noqa: PLR0911, RUF100 - explicit type cases keep canonicalization readable
    """Convert TimeF values into stable JSON-compatible structures.

    Returns:
        A recursively canonicalized value.
    """
    if value is None or isinstance(value, str | int | bool):
        return value
    if isinstance(value, float):
        return {"float_hex": value.hex()}
    if isinstance(value, Enum):
        return _canonical(value.value)
    if isinstance(value, type):
        return f"{value.__module__}.{value.__qualname__}"
    if is_dataclass(value):
        return {field.name: _canonical(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, dict):
        return {str(key): _canonical(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, list | tuple):
        return [_canonical(item) for item in value]
    return str(value)


def _dataset_dir(root: Path) -> Path:
    """Return the fixed corpus version directory."""
    return root / "timenet" / "end-to-end-benchmark" / "1.0.0"


def write_and_fingerprint(root: Path, case: MatrixCase, *, scale: int) -> tuple[str, dict[str, float | int]]:
    """Build, write, read, materialize, and fingerprint one matrix case.

    Returns:
        The logical SHA-256 and operation timings/counts.

    Raises:
        ValueError: If the selected backend is unavailable in this revision.
    """
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True)
    started = time.perf_counter_ns()
    dataset = build_corpus(profile=case.profile, scale=scale)
    converted_ns = time.perf_counter_ns() - started
    started = time.perf_counter_ns()
    if case.values_backend != "parquet":
        raise ValueError(f"this revision has no {case.values_backend!r} values backend")
    with TimeFWriter(
        root,
        dataset.metadata,
        chunk_max_bytes=case.chunk_max_bytes,
        row_group_target_bytes=case.row_group_target_bytes,
        shard_target_bytes=case.shard_target_bytes,
    ) as writer:
        writer.write(dataset)
    write_ns = time.perf_counter_ns() - started

    digest = hashlib.sha256()
    started = time.perf_counter_ns()
    series_count = 0
    value_bytes = 0
    with TimeFReader(_dataset_dir(root)) as reader:
        digest.update(
            json.dumps(
                {
                    "counts": _canonical(reader.counts()),
                    "dataset_annotations": _canonical(reader.annotations_for("dataset", "dataset")),
                    "tasks": _canonical(
                        [
                            {
                                "task_id": task.task_id,
                                "prompt": task.prompt,
                                "inputs": [item if isinstance(item, str) else item.record_id for item in task.inputs],
                                "target": [item if isinstance(item, str) else item.record_id for item in task.target],
                                "annotations": _canonical(task.annotations),
                            }
                            for task in reader.tasks(reader.task_ids())
                        ]
                    ),
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        )
        for record in reader.iter_records(batch_size=64):
            digest.update(
                json.dumps(
                    {
                        "record_id": record.record_id,
                        "start_time_us": record.start_time_us,
                        "annotations": _canonical(record.annotations),
                        "sources": _canonical(
                            [
                                {
                                    "name": source.name,
                                    "depth": source.depth,
                                    "annotations": _canonical(source.annotations),
                                }
                                for source in record.walk_sources()
                            ]
                        ),
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
            )
            for signal in record.signals():
                values = np.ascontiguousarray(reader.values(signal.signal_id))
                digest.update(
                    json.dumps(
                        {
                            "signal_id": signal.signal_id,
                            "name": signal.name,
                            "spec_type": signal.spec_type,
                            "unit": signal.unit,
                            "dtype": signal.dtype,
                            "axis_type": signal.axis_type,
                            "n_values": signal.n_values,
                            "annotations": _canonical(signal.annotations),
                            "values_dtype": values.dtype.str,
                            "shape": values.shape,
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode()
                )
                digest.update(values.tobytes())
                series_count += 1
                value_bytes += values.nbytes
    read_ns = time.perf_counter_ns() - started
    disk_bytes = sum(path.stat().st_size for path in _dataset_dir(root).rglob("*") if path.is_file())
    return digest.hexdigest(), {
        "convert_ns": converted_ns,
        "write_ns": write_ns,
        "read_ns": read_ns,
        "total_ns": converted_ns + write_ns + read_ns,
        "series": series_count,
        "value_bytes": value_bytes,
        "disk_bytes": disk_bytes,
    }


def case_to_json(case: MatrixCase) -> str:
    """Serialize one matrix case for a worker process.

    Returns:
        A stable JSON object.
    """
    return json.dumps(asdict(case), sort_keys=True)


def case_from_json(value: str) -> MatrixCase:
    """Deserialize one matrix case passed to a worker process.

    Returns:
        The decoded matrix case.
    """
    return MatrixCase(**json.loads(value))


def capabilities() -> dict[str, bool]:
    """Report format features supported by the active revision.

    Returns:
        Flags for selectable Zarr storage and rich dtype/N-D specs.
    """
    from timenet.types import TimeSeriesSpec  # noqa: PLC0415

    try:
        values_backends = importlib.import_module("timenet.values_backends")
    except ModuleNotFoundError:
        supported_values_backends = frozenset({"parquet"})
    else:
        supported_values_backends = values_backends.SUPPORTED_VALUES_BACKENDS

    spec_parameters = inspect.signature(TimeSeriesSpec).parameters
    return {
        "zarr": "zarr" in supported_values_backends,
        "rich": "dtype" in spec_parameters and "value_shape" in spec_parameters,
    }
