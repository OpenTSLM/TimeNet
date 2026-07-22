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
from timenet.reader import TimeFReader
from timenet.writer import TimeFWriter


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
    MatrixCase("zarr-small-chunks", "portable", "zarr", 64 * 1024, 512 * 1024, 2 * 1024 * 1024),
    MatrixCase("zarr-large-chunks", "portable", "zarr", 1024 * 1024, 4 * 1024 * 1024, 16 * 1024 * 1024),
    MatrixCase("zarr-rich", "rich", "zarr", 256 * 1024, 1024 * 1024, 4 * 1024 * 1024),
)
"""Default matrix covering both backends, two chunking regimes, and rich Zarr values."""


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
    writer_kwargs: dict[str, Any] = {
        "chunk_max_bytes": case.chunk_max_bytes,
        "row_group_target_bytes": case.row_group_target_bytes,
        "shard_target_bytes": case.shard_target_bytes,
    }
    if "values_backend" in inspect.signature(TimeFWriter).parameters:
        writer_kwargs["values_backend"] = case.values_backend
    elif case.values_backend != "parquet":
        raise ValueError("this revision does not support selectable values backends")
    with TimeFWriter(root, dataset, **writer_kwargs) as writer:
        writer.write()
    write_ns = time.perf_counter_ns() - started

    digest = hashlib.sha256()
    started = time.perf_counter_ns()
    with TimeFReader(_dataset_dir(root)) as reader:
        digest.update(
            json.dumps(
                {
                    "metadata": _canonical(reader.metadata),
                    "schema": _canonical(reader.schema),
                    "tasks": _canonical(reader.tasks),
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        )
        series_count = 0
        value_bytes = 0
        for sample in reader.iter_samples():
            digest.update(
                json.dumps(
                    {
                        "sample_id": sample.sample_id,
                        "subject_ids": sample.subject_ids,
                        "view": sample.view,
                        "task_ids": sample.task_ids,
                        "annotations": _canonical(sample.annotations),
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
            )
            for series in sample.time_series:
                values = np.ascontiguousarray(series.to_numpy())
                digest.update(
                    json.dumps(
                        {
                            "spec": _canonical(series.spec),
                            "channel": series.channel,
                            "source_id": series.source_id,
                            "time_series_id": series.time_series_id,
                            "sampling_rate_hz": float(series.sampling_rate_hz).hex(),
                            "t_start_s": float(series.t_start_s).hex(),
                            "t_end_s": None if series.t_end_s is None else float(series.t_end_s).hex(),
                            "dtype": values.dtype.str,
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
