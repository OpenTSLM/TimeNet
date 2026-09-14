"""Shared write, read, fingerprint, and timing operations for the regression suite."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, is_dataclass
from enum import Enum
import hashlib
import importlib
import importlib.util
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
    MatrixCase("zarr-small-chunks", "portable", "zarr", 64 * 1024, 512 * 1024, 2 * 1024 * 1024),
    MatrixCase("zarr-large-chunks", "portable", "zarr", 1024 * 1024, 4 * 1024 * 1024, 16 * 1024 * 1024),
    MatrixCase("zarr-rich", "rich", "zarr", 256 * 1024, 1024 * 1024, 4 * 1024 * 1024),
)
"""The matrix, over two chunking regimes, the richer value shapes, and both values backends.

The Zarr cases came back once ``signal_chunks`` stopped naming Parquet row groups. They share the
fingerprint with the Parquet cases on purpose: the two backends lay bytes out differently, and the
digest covers only the logical content, so a matching digest is what says they are interchangeable.
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


#: Tables that count physical placements rather than logical content. How many artifacts a values
#: plane wrote and how many chunks it split a signal into are exactly what a backend is free to
#: decide, so the fingerprint leaves them out and lets the values themselves carry the comparison.
_PHYSICAL_TABLES = frozenset({"values_artifacts", "signal_chunks"})


def _logical_counts(counts: dict[str, int]) -> dict[str, int]:
    """Drop the tables whose row counts describe layout rather than content.

    Returns:
        The counts a backend may not change.
    """
    return {table: count for table, count in counts.items() if table not in _PHYSICAL_TABLES}


def _values_options(case: MatrixCase) -> dict[str, int]:
    """Return the byte budgets one case's backend accepts.

    ``chunk_max_bytes`` and ``shard_target_bytes`` mean something to both backends. Row groups are
    Parquet's alone, so the Zarr cases do not pass that budget rather than passing one it ignores.

    Returns:
        The keyword arguments for the writer.

    Raises:
        ValueError: If the case names a backend this revision cannot write.
    """
    shared = {"chunk_max_bytes": case.chunk_max_bytes, "shard_target_bytes": case.shard_target_bytes}
    if case.values_backend == "zarr":
        return shared
    if case.values_backend == "parquet":
        return {**shared, "row_group_target_bytes": case.row_group_target_bytes}
    raise ValueError(f"this revision has no {case.values_backend!r} values backend")


def write_and_fingerprint(root: Path, case: MatrixCase, *, scale: int) -> tuple[str, dict[str, float | int]]:
    """Build, write, read, materialize, and fingerprint one matrix case.

    Returns:
        The logical SHA-256 and operation timings/counts.

    Raises:
        ValueError: If the selected backend is unavailable in this revision.
    """  # noqa: DOC502 - raised by _values_options
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True)
    started = time.perf_counter_ns()
    dataset = build_corpus(profile=case.profile, scale=scale)
    converted_ns = time.perf_counter_ns() - started
    started = time.perf_counter_ns()
    with TimeFWriter(root, dataset.metadata, values_backend=case.values_backend, **_values_options(case)) as writer:
        writer.write(dataset)
    write_ns = time.perf_counter_ns() - started

    # Every entity is identified by the id the corpus gave it. The surrogate the writer assigns is
    # dense and follows the walk, so it describes the build rather than the content.
    digest = hashlib.sha256()
    started = time.perf_counter_ns()
    series_count = 0
    value_bytes = 0
    with TimeFReader(_dataset_dir(root)) as reader:
        digest.update(
            json.dumps(
                {
                    "counts": _canonical(_logical_counts(reader.counts())),
                    "dataset_annotations": _canonical(reader.annotations_for("dataset")),
                    "tasks": _canonical(
                        [
                            {
                                "external_id": task.external_id,
                                "prompt": task.prompt,
                                "inputs": [item if isinstance(item, str) else item.external_id for item in task.inputs],
                                "target": [item if isinstance(item, str) else item.external_id for item in task.target],
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
                        "external_id": record.external_id,
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
                            "external_id": signal.external_id,
                            "name": signal.name,
                            "spec_type": signal.spec_type,
                            "unit": signal.unit,
                            "dtype": signal.dtype,
                            "axis_type": signal.axis_type,
                            # The axis itself, not just its kind: the period, the origin and the
                            # endpoints are what place a value in time, so a writer that halves
                            # every period must not fingerprint the same.
                            "time_axis": _canonical(signal.time_axis),
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

    supported_values_backends = frozenset({"parquet"})
    # Two revisions, two homes for the backend list: the control-plane branch keeps it beside the
    # locator it interprets, and the revision before it had a values_backends package.
    for module_name, attribute in (
        ("timenet.control_plane.values", "VALUES_BACKENDS"),
        ("timenet.values_backends", "SUPPORTED_VALUES_BACKENDS"),
    ):
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError:
            continue
        declared = getattr(module, attribute, None)
        if declared is not None:
            supported_values_backends = frozenset(declared)
            break

    spec_parameters = inspect.signature(TimeSeriesSpec).parameters
    return {
        # The backend is declared, and the optional extra that implements it is actually installed.
        "zarr": "zarr" in supported_values_backends and importlib.util.find_spec("zarr") is not None,
        "rich": "dtype" in spec_parameters and "value_shape" in spec_parameters,
    }
