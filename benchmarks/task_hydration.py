"""Time the task path end to end: register in memory, write to DuckDB, read back, iterate per Record.

Run it before and after a change to the task model or the control database::

    uv run python -m benchmarks.task_hydration --records 10000 --tasks-per-record 20
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from fractions import Fraction
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
import warnings

import pyarrow as pa

from timenet.dataset import Record, RegularAxis, Signal, Source, TimeFDataset
from timenet.format.control_reader import DuckDBControlReader
from timenet.format.control_writer import DuckDBControlWriter
from timenet.types import (
    Annotation,
    ClassificationTask,
    DatasetMetadata,
    License,
    TimeInterval,
    TimeSeriesSpec,
    Version,
    ureg,
)


_SPEC = TimeSeriesSpec(spec_type="voltage", name="Voltage", unit_value=ureg.millivolt, dtype="float32")
_N_VALUES = 1_000
_PERIOD_US = 2_000
_EPOCH_US = _N_VALUES * _PERIOD_US // 20
_BATCH = 32


@dataclass(frozen=True)
class TaskBenchmark:
    """Wall-clock seconds for each stage of the task path."""

    records: int
    annotations: int
    tasks: int
    register_seconds: float
    write_seconds: float
    read_records_seconds: float
    read_tasks_seconds: float
    task_table_seconds: float | None
    per_record_iteration_seconds: float | None
    per_batch_seconds: float | None

    def to_dict(self) -> dict[str, int | float | None]:
        """Return a JSON-compatible result.

        Returns:
            The configuration and every timing.
        """
        return asdict(self)


def _build(records: int, tasks_per_record: int, annotations_per_record: int) -> tuple[TimeFDataset, float]:
    axis = RegularAxis(axis_id="axis-1", period_us=Fraction(_PERIOD_US))
    values = pa.array([0.0] * _N_VALUES, type=pa.float32())
    dataset = TimeFDataset(
        metadata=DatasetMetadata(
            dataset_id="benchmark/tasks",
            dataset_version=Version(1, 0, 0),
            name="Task benchmark",
            description="Synthetic epochs",
            license=License.MIT,
        )
    )
    tasks: list[ClassificationTask] = []
    for index in range(records):
        signal = Signal.from_loader(
            id=f"signal-{index}",
            name="I",
            spec=_SPEC,
            time_axis=axis,
            n_values=_N_VALUES,
            loader=lambda: values,
        )
        record = Record(
            record_id=f"record-{index}", sources=(Source(id=f"source-{index}", name="ECG", signals=(signal,)),)
        )
        record.add_annotation(Annotation(id=f"age-{index}", key="age", value=40))
        for number in range(annotations_per_record - 1):
            record.add_annotation(
                Annotation(
                    id=f"event-{index}-{number}",
                    key="event",
                    value="arousal",
                    span=TimeInterval(start_us=number * _EPOCH_US, end_us=(number + 1) * _EPOCH_US),
                )
            )
        dataset.add_record(record=record)
        for epoch in range(tasks_per_record):
            tasks.append(
                ClassificationTask(
                    id=f"task-{index}-{epoch}",
                    inputs=(record,),
                    prompt="Which stage?",
                    targets=("W",),
                    scope=TimeInterval(start_us=epoch * _EPOCH_US, end_us=(epoch + 1) * _EPOCH_US),
                    input_annotations=(record.annotations[0],),
                    target_schema="aasm",
                )
            )
    started = perf_counter()
    dataset.add_tasks(tasks=tasks)
    return dataset, perf_counter() - started


def _time_task_access(reader: DuckDBControlReader, hydrated: tuple) -> tuple[float | None, float | None, float | None]:
    """Time the Arrow view, one-record reads, and batch reads when the reader offers them.

    Returns:
        Seconds for ``task_table``, per one-record ``read_tasks``, and per 32-record ``read_tasks``.
    """
    task_table_seconds = None
    table = getattr(reader, "task_table", None)
    if table is not None:
        started = perf_counter()
        table()
        task_table_seconds = perf_counter() - started
    if not hasattr(reader, "iter_tasks"):
        return task_table_seconds, None, None
    sample = hydrated[: min(len(hydrated), 100)]
    started = perf_counter()
    for record in sample:
        reader.read_tasks((record,))
    per_record_seconds = (perf_counter() - started) / len(sample)
    batches = [hydrated[start : start + _BATCH] for start in range(0, min(len(hydrated), 32 * _BATCH), _BATCH)]
    started = perf_counter()
    for batch in batches:
        reader.read_tasks(batch)
    per_batch_seconds = (perf_counter() - started) / len(batches)
    return task_table_seconds, per_record_seconds, per_batch_seconds


def run(records: int, tasks_per_record: int, annotations_per_record: int = 1) -> TaskBenchmark:
    """Build, write, and read one synthetic dataset.

    Returns:
        The stage timings.
    """
    dataset, register_seconds = _build(records, tasks_per_record, annotations_per_record)
    with TemporaryDirectory() as directory:
        path = Path(directory) / "control.duckdb"
        started = perf_counter()
        DuckDBControlWriter(path).write_hierarchy(dataset)
        write_seconds = perf_counter() - started

        with DuckDBControlReader(path) as reader:
            started = perf_counter()
            hydrated = reader.read_records()
            read_records_seconds = perf_counter() - started
            started = perf_counter()
            tasks = reader.read_tasks(hydrated)
            read_tasks_seconds = perf_counter() - started
            task_table_seconds, per_record_seconds, per_batch_seconds = _time_task_access(reader, hydrated)
    return TaskBenchmark(
        records=records,
        annotations=records * annotations_per_record,
        tasks=len(tasks),
        register_seconds=register_seconds,
        write_seconds=write_seconds,
        read_records_seconds=read_records_seconds,
        read_tasks_seconds=read_tasks_seconds,
        task_table_seconds=task_table_seconds,
        per_record_iteration_seconds=per_record_seconds,
        per_batch_seconds=per_batch_seconds,
    )


def main() -> None:
    """Run the benchmark from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=int, default=10_000)
    parser.add_argument("--tasks-per-record", type=int, default=20)
    parser.add_argument("--annotations-per-record", type=int, default=1)
    parser.add_argument("--json", action="store_true", help="Print the result as one JSON object.")
    arguments = parser.parse_args()
    warnings.simplefilter("ignore")
    result = run(arguments.records, arguments.tasks_per_record, arguments.annotations_per_record)
    if arguments.json:
        print(json.dumps(result.to_dict()))
        return
    print(f"{result.records} records, {result.annotations} annotations, {result.tasks} tasks")
    print(f"  register (add_tasks):      {result.register_seconds:8.2f} s")
    print(f"  write_hierarchy:           {result.write_seconds:8.2f} s")
    print(f"  read_records:              {result.read_records_seconds:8.2f} s")
    print(f"  read_tasks:                {result.read_tasks_seconds:8.2f} s")
    if result.task_table_seconds is not None:
        print(f"  task_table (Arrow):        {result.task_table_seconds:8.2f} s")
    if result.per_record_iteration_seconds is not None:
        print(f"  read_tasks, one record:    {result.per_record_iteration_seconds * 1000:8.2f} ms")
    if result.per_batch_seconds is not None:
        print(f"  read_tasks, {_BATCH} records:    {result.per_batch_seconds * 1000:8.2f} ms")


if __name__ == "__main__":
    main()
