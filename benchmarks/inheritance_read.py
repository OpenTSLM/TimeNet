"""Measure local reads of the same records through zero, one, and three parent layers."""

import argparse
import cProfile
from fractions import Fraction
import gc
import json
from pathlib import Path
import platform
import statistics
import time
from typing import Any

import numpy as np

from timenet.composition import BuildContext
from timenet.dataset import Record, RegularAxis, Signal, Source, TimeFDataset
from timenet.errors import TimeFValidationError
from timenet.registry import LocalRegistry
from timenet.types import Annotation, AnswerTask, DatasetMetadata, DatasetRef, License, TimeSeriesSpec, Version
from timenet.writer import TimeFWriter


DATASETS = {"flat": "benchmark/flat", "one-parent": "benchmark/layer-1", "three-parents": "benchmark/layer-3"}
MODES = ("selected", "records", "dataset", "values", "selected-warm")


def _metadata(dataset_id: str, parent: str | None = None) -> DatasetMetadata:
    return DatasetMetadata(
        dataset_id=dataset_id,
        dataset_version=Version(1, 0, 0),
        name="Inheritance read benchmark",
        description="Deterministic local read workload",
        license=License.MIT,
        parents=() if parent is None else (DatasetRef(dataset_id=parent, version=Version(1, 0, 0)),),
    )


def _annotation(identifier: str, key: str) -> Annotation:
    return Annotation(
        id=identifier,
        occurrence_id=identifier + "-occurrence",
        key=key,
        value=[f"label-{index}" for index in range(16)],
        metadata={"nested": {"labels": [f"tag-{index}" for index in range(16)]}},
    )


def prepare(root: Path, records: int) -> None:
    """Write one root and three children with identical lazy signal payloads."""
    dataset = TimeFDataset(metadata=_metadata("benchmark/flat"))
    spec = TimeSeriesSpec(spec_type="benchmark-signal", name="Signal", unit_value=None)
    for index in range(records):
        record_id = f"record-{index:05d}"
        axis = RegularAxis(axis_id=f"axis-{index}", period_us=Fraction(10_000))
        signals = tuple(
            Signal(
                id=f"signal-{index}-{channel}",
                name=f"channel-{channel}",
                spec=spec,
                time_axis=axis,
                data=np.arange(64, dtype=np.float32),
                annotations=(_annotation(f"signal-label-{index}-{channel}", "signal-label"),),
                metadata={"nested": {"channel": channel, "tags": ["a", "b", "c"]}},
            )
            for channel in range(4)
        )
        record = dataset.add_record(
            record=Record(
                record_id=record_id,
                sources=(Source(id=f"source-{index}", name="source", signals=signals),),
                annotations=tuple(_annotation(f"record-label-{index}-{label}", f"label-{label}") for label in range(4)),
            )
        )
        dataset.add_task(task=AnswerTask(id=f"task-{index}", inputs=(record,), targets=("normal",)))
    with TimeFWriter(root, dataset) as writer:
        writer.write()
    parent_id = dataset.metadata.dataset_id
    for depth in range(1, 4):
        child = TimeFDataset(metadata=_metadata(f"benchmark/layer-{depth}", parent_id))
        with BuildContext.open(child.metadata, LocalRegistry(root)) as context:
            for index, imported in enumerate(context.parent(parent_id).import_records(child)):
                imported.annotate(Annotation(key=f"layer-{depth}", value=depth))
                child.add_task(task=AnswerTask(id=f"task-{index}", inputs=(imported,), targets=("normal",)))
            context.verify_imports(child)
            child.set_dependencies(context.dependency_lock())
            with TimeFWriter(root, child) as writer:
                writer.write()
        parent_id = child.metadata.dataset_id
    (root / "workload.json").write_text(json.dumps({"records": records, "signals_per_record": 4, "samples": 64}))


def _measure_warm_selection(root: Path, dataset_id: str, selected: tuple[str, ...]) -> float:
    """Measure a selected batch after opening the same reader's parent connections.

    Returns:
        Read duration in seconds, excluding reader setup and closing.

    Raises:
        TimeFValidationError: If the reader returns different record IDs or loses shared axes.
    """
    with LocalRegistry(root).open_reader(dataset_id) as reader:
        tuple(reader.iter_records(selected))
        gc.collect()
        start = time.perf_counter()
        records = tuple(reader.iter_records(selected))
        elapsed = time.perf_counter() - start
    if tuple(record.id for record in records) != selected:
        raise TimeFValidationError("benchmark read returned incorrect record IDs")
    if records[0].signals[0].time_axis is not records[0].signals[1].time_axis:
        raise TimeFValidationError("benchmark read did not preserve shared axes")
    return elapsed


def measure(root: Path, dataset_id: str, mode: str, selected: tuple[str, ...], expected: int) -> float:
    """Measure one fresh reader, including opening and closing its dependency graph.

    Returns:
        Read duration in seconds.

    Raises:
        TimeFValidationError: If the read changes counts, references, or values.
    """
    if mode == "selected-warm":
        return _measure_warm_selection(root, dataset_id, selected)
    gc.collect()
    start = time.perf_counter()
    with LocalRegistry(root).open_reader(dataset_id) as reader:
        if mode == "dataset":
            dataset = reader.read()
            records = dataset.records
        else:
            records = tuple(reader.iter_records(selected if mode in {"selected", "values"} else None))
        if mode == "values":
            total = sum(
                float(signal.to_arrow().to_numpy(zero_copy_only=False).sum())
                for record in records
                for signal in record.signals
            )
    elapsed = time.perf_counter() - start
    if len(records) != (len(selected) if mode in {"selected", "values"} else expected):
        raise TimeFValidationError("benchmark read returned an incorrect record count")
    if records[0].signals[0].time_axis is not records[0].signals[1].time_axis:
        raise TimeFValidationError("benchmark read did not preserve shared axes")
    if mode == "dataset" and (len(dataset.tasks) != expected or dataset.tasks[0].inputs[0] is not records[0]):
        raise TimeFValidationError("benchmark read did not preserve task counts or record references")
    if mode == "dataset":
        dataset.check_records()
    if mode == "values" and total != len(selected) * 4 * sum(range(64)):
        raise TimeFValidationError("benchmark read returned incorrect signal values")
    return elapsed


def run(root: Path, repetitions: int) -> dict[str, Any]:
    """Run warm-ups and report every timing sample and its median.

    Returns:
        Workload description, machine details, and timing results.
    """
    workload = json.loads((root / "workload.json").read_text())
    results = []
    for name, dataset_id in DATASETS.items():
        with LocalRegistry(root).open_reader(dataset_id) as reader:
            selected = reader.record_ids()[:32]
        for mode in MODES:
            measure(root, dataset_id, mode, selected, workload["records"])
            samples = [measure(root, dataset_id, mode, selected, workload["records"]) for _ in range(repetitions)]
            result = {"dataset": name, "mode": mode, "median_seconds": statistics.median(samples), "samples": samples}
            results.append(result)
            print(f"{name:14} {mode:8} {result['median_seconds']:.4f} s", flush=True)
    return {
        "workload": workload,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "results": results,
    }


def main() -> None:
    """Prepare or measure the benchmark corpus."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--records", type=int, default=1_000)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--profile", type=Path)
    args = parser.parse_args()
    if args.prepare:
        prepare(args.root, args.records)
        return
    if args.profile is not None:
        cProfile.runctx("run(args.root, 1)", globals(), locals(), str(args.profile))
        return
    result = run(args.root, args.repetitions)
    if args.output is not None:
        args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
