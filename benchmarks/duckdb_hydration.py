"""Compare batched DuckDB Record hydration with repeated one-Record reads."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass
import json
from pathlib import Path
from statistics import median
from tempfile import TemporaryDirectory
from time import perf_counter

from timenet.dataset import OrdinalAxis, Record, Signal, Source, TimeFDataset
from timenet.format.control_reader import DuckDBControlReader
from timenet.format.control_writer import DuckDBControlWriter
from timenet.types import Annotation, DatasetMetadata, License, TimeSeriesSpec, Version, ureg


_SPEC = TimeSeriesSpec(
    spec_type="benchmark_value",
    name="Benchmark value",
    unit_value=ureg.dimensionless,
)


@dataclass(frozen=True)
class HydrationBenchmark:
    """Median timings for the two public read patterns."""

    records: int
    selected: int
    repeats: int
    batch_seconds: float
    per_record_seconds: float

    @property
    def speedup(self) -> float:
        """Return how many times faster the batched read is.

        Returns:
            The per-Record time divided by the batched time.
        """
        return self.per_record_seconds / self.batch_seconds

    def to_dict(self) -> dict[str, int | float]:
        """Return a JSON-compatible result.

        Returns:
            The benchmark configuration, timings, and speedup.
        """
        return {
            "records": self.records,
            "selected": self.selected,
            "repeats": self.repeats,
            "batch_seconds": self.batch_seconds,
            "per_record_seconds": self.per_record_seconds,
            "batch_speedup": self.speedup,
        }


def _dataset(size: int) -> TimeFDataset:
    """Build a metadata-heavy dataset without materializing a values plane.

    Returns:
        The benchmark dataset.
    """
    dataset = TimeFDataset(
        metadata=DatasetMetadata(
            dataset_id="timenet/duckdb-hydration-benchmark",
            dataset_version=Version(1, 0, 0),
            name="DuckDB hydration benchmark",
            description="Synthetic hierarchy for Record selection measurements.",
            license=License.MIT,
        )
    )
    for index in range(size):
        signal = Signal.from_values(
            [float(index)],
            spec=_SPEC,
            signal="value",
            time_axis=OrdinalAxis(axis_id=f"axis-{index:08d}"),
            time_series_id=f"signal-{index:08d}",
        )
        record = Record(
            record_id=f"record-{index:08d}",
            sources=(
                Source(
                    id=f"source-{index:08d}",
                    name="Synthetic sensor",
                    signals=(signal,),
                ),
            ),
        )
        record.annotate(Annotation(key="partition", value=index % 16))
        dataset.add_record(record=record)
    return dataset


def _measure(operation: Callable[[], object], repeats: int) -> float:
    """Measure one zero-argument callable and return its median duration.

    Returns:
        Median wall-clock seconds.
    """
    samples: list[float] = []
    for _ in range(repeats):
        started = perf_counter()
        operation()
        samples.append(perf_counter() - started)
    return median(samples)


def run_benchmark(*, records: int, selected: int, repeats: int, root: Path) -> HydrationBenchmark:
    """Write a corpus and compare one batched selection with repeated single reads.

    Returns:
        Median timings and their ratio.

    Raises:
        ValueError: If the benchmark dimensions are invalid.
    """
    if records <= 0 or selected <= 0 or selected > records or repeats <= 0:
        raise ValueError("records and repeats must be positive, and selected must be within records")
    path = root / "control.duckdb"
    DuckDBControlWriter(path).write_hierarchy(_dataset(records))
    step = max(records // selected, 1)
    record_ids = tuple(f"record-{index:08d}" for index in range(0, records, step))[:selected]
    with DuckDBControlReader(path) as reader:
        reader.read_records(record_ids, with_annotations=True)
        batch = _measure(lambda: reader.read_records(record_ids, with_annotations=True), repeats)
        per_record = _measure(
            lambda: tuple(reader.read_records((record_id,), with_annotations=True)[0] for record_id in record_ids),
            repeats,
        )
    return HydrationBenchmark(records, selected, repeats, batch, per_record)


def main() -> None:
    """Run the benchmark and print one JSON result."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=int, default=10_000)
    parser.add_argument("--selected", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args()
    with TemporaryDirectory(prefix="timenet-duckdb-hydration-") as temporary:
        result = run_benchmark(
            records=args.records,
            selected=args.selected,
            repeats=args.repeats,
            root=Path(temporary),
        )
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
