"""Measure DuckDB control-plane hierarchy write time."""

from __future__ import annotations

import argparse
from fractions import Fraction
import json
from pathlib import Path
from statistics import median
from tempfile import TemporaryDirectory
from time import perf_counter

import pyarrow as pa

from timenet.dataset import Record, RegularAxis, Signal, Source, TimeFDataset
from timenet.format.control_writer import DuckDBControlWriter
from timenet.types import Annotation, DatasetMetadata, License, TimeSeriesSpec, Version, ureg


_SPEC = TimeSeriesSpec(
    spec_type="voltage",
    name="Voltage",
    unit_value=ureg.millivolt,
    dtype="float32",
)
_VALUES = pa.array([0.0], type=pa.float32())


def _dataset(*, records: int, signals_per_record: int) -> TimeFDataset:
    """Build a metadata-heavy hierarchy without a values plane.

    Returns:
        A dataset with one source, one axis, and one annotation for each record.
    """
    dataset = TimeFDataset(
        metadata=DatasetMetadata(
            dataset_id="timenet/control-write-benchmark",
            dataset_version=Version(1, 0, 0),
            name="Control write benchmark",
            description="Synthetic hierarchy for control-write measurements.",
            license=License.MIT,
        )
    )
    annotation = Annotation(key="cohort", value="A", id="cohort-a")
    for record_index in range(records):
        axis = RegularAxis(
            axis_id=f"axis-{record_index:08d}",
            period_us=Fraction(2_000),
        )
        signals = tuple(
            Signal(
                id=f"signal-{record_index:08d}-{signal_index:02d}",
                name=f"lead-{signal_index}",
                spec=_SPEC,
                time_axis=axis,
                data=_VALUES,
            )
            for signal_index in range(signals_per_record)
        )
        record = Record(
            record_id=f"record-{record_index:08d}",
            sources=(
                Source(
                    id=f"source-{record_index:08d}",
                    name="monitor",
                    signals=signals,
                ),
            ),
        )
        record.annotate(annotation)
        dataset.add_record(record=record)
    return dataset


def run_benchmark(
    *,
    records: int,
    signals_per_record: int,
    warmups: int,
    repeats: int,
) -> dict[str, int | float | list[float]]:
    """Measure complete control database writes after untimed warmups.

    Returns:
        The hierarchy dimensions, measured durations, and median duration.

    Raises:
        ValueError: If a benchmark dimension is invalid.
    """
    if records <= 0 or signals_per_record <= 0 or repeats <= 0 or warmups < 0:
        raise ValueError("records, signals_per_record, and repeats must be positive; warmups must not be negative")
    dataset = _dataset(records=records, signals_per_record=signals_per_record)
    durations: list[float] = []
    with TemporaryDirectory(prefix="timenet-control-write-") as directory:
        root = Path(directory)
        for iteration in range(warmups + repeats):
            started = perf_counter()
            DuckDBControlWriter(root / f"control-{iteration}.duckdb").write_hierarchy(dataset)
            duration = perf_counter() - started
            if iteration >= warmups:
                durations.append(duration)
    return {
        "records": records,
        "sources": records,
        "signals": records * signals_per_record,
        "axes": records,
        "annotations": records,
        "repeats": repeats,
        "seconds": durations,
        "median_seconds": median(durations),
    }


def main() -> None:
    """Run the benchmark and print one JSON result."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=int, default=2_000)
    parser.add_argument("--signals-per-record", type=int, default=12)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    result = run_benchmark(
        records=args.records,
        signals_per_record=args.signals_per_record,
        warmups=args.warmups,
        repeats=args.repeats,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
