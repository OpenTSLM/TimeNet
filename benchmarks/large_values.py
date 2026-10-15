"""Compare local packed-value storage against a separate checkout of main.

Run ``uv run python benchmarks/large_values.py --baseline /path/to/main``.
Each sample writes a fresh release, reads every value, and reads all public text tables.
Conversion and process startup are excluded. Runs alternate between revisions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
from statistics import median
import subprocess  # noqa: S404 - benchmark workers use fixed argument vectors
import sys
from tempfile import TemporaryDirectory
import time
from typing import Any

import numpy as np

from timenet.dataset import Record, Signal, Source, TimeFDataset
from timenet.dataset.axis import RegularAxis
from timenet.reader import TimeFReader
from timenet.registry import LocalRegistry
from timenet.types import Annotation, AnswerTask, DatasetMetadata, License, TimeSeriesSpec, Version, ureg
from timenet.writer import TimeFWriter


_DATASET_ID = "benchmark/large-values"
_CASES = ("small-text", "large-text", "irregular")
_METRICS = ("write_s", "catalog_read_s", "full_read_s", "tables_s", "catalog_bytes", "total_bytes")
_WORDS = [
    "patient",
    "recording",
    "signal",
    "normal",
    "abnormal",
    "rhythm",
    "observed",
    "episode",
    "stable",
    "baseline",
    "clinical",
    "sensor",
    "measurement",
]


def _dataset(case: str, records: int, steps: int) -> TimeFDataset:
    dataset = TimeFDataset(
        metadata=DatasetMetadata(
            dataset_id=_DATASET_ID,
            dataset_version=Version(1, 0, 0),
            name="Large value benchmark",
            description="Deterministic text and irregular timestamp workloads.",
            license=License.MIT,
        )
    )
    spec = TimeSeriesSpec(spec_type="sensor", name="Sensor", unit_value=ureg.dimensionless)
    regular = RegularAxis.from_rate_hz(100)
    for index in range(records):
        count = steps if case == "irregular" else 32
        values = np.sin(np.arange(count, dtype=np.float64) / 100 + index / 7).astype(np.float32)
        if case == "irregular":
            offsets = np.arange(count, dtype=np.int64) * 1_000 + np.arange(count) % 17 + index % 23
            signal = Signal.from_irregular(
                values, time_offsets_us=offsets, spec=spec, name="sensor", id=f"signal-{index}"
            )
        else:
            signal = Signal.from_values(values, spec=spec, name="sensor", time_axis=regular, id=f"signal-{index}")
        record = dataset.add_record(
            record=Record(
                record_id=f"record-{index:06d}",
                sources=(Source(id=f"source-{index}", name="sensor", signals=(signal,)),),
            )
        )
        word_count = 2_048 if case == "large-text" else 8
        text = f"Patient {index}. " + " ".join(np.random.default_rng(index).choice(_WORDS, size=word_count))
        record.annotate(Annotation(key="report", value=text, id=f"report-{index}"))
        dataset.add_task(
            task=AnswerTask(id=f"task-{index:06d}", inputs=(record,), prompt=text, rationale=text, targets=(text,))
        )
    dataset.derive_schema()
    return dataset


def _text(value: Any) -> str:
    return value if isinstance(value, str) else value.read()


def _read_all(root: Path) -> tuple[str, float, float]:
    digest = hashlib.sha256()
    started = time.perf_counter()
    version = LocalRegistry(root).open_version(_DATASET_ID, "1.0.0")
    with TimeFReader(version) as reader:
        dataset = reader.read()
        catalog_seconds = time.perf_counter() - started
        for task in dataset.tasks:
            for value in (task.prompt, task.rationale, *(task.targets or ())):
                digest.update(_text(value).encode())
        for record in dataset.records:
            for annotation in record.annotations:
                digest.update(_text(annotation.value).encode())
            for signal in record.signals:
                digest.update(signal.to_arrow().to_numpy(zero_copy_only=False).tobytes())
                if signal.time_offsets_loader is not None:
                    digest.update(signal.time_offsets_us().tobytes())
        full_seconds = time.perf_counter() - started
    return digest.hexdigest(), catalog_seconds, full_seconds


def _read_tables(root: Path) -> tuple[str, float]:
    digest = hashlib.sha256()
    started = time.perf_counter()
    with TimeFReader(LocalRegistry(root).open_version(_DATASET_ID, "1.0.0")) as reader:
        for table, columns in (
            (reader.task_table(), ("prompt", "rationale")),
            (reader.target_table(), ("text_value",)),
            (reader.annotation_table(), ("text_value",)),
        ):
            for column in columns:
                for value in table[column].to_pylist():
                    digest.update(value.encode())
    return digest.hexdigest(), time.perf_counter() - started


def _worker(case: str, records: int, steps: int) -> dict[str, Any]:
    dataset = _dataset(case, records, steps)
    with TemporaryDirectory(prefix="timenet-large-values-") as directory:
        root = Path(directory)
        started = time.perf_counter()
        with TimeFWriter(root, dataset) as writer:
            writer.write()
        write_seconds = time.perf_counter() - started
        version = root / _DATASET_ID / "1.0.0"
        digest, catalog_seconds, full_seconds = _read_all(root)
        tables_digest, tables_seconds = _read_tables(root)
        return {
            "write_s": write_seconds,
            "catalog_read_s": catalog_seconds,
            "full_read_s": full_seconds,
            "tables_s": tables_seconds,
            "catalog_bytes": (version / "control.duckdb").stat().st_size,
            "total_bytes": sum(path.stat().st_size for path in version.rglob("*") if path.is_file()),
            "digest": digest,
            "tables_digest": tables_digest,
        }


def _sample(checkout: Path, case: str, records: int, steps: int) -> dict[str, Any]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(checkout / "packages/timenet/src")
    completed = subprocess.run(  # noqa: S603 - no shell; benchmark arguments are explicit
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            case,
            "--records",
            str(records),
            "--steps",
            str(steps),
        ],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def _compare(args: argparse.Namespace) -> dict[str, Any]:
    checkouts = {"main": args.baseline.resolve(), "packed": Path(__file__).resolve().parents[1]}
    results: dict[str, Any] = {}
    for case in args.cases:
        samples: dict[str, list[dict[str, Any]]] = {label: [] for label in checkouts}
        fingerprints: set[tuple[str, str]] = set()
        for repetition in range(args.warmups + args.repeats):
            labels = list(checkouts) if repetition % 2 == 0 else list(reversed(checkouts))
            for label in labels:
                sample = _sample(checkouts[label], case, args.records, args.steps)
                fingerprints.add((sample["digest"], sample["tables_digest"]))
                if repetition >= args.warmups:
                    samples[label].append(sample)
        if len(fingerprints) != 1:
            raise RuntimeError(f"decoded values differ between revisions for {case}")
        summary = {
            label: {metric: median(run[metric] for run in runs) for metric in _METRICS}
            for label, runs in samples.items()
        }
        results[case] = {"medians": summary, "samples": samples}
        print(f"{case}: {json.dumps(summary, sort_keys=True)}", flush=True)
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "records": args.records,
        "irregular_steps_per_record": args.steps,
        "warmups": args.warmups,
        "repeats": args.repeats,
        "method": "Alternating fresh processes, warm filesystem cache; full reads materialize and hash every value.",
        "results": results,
    }


def main() -> None:
    """Measure local payload storage with matching decoded-value fingerprints."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--output", type=Path, default=Path("large-value-results.json"))
    parser.add_argument("--cases", nargs="+", choices=_CASES, default=list(_CASES))
    parser.add_argument("--records", type=int, default=512)
    parser.add_argument("--steps", type=int, default=4_096)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--worker", choices=_CASES)
    args = parser.parse_args()
    if args.records < 1 or args.steps < 1 or args.repeats < 1 or args.warmups < 0:
        parser.error("records, steps, and repeats must be positive; warmups must be nonnegative")
    if args.worker:
        print(json.dumps(_worker(args.worker, args.records, args.steps)))
    else:
        if args.baseline is None:
            parser.error("--baseline must name a checkout of main")
        args.output.write_text(json.dumps(_compare(args), indent=2) + "\n")


if __name__ == "__main__":
    main()
