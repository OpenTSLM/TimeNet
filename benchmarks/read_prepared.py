"""Measure consumer reads and complete scalar-numeric dataset fidelity.

Run each read in a fresh process after all writers and tests finish. Compare snapshots
with ``compare``. See ``read_prepared.md`` for full ECG-QA measurements.
The output contains timings, counts, and fingerprints, without dataset bodies.
"""

import argparse
from dataclasses import fields, is_dataclass
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import platform
import resource
import sys
from time import perf_counter
from typing import Any

from pydantic import BaseModel

from timenet.client import TimeNet
from timenet.dataset import Signal
from timenet.errors import TimeFValidationError
from timenet.registry import DatasetVersion
from timenet.types import Annotation, Payload, Task


_SKIP = {"origin", "content_origin", "occurrence_origin", "axis_id", "occurrence_id", "loader", "time_offsets_loader"}


def _environment() -> dict[str, Any]:
    return {"python": platform.python_version(), "platform": platform.platform(), "machine": platform.machine()}


def _rss_bytes() -> int:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(rss if sys.platform == "darwin" else rss * 1024)


def _artifact(path: Path) -> dict[str, Any]:
    manifest = DatasetVersion.open_local(path).manifest
    return {
        "artifact_path": str(path),
        "artifact_bytes": sum(file.stat().st_size for file in path.rglob("*") if file.is_file()),
        "catalog_bytes": sum(part.size for part in manifest.files.control),
        "owned_signal_bytes": sum(part.size for part in manifest.files.time_series),
        "owned_text_bytes": sum(part.size for part in manifest.files.payloads),
        "physical_owners": [owner.model_dump(mode="json") for owner in manifest.payload_sources],
        "counts": manifest.counts.model_dump(mode="json"),
    }


def _canonical(value: Any) -> Any:  # noqa: PLR0911 - explicit type dispatch keeps each representation visible
    if isinstance(value, Payload):
        return _canonical(value.read())
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, Fraction):
        return [value.numerator, value.denominator]
    if isinstance(value, Task):
        # Tasks reference records. Do not expand a recording for every question.
        return {
            "class": type(value).__name__,
            "inputs": [record.id for record in value.inputs],
            "prompt": value.prompt(),
            "rationale": value.rationale(),
            "targets": _canonical(value.targets()),
            **{
                item.name: _canonical(getattr(value, item.name))
                for item in fields(value)
                if item.name not in _SKIP | {"inputs", "_prompt", "_rationale", "_targets"}
            },
        }
    if is_dataclass(value) and not isinstance(value, type):
        skipped = _SKIP | ({"values", "time_offsets"} if isinstance(value, Signal) else set())
        if isinstance(value, Annotation):
            skipped |= {"_value"}
        return {
            "class": type(value).__name__,
            **{item.name: _canonical(getattr(value, item.name)) for item in fields(value) if item.name not in skipped},
            **({"value": value.value()} if isinstance(value, Annotation) else {}),
        }
    if isinstance(value, dict):
        return {key: _canonical(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_canonical(item) for item in value]
    if isinstance(value, set | frozenset):
        return sorted(value)
    return value


def _json_bytes(value: Any) -> bytes:
    return json.dumps(_canonical(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def read(root: Path, reference: str) -> dict[str, Any]:
    """Load through the consumer SDK and fingerprint all logical content.

    Returns:
        Load and scan timings, complete content fingerprints, and entity counts.

    Raises:
        TimeFValidationError: If a Signal is not scalar numeric data.
    """
    started = perf_counter()
    dataset = TimeNet(registry=root, storage_path=root).load(reference, auto_build=False)
    load_seconds = perf_counter() - started
    print(f"Loaded {len(dataset.records)} records and {len(dataset.tasks)} tasks in {load_seconds:.3f}s.", flush=True)
    catalog = hashlib.sha256()
    waveforms = hashlib.sha256()
    tasks = hashlib.sha256()
    started = perf_counter()
    axis_aliases: dict[str, int] = {}
    for record in dataset.records:
        catalog.update(_json_bytes(record) + b"\n")
        for signal in record.signals:
            group = axis_aliases.setdefault(signal.time_axis.axis_id, len(axis_aliases))
            catalog.update(_json_bytes([signal.id, group]) + b"\n")
    catalog.update(_json_bytes(dataset.schema))
    catalog.update(_json_bytes(dataset.annotations))
    catalog_seconds = perf_counter() - started
    started = perf_counter()
    counts = {"samples": 0, "signals": 0, "null_samples": 0}
    for record in dataset.records:
        for signal in record.signals:
            if signal.spec.value_shape or signal.spec.dtype in {"str", "enum"}:
                raise TimeFValidationError("this read benchmark requires scalar numeric Signals")
            array = signal.to_arrow()
            waveforms.update(_json_bytes([signal.id, str(array.type), len(array), array.null_count]) + b"\n")
            waveforms.update(array.to_numpy(zero_copy_only=False).tobytes())
            waveforms.update(array.is_null().to_numpy(zero_copy_only=False).tobytes())
            counts["samples"] += len(array)
            counts["signals"] += 1
            counts["null_samples"] += array.null_count
    waveform_seconds = perf_counter() - started
    print(f"Read and hashed {counts['samples']} samples in {waveform_seconds:.3f}s.", flush=True)
    started = perf_counter()
    for task in dataset.tasks:
        tasks.update(_json_bytes(task) + b"\n")
    task_seconds = perf_counter() - started
    return {
        "dataset": reference,
        "environment": _environment(),
        "load_seconds": load_seconds,
        "catalog_fingerprint_seconds": catalog_seconds,
        "waveform_read_and_hash_seconds": waveform_seconds,
        "task_read_and_hash_seconds": task_seconds,
        "records": len(dataset.records),
        **counts,
        "tasks": len(dataset.tasks),
        "axis_groups": len(axis_aliases),
        "catalog_sha256": catalog.hexdigest(),
        "waveforms_sha256": waveforms.hexdigest(),
        "tasks_sha256": tasks.hexdigest(),
        "release_metadata": dataset.metadata.model_dump(mode="json"),
        "peak_rss_bytes": _rss_bytes(),
    }


def compare(before: Path, after: Path) -> dict[str, Any]:
    """Reject any difference in the complete consumer fingerprints or counts.

    Returns:
        The matching fingerprints and counts.

    Raises:
        TimeFValidationError: If a fingerprint or count differs.
    """
    old, new = (json.loads(path.read_text()) for path in (before, after))
    for result in (old, new):
        result["descriptive_metadata"] = {
            key: value for key, value in result["release_metadata"].items() if key not in {"dataset_version", "inputs"}
        }
    keys = (
        "descriptive_metadata",
        "records",
        "signals",
        "samples",
        "null_samples",
        "tasks",
        "axis_groups",
        "catalog_sha256",
        "waveforms_sha256",
        "tasks_sha256",
    )
    differences = {key: [old[key], new[key]] for key in keys if old[key] != new[key]}
    if differences:
        raise TimeFValidationError(f"ECG-QA consumer content differs: {differences}")
    return {"fidelity": "equal", **{key: old[key] for key in keys}}


def main() -> None:
    """Measure one consumer read or compare two complete fingerprints."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("read", "compare"))
    parser.add_argument("--registry", type=Path)
    parser.add_argument("--dataset")
    parser.add_argument("--before", type=Path)
    parser.add_argument("--after", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.phase == "read":
        if args.registry is None or args.dataset is None:
            parser.error("read requires --registry and --dataset")
        result = read(args.registry, args.dataset)
    else:
        if args.before is None or args.after is None:
            parser.error("compare requires --before and --after")
        result = compare(args.before, args.after)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
