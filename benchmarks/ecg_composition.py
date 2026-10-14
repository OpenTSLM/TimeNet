"""Compare standalone ECG-QA with a child that shares PTB-XL payloads.

Use the cached public source files. Build each variant in a fresh process, then
read each artifact in a fresh process. See ``ecg_composition.md`` for commands.
This benchmark is separate from CI and contains no dataset content.
"""

import argparse
from contextlib import ExitStack
from dataclasses import fields, is_dataclass
from fractions import Fraction
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import resource
import shutil
import subprocess  # noqa: S404 - Sapling runs with an argument vector
import sys
import tempfile
from time import perf_counter
from types import ModuleType
from typing import Any, cast

from pydantic import BaseModel

from timenet.client import TimeNet
from timenet.connectors import BuildContext
from timenet.dataset import Signal
from timenet.engine import store_dataset
from timenet.errors import TimeFValidationError
from timenet.reader import TimeFReader
from timenet.registry import DatasetVersion
from timenet.types import Annotation, Payload, Task
from timenet_connectors.datasets.physionet.ecg_qa_cot.connector import EcgQaCotConnector, EcgQaCotSource
from timenet_connectors.datasets.physionet.ptb_xl.connector import PtbXlConnector, PtbXlSource


_CONNECTOR = "packages/timenet-connectors/src/timenet_connectors/datasets/physionet/ecg_qa_cot"
_PTB_ROOT = "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
_SKIP = {"origin", "content_origin", "occurrence_origin", "axis_id", "occurrence_id", "loader", "time_offsets_loader"}


def _baseline(revision: str, directory: Path) -> ModuleType:
    sapling = shutil.which("sl")
    if sapling is None:
        raise TimeFValidationError("the baseline build requires Sapling on PATH")
    for name in ("connector.py", "dataset.yaml"):
        result = subprocess.run(  # noqa: S603 - no shell and no mutations
            [sapling, "cat", "-r", revision, f"{_CONNECTOR}/{name}"], capture_output=True, check=True
        )
        (directory / name).write_bytes(result.stdout)
    spec = importlib.util.spec_from_file_location("ecg_standalone_baseline", directory / "connector.py")
    if spec is None or spec.loader is None:
        raise TimeFValidationError("cannot load the standalone ECG-QA connector")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


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


def build(variant: str, cache: Path, work: Path, baseline_revision: str) -> dict[str, Any]:
    """Convert and write a full variant from cached raw files.

    Returns:
        Stage timings, owned bytes, and inherited payload read counts.

    Raises:
        TimeFValidationError: If a child reads or writes inherited waveforms.
    """
    root = work / ("before" if variant == "before" else "after")
    ptb = cache / _PTB_ROOT
    cot_csvs = tuple((split, cache / "ecg_qa_cot" / name) for split, name in EcgQaCotConnector._COT_CSVS)
    stored_reads = 0
    original_read = Payload.read

    def read(payload):
        nonlocal stored_reads
        if payload.reference is not None:
            stored_reads += 1
        return original_read(payload)

    with ExitStack() as stack:
        parent = (
            stack.enter_context(TimeFReader(DatasetVersion.open_local(root / "physionet/ptb-xl/1.0.0")))
            if variant == "after"
            else None
        )
        baseline = (
            _baseline(baseline_revision, Path(stack.enter_context(tempfile.TemporaryDirectory())))
            if variant == "before"
            else None
        )
        cast("Any", Payload).read = read
        try:
            started = perf_counter()
            if baseline is not None:
                source = baseline.EcgQaCotSource(
                    records_root=ptb / "records500",
                    answers_path=cache / "answers_for_each_template.csv",
                    cot_csvs=cot_csvs,
                )
                dataset = baseline.EcgQaCotConnector().convert([source])
            elif parent is not None:
                source = EcgQaCotSource(answers_path=cache / "answers_for_each_template.csv", cot_csvs=cot_csvs)
                dataset = EcgQaCotConnector().convert([source], context=BuildContext({"ptbxl": parent}))
            else:
                dataset = PtbXlConnector().convert([PtbXlSource(root=ptb, database_csv=ptb / "ptbxl_database.csv")])
            convert_seconds = perf_counter() - started
            print(f"Converted {len(dataset.records)} records in {convert_seconds:.3f}s.", flush=True)
            started = perf_counter()
            dataset.derive_schema()
            schema_seconds = perf_counter() - started
            started = perf_counter()
            path = store_dataset(dataset, root)
            write_seconds = perf_counter() - started
        finally:
            cast("Any", Payload).read = original_read
    artifact = _artifact(path)
    if variant == "after" and (stored_reads or artifact["owned_signal_bytes"]):
        raise TimeFValidationError("the ECG-QA child read or wrote inherited waveform payloads")
    return {
        "variant": variant,
        "environment": _environment(),
        "baseline_revision": baseline_revision if variant == "before" else None,
        "convert_seconds": convert_seconds,
        "schema_seconds": schema_seconds,
        "write_seconds": write_seconds,
        "stored_payload_reads": stored_reads,
        "peak_rss_bytes": _rss_bytes(),
        **artifact,
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


def read(variant: str, work: Path) -> dict[str, Any]:
    """Load through the consumer SDK and fingerprint all logical content.

    Returns:
        Load and scan timings, complete content fingerprints, and entity counts.
    """
    root = work / ("before" if variant == "before" else "after")
    version = "1.0.0" if variant == "before" else "1.1.0"
    started = perf_counter()
    dataset = TimeNet(registry=root, storage_path=root).load(f"physionet/ecg-qa-cot@{version}", auto_build=False)
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
        "variant": variant,
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
    """Run one phase and save measurements without dataset bytes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("build", "read", "compare"))
    parser.add_argument("--variant", choices=("before", "parent", "after"), default="before")
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--baseline-revision", default="34e2b55f0d62a610be8176bb7d3e3f4b9e289f7f")
    parser.add_argument("--before", type=Path)
    parser.add_argument("--after", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.phase == "build":
        if args.cache is None:
            parser.error("build requires --cache")
        result = build(args.variant, args.cache, args.work, args.baseline_revision)
    elif args.phase == "read":
        if args.variant == "parent":
            parser.error("read compares the two ECG-QA variants")
        result = read(args.variant, args.work)
    else:
        if args.before is None or args.after is None:
            parser.error("compare requires --before and --after")
        result = compare(args.before, args.after)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
