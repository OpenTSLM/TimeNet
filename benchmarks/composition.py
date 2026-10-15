"""Measure catalog copies and shared payloads across three dataset generations.

Run with ``uv run benchmarks/composition.py``. Timings are observations, not CI thresholds.
"""

import argparse
from collections import Counter
from collections.abc import Callable
from contextlib import ExitStack
import json
from pathlib import Path
import tempfile
import time
from typing import Any, cast

import numpy as np

from timenet.dataset import Record, RegularAxis, Signal, Source, TimeFDataset
from timenet.errors import TimeFValidationError
from timenet.reader import TimeFReader
from timenet.registry.version import DatasetVersion
from timenet.testing import make_dataset
from timenet.types import Annotation, AnswerTask, DatasetRef, Payload


_INLINE_BYTES = 4096


class CompositionBenchmark:
    """Track body reads and owned bytes for comparable catalog generations."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.reads: Counter[str] = Counter()

    def _measure(self, profile: str, generation: str, write: Callable[[], Path]) -> dict[str, Any]:
        self.reads.clear()
        started = time.perf_counter()
        path = write()
        elapsed = time.perf_counter() - started
        manifest = DatasetVersion.open_local(path).manifest
        stored = sum(count for name, count in self.reads.items() if name.startswith("stored_"))
        owned = sum(part.size for part in (*manifest.files.time_series, *manifest.files.payloads))
        if generation != "parent" and (stored or owned):
            raise TimeFValidationError(
                f"{profile}/{generation} read {stored} inherited bodies and wrote {owned} payload bytes"
            )
        return {
            "profile": profile,
            "generation": generation,
            "write_seconds": round(elapsed, 6),
            "catalog_bytes": sum(part.size for part in manifest.files.control),
            "owned_signal_bytes": sum(part.size for part in manifest.files.time_series),
            "owned_payload_bytes": sum(part.size for part in manifest.files.payloads),
            "physical_owners": len(manifest.payload_sources),
            "records": manifest.counts.records,
            "tasks": sum(manifest.counts.tasks.values()),
            "payload_reads": dict(self.reads),
            "stored_payload_reads": stored,
        }

    def run(self, *, n_values: int, n_tasks: int, text_bytes: int) -> list[dict[str, Any]]:
        """Write representative parents, children, and grandchildren.

        Returns:
            Measurements by profile and generation.
        """
        original = Payload.read

        def read(payload):
            descriptor = payload.descriptor
            kind = "offsets" if descriptor.first_us is not None else descriptor.kind
            storage = "stored" if payload.reference is not None else "native"
            self.reads[f"{storage}_{kind}"] += 1
            return original(payload)

        cast("Any", Payload).read = read
        results = []
        try:
            for profile in ("regular", "irregular", "annotations", "tasks"):
                parent = make_dataset()
                parent._metadata = parent.metadata.model_copy(update={"dataset_id": f"bench/{profile}-parent"})
                spec = parent.records[0].signals[0].spec
                values = np.sin(np.arange(n_values, dtype=np.float32) / 32)
                signal = (
                    Signal.from_irregular(
                        values, time_offsets_us=np.arange(n_values, dtype=np.int64) * 17, spec=spec, name="large"
                    )
                    if profile == "irregular"
                    else Signal.from_values(values, spec=spec, name="large", time_axis=RegularAxis.from_rate_hz(100))
                )
                record = parent.add_record(
                    record=Record(record_id="large", sources=(Source(name="monitor", signals=(signal,)),))
                )
                if profile == "annotations":
                    body = "\n".join(
                        f"Work order {index}: replace sensor {index * 104729 % 999983}; inspect alarm {index % 37}."
                        for index in range(text_bytes // 45 + 1)
                    )[:text_bytes]
                    annotation = Annotation(id="manual", key="manual", value=body)
                    record.annotate(annotation)
                    parent.records[0].annotate(annotation)
                    parent.add_task(
                        task=AnswerTask(id="long-text", inputs=(record,), prompt=body, rationale=body, targets=(body,))
                    )
                if profile == "tasks":
                    parent.add_task_stream(
                        [AnswerTask],
                        lambda record=record: (
                            AnswerTask(
                                id=f"question-{index}",
                                inputs=(record,),
                                prompt=f"Question {index}",
                                targets=("normal",),
                            )
                            for index in range(n_tasks)
                        ),
                    )
                paths: list[Path] = []

                def write_parent(parent=parent, paths=paths):
                    path = parent.write(path=self.root)
                    paths.append(path)
                    return path

                results.append(self._measure(profile, "parent", write_parent))
                with ExitStack() as stack:
                    for generation in ("child", "grandchild"):
                        reader = stack.enter_context(TimeFReader(DatasetVersion.open_local(paths[-1])))
                        identity = DatasetRef(
                            dataset_id=reader.metadata.dataset_id, version=reader.metadata.dataset_version
                        )
                        child = TimeFDataset(
                            metadata=reader.metadata.model_copy(
                                update={"dataset_id": f"bench/{profile}-{generation}", "inputs": {"base": identity}}
                            )
                        )

                        def write_child(child=child, reader=reader, paths=paths):
                            imported = child.copy_from(reader, namespace="base", include_tasks=True)
                            child.add_task(
                                task=AnswerTask(
                                    inputs=(next(iter(imported.records.values())),),
                                    prompt="A new task",
                                    targets=("added",),
                                )
                            )
                            path = child.write(path=self.root)
                            paths.append(path)
                            return path

                        results.append(self._measure(profile, generation, write_child))
        finally:
            cast("Any", Payload).read = original
        return results


def main() -> None:
    """Print measurements as JSON without leaving dataset artifacts behind."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--values", type=int, default=100_000)
    parser.add_argument("--tasks", type=int, default=5_000)
    parser.add_argument("--text-bytes", type=int, default=1_000_000)
    args = parser.parse_args()
    if args.values < 1 or args.tasks < 1 or args.text_bytes <= _INLINE_BYTES:
        parser.error("values and tasks must be positive; text-bytes must exceed the inline threshold")
    with tempfile.TemporaryDirectory(prefix="timenet-composition-") as directory:
        rows = CompositionBenchmark(Path(directory)).run(
            n_values=args.values, n_tasks=args.tasks, text_bytes=args.text_bytes
        )
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
