"""Compare selected windows and complete training batches on main and inheritance."""

import argparse
import hashlib
import json
from pathlib import Path
import random
from time import perf_counter

from timenet.reader import TimeFReader
from timenet.registry import DatasetVersion
from timenet.torch import TimeFTorchDataset


def benchmark(root: Path) -> dict[str, object]:
    """Measure identical selections and fingerprint their materialized values.

    Returns:
        Timings and hashes for windows and training batches.
    """
    digest = hashlib.sha256()
    with TimeFReader(DatasetVersion.open_local(root)) as reader:
        start = perf_counter()
        ids = reader.record_ids()
        positions = random.Random(42).sample(range(len(ids)), min(64, len(ids)))  # noqa: S311 - reproducible samples
        for position in positions:
            record = next(reader.iter_records([ids[position]]))
            values = record.signals[0].read_steps(100, 600)
            digest.update(values.to_numpy().tobytes())
        window_seconds = perf_counter() - start
    training_digest = hashlib.sha256()
    with TimeFReader(DatasetVersion.open_local(root)) as reader:
        start = perf_counter()
        dataset = TimeFTorchDataset.from_reader(reader)
        items = []
        for offset in range(0, len(positions), 16):
            items.extend(dataset.__getitems__(positions[offset : offset + 16]))
        training_seconds = perf_counter() - start
        for item in items:
            for values in item["series"]:
                training_digest.update(values.numpy().tobytes())
            for task in item["tasks"]:
                training_digest.update(json.dumps([task.prompt, task.targets, task.rationale], default=str).encode())
    return {
        "records": len(positions),
        "window_seconds": window_seconds,
        "training_seconds": training_seconds,
        "window_sha256": digest.hexdigest(),
        "training_sha256": training_digest.hexdigest(),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    print(json.dumps(benchmark(parser.parse_args().root), indent=2))
