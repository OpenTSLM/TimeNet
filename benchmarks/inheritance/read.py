"""Measure equivalent full reads in a fresh process on main or the inheritance stack.

Run with ``uv run benchmarks/inheritance/read.py /path/to/dataset/version``.
Retain all arrays to compare eager reads against main's lazy signal readers fairly.
"""

import argparse
import hashlib
import json
from pathlib import Path
import resource
from time import perf_counter

from timenet.reader import TimeFReader
from timenet.registry.version import DatasetVersion


def benchmark(root: Path) -> dict[str, object]:
    """Read all catalog objects and payloads, returning timings and content fingerprints.

    Returns:
        Timings, counts, memory use, and content hashes.
    """
    start = perf_counter()
    reader = TimeFReader(DatasetVersion.open_local(root))
    dataset = reader.read()
    catalog_done = perf_counter()
    arrays = [signal.to_arrow() for record in dataset.records for signal in record.signals]
    materialized = perf_counter()
    digest = hashlib.sha256()
    for array in arrays:
        digest.update(str(array.type).encode())
        for buffer in array.buffers():
            if buffer is not None:
                digest.update(buffer)
    text_digest = hashlib.sha256()
    for task in dataset.tasks:
        text_digest.update(json.dumps([task.prompt, task.targets, task.rationale], default=str).encode())
    result = {
        "read_seconds": catalog_done - start,
        "materialization_seconds": materialized - catalog_done,
        "full_read_seconds": materialized - start,
        "records": len(dataset.records),
        "signals": len(arrays),
        "values": sum(map(len, arrays)),
        "tasks": len(dataset.tasks),
        "payload_sha256": digest.hexdigest(),
        "text_sha256": text_digest.hexdigest(),
        "peak_rss_platform_units": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }
    reader.close()
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    print(json.dumps(benchmark(parser.parse_args().root), indent=2))
