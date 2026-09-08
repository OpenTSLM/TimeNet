"""Measure a local read without downloading, fingerprinting, or serializing the dataset.

Run each measurement in a fresh process, for example::

    uv run python -m benchmarks.read_load /path/to/dataset/1.0.0 --mode full

``full`` constructs records, annotations, and tasks but leaves signal values lazy. ``records``
streams records, and ``values`` also materializes every signal. These two streaming modes accept
``--without-annotations``. Peak RSS includes the interpreter and imported libraries.
"""

import argparse
import gc
import json
from pathlib import Path
import resource
import sys
import time

from timenet.reader import TimeFReader
from timenet.registry import DatasetVersion


def _read(path: Path, mode: str, *, with_annotations: bool) -> dict[str, int]:
    with TimeFReader(DatasetVersion.open_local(path)) as reader:
        if mode == "full":
            dataset = reader.read()
            return {"records": len(dataset.records), "tasks": len(dataset.tasks)}
        records = series = value_bytes = 0
        for record in reader.iter_records(with_annotations=with_annotations):
            records += 1
            for ts in record.time_series:
                series += 1
                if mode == "values":
                    value_bytes += ts.to_arrow().nbytes
        return {"records": records, "series": series, "value_bytes": value_bytes}


def main() -> None:
    """Print one read's wall time, CPU time, peak memory, and consumed item counts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--mode", choices=("full", "records", "values"), default="full")
    parser.add_argument("--without-annotations", action="store_true")
    args = parser.parse_args()
    if args.mode == "full" and args.without_annotations:
        parser.error("full loading includes annotations; use --mode records or --mode values to skip them")
    gc.collect()
    wall, cpu = time.perf_counter(), time.process_time()
    counts = _read(args.path, args.mode, with_annotations=not args.without_annotations)
    wall_s, cpu_s = time.perf_counter() - wall, time.process_time() - cpu
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(
        json.dumps(
            {
                "mode": args.mode,
                "with_annotations": not args.without_annotations,
                "wall_s": wall_s,
                "cpu_s": cpu_s,
                "peak_rss_bytes": peak_rss if sys.platform == "darwin" else peak_rss * 1024,
                **counts,
            }
        )
    )


if __name__ == "__main__":
    main()
