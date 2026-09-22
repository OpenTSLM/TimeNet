"""Time a fixed read workload against one TimeF version directory. Runs under v0.1 and this stack.

Stages: open and hydrate every record and task (``read``), load the values of the first N signals
(``values``), and hydrate M single records by id one at a time (``single``).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter
import warnings

from timenet.reader import TimeFReader
from timenet.registry.version import DatasetVersion


def _signals(record: object) -> tuple:
    signals = getattr(record, "signals", None)
    if signals is None or callable(signals):
        return tuple(getattr(record, "time_series", ()))
    return tuple(signals)


def main() -> None:
    """Run the workload and print one JSON object."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("version_dir")
    parser.add_argument("--signals", type=int, default=2000)
    parser.add_argument("--records", type=int, default=200)
    arguments = parser.parse_args()
    warnings.simplefilter("ignore")
    result: dict[str, float | int] = {}

    started = perf_counter()
    reader = TimeFReader(DatasetVersion.open_local(Path(arguments.version_dir)))
    dataset = reader.read()
    result["read_seconds"] = perf_counter() - started
    result["records"] = len(dataset.records)
    result["tasks"] = len(dataset.tasks)

    started = perf_counter()
    loaded = 0
    values = 0
    for record in dataset.records:
        for signal in _signals(record):
            values += len(signal.to_arrow())
            loaded += 1
            if loaded >= arguments.signals:
                break
        if loaded >= arguments.signals:
            break
    result["values_seconds"] = perf_counter() - started
    result["values_signals"] = loaded
    result["values_points"] = values

    ids = [record.record_id for record in dataset.records]
    step = max(1, len(ids) // arguments.records)
    sample = ids[::step][: arguments.records]
    started = perf_counter()
    for record_id in sample:
        for _ in reader.iter_records([record_id]):
            pass
    result["single_seconds"] = perf_counter() - started
    result["single_records"] = len(sample)
    result["single_ms_per_record"] = 1000 * result["single_seconds"] / max(1, len(sample))

    close = getattr(reader, "close", None)
    if close is not None:
        close()
    json.dump(result, sys.stdout)
    print()


if __name__ == "__main__":
    main()
