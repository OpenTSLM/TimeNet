"""Under the v0.1 environment: read a v0.1 version into memory, then time writing it back out."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from time import perf_counter
import warnings

from timenet.reader import TimeFReader
from timenet.registry.version import DatasetVersion
from timenet.writer import TimeFWriter


def main() -> None:
    """Load, write, and print one JSON object."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("version_dir")
    parser.add_argument("--out", help="Keep the written version under this directory instead of a temp dir.")
    arguments = parser.parse_args()
    warnings.simplefilter("ignore")
    started = perf_counter()
    dataset = TimeFReader(DatasetVersion.open_local(Path(arguments.version_dir))).read()
    dataset.derive_schema()
    load_seconds = perf_counter() - started

    def write(root: Path) -> float:
        started = perf_counter()
        with TimeFWriter(root, dataset) as writer:
            writer.write()
        return perf_counter() - started

    if arguments.out:
        write_seconds = write(Path(arguments.out))
    else:
        with TemporaryDirectory() as directory:
            write_seconds = write(Path(directory))
    json.dump(
        {
            "load_seconds": load_seconds,
            "write_seconds": write_seconds,
            "records": len(dataset.records),
            "tasks": len(dataset.tasks),
        },
        sys.stdout,
    )
    print()


if __name__ == "__main__":
    main()
