"""Build full ECG-QA benchmark variants without adding connectors to the core stack.

Export the three connector modules and cards with Sapling, as described in
``read_prepared.md``. Run each build in a fresh process against an empty output.
"""

import argparse
from contextlib import ExitStack
import importlib.util
import json
from pathlib import Path
import platform
import resource
import sys
from time import perf_counter
from typing import Any, cast

from timenet.connectors import BuildContext
from timenet.errors import TimeFValidationError
from timenet.reader import TimeFReader
from timenet.registry import DatasetVersion
from timenet.types import Payload
from timenet.writer import TimeFWriter


def main() -> None:  # noqa: PLR0914, PLR0915 - explicit build stages and separate metrics
    """Build one full ECG variant from cached sources and exported connector modules.

    Raises:
        TimeFValidationError: If an exported connector cannot be loaded or a child reads or writes inherited waveforms.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("variant", choices=["parent", "before", "after"])
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--connectors", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--group-bytes", type=int, default=4 * 2**20)
    parser.add_argument("--layout", choices=["packed", "record", "auto"], default="packed")
    parser.add_argument("--encoding", default="auto")
    parser.add_argument("--compression", default="zstd")
    parser.add_argument("--level", type=int)
    args = parser.parse_args()
    modules = {}
    for name in ("ptb_xl", "before", "child"):
        path = args.connectors / name / "connector.py"
        spec = importlib.util.spec_from_file_location("bench_" + name, path)
        if spec is None or spec.loader is None:
            raise TimeFValidationError(f"cannot load connector module {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        modules[name] = module
    cache = args.cache
    ptb = cache / "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
    root = args.work / ("before" if args.variant == "before" else "after")
    reads = 0
    original_read = Payload.read

    def tracked(payload):
        """Count actual inherited materializations while preserving the real read.

        Returns:
            The decoded payload content.
        """
        nonlocal reads
        if payload.reference is not None:
            reads += 1
        return original_read(payload)

    with ExitStack() as stack:
        cast("Any", Payload).read = tracked
        parent = None
        stack.callback(setattr, Payload, "read", original_read)
        started = perf_counter()
        if args.variant == "parent":
            module = modules["ptb_xl"]
            dataset = module.PtbXlConnector().convert(
                [module.PtbXlSource(root=ptb, database_csv=ptb / "ptbxl_database.csv")]
            )
        else:
            module = modules["before" if args.variant == "before" else "child"]
            kwargs = {
                "answers_path": cache / "answers_for_each_template.csv",
                "cot_csvs": tuple(
                    (split, cache / "ecg_qa_cot" / name) for split, name in module.EcgQaCotConnector._COT_CSVS
                ),
            }
            if args.variant == "before":
                kwargs["records_root"] = ptb / "records500"
                context = None
            else:
                parent = stack.enter_context(TimeFReader(DatasetVersion.open_local(root / "physionet/ptb-xl/1.0.0")))
                context = BuildContext({"ptbxl": parent})
            source = module.EcgQaCotSource(**kwargs)
            dataset = module.EcgQaCotConnector().convert([source], **({"context": context} if context else {}))
        convert = perf_counter() - started
        print("Converted", len(dataset.records), "records in", convert, flush=True)
        started = perf_counter()
        dataset.derive_schema()
        schema = perf_counter() - started
        started = perf_counter()
        writer_options = {
            "row_group_target_bytes": args.group_bytes,
            "value_encoding": args.encoding,
            "compression": args.compression,
        }
        if args.level is not None:
            writer_options["compression_level"] = args.level
        writer_options["row_group_layout"] = args.layout
        with TimeFWriter(root, dataset, **writer_options) as writer:
            writer.write()
        write = perf_counter() - started
        path = root / dataset.metadata.dataset_id / str(dataset.metadata.dataset_version)
        metrics = {
            "variant": args.variant,
            "convert_seconds": convert,
            "schema_seconds": schema,
            "write_seconds": write,
            "stored_payload_reads": reads,
            "artifact_bytes": sum(file.stat().st_size for file in path.rglob("*") if file.is_file()),
            "catalog_bytes": (path / "control.duckdb").stat().st_size,
            "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            * (1 if sys.platform == "darwin" else 1024),
            "python": platform.python_version(),
            "layout": args.layout,
            "compression": args.compression,
            "encoding": args.encoding,
            "artifact_path": str(path),
        }
        if args.variant == "after" and (reads or DatasetVersion.open_local(path).manifest.files.time_series):
            raise TimeFValidationError("child construction read or wrote inherited waveform bodies")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(metrics, indent=2) + "\n")
        print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
