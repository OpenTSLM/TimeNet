"""Write a run to JSON, and print it plainly."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import TextIO

from evaluations.errors import EvaluationError
from evaluations.harness import Operation
from evaluations.result import EvaluationResult


def write_json(result: EvaluationResult, out_dir: Path) -> Path:
    """Write the result as JSON under its own run directory.

    Each run writes to its own directory, so runs accumulate rather than overwrite. Two runs of
    the same source do not produce identical files: ``run_id``, ``started_at`` and the timings
    themselves all differ by nature. What is stable is the shape, so a reader can diff two runs
    field by field and see only what actually changed.

    Args:
        result: The run to write.
        out_dir: Parent directory. The run directory is created beneath it.

    Returns:
        The path written.
    """
    run_dir = out_dir / result.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "result.json"
    path.write_text(result.model_dump_json(indent=2) + "\n")

    return path


def print_summary(result: EvaluationResult, file: TextIO | None = None) -> None:
    """Print one line per format.

    Deliberately plain: no tables, no colours, no formatting library. The JSON is the record;
    this is for reading at the end of a run.

    Args:
        result: The run to summarize.
        file: Where to write. Defaults to standard output.

    Raises:
        EvaluationError: If the measurements do not cover every artifact at both operations.
            A summary that silently omits one is a partial report presented as complete.
    """
    stream = file if file is not None else sys.stdout
    elapsed = {(one.format, one.operation): one.elapsed_ns for one in result.measurements}
    # Every format that was written must carry a write and a read, or a line below would report
    # a number it does not have.
    wanted = {(artifact.format, operation) for artifact in result.artifacts for operation in Operation}

    if elapsed.keys() != wanted:
        raise EvaluationError(
            f"the measurements do not match the artifacts: measured {sorted(elapsed)}, expected {sorted(wanted)}"
        )

    print(
        f"run {result.run_id}  {result.environment.platform}  python {result.environment.python}",
        file=stream,
    )

    for artifact in result.artifacts:
        write_s = elapsed[artifact.format, Operation.WRITE] / 1e9
        read_s = elapsed[artifact.format, Operation.READ] / 1e9
        megabytes = artifact.size_bytes / 1e6
        print(
            f"{artifact.format:<8} write {write_s:7.3f} s   read {read_s:7.3f} s   size {megabytes:9.1f} MB",
            file=stream,
        )
