"""A run must serialize with a stable shape and print one line per format."""

from datetime import UTC, datetime
import io
import json
from pathlib import Path

import pytest

from evaluations.errors import EvaluationError
from evaluations.formats.base import FormatName
from evaluations.harness import Artifact, Measurement, Operation
from evaluations.report import print_summary, write_json
from evaluations.result import Environment, EvaluationResult, collect_environment


def _result(run_id: str = "abc123", started: datetime | None = None) -> EvaluationResult:
    return EvaluationResult(
        run_id=run_id,
        started_at=started or datetime(2026, 8, 29, 12, 0, tzinfo=UTC),
        source_path=Path("/data/sleep-edfx"),
        environment=Environment(python="3.13.5", platform="darwin-arm64", cpu_count=8, packages={"torch": "2.7.1"}),
        artifacts=(
            Artifact(format=FormatName.PANDAS, path=Path("artifacts/a.parquet"), size_bytes=2_100_000),
            Artifact(format=FormatName.TIMEF, path=Path("artifacts/v"), size_bytes=1_830_000),
        ),
        measurements=(
            _write(FormatName.PANDAS, 8_400_000_000),
            _read(FormatName.PANDAS, 2_950_000_000),
            _write(FormatName.TIMEF, 9_100_000_000),
            _read(FormatName.TIMEF, 4_100_000_000),
        ),
    )


def _write(name: FormatName, elapsed_ns: int) -> Measurement:
    return Measurement(format=name, operation=Operation.WRITE, elapsed_ns=elapsed_ns, repetitions=1, warmups=0)


def _read(name: FormatName, elapsed_ns: int) -> Measurement:
    return Measurement(format=name, operation=Operation.READ, elapsed_ns=elapsed_ns, repetitions=5, warmups=1)


def test_two_runs_share_a_shape_but_not_content(tmp_path: Path) -> None:
    # Two genuinely different runs: different id, different clock. The files must
    # differ, and their field order must not, so a diff shows only what changed.
    first = json.loads(write_json(_result("run1"), tmp_path).read_text())
    second = json.loads(write_json(_result("run2", datetime(2026, 8, 30, tzinfo=UTC)), tmp_path).read_text())

    assert first != second
    assert list(first) == list(second)


def test_each_run_gets_its_own_directory(tmp_path: Path) -> None:
    first = write_json(_result("run1"), tmp_path)
    second = write_json(_result("run2"), tmp_path)

    assert first == tmp_path / "run1" / "result.json"
    assert second == tmp_path / "run2" / "result.json"
    assert first.exists() and second.exists()


def test_summary_prints_one_line_per_format() -> None:
    stream = io.StringIO()
    print_summary(_result(), file=stream)
    lines = stream.getvalue().strip().splitlines()

    assert len(lines) == 3
    assert "run abc123" in lines[0]
    assert lines[1].startswith("pandas")
    assert lines[2].startswith("timef")


def test_summary_reports_both_the_write_and_the_read() -> None:
    stream = io.StringIO()
    print_summary(_result(), file=stream)
    pandas_line = stream.getvalue().strip().splitlines()[1]

    assert "write   8.400 s" in pandas_line
    assert "read   2.950 s" in pandas_line
    assert "size       2.1 MB" in pandas_line


def test_summary_refuses_a_report_missing_an_operation() -> None:
    result = _result()
    # Drop the read of the last format. Its line would otherwise report a number it does not have.
    partial = result.model_copy(update={"measurements": result.measurements[:-1]})

    with pytest.raises(EvaluationError, match="do not match the artifacts"):
        print_summary(partial, file=io.StringIO())


def test_summary_refuses_a_report_missing_a_format() -> None:
    result = _result()
    partial = result.model_copy(update={"measurements": result.measurements[:2]})

    with pytest.raises(EvaluationError, match="do not match the artifacts"):
        print_summary(partial, file=io.StringIO())


def test_environment_records_the_interpreter_and_packages() -> None:
    environment = collect_environment()

    assert environment.python
    assert environment.cpu_count > 0
    assert environment.packages
