"""The harness must measure a write, discard read warm-ups, take a median, and refuse a bad run."""

from pathlib import Path
import time

import numpy as np
import pytest

from evaluations.cli import run_evaluation
from evaluations.errors import EvaluationError
from evaluations.formats.base import FormatName
from evaluations.harness import Operation, directory_size, measure_read, measure_write


class ScriptedFormat:
    """A format whose reads take a scripted sequence of durations."""

    name = FormatName.PANDAS

    def __init__(self, sleeps_ms: tuple[float, ...] = (), write_bytes: int = 1) -> None:
        self.reads = 0
        self._sleeps_ms = sleeps_ms
        self._write_bytes = write_bytes

    def write(self, source: Path, out: Path) -> Path:
        out.mkdir(parents=True, exist_ok=True)
        path = out / "written.bin"
        path.write_bytes(b"x" * self._write_bytes)
        return path

    def read_all(self, path: Path) -> list[np.ndarray]:
        if self.reads < len(self._sleeps_ms):
            time.sleep(self._sleeps_ms[self.reads] / 1000)
        self.reads += 1
        return [np.zeros(1, dtype=np.float32)]


def test_a_write_is_measured_once_and_sized(tmp_path: Path) -> None:
    fmt = ScriptedFormat(write_bytes=64)
    artifact, measurement = measure_write(fmt, tmp_path / "release", tmp_path / "out")

    assert artifact.format is FormatName.PANDAS
    assert artifact.size_bytes == 64
    assert artifact.path.is_file()
    assert measurement.operation is Operation.WRITE
    assert measurement.repetitions == 1
    assert measurement.warmups == 0
    assert measurement.elapsed_ns > 0


def test_the_size_comes_from_the_files_not_the_format(tmp_path: Path) -> None:
    artifact, _ = measure_write(ScriptedFormat(write_bytes=1234), tmp_path / "release", tmp_path / "out")

    assert artifact.size_bytes == artifact.path.stat().st_size


def test_warmups_are_discarded_and_repetitions_counted(tmp_path: Path) -> None:
    fmt = ScriptedFormat()
    measurement = measure_read(fmt, tmp_path, repeats=5, warmups=2)

    assert fmt.reads == 7
    assert measurement.repetitions == 5
    assert measurement.warmups == 2
    assert measurement.operation is Operation.READ


def test_elapsed_is_a_median_not_a_mean(tmp_path: Path) -> None:
    # One warm-up, then four fast reads and one slow one. A mean would be dragged
    # above 10 ms by the outlier; a median stays with the fast reads.
    fmt = ScriptedFormat(sleeps_ms=(0, 0, 0, 0, 0, 60))
    measurement = measure_read(fmt, tmp_path, repeats=5, warmups=1)

    elapsed_ms = measurement.elapsed_ns / 1e6
    assert elapsed_ms < 10, f"median dragged by the outlier: {elapsed_ms:.1f} ms"


def test_warmups_below_one_is_refused(tmp_path: Path) -> None:
    with pytest.raises(EvaluationError, match="warmups must be at least 1"):
        measure_read(ScriptedFormat(), tmp_path, repeats=3, warmups=0)


def test_repeats_below_one_is_refused(tmp_path: Path) -> None:
    with pytest.raises(EvaluationError, match="repeats must be at least 1"):
        measure_read(ScriptedFormat(), tmp_path, repeats=0, warmups=1)


def test_directory_size_sums_every_file(tmp_path: Path) -> None:
    (tmp_path / "nested").mkdir()
    (tmp_path / "a.bin").write_bytes(b"x" * 10)
    (tmp_path / "nested" / "b.bin").write_bytes(b"y" * 25)

    assert directory_size(tmp_path) == 35
    assert directory_size(tmp_path / "a.bin") == 10


def test_missing_source_raises(tmp_path: Path) -> None:
    with pytest.raises(EvaluationError, match="does not exist"):
        run_evaluation(tmp_path / "absent", tmp_path / "out")
