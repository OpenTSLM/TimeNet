"""Unit tests for the synthetic corpus and regression calculations."""

from pathlib import Path

import pytest

from benchmarks.end_to_end.benchmark import regression_percent
from benchmarks.end_to_end.core import MATRIX, capabilities, write_and_fingerprint
from benchmarks.end_to_end.corpus import build_corpus


def test_portable_corpus_covers_the_shapes_that_break():
    dataset = build_corpus()
    assert dataset.metadata is not None
    assert len(dataset.records) == 4
    assert len(dataset.tasks) == 3
    # a source tree deeper than one level
    assert max(len(record.walk_sources()) for record in dataset.records) >= 3
    # one series shared by two records, so the writer must store it once and link it twice
    shared = [signal for record in dataset.records for signal in record.signals() if signal.id == "shared-series"]
    assert len(shared) == 2
    assert shared[0] is shared[1]
    # annotations at every level
    assert any(record.annotations for record in dataset.records)
    assert any(source.annotations for record in dataset.records for source in record.walk_sources())
    assert any(signal.annotations for record in dataset.records for signal in record.signals())
    assert any(task.annotations for task in dataset.tasks)
    assert dataset.annotations
    # a task whose two inputs are ordered
    assert len(dataset.tasks[2].inputs) == 2


def test_rich_corpus_adds_an_ordinal_axis_and_a_wider_dtype():
    if not capabilities()["rich"]:
        pytest.skip("active revision does not support the rich profile")
    dataset = build_corpus(profile="rich")
    dtypes = {signal.spec.dtype for record in dataset.records for signal in record.signals()}
    axes = {type(signal.time_axis).__name__ for record in dataset.records for signal in record.signals()}
    assert "float64" in dtypes
    assert "OrdinalAxis" in axes


def test_every_axis_shape_is_covered():
    axes = {type(signal.time_axis).__name__ for record in build_corpus().records for signal in record.signals()}
    assert {"RegularAxis", "IrregularAxis"} <= axes


@pytest.mark.parametrize("scale", (0, -1))
def test_scale_must_be_positive(scale):
    with pytest.raises(ValueError, match="scale"):
        build_corpus(scale=scale)


def test_profile_must_be_known():
    with pytest.raises(ValueError, match="profile"):
        build_corpus(profile="nope")


def test_regression_percent():
    assert regression_percent(100.0, 90.0) == pytest.approx(-10.0)
    assert regression_percent(100.0, 101.0) == pytest.approx(1.0)


@pytest.mark.parametrize("case", MATRIX, ids=lambda case: case.name)
def test_each_matrix_case_round_trips_deterministically(tmp_path: Path, case):
    supported = capabilities()
    if case.values_backend != "parquet" and not supported.get(case.values_backend):
        pytest.skip(f"active revision does not support {case.values_backend}")
    if case.profile == "rich" and not supported["rich"]:
        pytest.skip("active revision does not support the rich profile")
    first, metrics = write_and_fingerprint(tmp_path / "first", case, scale=1)
    second, _ = write_and_fingerprint(tmp_path / "second", case, scale=1)
    assert first == second
    assert metrics["series"] > 0
    assert metrics["value_bytes"] > 0
    assert metrics["disk_bytes"] > 0


def test_chunking_does_not_change_the_logical_content(tmp_path: Path):
    """The point of the fingerprint: physical layout may differ, logical content may not."""
    small, large = MATRIX[0], MATRIX[1]
    assert small.profile == large.profile
    assert small.chunk_max_bytes != large.chunk_max_bytes
    first, _ = write_and_fingerprint(tmp_path / "small", small, scale=1)
    second, _ = write_and_fingerprint(tmp_path / "large", large, scale=1)
    assert first == second
