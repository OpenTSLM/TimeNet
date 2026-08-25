"""Unit tests for the synthetic corpus and regression calculations."""

from pathlib import Path

import pytest

from benchmarks.end_to_end.benchmark import regression_percent
from benchmarks.end_to_end.core import MATRIX, capabilities, write_and_fingerprint
from benchmarks.end_to_end.corpus import SCENARIOS, build_corpus


def test_portable_corpus_covers_scenarios_connector_patterns_and_tasks():
    dataset = build_corpus()
    schema = dataset.schema
    assert schema is not None
    assert len(SCENARIOS) == 8
    assert len(dataset.samples) > 100
    assert {type(task) for task in dataset.tasks} == set(schema.tasks)
    # The portable corpus stays scalar (no N-D value_shape) and adds scalar non-float channels.
    # The default (parquet) build keeps the str channel; a zarr build must drop it, because the
    # Zarr backend rejects the str dtype.
    for spec in schema.time_series_specs:
        assert spec.value_shape == ()
        if spec.dtype != "float32":
            assert spec.dtype in {"int16", "bool", "float64", "str"}
    zarr_schema = build_corpus(values_backend="zarr").schema
    assert zarr_schema is not None
    zarr_dtypes = {s.dtype for s in zarr_schema.time_series_specs}
    assert "str" not in zarr_dtypes


def test_rich_corpus_adds_varied_dtypes_and_shapes():
    if not capabilities()["rich"]:
        pytest.skip("active revision does not support rich dtype/N-D specs")
    dataset = build_corpus(profile="rich")
    schema = dataset.schema
    assert schema is not None
    contracts = {
        (getattr(spec, "dtype", None), getattr(spec, "value_shape", None)) for spec in schema.time_series_specs
    }
    assert ("float64", (12,)) in contracts
    assert ("uint16", (32,)) in contracts
    assert ("int16", (2, 2)) in contracts


@pytest.mark.parametrize("scale", (0, -1))
def test_scale_must_be_positive(scale):
    with pytest.raises(ValueError, match="scale"):
        build_corpus(scale=scale)


def test_regression_percent():
    assert regression_percent(100.0, 90.0) == pytest.approx(-10.0)
    assert regression_percent(100.0, 101.0) == pytest.approx(1.0)


@pytest.mark.parametrize("case", MATRIX, ids=lambda case: case.name)
def test_each_matrix_case_round_trips_deterministically(tmp_path: Path, case):
    supported = capabilities()
    if case.values_backend == "zarr" and not supported["zarr"]:
        pytest.skip("active revision does not support Zarr")
    if case.profile == "rich" and not supported["rich"]:
        pytest.skip("active revision does not support rich dtype/N-D specs")
    first, metrics = write_and_fingerprint(tmp_path / "first", case, scale=1)
    second, _ = write_and_fingerprint(tmp_path / "second", case, scale=1)
    assert first == second
    assert metrics["series"] > 100
    assert metrics["value_bytes"] > 0
    assert metrics["disk_bytes"] > 0
