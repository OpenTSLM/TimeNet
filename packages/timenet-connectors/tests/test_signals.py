import numpy as np
import pyarrow as pa
import pytest

from timenet.errors import TimeNetConnectorError
from timenet.types import TimeSeriesSpec
from timenet_connectors.sources.fields import error_context
from timenet_connectors.sources.signals import check_channel_matrix, numeric_signal


def spec(dtype="float32", **kwargs):
    return TimeSeriesSpec(spec_type="test", name="Test", unit_value=None, dtype=dtype, **kwargs)


@pytest.mark.parametrize("values", [[1, 2], (1, 2), np.array([1, 2]), pa.array([1, 2])])
def test_input_representations(values):
    result = numeric_signal(values, "signal", spec=spec(), n_values=2)
    assert result.type == pa.float32()
    assert result.to_pylist() == [1.0, 2.0]


@pytest.mark.parametrize(
    "values",
    [
        [1, True],
        [1, "2"],
        [1, complex(2)],
        [[1]],
        np.array([1], dtype=object),
        np.ma.array([1]),
        pa.chunked_array([[1]]),
        np.zeros((2, 1)),
    ],
)
def test_unsupported_inputs(values):
    with pytest.raises(TimeNetConnectorError):
        numeric_signal(values, "signal", spec=spec())


@pytest.mark.parametrize(
    "dtype,value",
    [
        ("float32", 2**24 + 1),
        ("float64", 2**53 + 1),
        ("float64", 10**400),
        ("uint8", -1),
        ("int8", 128),
        ("int32", 1.5),
    ],
)
@pytest.mark.parametrize("casting", ["exact", "float_rounding"])
def test_integer_precision_and_range(dtype, value, casting):
    with pytest.raises(TimeNetConnectorError, match=r"signal\[1\]"):
        numeric_signal([1, value], "signal", spec=spec(dtype), casting=casting)


@pytest.mark.parametrize(
    "dtype,value", [("float32", 2**24), ("float64", 2**60), ("int8", -128), ("uint32", 2**32 - 1), ("int32", 1.0)]
)
def test_exact_values(dtype, value):
    assert numeric_signal([value], "signal", spec=spec(dtype)).to_pylist() == [value]


def test_rounding_is_explicit():
    with pytest.raises(TimeNetConnectorError, match="exactly"):
        numeric_signal([0.1], "signal", spec=spec())
    assert numeric_signal([0.1], "signal", spec=spec(), casting="float_rounding").to_pylist() == [
        float(np.float32(0.1))
    ]
    assert numeric_signal([1e-100], "signal", spec=spec(), casting="float_rounding").to_pylist() == [0.0]
    with pytest.raises(TimeNetConnectorError, match="overflows"):
        numeric_signal([1e100], "signal", spec=spec(), casting="float_rounding")


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_is_never_missing(value):
    with pytest.raises(TimeNetConnectorError, match=r"signal\[0\]"):
        numeric_signal([value], "signal", spec=spec(nullable=True))


def test_nulls_and_empty():
    for values in ([None, 1], pa.array([None, 1])):
        assert numeric_signal(values, "signal", spec=spec(nullable=True)).to_pylist() == [None, 1.0]
        with pytest.raises(TimeNetConnectorError, match="nullable"):
            numeric_signal(values, "signal", spec=spec())
    assert numeric_signal([], "signal", spec=spec(), n_values=0).type == pa.float32()
    with pytest.raises(TimeNetConnectorError, match="expected 1 samples, got 0"):
        numeric_signal([], "signal", spec=spec(), n_values=1)


def test_input_storage_is_not_shared_even_through_arrow():
    values = np.array([1, 2], dtype=np.float32)
    for source in (values, pa.array(values)):
        result = numeric_signal(source, "signal", spec=spec())
        values[0] = 9
        assert result.to_pylist() == [1, 2]
        values[0] = 1


def test_context_and_first_invalid_position():
    with (
        pytest.raises(TimeNetConnectorError, match=r"file: row 2: signal\[0\]"),
        error_context("file"),
        error_context("row 2"),
    ):
        numeric_signal([float("inf"), True], "signal", spec=spec())


@pytest.mark.parametrize("kwargs", [{"dtype": "bool"}, {"value_shape": (2,)}])
def test_unsupported_specs(kwargs):
    with pytest.raises(TimeNetConnectorError, match="scalar numeric spec"):
        numeric_signal([], "signal", spec=spec(**kwargs))


@pytest.mark.parametrize("kwargs", [{"n_values": True}, {"n_values": -1}, {"casting": "unsafe"}])
def test_configuration(kwargs):
    with pytest.raises(TypeError):
        numeric_signal([], "signal", spec=spec(), **kwargs)


def test_matrix_layout_does_not_scan_or_mutate():
    matrix = np.full((3, 2), np.nan)
    check_channel_matrix(matrix, "channels", sample_axis=0, n_channels=2, n_values=3)
    check_channel_matrix(matrix.T, "channels", sample_axis=1, n_channels=2, n_values=3)
    with pytest.raises(TimeNetConnectorError, match="expected 3 channels, got 2"):
        check_channel_matrix(matrix, "channels", sample_axis=0, n_channels=3)
    with pytest.raises(TimeNetConnectorError, match="expected 4 samples, got 3"):
        check_channel_matrix(matrix, "channels", sample_axis=0, n_channels=2, n_values=4)
    with pytest.raises(TypeError):
        check_channel_matrix(matrix, "channels", sample_axis=True, n_channels=2)  # ty: ignore[invalid-argument-type]
