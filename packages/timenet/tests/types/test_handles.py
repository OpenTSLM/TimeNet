import copy
from dataclasses import dataclass
from functools import partial
import pickle

import numpy as np
import pyarrow as pa
import pytest

from timenet.errors import TimeFNullValuesError, TimeFValidationError
from timenet.types import ArrayHandle, ValueHandle
from timenet.types.handles import ArrowArray, CallableArray, NumpyArray
from timenet.types.text import InlineText, InlineTextList


@dataclass(eq=False)
class _StoredArray(ArrayHandle):
    """A stored array that records every read and relies on the default read_steps and read_numpy."""

    values: pa.Array
    reads: int = 0

    @property
    def is_loaded(self) -> bool:
        return False

    def read(self) -> pa.Array:
        self.reads += 1
        return self.values


def _tensor(frames: np.ndarray) -> pa.Array:
    storage = pa.FixedSizeListArray.from_arrays(pa.array(frames.ravel()), frames[0].size)
    tensor_type = pa.fixed_shape_tensor(pa.from_numpy_dtype(frames.dtype), frames.shape[1:])
    return pa.ExtensionArray.from_storage(tensor_type, storage)


def _no_arrow(*_args, **_kwargs):
    raise AssertionError("built Arrow")


def _counting_function():
    calls = []

    def function() -> pa.Array:
        calls.append(None)
        return pa.array([1.0, 2.0, 3.0], type=pa.float32())

    return function, calls


def test_the_default_read_steps_slices_one_read():
    handle = _StoredArray(pa.array([0, 1, 2, 3, 4]))
    assert handle.read_steps(1, 3).to_pylist() == [1, 2]
    assert handle.reads == 1


@pytest.mark.parametrize(
    ("start", "stop", "expected"),
    [(0, 0, []), (3, 99, [3, 4]), (5, 7, []), (9, 12, [])],
)
def test_the_default_read_steps_reads_to_the_end_at_most(start, stop, expected):
    assert _StoredArray(pa.array([0, 1, 2, 3, 4])).read_steps(start, stop).to_pylist() == expected


def test_the_default_read_numpy_converts_one_read():
    handle = _StoredArray(pa.array([1.0, 2.0], type=pa.float32()))
    values = handle.read_numpy()
    assert values.dtype == np.float32
    assert values.tolist() == [1.0, 2.0]
    assert handle.reads == 1


@pytest.mark.parametrize("dtype", [np.int16, np.float32, np.bool_])
def test_the_default_read_numpy_gives_tensors_their_value_shape(dtype):
    frames = (np.arange(12) % 3).astype(dtype).reshape(2, 2, 3)
    values = _StoredArray(_tensor(frames)).read_numpy()
    assert values.dtype == dtype
    np.testing.assert_array_equal(values, frames)


@pytest.mark.parametrize("values", [pa.array([1.0, None]), pa.array([True, None]), pa.array(["a", None])])
def test_the_default_read_numpy_rejects_nulls(values):
    with pytest.raises(TimeFNullValuesError, match="1 of 2 values are null"):
        _StoredArray(values).read_numpy()


def test_numpy_array_holds_loaded_numpy_values():
    values = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    handle = NumpyArray(values)
    assert isinstance(handle, ArrayHandle)
    assert isinstance(handle, ValueHandle)
    assert handle.is_loaded
    np.testing.assert_array_equal(handle.read_numpy(), values)


def test_numpy_array_reads_numpy_without_building_arrow(monkeypatch):
    handle = NumpyArray(np.array([True, False, True]))
    monkeypatch.setattr(pa, "array", _no_arrow)
    assert handle.read_numpy().tolist() == [True, False, True]


def test_numpy_array_builds_arrow_once_on_the_first_read():
    values = np.array([1.0, 2.0], dtype=np.float32)
    handle = NumpyArray(values)
    first = handle.read()
    assert first.type == pa.float32()
    assert first.to_pylist() == [1.0, 2.0]
    assert handle.read() is first


def test_numpy_array_values_cannot_be_changed_through_a_read():
    handle = NumpyArray(np.array([1, 2, 3], dtype=np.int64))
    with pytest.raises(ValueError, match="read-only"):
        handle.read_numpy()[0] = 9


@pytest.mark.parametrize("dtype", [np.bool_, np.int64, np.uint8, np.float16])
def test_numpy_array_takes_boolean_integer_and_float_values(dtype):
    values = np.array([1, 0], dtype=dtype)
    np.testing.assert_array_equal(NumpyArray(values).read_numpy(), values)


@pytest.mark.parametrize(
    "values",
    [
        np.zeros((2, 2), dtype=np.float32),
        np.array(1.0),
        np.array([1, None], dtype=object),
        np.array(["a", "b"]),
        np.array([1 + 2j]),
        np.array(["2026-01-01"], dtype="datetime64[D]"),
    ],
)
def test_numpy_array_takes_one_boolean_integer_or_float_value_per_step(values):
    with pytest.raises(TimeFValidationError, match="one-dimensional, one value per step, with a boolean"):
        NumpyArray(values)


def test_numpy_array_shares_a_contiguous_numeric_array_with_arrow():
    values = np.arange(4, dtype=np.float32)
    handle = NumpyArray(values)
    assert np.shares_memory(handle.read_numpy(), values)
    assert np.shares_memory(handle.read().to_numpy(), values)


def test_numpy_array_takes_only_a_numpy_array():
    with pytest.raises(TimeFValidationError, match="holds a NumPy array"):
        NumpyArray([1.0, 2.0])  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize(("start", "stop", "expected"), [(1, 3, [2, 3]), (2, 99, [3, 4]), (6, 8, [])])
def test_numpy_array_reads_steps(start, stop, expected):
    assert NumpyArray(np.array([1, 2, 3, 4])).read_steps(start, stop).to_pylist() == expected


def test_numpy_array_pickles_and_copies_its_values_but_not_its_arrow():
    handle = NumpyArray(np.arange(1_000, dtype=np.float64))
    size = len(pickle.dumps(handle))
    handle.read()
    assert len(pickle.dumps(handle)) == size
    for back in (pickle.loads(pickle.dumps(handle)), copy.copy(handle), copy.deepcopy(handle)):
        np.testing.assert_array_equal(back.read_numpy(), handle.read_numpy())
        assert back.read().equals(handle.read())
        assert not back.read_numpy().flags.writeable


def test_arrow_array_holds_a_loaded_arrow_array():
    values = pa.array(["a", "b", None])
    handle = ArrowArray(values)
    assert isinstance(handle, ArrayHandle)
    assert handle.is_loaded
    assert handle.read() is values


def test_arrow_array_reads_steps_and_numpy():
    frames = np.arange(12, dtype=np.float32).reshape(3, 4)
    handle = ArrowArray(_tensor(frames))
    np.testing.assert_array_equal(handle.read_numpy(), frames)
    np.testing.assert_array_equal(handle.read_steps(1, 9).to_numpy_ndarray(), frames[1:])


def test_arrow_array_takes_only_an_arrow_array():
    with pytest.raises(TimeFValidationError, match="pyarrow Array"):
        ArrowArray(np.array([1.0]))


@pytest.mark.parametrize(
    "handle",
    [NumpyArray(np.array([1.0, 2.0])), ArrowArray(pa.array([1.0, 2.0])), CallableArray(partial(pa.array, [1.0, 2.0]))],
)
def test_in_memory_handles_survive_pickling_and_copies(handle):
    for back in (pickle.loads(pickle.dumps(handle)), copy.copy(handle), copy.deepcopy(handle)):
        assert type(back) is type(handle)
        assert back.read().equals(handle.read())


@pytest.mark.parametrize(
    "handle",
    [
        NumpyArray(np.array([1.0])),
        ArrowArray(pa.array([1.0])),
        CallableArray(partial(pa.array, [1.0])),
        InlineText("note"),
        InlineTextList(["a"]),
    ],
)
def test_in_memory_handles_have_no_stored_location(handle):
    assert isinstance(handle, ValueHandle)
    assert handle._location is None
    assert not hasattr(handle, "__dict__")


def test_callable_array_reads_through_its_function_on_every_read():
    function, calls = _counting_function()
    handle = CallableArray(function)
    assert isinstance(handle, ArrayHandle)
    assert not handle.is_loaded
    assert not calls
    assert handle.read().to_pylist() == [1.0, 2.0, 3.0]
    assert handle.read_numpy().tolist() == [1.0, 2.0, 3.0]
    assert handle.read_steps(1, 2).to_pylist() == [2.0]
    assert len(calls) == 3
    assert not handle.is_loaded


def test_callable_arrays_over_one_function_compare_equal():
    function, _ = _counting_function()
    other, _ = _counting_function()
    assert CallableArray(function) == CallableArray(function)
    assert hash(CallableArray(function)) == hash(CallableArray(function))
    assert CallableArray(function) != CallableArray(other)


def test_in_memory_arrays_compare_by_identity():
    values = pa.array([1.0])
    assert ArrowArray(values) != ArrowArray(values)
    assert NumpyArray(np.array([1.0])) != NumpyArray(np.array([1.0]))


def test_callable_array_takes_only_a_callable():
    with pytest.raises(TimeFValidationError, match="takes a function"):
        CallableArray(pa.array([1.0]))
