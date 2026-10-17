"""Window selection, scalar fidelity, reduction, and bounded value reads."""

import numpy as np
import pyarrow as pa
import pytest

from timenet.dataset import Record, Signal, Source, TimeFDataset
from timenet.dataset.axis import RegularAxis
from timenet.errors import TimeFValidationError
from timenet.testing import make_dataset
from timenet.types import TimeSeriesSpec
from timenet.types.wire import ValueDtype
from timenet.viewer.schemas import WindowQuery
from timenet.viewer.windows import window_result


def dataset_with_signals(*signals: Signal) -> TimeFDataset:
    dataset = TimeFDataset(metadata=make_dataset().metadata)
    dataset.add_record(
        record=Record(record_id="record", sources=(Source(id="source", name="Sensor", signals=signals),))
    )
    return dataset


def signal(values: pa.Array, *, signal_id: str = "signal", rate: int = 100, dtype: ValueDtype = "float64") -> Signal:
    return Signal(
        id=signal_id,
        name=signal_id,
        spec=TimeSeriesSpec(spec_type=signal_id, name=signal_id, dtype=dtype, unit_value=None, nullable=True),
        time_axis=RegularAxis.from_rate_hz(rate),
        data=values,
    )


def test_reduction_preserves_order_endpoints_extrema_and_gaps(open_dataset):
    # 256 buckets of ten samples force the production streaming reduction path.
    values = [8, 7, None, 100, -10, 3, 2, 1, 0, 9] * 256
    dataset = dataset_with_signals(signal(pa.array(values, type=pa.float64())))
    reader = open_dataset(dataset)
    result = window_result(reader.inspect(), WindowQuery(record_id="record", signal_id="signal", full=True, width=256))
    assert result["reduced"] is True
    expected = [base + offset for base in range(0, len(values), 10) for offset in (0, 2, 3, 4, 9)]
    assert [int(item["index"]) for item in result["items"]] == expected
    assert [item["value"] for item in result["items"][:5]] == ["8.0", None, "100.0", "-10.0", "9.0"]
    assert result["coverage"]["null"] == "256"
    assert result["coverage"]["finite"] == "2304"


def test_plot_read_limit_and_raw_continuation(open_dataset, monkeypatch):
    dataset = dataset_with_signals(signal(pa.array(range(20_001), type=pa.float64())))
    reads = []
    original = Signal.read_steps

    def bounded(self, start, stop):
        reads.append(stop - start)
        assert stop - start <= 20_000
        return original(self, start, stop)

    monkeypatch.setattr(Signal, "read_steps", bounded)
    reader = open_dataset(dataset)
    with pytest.raises(TimeFValidationError, match="20,000 steps"):
        window_result(reader.inspect(), WindowQuery(record_id="record", signal_id="signal", full=True, width=256))
    assert reads == []
    window = window_result(
        reader.inspect(), WindowQuery(record_id="record", signal_id="signal", stop=20_000, width=256)
    )
    assert window["scanned_count"] == "20000"
    assert window["reduced"]
    assert len(window["items"]) <= 5 * 256
    assert window["items"][-1]["index"] == "19999"
    raw = window_result(reader.inspect(), WindowQuery(record_id="record", signal_id="signal", stop=20001, mode="raw"))
    assert len(raw["items"]) == 200
    assert raw["next_start"] == "200"
    page = window_result(
        reader.inspect(), WindowQuery(record_id="record", signal_id="signal", start=200, stop=20001, mode="raw")
    )
    assert page["items"][0]["index"] == "200"
    initial = window_result(reader.inspect(), WindowQuery(record_id="record", signal_id="signal"))
    assert initial["scanned_count"] == "1000"
    assert max(reads) <= 20_000


def test_time_window_resolves_each_sample_rate(open_dataset):
    dataset = dataset_with_signals(
        signal(pa.array(range(1000), type=pa.float64()), signal_id="slow"),
        signal(pa.array(range(2500), type=pa.float64()), signal_id="fast", rate=250),
    )
    reader = open_dataset(dataset)
    results = [
        window_result(
            reader.inspect(), WindowQuery(record_id="record", signal_id=name, start_us=1_000_000, end_us=2_000_000)
        )
        for name in ("slow", "fast")
    ]
    assert [result["scanned_count"] for result in results] == ["100", "250"]
    for result, rate in zip(results, (100, 250), strict=True):
        assert [(item["index"], item["x"], item["value"]) for item in result["items"]] == [
            (str(index), str(index * (1_000_000 // rate)), str(float(index))) for index in range(rate, 2 * rate)
        ]


@pytest.mark.parametrize(
    ("dtype", "values", "expected"),
    [
        ("str", pa.array(["hello", None, "world"]), ["hello", None, "world"]),
        ("bool", pa.array([True, None, False]), ["True", None, "False"]),
        ("int32", pa.array([2**30 + 1, None, -(2**30)], type=pa.int32()), [str(2**30 + 1), None, str(-(2**30))]),
    ],
)
def test_typed_raw_values(open_dataset, dtype, values, expected):
    dataset = dataset_with_signals(signal(values, dtype=dtype))
    reader = open_dataset(dataset)
    result = window_result(reader.inspect(), WindowQuery(record_id="record", signal_id="signal", mode="raw"))
    assert [item["value"] for item in result["items"]] == expected
    assert [item["display"] for item in result["items"]] == ([None] * 3 if dtype == "str" else expected)
    assert [item["kind"] for item in result["items"]] == ["finite", "null", "finite"]


def test_irregular_window_preserves_duplicate_offsets_and_slice_boundaries(open_dataset):
    irregular = Signal.from_irregular(
        [1.0, 2.0, 3.0],
        time_offsets_us=[0, 0, 1000],
        id="signal",
        name="Irregular",
        spec=TimeSeriesSpec(spec_type="irregular", name="Irregular", unit_value=None),
    )
    dataset = dataset_with_signals(irregular)
    reader = open_dataset(dataset)
    by_step = window_result(reader.inspect(), WindowQuery(record_id="record", signal_id="signal", start=1, stop=3))
    assert [(item["index"], item["x"], item["value"]) for item in by_step["items"]] == [
        ("1", "0", "2.0"),
        ("2", "1000", "3.0"),
    ]
    by_time = window_result(
        reader.inspect(), WindowQuery(record_id="record", signal_id="signal", start_us=0, end_us=1000)
    )
    assert [(item["index"], item["x"], item["value"]) for item in by_time["items"]] == [
        ("0", "0", "1.0"),
        ("1", "0", "2.0"),
    ]


@pytest.mark.parametrize("mode", ["raw", "plot"])
@pytest.mark.parametrize("component", [0, 1])
def test_tensor_window_preserves_missing_timesteps(open_dataset, mode, component):
    storage = pa.array(
        [[9.0, 90.0], [1.0, 10.0], None, [0.0, 0.0], [3.0, 30.0], [float("nan"), float("inf")]],
        type=pa.list_(pa.float32(), 2),
    )
    values = pa.FixedShapeTensorArray.from_storage(pa.fixed_shape_tensor(pa.float32(), [2]), storage)
    signal = Signal(
        id="tensor",
        name="Tensor",
        spec=TimeSeriesSpec(spec_type="tensor", name="Tensor", unit_value=None, value_shape=(2,), nullable=True),
        time_axis=RegularAxis.from_rate_hz(1),
        data=values,
    )
    dataset = TimeFDataset(metadata=make_dataset().metadata)
    dataset.add_record(record=Record(record_id="record", sources=(Source(name="Sensor", signals=(signal,)),)))

    reader = open_dataset(dataset)
    result = window_result(
        reader.inspect(),
        WindowQuery(
            record_id="record",
            signal_id="tensor",
            component=(component,),
            start=1,
            stop=6,
            mode=mode,
        ),
    )
    expected_values = ["1.0", None, "0.0", "3.0", None] if component == 0 else ["10.0", None, "0.0", "30.0", None]
    assert [item["index"] for item in result["items"]] == ["1", "2", "3", "4", "5"]
    assert [item["value"] for item in result["items"]] == expected_values
    assert result["coverage"] == {
        "null": "1",
        "nan": "1" if component == 0 else "0",
        "positive_infinity": "0" if component == 0 else "1",
        "negative_infinity": "0",
        "finite": "3",
    }


@pytest.mark.parametrize("mode", ["plot", "raw"])
def test_numeric_budget_counts_all_tensor_components_before_reading(open_dataset, monkeypatch, mode):
    values = pa.FixedShapeTensorArray.from_numpy_ndarray(np.zeros((129, 8192), dtype=np.float32))
    signal = Signal(
        id="tensor",
        name="Tensor",
        spec=TimeSeriesSpec(spec_type="tensor", name="Tensor", unit_value=None, value_shape=(8192,)),
        time_axis=RegularAxis.from_rate_hz(1),
        data=values,
    )
    dataset = TimeFDataset(metadata=make_dataset().metadata)
    dataset.add_record(record=Record(record_id="record", sources=(Source(name="Sensor", signals=(signal,)),)))
    reads = []
    original = Signal.read_steps

    def capture(self, start, stop):
        reads.append((start, stop))
        return original(self, start, stop)

    monkeypatch.setattr(Signal, "read_steps", capture)
    reader = open_dataset(dataset)
    query = WindowQuery(record_id="record", signal_id="tensor", component=(0,), mode=mode, stop=129)
    with pytest.raises(TimeFValidationError, match="8 MiB"):
        window_result(reader.inspect(), query)
    assert reads == []
    accepted = window_result(reader.inspect(), query.model_copy(update={"stop": 128}))
    assert accepted["scanned_count"] == "128"
    assert reads == [(0, 128)]


def test_raw_unicode_preview_preserves_original_size(open_dataset):
    value = "界" * (2 * 2**20)
    reader = open_dataset(dataset_with_signals(signal(pa.array([value]), dtype="str")))
    result = window_result(reader.inspect(), WindowQuery(record_id="record", signal_id="signal", mode="raw"))
    item = result["items"][0]
    assert item["value"] == "界" * 1024
    assert item["original_bytes"] == str(len(value.encode("utf-8")))
    assert item["value_truncated"]
