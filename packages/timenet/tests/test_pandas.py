from fractions import Fraction
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest


pd = pytest.importorskip("pandas")

from timenet.dataset import TimeFDataset, TimeSeries  # noqa: E402
from timenet.dataset.axis import OrdinalAxis, RegularAxis  # noqa: E402
from timenet.errors import TimeFValidationError  # noqa: E402
from timenet.pandas import (  # noqa: E402
    ARRAY_META_COLUMNS,
    iter_record_frames,
    record_array_frame,
    series_frame,
)
from timenet.reader import TimeFReader  # noqa: E402
from timenet.registry import DatasetVersion  # noqa: E402
from timenet.testing import make_dataset  # noqa: E402
from timenet.types import DatasetMetadata, Domain, License, TimeSeriesSpec, Version, ureg  # noqa: E402
from timenet.writer import TimeFWriter  # noqa: E402


def _spec(dtype: str = "float32", value_shape: tuple[int, ...] = ()) -> TimeSeriesSpec:
    return TimeSeriesSpec(
        spec_type="eeg",
        name="EEG",
        unit_value=ureg.microvolt,
        dtype=dtype,
        value_shape=value_shape,
    )


def _dataset(dataset_id: str, description: str) -> TimeFDataset:
    return TimeFDataset(
        metadata=DatasetMetadata(
            dataset_id=dataset_id,
            dataset_version=Version(1, 0, 0),
            name="Fixture",
            description=description,
            license=License.CC_BY_4_0,
            domains=(Domain.GENERAL,),
        )
    )


def test_a_regular_series_computes_its_time_offsets_from_the_cadence():
    series = TimeSeries.from_values(
        np.array([1.0, 2.0, 3.0], dtype=np.float32),
        spec=_spec(),
        signal="Fpz-Cz",
        time_axis=RegularAxis.from_rate_hz(2),
    )
    frame = series_frame(series)
    assert list(frame["time_us"]) == [0, 500_000, 1_000_000]
    assert list(frame["value"]) == [1.0, 2.0, 3.0]


def test_a_non_integral_rate_floors_the_same_way_the_axis_does():
    # 360 Hz is 25000/9 us, so the offsets are not whole microseconds before the floor.
    axis = RegularAxis(period_us=Fraction(25_000, 9))
    series = TimeSeries.from_values(
        np.arange(5, dtype=np.float32),
        spec=_spec(),
        signal="II",
        time_axis=axis,
    )
    assert list(series_frame(series)["time_us"]) == [axis.time_offset_us(i) for i in range(5)]


def test_a_windowed_series_starts_at_its_own_origin():
    series = TimeSeries.from_values(
        np.arange(3, dtype=np.float32),
        spec=_spec(),
        signal="Fpz-Cz",
        time_axis=RegularAxis(period_us=Fraction(1_000_000), start_index=10),
    )
    assert list(series_frame(series)["time_us"]) == [10_000_000, 11_000_000, 12_000_000]


def test_an_irregular_series_reads_its_stored_offsets():
    series = TimeSeries.from_irregular(
        np.array([1.0, 2.0], dtype=np.float32),
        time_offsets_us=np.array([0, 7], dtype=np.int64),
        spec=_spec(),
        signal="events",
    )
    assert list(series_frame(series)["time_us"]) == [0, 7]


def test_an_ordinal_series_has_no_timeline():
    series = TimeSeries.from_values(
        np.array([1.0, 2.0], dtype=np.float32),
        spec=_spec(),
        signal="tokens",
        time_axis=OrdinalAxis(),
    )
    assert series_frame(series)["time_us"].isna().all()


def test_an_n_dimensional_series_spreads_over_one_column_per_component():
    values = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    series = TimeSeries(
        spec=_spec(value_shape=(2,)),
        signal="xy",
        time_axis=RegularAxis.from_rate_hz(1),
        loader=lambda: pa.FixedShapeTensorArray.from_numpy_ndarray(values),
        n_values=2,
        time_series_id="xy",
    )
    frame = series_frame(series)
    assert list(frame.columns) == ["time_us", "value_0", "value_1"]
    assert list(frame["value_1"]) == [2.0, 4.0]


def test_the_frame_keeps_the_stored_dtype():
    series = TimeSeries.from_values(
        np.array([1, 2, 3], dtype=np.int16),
        spec=_spec(dtype="int16"),
        signal="raw",
        time_axis=RegularAxis.from_rate_hz(1),
    )
    assert series_frame(series)["value"].dtype == np.int16


def test_a_record_becomes_one_row_with_a_column_per_signal():
    dataset = make_dataset()
    record = dataset.records[0]
    frame = record_array_frame(record)
    assert len(frame) == 1
    assert list(frame.columns)[: len(ARRAY_META_COLUMNS)] == list(ARRAY_META_COLUMNS)
    assert list(frame.columns)[len(ARRAY_META_COLUMNS) :] == [ts.signal for ts in record.time_series]
    cell = frame.iloc[0][record.time_series[0].signal]
    assert list(cell) == pytest.approx(list(record.time_series[0].to_numpy()))


def test_a_cell_keeps_the_stored_dtype_and_the_signals_stay_ragged():
    # Array cells put two signals of different length and dtype in one row, with no cast.
    fast = TimeSeries.from_values(
        np.ones(100, dtype=np.float32),
        spec=_spec(),
        signal="eeg",
        time_axis=RegularAxis.from_rate_hz(100),
        time_series_id="fast",
    )
    slow = TimeSeries.from_values(
        np.ones(3, dtype=np.int16),
        spec=_spec(dtype="int16"),
        signal="temp",
        time_axis=RegularAxis.from_rate_hz(1),
        time_series_id="slow",
    )
    dataset = _dataset("timenet/rates", "Two signals at 100 Hz and 1 Hz.")
    record = dataset.add_record(time_series=(fast, slow), record_id="night-0")
    row = record_array_frame(record).iloc[0]
    # 100 and 3, not 100 and 100: nothing is padded or resampled to the record's highest rate.
    assert (len(row["eeg"]), len(row["temp"])) == (100, 3)
    assert (row["eeg"].dtype, row["temp"].dtype) == (np.float32, np.int16)


def test_the_axis_cell_names_every_signal_column():
    dataset = make_dataset()
    record = dataset.records[0]
    axes = record_array_frame(record).iloc[0]["time_axis"]
    assert set(axes) == {ts.signal for ts in record.time_series}
    assert axes[record.time_series[0].signal] == record.time_series[0].time_axis


def test_a_repeated_signal_name_carries_its_series_id():
    # One record can hold the same signal twice, at two rates. Two columns need two names.
    pair = tuple(
        TimeSeries.from_values(
            np.ones(4, dtype=np.float32),
            spec=_spec(),
            signal="II",
            time_axis=RegularAxis.from_rate_hz(rate),
            time_series_id=f"lead-{rate}",
        )
        for rate in (100, 500)
    )
    dataset = _dataset("timenet/repeat", "One signal name twice, at two rates.")
    record = dataset.add_record(time_series=pair, record_id="r")
    columns = list(record_array_frame(record).columns)[len(ARRAY_META_COLUMNS) :]
    assert columns == ["II (lead-100)", "II (lead-500)"]


def test_frames_stream_off_a_reader_without_materializing_the_dataset(tmp_path: Path):
    dataset = make_dataset()
    with TimeFWriter(tmp_path, dataset) as writer:
        writer.write()
    root = tmp_path / dataset.metadata.dataset_id / str(dataset.metadata.dataset_version)

    with TimeFReader(DatasetVersion.open_local(root)) as reader:
        streamed = dict(iter_record_frames(reader))

    assert set(streamed) == {record.record_id for record in dataset.records}
    first = dataset.records[0].time_series[0]
    cell = streamed["record-0"].iloc[0][first.signal]
    assert list(cell) == pytest.approx(list(first.to_numpy()))


def test_streaming_one_record_reads_only_that_record(tmp_path: Path):
    dataset = make_dataset()
    with TimeFWriter(tmp_path, dataset) as writer:
        writer.write()
    root = tmp_path / dataset.metadata.dataset_id / str(dataset.metadata.dataset_version)

    with TimeFReader(DatasetVersion.open_local(root)) as reader:
        streamed = list(iter_record_frames(reader, ["record-2"]))

    assert [record_id for record_id, _ in streamed] == ["record-2"]


def test_a_record_without_a_series_has_no_frame():
    class Empty:
        record_id = "r"
        time_series = ()

    with pytest.raises(TimeFValidationError, match="holds no series"):
        record_array_frame(Empty())  # ty: ignore[invalid-argument-type]
