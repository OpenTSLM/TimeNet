"""Round-trips for a series whose time offsets are stored rather than computed (TimeF cases 3 and 4)."""

import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from timenet.dataset import TimeFDataset, TimeSeries
from timenet.dataset.axis import IrregularAxis, OrdinalAxis, RegularAxis
from timenet.dataset.dataset import View
from timenet.errors import TimeFFormatError, TimeFValidationError
from timenet.reader import TimeFReader
from timenet.types import DatasetMetadata, License, TimeSeriesSpec, Version, ureg
from timenet.writer import TimeFWriter


def _spec(spec_type="env"):
    return TimeSeriesSpec(
        spec_type=spec_type,
        name=spec_type,
        unit_value=ureg.dimensionless,
    )


def _dataset():
    return TimeFDataset(
        metadata=DatasetMetadata(
            dataset_id="demo/irregular",
            dataset_version=Version(1, 0, 0),
            name="Irregular",
            description="d",
            license=License.MIT,
        )
    )


def _written(tmp_path, dataset, **kwargs):
    with TimeFWriter(tmp_path, dataset, **kwargs) as writer:
        writer.write()
    return next(tmp_path.rglob("manifest.json")).parent


# Machine hatch sensor: a PLC row only on state change, gaps from seconds to hours.
_HATCH_US = [0, 1_655_412_000, 6_486_006_000, 6_527_881_000]
_HATCH_VALUES = [1.0, 0.0, 1.0, 0.0]


@pytest.mark.parametrize("values_backend", ["parquet", "zarr"])
def test_an_irregular_series_round_trips(tmp_path, values_backend):
    dataset = _dataset()
    ts = TimeSeries.from_irregular(_HATCH_VALUES, time_offsets_us=_HATCH_US, spec=_spec(), channel="hatch")
    dataset.add_sample(time_series=(ts,), view=View.WINDOW)
    dataset.derive_schema()

    reader = TimeFReader(_written(tmp_path, dataset, values_backend=values_backend))
    back = next(iter(reader.iter_samples())).time_series[0]
    assert back.time_axis == IrregularAxis(first_us=0, last_us=6_527_881_000)
    assert back.time_offsets_us().tolist() == _HATCH_US
    assert back.to_numpy().tolist() == _HATCH_VALUES


def test_the_window_comes_from_the_stored_endpoints():
    ts = TimeSeries.from_irregular(_HATCH_VALUES, time_offsets_us=_HATCH_US, spec=_spec(), channel="hatch")
    # One microsecond past the last time_offset: no cadence means no next time offset to end on.
    assert ts.span_us == (0, 6_527_881_001)


def test_a_mixed_sample_keeps_each_series_on_its_own_axis(tmp_path):
    # Temperature and humidity written together on change, a regular signal, and a second irregular
    # signal on a different clock. One sample, four axes, one shared timeline.
    dataset = _dataset()
    shared_us = [0, 1_655_412_000, 6_486_006_000]
    series = (
        TimeSeries.from_irregular([21.3, 21.4, 21.9], time_offsets_us=shared_us, spec=_spec(), channel="temp"),
        TimeSeries.from_irregular([55.1, 55.3, 54.8], time_offsets_us=shared_us, spec=_spec(), channel="humidity"),
        TimeSeries.from_values(
            np.arange(500, dtype=np.float32),
            spec=_spec("vib"),
            channel="z",
            time_axis=RegularAxis.from_rate_hz(500),
        ),
        TimeSeries.from_irregular(_HATCH_VALUES, time_offsets_us=_HATCH_US, spec=_spec("mach"), channel="hatch"),
    )
    dataset.add_sample(time_series=series, view=View.WINDOW)
    dataset.derive_schema()

    reader = TimeFReader(_written(tmp_path, dataset))
    back = {ts.channel: ts for ts in next(iter(reader.iter_samples())).time_series}
    assert back["temp"].time_offsets_us().tolist() == shared_us
    assert back["humidity"].time_offsets_us().tolist() == shared_us
    assert back["hatch"].time_offsets_us().tolist() == _HATCH_US
    assert back["z"].time_axis == RegularAxis.from_rate_hz(500)
    # The regular series computes its time offsets and stores none, in the same shard as those that do.
    assert back["z"].time_offsets_loader is None
    assert back["z"].span_us == (0, 1_000_000)


def test_series_sharing_time_offsets_each_store_their_own_copy(tmp_path):
    # Documented behaviour, not an accident: an axis is a value, repeated per series, exactly as two
    # 500 Hz channels each carry their own RegularAxis. Writer-side dedupe of identical streams stays
    # available later without a schema change, since two index rows may name one chunk locator.
    dataset = _dataset()
    shared_us = [0, 5_000, 9_000]
    for channel in ("temp", "humidity"):
        dataset_series = TimeSeries.from_irregular(
            [1.0, 2.0, 3.0], time_offsets_us=shared_us, spec=_spec(), channel=channel
        )
        dataset.add_sample(time_series=(dataset_series,), view=View.WINDOW)
    dataset.derive_schema()

    reader = TimeFReader(_written(tmp_path, dataset))
    streams = [s.time_series[0].time_offsets_us().tolist() for s in reader.iter_samples()]
    assert streams == [shared_us, shared_us]


def test_chunking_splits_values_and_time_offsets_at_the_same_boundary(tmp_path):
    # The whole reason one chunk locator can address both columns.
    dataset = _dataset()
    time_offsets = list(range(0, 4000 * 1000, 1000))
    ts = TimeSeries.from_irregular(
        np.arange(4000, dtype=np.float32), time_offsets_us=time_offsets, spec=_spec(), channel="ibi"
    )
    dataset.add_sample(time_series=(ts,), view=View.WINDOW)
    dataset.derive_schema()

    # 480 bytes per chunk => 40 steps per chunk at 12 bytes a step, so ~100 chunks.
    reader = TimeFReader(_written(tmp_path, dataset, chunk_max_bytes=480))
    back = next(iter(reader.iter_samples())).time_series[0]
    assert back.time_offsets_us().tolist() == time_offsets
    assert back.to_numpy().tolist() == list(range(4000))


def test_an_irregular_series_needs_its_time_offsets():
    with pytest.raises(TimeFValidationError, match="must carry time_offsets_loader"):
        TimeSeries(
            spec=_spec(),
            channel="c",
            time_axis=IrregularAxis(first_us=0, last_us=1),
            loader=lambda: None,
            n_values=2,
        )


@pytest.mark.parametrize("axis", [RegularAxis.from_rate_hz(1), OrdinalAxis()])
def test_only_an_irregular_series_may_carry_time_offsets(axis):
    with pytest.raises(TimeFValidationError, match="only for an IrregularAxis"):
        TimeSeries(
            spec=_spec(),
            channel="c",
            time_axis=axis,
            loader=lambda: None,
            time_offsets_loader=lambda: None,
            n_values=2,
        )


def test_time_offsets_must_match_the_value_count():
    with pytest.raises(TimeFValidationError, match="one time offset per value"):
        TimeSeries.from_irregular([1.0, 2.0], time_offsets_us=[0, 1, 2], spec=_spec(), channel="c")


def test_a_stream_disagreeing_with_its_axis_is_refused(tmp_path):
    # first_time_offset_us / last_time_offset_us are verified metadata: the writer checks the claim against
    # the stream rather than trusting it.
    dataset = _dataset()
    values = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    ts = TimeSeries(
        spec=_spec(),
        channel="c",
        time_axis=IrregularAxis(first_us=0, last_us=999),  # the stream ends at 20
        loader=lambda: pa.array(values),
        time_offsets_loader=lambda: pa.array(np.array([0, 10, 20], dtype=np.int64)),
        n_values=3,
    )
    dataset.add_sample(time_series=(ts,), view=View.WINDOW)
    dataset.derive_schema()
    with pytest.raises(TimeFValidationError, match="its axis claims the stream runs"):
        _written(tmp_path, dataset)


@pytest.mark.parametrize("chunk_max_bytes", [64, 480, 1000, 4096])
def test_zarr_time_offsets_survive_a_tuned_chunk_size(tmp_path, chunk_max_bytes):
    # Zarr requires the shard shape to be a multiple of the chunk shape. The time offsets array has its
    # own byte width, so it needs the same rounding the values array does; without it every byte
    # target where the two do not divide aborts create_array.
    dataset = _dataset()
    ts = TimeSeries.from_irregular(_HATCH_VALUES, time_offsets_us=_HATCH_US, spec=_spec(), channel="hatch")
    dataset.add_sample(time_series=(ts,))
    dataset.derive_schema()

    version = _written(tmp_path, dataset, values_backend="zarr", chunk_max_bytes=chunk_max_bytes)
    back = next(iter(TimeFReader(version).iter_samples())).time_series[0]
    assert back.time_offsets_us().tolist() == _HATCH_US


def test_zarr_keeps_regular_and_irregular_arrays_apart(tmp_path):
    # The partition is what lets one element offset address both a partition's values and its
    # time_offsets: mixing the two in one array would let the offsets drift apart silently.
    dataset = _dataset()
    dataset.add_sample(
        time_series=(
            TimeSeries.from_values(
                np.arange(8, dtype=np.float32),
                spec=_spec(),
                channel="steady",
                time_axis=RegularAxis.from_rate_hz(2),
            ),
            TimeSeries.from_irregular(_HATCH_VALUES, time_offsets_us=_HATCH_US, spec=_spec(), channel="hatch"),
        ),
        view=View.WINDOW,
    )
    dataset.derive_schema()
    version = _written(tmp_path, dataset, values_backend="zarr")

    store = version / "time_series.zarr"
    assert (store / "env").is_dir()  # regular values keep today's top-level path
    assert (store / "_irregular" / "env").is_dir()
    assert (store / "_time_offsets" / "env").is_dir()

    back = {ts.channel: ts for ts in next(iter(TimeFReader(version).iter_samples())).time_series}
    assert back["hatch"].time_offsets_us().tolist() == _HATCH_US
    assert back["steady"].to_numpy().tolist() == list(range(8))
    assert back["steady"].time_offsets_loader is None


def test_a_spec_type_cannot_collide_with_the_zarr_groups():
    # quote() leaves underscores alone, so these would land on the groups the writer owns.
    for reserved in ("_irregular", "_time_offsets"):
        with pytest.raises(TimeFValidationError, match="must not be one of"):
            _spec(reserved)


def test_reader_rejects_time_offsets_disagreeing_with_the_stored_axis(tmp_path):
    # The writer verifies endpoints, but a corrupt shard could hand back a stream whose last time offset
    # no longer matches the axis. The reader re-checks on read and refuses it.
    dataset = _dataset()
    ts = TimeSeries.from_irregular(_HATCH_VALUES, time_offsets_us=_HATCH_US, spec=_spec(), channel="hatch")
    dataset.add_sample(time_series=(ts,), view=View.WINDOW)
    dataset.derive_schema()
    version_dir = _written(tmp_path, dataset)
    samples_path = version_dir / "samples.parquet"
    table = pq.read_table(samples_path)
    rows = table.to_pylist()
    rows[0]["time_series"][0]["last_time_offset_us"] = _HATCH_US[-1] + 1_000  # axis now disagrees with the stream
    pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), samples_path)
    back = next(iter(TimeFReader(version_dir).iter_samples())).time_series[0]
    with pytest.raises(TimeFFormatError, match="disagreeing with its axis endpoints"):
        back.time_offsets_us()


def test_zarr_time_offsets_persist_the_delta_filter(tmp_path):
    # The time offsets array claims a Delta filter for its monotonic stream; confirm it lands in the
    # written zarr.json rather than trusting the create_array call that requested it.
    dataset = _dataset()
    ts = TimeSeries.from_irregular(_HATCH_VALUES, time_offsets_us=_HATCH_US, spec=_spec(), channel="hatch")
    dataset.add_sample(time_series=(ts,))
    dataset.derive_schema()
    version = _written(tmp_path, dataset, values_backend="zarr")

    meta = json.loads((version / "time_series.zarr" / "_time_offsets" / "env" / "zarr.json").read_text())
    inner = meta["codecs"][0]["configuration"]["codecs"]  # inside the sharding codec
    assert any(codec["name"] == "numcodecs.delta" for codec in inner)
