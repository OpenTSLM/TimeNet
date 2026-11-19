from fractions import Fraction

import pyarrow as pa
import pytest

from timenet.dataset import Record, RegularAxis, Signal, Source, TimeFDataset
from timenet.errors import TimeFValidationError
from timenet.testing import make_dataset
from timenet.types import Annotation, TimeSeriesSpec, ureg


SPEC = TimeSeriesSpec(spec_type="acceleration", name="Acceleration", unit_value=ureg.Unit("m/s^2"))


def _signal(name: str) -> Signal:
    return Signal.from_loader(
        spec=SPEC,
        name=name,
        time_axis=RegularAxis(period_us=Fraction(1_000)),
        loader=lambda: pa.array([1.0], type=pa.float32()),
        n_values=1,
    )


def test_record_walks_recursive_sources_and_signals():
    accelerometer = Source(name="Accelerometer", signals=(_signal("X"), _signal("Y"), _signal("Z")))
    gyroscope = Source(name="Gyroscope", signals=(_signal("GX"),))
    imu = Source(name="IMU", sources=(gyroscope, accelerometer))
    record = Record(sources=(imu,))

    assert [source.name for source in record.walk_sources()] == ["IMU", "Accelerometer", "Gyroscope"]
    assert [signal.name for signal in record.walk_signals()] == ["X", "Y", "Z", "GX"]


def test_record_rejects_source_reused_under_two_parents():
    shared = Source(name="Shared")
    left = Source(name="Left", sources=(shared,))
    right = Source(name="Right", sources=(shared,))

    with pytest.raises(TimeFValidationError, match="attached more than once"):
        Record(sources=(Source(name="Root", sources=(left, right)),))


def test_record_rejects_source_cycle():
    root = Source(name="Root")
    child = Source(name="Child", sources=(root,))
    root.sources = (child,)

    with pytest.raises(TimeFValidationError, match="cycle"):
        Record(sources=(root,))


def test_a_source_rejects_child_sources_of_one_name():
    with pytest.raises(TimeFValidationError, match="source 'IMU' holds more than one Source named 'Gyroscope'"):
        Source(name="IMU", sources=(Source(name="Gyroscope"), Source(name="Accelerometer"), Source(name="Gyroscope")))


def test_a_record_rejects_root_sources_of_one_name():
    with pytest.raises(TimeFValidationError, match="record holds more than one Source named 'IMU'"):
        Record(sources=(Source(name="IMU"), Source(name="IMU")))
    # Sources of one name below different owners are not siblings.
    record = Record(sources=(Source(name="Left", sources=(Source(name="IMU"),)), Source(name="IMU")))
    assert [source.name for source in record.walk_sources()] == ["IMU", "Left", "IMU"]


def _empty_dataset():
    return TimeFDataset(metadata=make_dataset().metadata)


def test_add_record_rejects_sibling_sources_of_one_name_that_a_caller_assigned():
    imu = Source(name="IMU", signals=(_signal("X"),))
    record = Record(sources=(imu,))
    dataset = _empty_dataset()
    imu.sources = (Source(name="Gyroscope"), Source(name="Gyroscope"))
    with pytest.raises(TimeFValidationError, match="source 'IMU' holds more than one Source named 'Gyroscope'"):
        dataset.add_record(record=record)
    imu.sources = ()
    record.sources = (imu, Source(name="IMU"))
    with pytest.raises(TimeFValidationError, match="record holds more than one Source named 'IMU'"):
        dataset.add_record(record=record)
    assert (dataset.records, record._id, imu.signals[0]._id) == ((), None, None)


def test_collection_fields_take_any_iterable_and_store_a_tuple():
    signals = [_signal("X"), _signal("Y")]
    gyroscope = Source("Gyroscope")
    source = Source("IMU", signals=(signal for signal in signals), sources=[gyroscope])
    record = Record(sources=[source], annotations=[Annotation("site", value="lab")])

    assert source.name == "IMU"
    assert source.signals == tuple(signals)
    assert source.sources == (gyroscope,)
    assert record.sources == (source,)
    assert isinstance(record.annotations, tuple)
    record.sources = [source]
    assert record.sources == (source,)
    annotated = Signal.from_loader(
        spec=SPEC,
        name="Z",
        time_axis=RegularAxis(period_us=Fraction(1_000)),
        loader=lambda: pa.array([1.0], type=pa.float32()),
        n_values=1,
        annotations=[Annotation("site", value="lab")],
    )
    assert isinstance(annotated.annotations, tuple)


@pytest.mark.parametrize("value", ["IMU", {"IMU": 1}, Source("IMU")])
def test_a_collection_field_rejects_a_string_a_mapping_and_a_single_item(value):
    with pytest.raises(TimeFValidationError, match="takes an iterable"):
        Record(sources=value)
    with pytest.raises(TimeFValidationError, match="takes an iterable"):
        Source("Parent", sources=value)


@pytest.mark.parametrize("annotations", ["ab", {"a": 1}, Annotation("site", value="lab")])
def test_a_signal_rejects_annotations_that_are_not_an_iterable_of_items(annotations):
    with pytest.raises(TimeFValidationError, match="takes an iterable"):
        Signal.from_loader(
            spec=SPEC,
            name="Z",
            time_axis=RegularAxis(period_us=Fraction(1_000)),
            loader=lambda: pa.array([1.0], type=pa.float32()),
            n_values=1,
            annotations=annotations,
        )


def test_only_the_source_name_is_positional():
    with pytest.raises(TypeError):
        Source("IMU", ())  # ty: ignore[too-many-positional-arguments]
