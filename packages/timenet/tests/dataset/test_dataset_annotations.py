from collections import UserList
from fractions import Fraction

import pytest

from timenet.dataset import Record, RegularAxis, Signal, Source, TimeFDataset
from timenet.errors import TimeFValidationError
from timenet.types import (
    Annotation,
    DatasetMetadata,
    License,
    TimePoint,
    TimeSeriesSpec,
    Version,
    read_value,
    ureg,
)


_SPEC = TimeSeriesSpec(spec_type="s", name="S", unit_value=ureg.dimensionless)


def _dataset() -> TimeFDataset:
    metadata = DatasetMetadata(
        dataset_id="timenet/annotations",
        dataset_version=Version(1, 0, 0),
        name="Annotations",
        description="test",
        license=License.CC_BY_4_0,
    )
    return TimeFDataset(metadata=metadata)


def _add_record(dataset: TimeFDataset) -> Record:
    signal = Signal(spec=_SPEC, name="a", time_axis=RegularAxis(period_us=Fraction(1000)), data=[1.0, 2.0])
    return dataset.add_record(record=Record(sources=(Source(name="device", signals=(signal,)),)))


def test_add_annotations_attaches_new_occurrences_in_order():
    dataset = _dataset()
    given = (Annotation(key="a", value=1), Annotation(key="b", value="x"))
    attached = dataset.add_annotations(given)
    assert [(one.key, one.value) for one in attached] == [("a", 1), ("b", "x")]
    assert all(new is not old for new, old in zip(attached, given, strict=True))
    assert dataset.annotations == attached


def test_add_annotations_attaches_nothing_if_one_fails():
    dataset = _dataset()
    dataset.annotate(Annotation(key="kept", value=1))
    batch = [Annotation(key="a", value=1), Annotation(key="b", value=2, span=TimePoint.seconds(1.0))]
    with pytest.raises(TimeFValidationError, match="cannot have a time span"):
        dataset.add_annotations(batch)
    assert [one.key for one in dataset.annotations] == ["kept"]


def test_add_annotations_takes_a_mapping_of_keys_to_values():
    dataset = _dataset()
    attached = dataset.add_annotations({"sampling_rate_hz": 50, "device": "wrist", "calibrated": True})
    assert [(one.key, read_value(one.value)) for one in attached] == [
        ("sampling_rate_hz", 50),
        ("device", "wrist"),
        ("calibrated", True),
    ]
    assert dataset.annotations == attached


def test_record_add_annotations_takes_a_mapping_of_keys_to_values():
    record = _add_record(_dataset())
    attached = record.add_annotations({"sensor_location": "wrist", "age": 54})
    assert [(one.key, read_value(one.value)) for one in record.annotations] == [
        ("sensor_location", "wrist"),
        ("age", 54),
    ]
    assert record.annotations == attached


@pytest.mark.parametrize(
    ("mapping", "message"),
    [({"a": 1, "b": {"nested": 1}}, "unsupported annotation value type"), ({"a": 1, 3: "x"}, "key must be text")],
)
def test_add_annotations_attaches_nothing_from_a_bad_mapping(mapping, message):
    dataset = _dataset()
    record = _add_record(dataset)
    with pytest.raises(TimeFValidationError, match=message):
        dataset.add_annotations(mapping)
    with pytest.raises(TimeFValidationError, match=message):
        record.add_annotations(mapping)
    assert dataset.annotations == ()
    assert record.annotations == ()


@pytest.mark.parametrize(
    ("batch", "message"),
    [("ab", "not str"), ([Annotation(key="a", value=1), "b"], "must be an Annotation; got str")],
)
def test_add_annotations_rejects_a_batch_of_other_objects(batch, message):
    dataset = _dataset()
    record = _add_record(dataset)
    with pytest.raises(TimeFValidationError, match=message):
        dataset.add_annotations(batch)
    with pytest.raises(TimeFValidationError, match=message):
        record.add_annotations(batch)
    assert dataset.annotations == ()
    assert record.annotations == ()


def test_add_annotations_takes_lists_and_tuples_of_text_but_not_other_sequences():
    dataset = _dataset()
    attached = dataset.add_annotations({"tags": ["a", "b"], "pair": ("c", "d")})
    assert [read_value(one.value) for one in attached] == [["a", "b"], ["c", "d"]]
    with pytest.raises(TimeFValidationError, match="unsupported annotation value type: UserList"):
        dataset.add_annotations({"tags": UserList(["a", "b"])})  # ty: ignore[invalid-argument-type]
    assert len(dataset.annotations) == 2
