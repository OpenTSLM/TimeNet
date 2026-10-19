from fractions import Fraction

import pytest

from timenet.dataset import OrdinalAxis, Record, RegularAxis, Signal, Source, TimeFDataset
from timenet.errors import TimeFValidationError
from timenet.format.duckdb import connect_control
from timenet.types import Annotation, DatasetMetadata, License, TimeSeriesSpec, Version, ureg


SPEC = TimeSeriesSpec(spec_type="s", name="S", unit_value=ureg.dimensionless)


def _dataset():
    metadata = DatasetMetadata(
        dataset_id="timenet/dedupe",
        dataset_version=Version(1, 0, 0),
        name="Dedupe",
        description="test",
        license=License.CC_BY_4_0,
    )
    return TimeFDataset(metadata=metadata)


def _record(axis=None, *annotations):
    axis = axis or RegularAxis(period_us=Fraction(1000))
    signal = Signal(spec=SPEC, name="a", time_axis=axis, data=[1.0, 2.0])
    record = Record(sources=(Source(name="device", signals=(signal,)),))
    for annotation in annotations:
        record.annotate(annotation)
    return record


def test_equal_contents_share_one_content_id():
    dataset = _dataset()
    first = dataset.add_record(record=_record(None, Annotation(key="sex", value="f")))
    second = dataset.add_record(record=_record(None, Annotation(key="sex", value="f")))
    third = dataset.add_record(record=_record(None, Annotation(key="sex", value="m")))

    assert first.annotations[0].id == second.annotations[0].id
    assert third.annotations[0].id != first.annotations[0].id
    assert first.annotations[0].occurrence_id != second.annotations[0].occurrence_id


@pytest.mark.parametrize(
    ("left", "right", "shared"),
    [
        (0.0, -0.0, True),
        (float("nan"), float("nan"), True),
        (True, 1, False),
        (1, 1.0, False),
        ("a", ["a"], False),
    ],
)
def test_content_equality_follows_the_stored_value(left, right, shared):
    dataset = _dataset()
    first = dataset.annotate(Annotation(key="k", value=left))
    second = dataset.annotate(Annotation(key="k", value=right))

    assert (first.id == second.id) is shared


def test_equal_regular_and_ordinal_axes_share_one_axis_id():
    dataset = _dataset()
    regular = [dataset.add_record(record=_record(RegularAxis(period_us=Fraction(1000)))) for _ in range(2)]
    other = dataset.add_record(record=_record(RegularAxis(period_us=Fraction(2000))))
    ordinal = [dataset.add_record(record=_record(OrdinalAxis())) for _ in range(2)]

    assert regular[0].signals[0].time_axis.id == regular[1].signals[0].time_axis.id
    assert other.signals[0].time_axis.id != regular[0].signals[0].time_axis.id
    assert ordinal[0].signals[0].time_axis.id == ordinal[1].signals[0].time_axis.id


def test_irregular_axes_keep_their_own_ids():
    dataset = _dataset()
    signals = [
        Signal.from_irregular([1.0, 2.0, 3.0], time_offsets_us=offsets, spec=SPEC, name="a")
        for offsets in ([0, 100, 1000], [0, 900, 1000])
    ]
    for signal in signals:
        dataset.add_record(record=Record(sources=(Source(name="device", signals=(signal,)),)))

    assert signals[0].time_axis == signals[1].time_axis
    assert signals[0].time_axis.id != signals[1].time_axis.id


def test_an_explicit_content_id_with_a_different_payload_is_rejected():
    dataset = _dataset()
    known = dataset.annotate(Annotation(key="sex", value="f"))

    with pytest.raises(TimeFValidationError, match="already used by different content"):
        dataset.annotate(Annotation(key="sex", value="m", id=known.id))


def test_shared_ids_are_stored_once_and_read_back(tmp_path):
    dataset = _dataset()
    for _ in range(3):
        dataset.add_record(record=_record(None, Annotation(key="sex", value="f")))
    version = dataset.write(path=tmp_path)

    with connect_control(version / "control.duckdb") as connection:
        counts = connection.execute(
            "SELECT (SELECT count(*) FROM axes), (SELECT count(*) FROM annotation_contents),"
            " (SELECT count(*) FROM annotation_occurrences)"
        ).fetchone()
    assert counts == (1, 1, 3)
    back = TimeFDataset.open(path=version)
    assert {record.annotations[0].id for record in back.records} == {dataset.records[0].annotations[0].id}
