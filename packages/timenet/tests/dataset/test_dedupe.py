from fractions import Fraction

import pytest

from timenet.dataset import OrdinalAxis, Record, RegularAxis, Signal, Source, TimeFDataset
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


def _stored(dataset, root):
    """Write ``dataset`` and read back the content id of each record's annotation and its axis id."""
    back = TimeFDataset.open(path=dataset.write(path=root))
    return [
        (record.annotations[0]._content_id if record.annotations else None, record.signals[0].time_axis._stored_id)
        for record in back.records
    ]


def test_equal_contents_share_one_stored_content(tmp_path):
    dataset = _dataset()
    for value in ("f", "f", "m"):
        dataset.add_record(record=_record(None, Annotation(key="sex", value=value)))

    first, second, third = (content_id for content_id, _ in _stored(dataset, tmp_path))

    assert first == second != third


@pytest.mark.parametrize(
    ("left", "right", "shared"),
    [
        ({"value": 0.0}, {"value": -0.0}, True),
        ({"value": float("nan")}, {"value": float("nan")}, True),
        ({"value": True}, {"value": 1}, False),
        ({"value": ["a", "b"]}, {"value": ["a", "b"]}, True),
        ({"value": "a", "metadata": {"x": 1}}, {"value": "a", "metadata": {"x": 2}}, False),
    ],
)
def test_content_equality_follows_the_stored_value(left, right, shared):
    first = Annotation(key="k", **left)
    second = Annotation(key="k", **right)

    assert (first._content_identity == second._content_identity) is shared


def test_equal_regular_and_ordinal_axes_share_one_stored_axis(tmp_path):
    dataset = _dataset()
    axes = (RegularAxis(period_us=Fraction(1000)),) * 2 + (RegularAxis(period_us=Fraction(2000)),)
    for axis in (*axes, OrdinalAxis(), OrdinalAxis()):
        dataset.add_record(record=_record(axis))

    regular, same, other, ordinal, same_ordinal = (axis_id for _, axis_id in _stored(dataset, tmp_path))

    assert regular == same != other
    assert ordinal == same_ordinal


def test_irregular_axes_keep_their_own_rows(tmp_path):
    dataset = _dataset()
    signals = [
        Signal.from_irregular([1.0, 2.0, 3.0], time_offsets_us=offsets, spec=SPEC, name="a")
        for offsets in ([0, 100, 1000], [0, 900, 1000])
    ]
    for signal in signals:
        dataset.add_record(record=Record(sources=(Source(name="device", signals=(signal,)),)))

    first, second = (axis_id for _, axis_id in _stored(dataset, tmp_path))

    assert signals[0].time_axis == signals[1].time_axis
    assert first != second


def test_shared_rows_are_stored_once_and_kept_after_reopening(tmp_path):
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
    back.add_record(record=_record(None, Annotation(key="sex", value="f")))
    edited = TimeFDataset.from_parts(
        metadata=back.metadata.model_copy(update={"dataset_version": Version(1, 1, 0)}),
        records=back.records,
        tasks=(),
        schema=back.derive_schema(),
        last_ids=back.last_ids,
    )
    stored = _stored(edited, tmp_path)
    assert stored[-1] == stored[0]
