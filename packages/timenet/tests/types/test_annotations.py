from dataclasses import dataclass
import pickle

import pytest

from timenet.errors import TimeFValidationError
from timenet.types import (
    Annotation,
    AnnotationDescriptor,
    AnnotationType,
    TimeInterval,
    TimePoint,
    annotation_type_of,
    ureg,
)


def test_an_annotation_needs_a_value_or_a_span():
    # With neither, it carries a key and nothing else.
    with pytest.raises(TimeFValidationError, match="says nothing"):
        Annotation(key="age")


def test_a_span_alone_is_enough():
    # A pure marker: no payload, but it marks a region, which is information.
    assert Annotation(key="artifact", span=TimePoint.seconds(0.5)).value is None


def test_static_with_value():
    ann = Annotation(key="age", value=64, unit="years")
    assert ann.value == 64
    assert ann.unit == "years"  # a recognized unit string is kept as written


def test_unit_accepts_a_pint_unit_and_stores_its_canonical_name():
    assert Annotation(key="v", value=1.0, unit=ureg.millivolt).unit == "millivolt"


def test_unknown_unit_string_raises():
    with pytest.raises(ValueError, match="unknown unit"):
        Annotation(key="x", value=1, unit="foobar")


def test_point_is_a_pure_marker_by_default():
    ann = Annotation(key="stimulus", span=TimePoint.seconds(4.0))
    assert ann.value is None
    assert ann.span is not None
    assert ann.span.start_us == 4_000_000
    assert ann.span is not None
    assert ann.span.time_series_ids is None


def test_interval_requires_end_after_start():
    with pytest.raises(TimeFValidationError):
        Annotation(key="artifact", span=TimeInterval.seconds(10.0, 8.0))
    ann = Annotation(key="artifact", span=TimeInterval.seconds(10.0, 12.0))
    assert isinstance(ann.span, TimeInterval)
    assert ann.span.end_us == 12_000_000


def test_point_rejects_empty_time_series_ids():
    with pytest.raises(TimeFValidationError, match="time_series_ids"):
        Annotation(key="stimulus", span=TimePoint.seconds(4.0, time_series_ids=()))


def test_interval_rejects_empty_time_series_ids():
    with pytest.raises(TimeFValidationError, match="time_series_ids"):
        Annotation(key="artifact", span=TimeInterval.seconds(1.0, 2.0, time_series_ids=()))


def test_annotation_type_of():
    assert annotation_type_of(Annotation(key="a", value=1)) is AnnotationType.STATIC
    assert annotation_type_of(Annotation(key="a", span=TimePoint.seconds(1.0))) is AnnotationType.POINT
    assert annotation_type_of(Annotation(key="a", span=TimeInterval.seconds(1.0, 2.0))) is AnnotationType.INTERVAL


def test_auto_id_unique():
    a = Annotation(key="age", value=1)
    b = Annotation(key="age", value=1)
    assert a.id != b.id


def test_sequence_value_canonicalized_to_list():
    # value_type is "list" for any sequence, and the manifest stores it as a JSON array that decodes
    # back to a list; normalizing at construction keeps an annotation equal to its read-back form.
    ann = Annotation(key="labels", value=("a", "b"))
    assert ann.value == ["a", "b"]
    assert isinstance(ann.value, list)


def test_frozen():
    ann = Annotation(key="age", value=1)
    with pytest.raises(AttributeError):
        ann.value = 2  # ty: ignore[invalid-assignment]


def test_picklable():
    ann = Annotation(key="artifact", span=TimeInterval.seconds(1.0, 2.0, time_series_ids=("s1",)))
    assert pickle.loads(pickle.dumps(ann)) == ann


def test_subclass_with_field_defaults_authoring():
    @dataclass(frozen=True, kw_only=True)
    class Age(Annotation):
        key: str = "age"
        unit: str | None = "years"

    age = Age(value=64)
    assert age.key == "age"
    assert age.unit == "years"
    assert age.value == 64
    assert annotation_type_of(age) is AnnotationType.STATIC


def test_annotation_descriptor():
    d = AnnotationDescriptor(
        key="age",
        annotation_type=AnnotationType.STATIC,
        value_type="int",
        unit="years",
        description=None,
    )
    assert d.key == "age"
    assert d.annotation_type is AnnotationType.STATIC


def test_base_annotation_not_typeable():
    # The base class has no shape; annotation_type_of only recognizes the three subtypes.
    with pytest.raises(ValueError):
        annotation_type_of(Annotation(key="x"))


def test_span_must_be_a_span():
    with pytest.raises(TimeFValidationError, match="must be a TimePoint or a TimeInterval"):
        Annotation(key="bad", span="not-a-span")  # ty: ignore[invalid-argument-type]
