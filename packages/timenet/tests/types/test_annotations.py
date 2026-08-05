from dataclasses import dataclass
import pickle

import pytest

from timenet.errors import TimeFValidationError
from timenet.types import (
    ANNOTATION_BASES,
    Annotation,
    AnnotationDescriptor,
    AnnotationType,
    IntervalAnnotation,
    IntervalSpan,
    PointAnnotation,
    PointSpan,
    StaticAnnotation,
    annotation_type_of,
    ureg,
)


def test_static_requires_value():
    with pytest.raises(TimeFValidationError, match="requires a value"):
        StaticAnnotation(key="age")  # type: ignore[call-arg]


def test_static_with_value():
    ann = StaticAnnotation(key="age", value=64, unit="years")
    assert ann.value == 64
    assert ann.unit == "years"  # a recognized unit string is kept as written


def test_unit_accepts_a_pint_unit_and_stores_its_canonical_name():
    assert StaticAnnotation(key="v", value=1.0, unit=ureg.millivolt).unit == "millivolt"


def test_unknown_unit_string_raises():
    with pytest.raises(ValueError, match="unknown unit"):
        StaticAnnotation(key="x", value=1, unit="foobar")


def test_point_is_a_pure_marker_by_default():
    ann = PointAnnotation(key="stimulus", span=PointSpan.seconds(4.0))
    assert ann.value is None
    assert ann.span.start == 4_000_000
    assert ann.span.time_series_ids is None


def test_interval_requires_end_after_start():
    with pytest.raises(TimeFValidationError):
        IntervalAnnotation(key="artifact", span=IntervalSpan.seconds(10.0, 8.0))
    ann = IntervalAnnotation(key="artifact", span=IntervalSpan.seconds(10.0, 12.0))
    assert ann.span.end == 12_000_000


def test_point_rejects_empty_time_series_ids():
    with pytest.raises(TimeFValidationError, match="time_series_ids"):
        PointAnnotation(key="stimulus", span=PointSpan.seconds(4.0, time_series_ids=()))


def test_interval_rejects_empty_time_series_ids():
    with pytest.raises(TimeFValidationError, match="time_series_ids"):
        IntervalAnnotation(key="artifact", span=IntervalSpan.seconds(1.0, 2.0, time_series_ids=()))


def test_annotation_type_of():
    assert annotation_type_of(StaticAnnotation(key="a", value=1)) is AnnotationType.STATIC
    assert annotation_type_of(PointAnnotation(key="a", span=PointSpan.seconds(1.0))) is AnnotationType.POINT
    assert (
        annotation_type_of(IntervalAnnotation(key="a", span=IntervalSpan.seconds(1.0, 2.0))) is AnnotationType.INTERVAL
    )


def test_annotation_bases_map():
    assert ANNOTATION_BASES[AnnotationType.STATIC] is StaticAnnotation
    assert ANNOTATION_BASES[AnnotationType.POINT] is PointAnnotation
    assert ANNOTATION_BASES[AnnotationType.INTERVAL] is IntervalAnnotation


def test_auto_id_unique():
    a = StaticAnnotation(key="age", value=1)
    b = StaticAnnotation(key="age", value=1)
    assert a.id != b.id


def test_sequence_value_canonicalized_to_list():
    # value_type is "list" for any sequence, and the manifest stores it as a JSON array that decodes
    # back to a list; normalizing at construction keeps an annotation equal to its read-back form.
    ann = StaticAnnotation(key="labels", value=("a", "b"))
    assert ann.value == ["a", "b"]
    assert isinstance(ann.value, list)


def test_frozen():
    ann = StaticAnnotation(key="age", value=1)
    with pytest.raises(AttributeError):
        ann.value = 2  # ty: ignore[invalid-assignment]


def test_picklable():
    ann = IntervalAnnotation(key="artifact", span=IntervalSpan.seconds(1.0, 2.0, time_series_ids=("s1",)))
    assert pickle.loads(pickle.dumps(ann)) == ann


def test_subclass_with_field_defaults_authoring():
    @dataclass(frozen=True, kw_only=True)
    class Age(StaticAnnotation):
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
