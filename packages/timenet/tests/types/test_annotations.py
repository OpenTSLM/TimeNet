from dataclasses import dataclass
import pickle

import pytest

from timenet.types import (
    ANNOTATION_BASES,
    Annotation,
    AnnotationDescriptor,
    AnnotationType,
    IntervalAnnotation,
    PointAnnotation,
    StaticAnnotation,
    annotation_type_of,
)


def test_static_requires_value():
    with pytest.raises((TypeError, ValueError)):
        StaticAnnotation(key="age")  # type: ignore[call-arg]


def test_static_with_value():
    ann = StaticAnnotation(key="age", value=64, unit="years")
    assert ann.value == 64
    assert ann.unit == "years"


def test_point_is_a_pure_marker_by_default():
    ann = PointAnnotation(key="stimulus", start_time_s=4.0)
    assert ann.value is None
    assert ann.start_time_s == pytest.approx(4.0)
    assert ann.time_series_ids is None


def test_interval_requires_end_after_start():
    with pytest.raises(ValueError):
        IntervalAnnotation(key="artifact", start_time_s=10.0, end_time_s=8.0)
    ann = IntervalAnnotation(key="artifact", start_time_s=10.0, end_time_s=12.0)
    assert ann.end_time_s == pytest.approx(12.0)


def test_annotation_type_of():
    assert annotation_type_of(StaticAnnotation(key="a", value=1)) is AnnotationType.STATIC
    assert annotation_type_of(PointAnnotation(key="a", start_time_s=1.0)) is AnnotationType.POINT
    assert annotation_type_of(IntervalAnnotation(key="a", start_time_s=1.0, end_time_s=2.0)) is AnnotationType.INTERVAL


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
    ann = IntervalAnnotation(key="artifact", start_time_s=1.0, end_time_s=2.0, time_series_ids=("s1",))
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
