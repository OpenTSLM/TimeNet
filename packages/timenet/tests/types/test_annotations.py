from dataclasses import dataclass, replace
import math
import pickle
from typing import Literal

from pydantic import ValidationError
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
    value_type_of,
)
from timenet.types.text import InlineText, InlineTextList


# A constant to reuse
HEALTHY_ANNOTATION = Annotation(key="condition", value="healthy")


def quality_annotation(level: Literal["low", "medium", "high"]) -> Annotation:
    return Annotation(key="quality", value=level)


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
    assert ann.span.time_series is None


def test_interval_requires_end_after_start():
    with pytest.raises(TimeFValidationError):
        Annotation(key="artifact", span=TimeInterval.seconds(10.0, 8.0))
    ann = Annotation(key="artifact", span=TimeInterval.seconds(10.0, 12.0))
    assert isinstance(ann.span, TimeInterval)
    assert ann.span.end_us == 12_000_000


def test_point_rejects_an_empty_time_series():
    with pytest.raises(TimeFValidationError, match="time_series"):
        Annotation(key="stimulus", span=TimePoint.seconds(4.0, time_series=()))


def test_interval_rejects_an_empty_time_series():
    with pytest.raises(TimeFValidationError, match="time_series"):
        Annotation(key="artifact", span=TimeInterval.seconds(1.0, 2.0, time_series=()))


def test_annotation_type_of():
    assert annotation_type_of(Annotation(key="a", value=1)) is AnnotationType.STATIC
    assert annotation_type_of(Annotation(key="a", span=TimePoint.seconds(1.0))) is AnnotationType.POINT
    assert annotation_type_of(Annotation(key="a", span=TimeInterval.seconds(1.0, 2.0))) is AnnotationType.INTERVAL


def test_sequence_value_canonicalized_to_list():
    # value_type is "list" for any sequence, and the manifest stores it as a JSON array that decodes
    # back to a list; normalizing at construction keeps an annotation equal to its read-back form.
    ann = Annotation(key="labels", value=("a", "b"))
    assert isinstance(ann.value, InlineTextList)
    assert ann.value.read() == ["a", "b"]


def test_frozen():
    ann = Annotation(key="age", value=1)
    with pytest.raises(AttributeError):
        ann.value = 2  # ty: ignore[invalid-assignment]


def test_picklable():
    ann = Annotation(key="artifact", span=TimeInterval.seconds(1.0, 2.0))
    assert pickle.loads(pickle.dumps(ann)) == ann


def test_presets_are_constants_or_builder_functions():
    # A constant is one piece of content: every occurrence shares its content key.
    assert type(HEALTHY_ANNOTATION) is Annotation
    assert isinstance(HEALTHY_ANNOTATION.value, InlineText)
    assert HEALTHY_ANNOTATION.value.read() == "healthy"
    first, second = HEALTHY_ANNOTATION._new_occurrence(), HEALTHY_ANNOTATION._new_occurrence()
    assert first == second == HEALTHY_ANNOTATION
    assert first._content_key == second._content_key == HEALTHY_ANNOTATION._content_key
    # A builder makes a new annotation on every call, with its value normalized like any other.
    # Equal arguments give equal content, which the writer stores once.
    high, low = quality_annotation("high"), quality_annotation("low")
    assert (high.key, type(high)) == ("quality", Annotation)
    assert isinstance(high.value, InlineText)
    assert high.value.read() == "high"
    assert quality_annotation("high")._content_key == high._content_key != low._content_key
    assert low.value == "low"


def test_annotation_cannot_be_subclassed():
    # A preset subclass that redeclared value would replace the field that wraps text in a handle.
    with pytest.raises(TypeError, match="Annotation can't be subclassed, so HealthyCondition is refused"):

        @dataclass(frozen=True, kw_only=True)
        class HealthyCondition(Annotation):  # ty: ignore[subclass-of-final-class]
            key: str = "condition"


def test_annotation_is_final_for_type_checkers():
    assert getattr(Annotation, "__final__", False) is True


def test_replace_and_pickle_keep_a_constant_annotation_intact():
    sick = replace(HEALTHY_ANNOTATION, value="sick")
    assert type(sick) is Annotation
    assert isinstance(sick.value, InlineText)
    assert sick.value.read() == "sick"
    restored = pickle.loads(pickle.dumps(HEALTHY_ANNOTATION))
    assert type(restored) is Annotation
    assert isinstance(restored.value, InlineText)
    assert restored == HEALTHY_ANNOTATION


def test_equal_text_gives_equal_content_keys():
    labels = Annotation(key="labels", value=["a", "b"])._content_key
    assert labels == Annotation(key="labels", value=("a", "b"))._content_key
    assert hash(labels) == hash(Annotation(key="labels", value=["a", "b"])._content_key)
    assert labels != Annotation(key="labels", value=["a", "c"])._content_key
    assert HEALTHY_ANNOTATION._content_key == Annotation(key="condition", value="healthy")._content_key
    assert HEALTHY_ANNOTATION._content_key != Annotation(key="condition", value="sick")._content_key
    # A string and a list holding only that string are different content.
    assert Annotation(key="x", value="a")._content_key != Annotation(key="x", value=["a"])._content_key


@pytest.mark.parametrize(
    ("first", "second", "equal"),
    [
        (True, 1, False),
        (1, 1.0, False),
        (0.0, -0.0, False),
        (float("nan"), -math.nan, True),
        (1.5, 1.5, True),
        ("1", 1, False),
    ],
)
def test_scalar_content_keys_compare_type_and_bits(first, second, equal):
    key = Annotation(key="x", value=first)._content_key
    assert (key == Annotation(key="x", value=second)._content_key) is equal


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


def test_annotation_descriptor_rejects_invalid_unit():
    with pytest.raises(ValidationError, match="unknown unit"):
        AnnotationDescriptor(key="x", annotation_type=AnnotationType.STATIC, unit="not_a_unit")


def test_annotation_descriptor_accepts_valid_unit():
    d = AnnotationDescriptor(key="x", annotation_type=AnnotationType.STATIC, unit="years")
    assert d.unit == "years"


def test_annotation_descriptor_normalizes_pint_unit_to_string():
    d = AnnotationDescriptor(key="x", annotation_type=AnnotationType.STATIC, unit=ureg.millivolt)  # ty: ignore[invalid-argument-type]
    assert d.unit == "millivolt"


def test_value_type_of_rejects_maps_and_non_string_lists():
    with pytest.raises(TimeFValidationError):
        value_type_of({"panas_pa": 30, "panas_na": 12})
    with pytest.raises(TimeFValidationError, match="strings only"):
        value_type_of([1, 2])


def test_value_type_of_rejects_unsupported_type():
    with pytest.raises(TimeFValidationError, match="unsupported annotation value type"):
        value_type_of(object())
