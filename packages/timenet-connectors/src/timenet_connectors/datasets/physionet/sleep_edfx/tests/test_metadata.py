from collections import Counter
from datetime import UTC, datetime, time

import pytest

from timenet.errors import TimeFValidationError
from timenet.types import US_PER_S
from timenet_connectors.datasets.physionet.sleep_edfx import metadata
from timenet_connectors.datasets.physionet.sleep_edfx.keys import AnnotationKey


# An invented recording id. It keeps the shape of a real id: three study characters, two subject
# digits, a night, and two more characters. It names no recording of the release, because the
# release numbers no cassette subject 90. Every age, clock time and header name in this file is
# invented for these tests in the same way.
_RECORD_ID = "sleep-edfx-SC4901E0"


def test_one_value_gives_one_instance():
    metadata_annotation = metadata.MetadataAnnotation()
    assert metadata_annotation.get_annotation(AnnotationKey.SEX, "F") is metadata_annotation.get_annotation(
        AnnotationKey.SEX, "F"
    )


def test_two_builds_of_one_value_state_the_same_thing():
    first = metadata.MetadataAnnotation().get_annotation(AnnotationKey.SEX, "F")
    second = metadata.MetadataAnnotation().get_annotation(AnnotationKey.SEX, "F")
    assert (first.key, first.value, first.description) == (second.key, second.value, second.description)


def test_a_shared_annotation_carries_its_key_and_value():
    metadata_annotation = metadata.MetadataAnnotation()
    built = (
        metadata_annotation.get_annotation(AnnotationKey.STUDY, "sleep-cassette"),
        metadata_annotation.get_annotation(AnnotationKey.NIGHT, 1),
        metadata_annotation.get_annotation(AnnotationKey.SEX, "M"),
        metadata_annotation.get_annotation(AnnotationKey.CONDITION, "placebo"),
    )
    assert tuple((one.key, one.value) for one in built) == (
        ("study", "sleep-cassette"),
        ("night", 1),
        ("sex", "M"),
        ("condition", "placebo"),
    )


def test_a_shared_annotation_carries_no_span():
    assert metadata.MetadataAnnotation().get_annotation(AnnotationKey.STUDY, "sleep-cassette").span is None


def test_the_counts_state_how_many_distinct_annotations_were_built():
    metadata_annotation = metadata.MetadataAnnotation()
    for sex in ("F", "M", "F", "F"):
        metadata_annotation.get_annotation(AnnotationKey.SEX, sex)
    metadata_annotation.get_annotation(AnnotationKey.STUDY, "sleep-cassette")
    built = Counter(key for key, _ in metadata_annotation._built)
    assert built == {"sex": 2, "study": 1}


def test_age_carries_its_value_and_the_unit():
    annotation = metadata.build_age(44)
    assert (annotation.key, annotation.value, annotation.span) == ("age", 44, None)
    assert annotation.unit is not None


def test_age_has_no_upper_bound():
    assert metadata.build_age(103).value == 103


def test_the_header_clock_is_carried_as_text_with_no_zone():
    annotation = metadata.build_recording_start_local(_RECORD_ID, datetime(1992, 3, 11, 21, 40, 0))
    assert annotation.key == "recording_start_local"
    assert annotation.value == "1992-03-11T21:40:00"
    assert annotation.span is None


def test_a_start_moment_that_names_a_zone_raises():
    with pytest.raises(TimeFValidationError, match="zone"):
        metadata.build_recording_start_local(_RECORD_ID, datetime(1992, 3, 11, 21, 40, 0, tzinfo=UTC))


def test_two_readings_that_agree_give_no_note():
    assert metadata.build_demographics_note("X F X Female_44yr", 44, "F") is None


def test_a_header_with_no_second_reading_gives_no_note():
    assert metadata.build_demographics_note("X X X X", 44, "F") is None


def test_a_sex_that_disagrees_is_recorded():
    note = metadata.build_demographics_note("X M X Male_44yr", 44, "F")
    assert note is not None
    assert note.key == "demographics_note"
    assert "F" in str(note.value)


def test_a_note_gives_both_readings():
    note = metadata.build_demographics_note("X M X Male_57yr", 58, "F")
    assert note is not None
    assert "57" in str(note.value)
    assert "58" in str(note.value)


def test_a_lights_off_time_after_midnight_wraps_forward():
    span = metadata.build_lights_off(time(1, 5, 0), datetime(1992, 3, 11, 21, 40, 0)).span
    assert span is not None
    assert span.start_us == 12300 * US_PER_S


def test_a_lights_off_time_minutes_after_the_start_does_not_wrap():
    span = metadata.build_lights_off(time(22, 56, 0), datetime(1992, 3, 11, 22, 55, 0)).span
    assert span is not None
    assert span.start_us == 60 * US_PER_S


def test_a_lights_off_time_at_the_start_is_the_first_moment():
    span = metadata.build_lights_off(time(22, 55, 0), datetime(1992, 3, 11, 22, 55, 0)).span
    assert span is not None
    assert span.start_us == 0


def test_lights_off_keeps_the_clock_and_places_the_point():
    annotation = metadata.build_lights_off(time(1, 5, 0), datetime(1992, 3, 11, 21, 40, 0))
    assert annotation.value == "01:05:00"
    assert annotation.span is not None
    assert annotation.span.start_us == 12300 * US_PER_S
    assert annotation.span.is_point
    assert annotation.span.time_series_ids is None
