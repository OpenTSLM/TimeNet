from datetime import UTC, datetime, time

import pytest

from timenet.errors import TimeFValidationError
from timenet.types import US_PER_S
from timenet_connectors.datasets.physionet.sleep_edfx import records as metadata
from timenet_connectors.datasets.physionet.sleep_edfx.labels import AnnotationKey
from timenet_connectors.demographics import Sex


# An invented recording id. It keeps the shape of a real id: three study characters, two subject
# digits, a night, and two more characters. It names no recording of the release, because the
# release numbers no cassette subject 90. Every age, clock time and header name in this file is
# invented for these tests in the same way.
_RECORDING_ID = "SC4901E0"


def test_two_builds_of_one_value_are_equal():
    assert metadata.build_closed_set(AnnotationKey.SEX, "female") == metadata.build_closed_set(
        AnnotationKey.SEX, "female"
    )


def test_a_closed_set_annotation_carries_its_key_and_value():
    built = (
        metadata.build_closed_set(AnnotationKey.STUDY, "sleep-cassette"),
        metadata.build_closed_set(AnnotationKey.NIGHT, 1),
        metadata.build_closed_set(AnnotationKey.SEX, "male"),
        metadata.build_closed_set(AnnotationKey.CONDITION, "placebo"),
    )
    assert tuple((one.key, one.value) for one in built) == (
        ("study", "sleep-cassette"),
        ("night", 1),
        ("sex", "male"),
        ("condition", "placebo"),
    )


def test_a_closed_set_annotation_carries_no_span():
    assert metadata.build_closed_set(AnnotationKey.STUDY, "sleep-cassette").span is None


def test_age_carries_its_value_and_the_unit():
    annotation = metadata.build_age(44)
    assert (annotation.key, annotation.value, annotation.span) == ("age", 44, None)
    assert annotation.unit is not None


def test_age_has_no_upper_bound():
    assert metadata.build_age(103).value == 103


def test_the_header_clock_is_carried_as_text_with_no_zone():
    annotation = metadata.build_recording_start_local(_RECORDING_ID, datetime(1992, 3, 11, 21, 40, 0))
    assert annotation.key == "recording_start_local"
    assert annotation.value == "1992-03-11T21:40:00"
    assert annotation.span is None


def test_a_start_moment_that_names_a_zone_raises():
    with pytest.raises(TimeFValidationError, match="zone"):
        metadata.build_recording_start_local(_RECORDING_ID, datetime(1992, 3, 11, 21, 40, 0, tzinfo=UTC))


def test_two_readings_that_agree_give_no_note():
    assert metadata.build_demographics_note("X F X Female_44yr", 44, Sex.FEMALE) is None


def test_a_header_with_no_second_reading_gives_no_note():
    assert metadata.build_demographics_note("X X X X", 44, Sex.FEMALE) is None


def test_a_sex_that_disagrees_is_recorded():
    note = metadata.build_demographics_note("X M X Male_44yr", 44, Sex.FEMALE)
    assert note is not None
    assert note.key == "demographics_note"
    text = str(note.value)
    assert "the EDF header states sex male" in text
    assert "The subject table states sex female" in text


def test_a_note_gives_both_readings():
    note = metadata.build_demographics_note("X M X Male_57yr", 58, Sex.FEMALE)
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
    assert annotation.span.time_series is None
