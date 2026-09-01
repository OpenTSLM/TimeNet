from datetime import UTC, datetime, time

import pytest

from timenet.errors import TimeFValidationError
from timenet.types import US_PER_S, Annotation, TimeInterval
from timenet_connectors.datasets.physionet.sleep_edfx import metadata


_PREFIX = "sleep-edfx"

# An invented recording id. It keeps the shape the release uses, three study characters then two
# subject digits, a night and two more characters, and it names no recording of the release. The
# release numbers no cassette subject 90. Every age, clock time and header name below is invented
# for these tests in the same way.
_SAMPLE_ID = "sleep-edfx-SC4901E0"


def _stage(label: str, start_s: int, end_s: int) -> Annotation:
    # The shape the scoring module gives a stage: an interval that names the four scoring
    # channels. The sleep period reads the bounds and drops the scope.
    return Annotation(
        key="sleep_stage",
        value=label,
        span=TimeInterval.micros(start_s * US_PER_S, end_s * US_PER_S, time_series_ids=("a", "b")),
        id=f"{_SAMPLE_ID}-stage-{start_s}",
    )


def test_one_value_gives_one_instance():
    metadata_annotation = metadata.MetadataAnnotation(_PREFIX)
    assert metadata_annotation.get_sex("F") is metadata_annotation.get_sex("F")


def test_two_builds_of_one_value_are_field_equal():
    first = metadata.MetadataAnnotation(_PREFIX).get_sex("F")
    second = metadata.MetadataAnnotation(_PREFIX).get_sex("F")
    assert first == second
    assert first.id == "sleep-edfx-sex-F"


def test_a_shared_id_names_its_value():
    metadata_annotation = metadata.MetadataAnnotation(_PREFIX)
    ids = (
        metadata_annotation.get_study("sleep-cassette").id,
        metadata_annotation.get_night(1).id,
        metadata_annotation.get_sex("M").id,
        metadata_annotation.get_condition("placebo").id,
    )
    assert ids == (
        "sleep-edfx-study-sleep-cassette",
        "sleep-edfx-night-1",
        "sleep-edfx-sex-M",
        "sleep-edfx-condition-placebo",
    )


def test_a_shared_annotation_carries_no_span():
    assert metadata.MetadataAnnotation(_PREFIX).get_study("sleep-cassette").span is None


def test_the_counts_state_how_many_distinct_annotations_were_built():
    metadata_annotation = metadata.MetadataAnnotation(_PREFIX)
    for sex in ("F", "M", "F", "F"):
        metadata_annotation.get_sex(sex)
    metadata_annotation.get_study("sleep-cassette")
    assert metadata_annotation.count_built() == {"sex": 2, "study": 1}


def test_age_is_per_sample_and_carries_the_unit():
    annotation = metadata.build_age(_SAMPLE_ID, 44)
    assert (annotation.key, annotation.value, annotation.span) == ("age", 44, None)
    assert annotation.id == "sleep-edfx-SC4901E0-age"
    assert annotation.unit is not None


def test_age_has_no_upper_bound():
    assert metadata.build_age(_SAMPLE_ID, 103).value == 103


def test_the_header_clock_is_carried_as_text_with_no_zone():
    annotation = metadata.build_recording_start_local(_SAMPLE_ID, datetime(1992, 3, 11, 21, 40, 0))
    assert annotation.key == "recording_start_local"
    assert annotation.value == "1992-03-11T21:40:00"
    assert annotation.span is None
    assert annotation.id == "sleep-edfx-SC4901E0-start-local"


def test_a_start_moment_that_names_a_zone_raises():
    with pytest.raises(TimeFValidationError, match="zone"):
        metadata.build_recording_start_local(_SAMPLE_ID, datetime(1992, 3, 11, 21, 40, 0, tzinfo=UTC))


def test_two_readings_that_agree_give_no_note():
    assert metadata.build_demographics_note(_SAMPLE_ID, "X F X Female_44yr", 44, "F") is None


def test_a_header_with_no_second_reading_gives_no_note():
    assert metadata.build_demographics_note(_SAMPLE_ID, "X X X X", 44, "F") is None


def test_a_sex_that_disagrees_is_recorded():
    note = metadata.build_demographics_note(_SAMPLE_ID, "X M X Male_44yr", 44, "F")
    assert note is not None
    assert note.id == "sleep-edfx-SC4901E0-demographics_note"
    assert "F" in str(note.value)


def test_a_note_gives_both_readings():
    note = metadata.build_demographics_note(_SAMPLE_ID, "X M X Male_57yr", 58, "F")
    assert note is not None
    assert "57" in str(note.value)
    assert "58" in str(note.value)


def test_a_lights_off_time_after_midnight_wraps_forward():
    at = metadata.find_lights_off_offset(datetime(1992, 3, 11, 21, 40, 0), time(1, 5, 0))
    assert at == 12300 * US_PER_S


def test_a_lights_off_time_minutes_after_the_start_does_not_wrap():
    at = metadata.find_lights_off_offset(datetime(1992, 3, 11, 22, 55, 0), time(22, 56, 0))
    assert at == 60 * US_PER_S


def test_a_lights_off_time_at_the_start_is_the_first_moment():
    at = metadata.find_lights_off_offset(datetime(1992, 3, 11, 22, 55, 0), time(22, 55, 0))
    assert at == 0


def test_lights_off_keeps_the_clock_and_places_the_point():
    annotation = metadata.build_lights_off(_SAMPLE_ID, time(1, 5, 0), 12300 * US_PER_S)
    assert annotation.value == "01:05:00"
    assert annotation.span is not None
    assert annotation.span.start_us == 12300 * US_PER_S
    assert annotation.span.is_point
    assert annotation.span.time_series_ids is None
    assert annotation.id == "sleep-edfx-SC4901E0-lights_off"
