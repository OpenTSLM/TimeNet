from datetime import date, time
from pathlib import Path

import edfio
import numpy as np
import pytest

from timenet.dataset import TimeFDataset
from timenet.engine import store_dataset
from timenet.errors import TimeFFormatError
from timenet.reader import TimeFReader
from timenet.registry import DatasetVersion
from timenet.types import (
    US_PER_S,
    Annotation,
    ClassificationTask,
    ScalarPredictionTask,
    TemporalLocalizationTask,
    TimeInterval,
)
from timenet_connectors.datasets.physionet.sleep_edfx import reader, tables
from timenet_connectors.datasets.physionet.sleep_edfx.connector import (
    SleepEdfxConnector,
    SleepEdfxSource,
    _parse_recording_number,
    _parse_subject_id,
)
from timenet_connectors.datasets.physionet.sleep_edfx.keys import AnnotationKey


# A synthetic release of two cassette recordings, hand-written for these tests and derived from
# no byte of the real one. Both belong to one subject, one for each of their two nights.
#
# Every recording id here is invented. Each one keeps the shape the release uses, three study
# characters then two subject digits, a night and two more characters, because the parser needs
# that shape. Each one names no recording of the release, and the release numbers no subject
# above 89. The age, the sex, the clock times and the header names are invented in the same way.
_STUDY = "sleep-cassette"
_CHANNELS = ("EEG Fpz-Cz", "EEG Pz-Oz", "EOG horizontal", "EMG submental")
_RECORD_SECONDS = 30
_RECORDS = 120  # a session of 3600 s, short enough to write in a test
_SESSION_SECONDS = _RECORD_SECONDS * _RECORDS
_START = time(22, 0, 0)
_LIGHTS_OFF_SECONDS = 22 * 60 * 60 + 30 * 60  # 22:30:00, half an hour into the session

# The scoring of each recording: wake, then one stretch of sleep, then an unscored entry that
# pads the file. The sleep period must cover the middle stretch alone.
_SCORING = (
    ("Sleep stage W", 0, 1800),
    ("Sleep stage 2", 1800, 2700),
    ("Sleep stage ?", 2700, 3600),
)

# The header of the first recording agrees with the table. The second states a male subject
# where the table states a female one, so that build carries a note and warns one time.
_HEADER_NAMES = {"SC4901E0": "Female_44yr", "SC4902E0": "Male_44yr"}

# Rows in the shape the cassette sheet gives back, with the lights-off time as a day fraction.
_TABLE_ROWS = [
    ("subject", "night", "age", "sex (F=1)", "LightsOff"),
    (90.0, 1.0, 44.0, 1.0, _LIGHTS_OFF_SECONDS / 86400),
    (90.0, 2.0, 44.0, 1.0, _LIGHTS_OFF_SECONDS / 86400),
]


def _write_recording(study_dir: Path, recording_id: str) -> None:
    rng = np.random.default_rng(0)
    signals = [
        edfio.EdfSignal(rng.standard_normal(_SESSION_SECONDS * 10), 10.0, label=channel, physical_dimension="uV")
        for channel in _CHANNELS
    ]
    psg = edfio.Edf(
        signals,
        patient=edfio.Patient(code="X", sex="F", name=_HEADER_NAMES[recording_id]),
        starttime=_START,
        data_record_duration=float(_RECORD_SECONDS),
    )
    psg.startdate = date(1992, 3, 11)
    psg.write(study_dir / f"{recording_id}-PSG.edf")

    scoring = edfio.Edf(
        [],
        annotations=[edfio.EdfAnnotation(float(start), float(end - start), label) for label, start, end in _SCORING],
    )
    scoring.startdate = date(1992, 3, 11)
    # A scoring is named after the technician who wrote it, and the PSG name does not predict
    # that letter. The connector pairs the two on the seven characters they share.
    scoring.write(study_dir / f"{recording_id[:7]}C-Hypnogram.edf")


@pytest.fixture(scope="session")
def release(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("sleep_edfx_release")
    study_dir = root / _STUDY
    study_dir.mkdir()
    for recording_id in _HEADER_NAMES:
        _write_recording(study_dir, recording_id)
    return root


def _source(release: Path) -> SleepEdfxSource:
    return SleepEdfxSource(
        studies=((_STUDY, release / _STUDY),),
        subject_tables=((_STUDY, release / tables.CASSETTE_TABLE_NAME),),
    )


def _convert(release: Path, monkeypatch, rows=None) -> TimeFDataset:
    # The workbook seam. The mapping after it takes rows, so this build needs no .xls binary.
    monkeypatch.setattr(reader, "read_table_rows", lambda path: _TABLE_ROWS if rows is None else rows)
    return SleepEdfxConnector().convert([_source(release)])


def _streamed(dataset: TimeFDataset) -> list:
    # _task_stream is Callable | None, so narrow it before calling.
    source = dataset._task_stream
    assert source is not None
    return list(source())


def _annotations(dataset: TimeFDataset, sample_id: str, key: str) -> list:
    sample = next(one for one in dataset.samples if one.sample_id == sample_id)
    return [annotation for annotation in sample.annotations if annotation.key == key]


def test_a_recording_name_states_a_subject_number_and_a_night():
    assert _parse_recording_number(_STUDY, "SC4902E0", Path("SC4902E0-PSG.edf")) == (90, 2)


def test_a_telemetry_recording_name_states_its_own_numbers():
    assert _parse_recording_number("sleep-telemetry", "ST7911J0", Path("ST7911J0-PSG.edf")) == (91, 1)


def test_the_subject_id_names_the_study_and_the_number():
    assert _parse_subject_id(_STUDY, "SC4931E0", Path("SC4931E0-PSG.edf")) == "sleep-cassette-93"


def test_a_name_that_does_not_fit_the_shape_raises():
    with pytest.raises(TimeFFormatError, match="names no subject"):
        _parse_recording_number(_STUDY, "SCXXXXX0", Path("SCXXXXX0-PSG.edf"))


def test_one_sample_for_each_recording(release, monkeypatch):
    dataset = _convert(release, monkeypatch)
    assert {sample.sample_id for sample in dataset.samples} == {"sleep-edfx-SC4901E0", "sleep-edfx-SC4902E0"}


def test_every_sample_carries_the_same_study_annotation(release, monkeypatch):
    dataset = _convert(release, monkeypatch)
    study = [_annotations(dataset, sample.sample_id, "study")[0] for sample in dataset.samples]
    assert study[0] is study[1]
    assert study[0].value == _STUDY
    assert study[0].id == "sleep-edfx-study-sleep-cassette"


def test_the_night_comes_from_the_filename(release, monkeypatch):
    dataset = _convert(release, monkeypatch)
    assert _annotations(dataset, "sleep-edfx-SC4901E0", "night")[0].value == 1
    assert _annotations(dataset, "sleep-edfx-SC4902E0", "night")[0].id == "sleep-edfx-night-2"


def test_the_sex_of_two_samples_is_one_annotation(release, monkeypatch):
    dataset = _convert(release, monkeypatch)
    first = _annotations(dataset, "sleep-edfx-SC4901E0", "sex")[0]
    second = _annotations(dataset, "sleep-edfx-SC4902E0", "sex")[0]
    assert first is second
    assert first.id == "sleep-edfx-sex-F"


def test_the_age_comes_from_the_table_and_names_its_recording(release, monkeypatch):
    dataset = _convert(release, monkeypatch)
    annotation = _annotations(dataset, "sleep-edfx-SC4901E0", "age")[0]
    assert annotation.value == 44
    assert annotation.id == "sleep-edfx-SC4901E0-age"


def test_the_header_clock_is_carried_and_no_start_time_is_set(release, monkeypatch):
    dataset = _convert(release, monkeypatch)
    sample = next(one for one in dataset.samples if one.sample_id == "sleep-edfx-SC4901E0")
    assert sample.start_time is None
    clock = _annotations(dataset, sample.sample_id, "recording_start_local")
    assert [one.value for one in clock] == ["1992-03-11T22:00:00"]


def test_a_recording_that_agrees_carries_no_note(release, monkeypatch):
    dataset = _convert(release, monkeypatch)
    assert _annotations(dataset, "sleep-edfx-SC4901E0", "demographics_note") == []


def test_a_recording_that_disagrees_carries_a_note_and_keeps_the_table_value(release, monkeypatch):
    dataset = _convert(release, monkeypatch)
    note = _annotations(dataset, "sleep-edfx-SC4902E0", "demographics_note")
    assert len(note) == 1
    assert _annotations(dataset, "sleep-edfx-SC4902E0", "sex")[0].value == "F"


def test_lights_off_is_placed_on_the_timeline_and_names_no_series(release, monkeypatch):
    dataset = _convert(release, monkeypatch)
    annotation = _annotations(dataset, "sleep-edfx-SC4901E0", "lights_off")[0]
    assert annotation.value == "22:30:00"
    assert annotation.span.start_us == 1800 * US_PER_S
    assert annotation.span.time_series_ids is None


def test_a_cassette_sample_carries_no_condition(release, monkeypatch):
    dataset = _convert(release, monkeypatch)
    assert _annotations(dataset, "sleep-edfx-SC4901E0", "condition") == []


def test_a_second_build_gives_the_same_ids(release, monkeypatch):
    first = {one.id for sample in _convert(release, monkeypatch).samples for one in sample.annotations}
    second = {one.id for sample in _convert(release, monkeypatch).samples for one in sample.annotations}
    assert first == second


def test_a_recording_with_no_row_raises(release, monkeypatch):
    rows = [_TABLE_ROWS[0], _TABLE_ROWS[1]]  # the second night has no row
    with pytest.raises(TimeFFormatError, match="SC4902E0"):
        _convert(release, monkeypatch, rows=rows)


def test_convert_round_trips_through_the_writer(release, monkeypatch, tmp_path):
    dataset = _convert(release, monkeypatch)
    dataset.derive_schema()
    version_dir = store_dataset(dataset, tmp_path / "out")
    with TimeFReader(DatasetVersion.open_local(version_dir)) as reader:
        restored = reader.read()
    assert {sample.sample_id for sample in restored.samples} == {"sleep-edfx-SC4901E0", "sleep-edfx-SC4902E0"}
    sample = next(one for one in restored.samples if one.sample_id == "sleep-edfx-SC4901E0")
    assert len(sample.time_series) == len(_CHANNELS)
    lights_off = next(one for one in sample.annotations if one.key == "lights_off")
    assert lights_off.value == "22:30:00"
    assert lights_off.span is not None
    assert lights_off.span.start_us == 1800 * US_PER_S
    assert len([one for one in sample.annotations if one.key == "sleep_stage"]) == len(_SCORING)
    # The tasks stream, so the writer drains them on the way out. Nothing else proves they
    # survive the round trip.
    written = {one.id: one for one in _streamed(dataset)}
    restored_tasks = {one.id: one for one in restored.tasks}
    assert set(restored_tasks) == set(written)
    for task_id, before in written.items():
        after = restored_tasks[task_id]
        assert type(after) is type(before)
        assert (after.target, after.scope, after.sample_ids, after.prompt) == (
            before.target,
            before.scope,
            before.sample_ids,
            before.prompt,
        )


def test_the_tasks_stream_and_are_not_materialized(release, monkeypatch):
    dataset = _convert(release, monkeypatch)
    # A streamed task never lands in Sample.task_ids, which is what tells the two apart.
    assert all(not sample.task_ids for sample in dataset.samples)
    assert dataset._task_stream is not None


def test_the_source_gives_a_fresh_iterator_each_call(release, monkeypatch):
    dataset = _convert(release, monkeypatch)
    first = [one.id for one in _streamed(dataset)]
    second = [one.id for one in _streamed(dataset)]
    assert first == second
    assert first


def test_the_schema_names_every_task_type(release, monkeypatch):
    dataset = _convert(release, monkeypatch)
    assert set(dataset._streamed_task_types) == {
        ClassificationTask,
        ScalarPredictionTask,
        TemporalLocalizationTask,
    }


def test_every_streamed_task_carries_its_own_sample_ids(release, monkeypatch):
    dataset = _convert(release, monkeypatch)
    known = {sample.sample_id for sample in dataset.samples}
    for task in _streamed(dataset):
        assert task.sample_ids
        assert set(task.sample_ids) <= known


def test_the_vocabularies_are_registered_and_belong_to_no_sample(release, monkeypatch):
    dataset = _convert(release, monkeypatch)
    registered = {one.id for one in dataset._registered_annotations.values()}
    assert registered == {
        "sleep-edfx-vocabulary-sleep_stage",
        "sleep-edfx-vocabulary-sex",
        "sleep-edfx-vocabulary-condition",
    }
    for sample in dataset.samples:
        assert not registered & {one.id for one in sample.annotations}


def test_an_epoch_task_covers_each_scored_epoch(release, monkeypatch):
    dataset = _convert(release, monkeypatch)
    epochs = [one for one in _streamed(dataset) if "-epoch-" in one.id]
    scored = sum(
        (one.span.exclusive_end - one.span.start_us) // (30 * US_PER_S)
        for sample in dataset.samples
        for one in sample.annotations
        if one.key == "sleep_stage" and one.span is not None
    )
    assert len(epochs) == scored


def test_a_scoring_that_cannot_be_expanded_fails_the_write(release, monkeypatch, tmp_path):
    # The stream is drained by the writer, after convert has returned, so this is the one
    # failure path that exists only because the tasks stream. Nothing may swallow it.
    dataset = _convert(release, monkeypatch)
    sample = dataset.samples[0]
    ragged = Annotation(
        key=AnnotationKey.SLEEP_STAGE,
        value="Sleep stage W",
        span=TimeInterval.micros(0, 45 * US_PER_S),
        id=f"{sample.sample_id}-stage-ragged",
    )
    sample.add_annotations([ragged])
    dataset.derive_schema()
    with pytest.raises(TimeFFormatError, match=sample.sample_id):
        store_dataset(dataset, tmp_path / "out")


def test_a_span_carrying_annotation_asks_for_a_region_and_not_a_value(release, monkeypatch):
    # lights_off states a fact about the recording and carries a span. It asks where that
    # moment is, and never becomes a whole-sample question like age or sex.
    dataset = _convert(release, monkeypatch)
    assert _annotations(dataset, "sleep-edfx-SC4901E0", "lights_off")[0].span is not None
    asked = [one for one in _streamed(dataset) if one.id.endswith("-lights_off")]
    assert asked
    assert all(isinstance(one, TemporalLocalizationTask) for one in asked)
