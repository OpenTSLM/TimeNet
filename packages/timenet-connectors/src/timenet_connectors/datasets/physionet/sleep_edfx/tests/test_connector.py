from datetime import date, time
from pathlib import Path

import edfio
import numpy as np
import pytest

from timenet.dataset import TimeFDataset
from timenet.engine import store_dataset
from timenet.errors import TimeFFormatError, TimeNetDownloadError
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
from timenet_connectors.bases import excel
from timenet_connectors.datasets.physionet.sleep_edfx import tables
from timenet_connectors.datasets.physionet.sleep_edfx.connector import (
    SleepEdfxConnector,
    SleepEdfxSource,
    _find_hypnogram,
    _parse_recording_number,
    _parse_subject_id,
)
from timenet_connectors.datasets.physionet.sleep_edfx.keys import AnnotationKey


# This is a synthetic release of two cassette recordings. These tests write it, and it holds no
# byte of the real release. Both recordings belong to one subject, and each one covers one of
# the two nights of that subject.
#
# Every recording id here is invented. Each one keeps the shape of a real id, because the parser
# needs that shape. A real id is three study characters, two subject digits, a night, and two
# more characters. No id here names a recording of the release. Every subject number is more
# than 89, and the release numbers no subject that high. The age, the sex, the clock times and
# the header names are invented in the same way.
_STUDY = "sleep-cassette"
_CHANNELS = ("EEG Fpz-Cz", "EEG Pz-Oz", "EOG horizontal", "EMG submental")
_RECORD_SECONDS = 30
_RECORDS = 120  # a session of 3600 s, which is short enough to write in a test
_SESSION_SECONDS = _RECORD_SECONDS * _RECORDS
_START = time(22, 0, 0)
_LIGHTS_OFF_SECONDS = 22 * 60 * 60 + 30 * 60  # 22:30:00, half an hour into the session

# Each recording gets this scoring: wake, then one stretch of sleep, then an unscored entry
# that pads the file. The sleep period must cover the middle stretch alone.
_SCORING = (
    ("Sleep stage W", 0, 1800),
    ("Sleep stage 2", 1800, 2700),
    ("Sleep stage ?", 2700, 3600),
)

# The header of the first recording agrees with the table. The header of the second states a
# male subject, and the table states a female one. That build carries a note.
_HEADER_NAMES = {"SC4901E0": "Female_44yr", "SC4902E0": "Male_44yr"}

# These rows have the shape that the cassette subject table gives. The lights-off time is a day
# fraction.
_TABLE_ROWS = [
    ("subject", "night", "age", "sex (F=1)", "LightsOff"),
    (90.0, 1.0, 44.0, 1.0, _LIGHTS_OFF_SECONDS / 86400),
    (90.0, 2.0, 44.0, 1.0, _LIGHTS_OFF_SECONDS / 86400),
]


# These rows have the shape that the telemetry subject table gives: two header rows, then one
# row for the subject. The group labels of the first header row are merged cells, and each one
# reads as empty. The numbers are invented, and no row of the real table holds them. The row of
# subject 91 puts night 1 in the placebo column.
_TELEMETRY_STUDY = "sleep-telemetry"
_TELEMETRY_ROWS = [
    ("Subject - age - sex", "", "", "Placebo night", "", "Temazepam night", ""),
    ("Nr", "Age", "M1/F2", "night nr", "lights off", "night nr", "lights off"),
    (91.0, 62.0, 1.0, 1.0, _LIGHTS_OFF_SECONDS / 86400, 2.0, 0.9861111111111112),
]


def _write_recording(
    study_dir: Path, recording_id: str, channels: tuple[str, ...] = _CHANNELS, patient_name: str | None = None
) -> None:
    rng = np.random.default_rng(0)
    signals = [
        edfio.EdfSignal(rng.standard_normal(_SESSION_SECONDS * 10), 10.0, label=channel, physical_dimension="uV")
        for channel in channels
    ]
    psg = edfio.Edf(
        signals,
        patient=edfio.Patient(code="X", sex="F", name=patient_name or _HEADER_NAMES[recording_id]),
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
    # The release ends the name of a scoring with the initial of the technician who wrote it.
    # The PSG name does not predict that initial. The connector pairs the two files on the seven
    # characters they share.
    scoring.write(study_dir / f"{recording_id[:7]}C-Hypnogram.edf")


@pytest.fixture(scope="session")
def release(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("sleep_edfx_release")
    study_dir = root / _STUDY
    study_dir.mkdir()
    for recording_id in _HEADER_NAMES:
        _write_recording(study_dir, recording_id)
    return root


def _convert(
    release: Path, monkeypatch, rows=None, study: str = _STUDY, table: str = tables.CASSETTE_TABLE_NAME
) -> TimeFDataset:
    # This line replaces the function that reads the .xls subject table. The code that follows
    # takes rows, so this build needs no .xls file.
    monkeypatch.setattr(excel, "read_table_rows", lambda path: _TABLE_ROWS if rows is None else rows)
    source = SleepEdfxSource(
        studies=((study, release / study),),
        subject_tables=((study, release / table),),
    )
    return SleepEdfxConnector().convert([source])


def _streamed(dataset: TimeFDataset) -> list:
    # _task_stream is Callable | None, so narrow the type before you call it.
    source = dataset._task_stream
    assert source is not None
    return list(source())


def _annotations(dataset: TimeFDataset, sample_id: str, key: str) -> list:
    sample = next(one for one in dataset.samples if one.sample_id == sample_id)
    return [annotation for annotation in sample.annotations if annotation.key == key]


def _asked(task) -> tuple[str, ...]:
    # A task id is generated, so two builds never share one. What a task asks is what a
    # consumer reads, and this is that. Every part is text, so a list of these sorts.
    return (type(task).__name__, str(task.target), str(task.scope), str(task.sample_ids), str(task.prompt))


def test_a_recording_name_states_a_subject_number_and_a_night():
    assert _parse_recording_number(_STUDY, "SC4902E0", Path("SC4902E0-PSG.edf")) == (90, 2)


def test_a_telemetry_recording_name_states_its_own_numbers():
    assert _parse_recording_number("sleep-telemetry", "ST7911J0", Path("ST7911J0-PSG.edf")) == (91, 1)


def test_the_subject_id_names_the_study_and_the_number():
    assert _parse_subject_id(_STUDY, "SC4931E0", Path("SC4931E0-PSG.edf")) == "sleep-cassette-93"


def test_a_name_that_does_not_fit_the_shape_raises():
    with pytest.raises(TimeFFormatError, match="names no subject"):
        _parse_recording_number(_STUDY, "SCXXXXX0", Path("SCXXXXX0-PSG.edf"))


def test_a_study_this_release_does_not_hold_raises():
    with pytest.raises(TimeFFormatError, match="not a study of this release"):
        _parse_recording_number("sleep-apnea", "SC4901E0", Path("SC4901E0-PSG.edf"))


def test_a_name_stating_a_third_night_raises():
    with pytest.raises(TimeFFormatError, match="states night 3"):
        _parse_recording_number(_STUDY, "SC4903E0", Path("SC4903E0-PSG.edf"))


def test_a_recording_with_no_scoring_beside_it_raises(tmp_path: Path):
    psg = tmp_path / "SC4901E0-PSG.edf"
    psg.touch()
    with pytest.raises(TimeNetDownloadError, match="found none"):
        _find_hypnogram(psg)


def test_a_recording_with_two_scorings_raises(tmp_path: Path):
    psg = tmp_path / "SC4901E0-PSG.edf"
    psg.touch()
    (tmp_path / "SC4901EC-Hypnogram.edf").touch()
    (tmp_path / "SC4901ED-Hypnogram.edf").touch()
    with pytest.raises(TimeNetDownloadError, match=r"SC4901EC-Hypnogram\.edf, SC4901ED-Hypnogram\.edf"):
        _find_hypnogram(psg)


def test_one_sample_for_each_recording(release, monkeypatch):
    dataset = _convert(release, monkeypatch)
    assert {sample.sample_id for sample in dataset.samples} == {"sleep-edfx-SC4901E0", "sleep-edfx-SC4902E0"}


def test_every_sample_carries_the_same_study_annotation(release, monkeypatch):
    dataset = _convert(release, monkeypatch)
    study = [_annotations(dataset, sample.sample_id, "study")[0] for sample in dataset.samples]
    assert study[0] is study[1]
    assert study[0].value == _STUDY


def test_the_night_comes_from_the_filename(release, monkeypatch):
    dataset = _convert(release, monkeypatch)
    assert _annotations(dataset, "sleep-edfx-SC4901E0", "night")[0].value == 1
    assert _annotations(dataset, "sleep-edfx-SC4902E0", "night")[0].value == 2


def test_the_sex_of_two_samples_is_one_annotation(release, monkeypatch):
    dataset = _convert(release, monkeypatch)
    first = _annotations(dataset, "sleep-edfx-SC4901E0", "sex")[0]
    second = _annotations(dataset, "sleep-edfx-SC4902E0", "sex")[0]
    assert first is second
    assert first.value == "F"


def test_the_age_comes_from_the_table(release, monkeypatch):
    dataset = _convert(release, monkeypatch)
    assert _annotations(dataset, "sleep-edfx-SC4901E0", "age")[0].value == 44


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


def test_a_second_build_names_the_same_samples_and_asks_the_same_questions(release, monkeypatch):
    # An annotation and a task each take the generated id, which is a fresh uuid on every
    # build. A sample id is named, and it must survive a rebuild. So must every question the
    # build asks, even though no two runs name one the same.
    first = _convert(release, monkeypatch)
    second = _convert(release, monkeypatch)
    assert {one.sample_id for one in first.samples} == {one.sample_id for one in second.samples}
    assert [_asked(one) for one in _streamed(first)] == [_asked(one) for one in _streamed(second)]


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
    # The tasks stream, so the writer reads them one time as it writes. No other test proves
    # that they survive the round trip. The writer wrote the ids of the pass it consumed, and a
    # fresh pass generates new ones, so compare what each task asks rather than its id.
    written = sorted(_asked(one) for one in _streamed(dataset))
    assert sorted(_asked(one) for one in restored.tasks) == written
    assert written


def test_the_tasks_stream_and_are_not_materialized(release, monkeypatch):
    dataset = _convert(release, monkeypatch)
    # A streamed task never reaches Sample.task_ids. This difference separates a streamed task
    # from a materialized one.
    assert all(not sample.task_ids for sample in dataset.samples)
    assert dataset._task_stream is not None


def test_the_source_gives_a_fresh_iterator_each_call(release, monkeypatch):
    dataset = _convert(release, monkeypatch)
    first = [_asked(one) for one in _streamed(dataset)]
    second = [_asked(one) for one in _streamed(dataset)]
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
    # An epoch task is the only classification task that carries a scope. Age, sex and the drug
    # condition are asked of the whole sample.
    epochs = [one for one in _streamed(dataset) if isinstance(one, ClassificationTask) and one.scope is not None]
    scored = sum(
        (one.span.exclusive_end - one.span.start_us) // (30 * US_PER_S)
        for sample in dataset.samples
        for one in sample.annotations
        if one.key == "sleep_stage" and one.span is not None
    )
    assert len(epochs) == scored


def test_a_scoring_that_cannot_be_expanded_fails_the_write(release, monkeypatch, tmp_path):
    # The writer reads the stream after convert returns. This error path exists only because
    # the tasks stream. Nothing hides this error.
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
    # lights_off states a fact about the recording, and it carries a span. The task asks where
    # that moment is. The task is never a whole-sample question like age or sex.
    dataset = _convert(release, monkeypatch)
    moment = _annotations(dataset, "sleep-edfx-SC4901E0", "lights_off")[0].span
    assert moment is not None
    asked = [one for one in _streamed(dataset) if isinstance(one, TemporalLocalizationTask) and one.target == (moment,)]
    assert asked


def test_a_recording_missing_a_scoring_channel_fails_the_build(tmp_path: Path, monkeypatch):
    study_dir = tmp_path / _STUDY
    study_dir.mkdir()
    _write_recording(study_dir, "SC4901E0", channels=_CHANNELS[:3])
    with pytest.raises(TimeFFormatError, match="does not hold the channels"):
        _convert(tmp_path, monkeypatch)


def test_a_telemetry_sample_carries_the_condition_of_its_night(tmp_path: Path, monkeypatch):
    study_dir = tmp_path / _TELEMETRY_STUDY
    study_dir.mkdir()
    _write_recording(study_dir, "ST7911J0", patient_name="Male_62yr")
    dataset = _convert(
        tmp_path,
        monkeypatch,
        rows=_TELEMETRY_ROWS,
        study=_TELEMETRY_STUDY,
        table=tables.TELEMETRY_TABLE_NAME,
    )
    conditions = _annotations(dataset, "sleep-edfx-ST7911J0", AnnotationKey.CONDITION)
    assert [one.value for one in conditions] == ["placebo"]


def test_the_series_values_match_the_recording_they_were_read_from(release, monkeypatch):
    # The count of series says nothing about what the series hold. This test reads the same
    # channel again with edfio and compares the values. A build that decodes every sample wrong
    # fails here.
    dataset = _convert(release, monkeypatch)
    sample = next(one for one in dataset.samples if one.sample_id == "sleep-edfx-SC4901E0")
    series = next(one for one in sample.time_series if one.channel == _CHANNELS[0])
    written = edfio.read_edf(release / _STUDY / "SC4901E0-PSG.edf").signals[0].data
    read_back = series.to_numpy()
    assert len(read_back) == len(written)
    assert float(read_back[0]) == pytest.approx(float(written[0]), rel=1e-3)
    assert float(read_back[-1]) == pytest.approx(float(written[-1]), rel=1e-3)
    assert series.spec.spec_type == "eeg"
    assert str(series.spec.unit_value) == "microvolt"
