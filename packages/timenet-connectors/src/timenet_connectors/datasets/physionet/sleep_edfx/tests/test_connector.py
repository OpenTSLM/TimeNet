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
from timenet_connectors.datasets.physionet.sleep_edfx import specs, tables
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
_SIGNALS = ("EEG Fpz-Cz", "EEG Pz-Oz", "EOG horizontal", "EMG submental")
_RECORD_SECONDS = 30
_RECORDS = 120  # a session of 3600 s, which is short enough to write in a test
_SESSION_SECONDS = _RECORD_SECONDS * _RECORDS
_START = time(22, 0, 0)
_LIGHTS_OFF_SECONDS = 22 * 60 * 60 + 30 * 60  # 22:30:00, half an hour into the session

# The EDF prefiltering field, which the connector carries as an annotation. This string keeps the
# shape of a real one and states no filter that the release states.
_PREFILTERING = "HP:1Hz LP:90Hz"

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
    study_dir: Path,
    recording_id: str,
    signals: tuple[str, ...] = _SIGNALS,
    patient_name: str | None = None,
    prefiltering: str = _PREFILTERING,
) -> None:
    rng = np.random.default_rng(0)
    edf_signals = [
        edfio.EdfSignal(
            rng.standard_normal(_SESSION_SECONDS * 10),
            10.0,
            label=signal,
            physical_dimension="uV",
            prefiltering=prefiltering,
        )
        for signal in signals
    ]
    psg = edfio.Edf(
        edf_signals,
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


def _annotations(dataset: TimeFDataset, record_id: str, key: str) -> list:
    record = next(one for one in dataset.records if one.record_id == record_id)
    return [annotation for annotation in record.annotations if annotation.key == key]


def _asked(task) -> tuple[str, ...]:
    # A task id is generated, so two builds never share one. What a task asks is what a
    # consumer reads, and this is that. Every part is text, so a list of these sorts.
    return (type(task).__name__, str(task.target), str(task.scope), str(task.record_ids), str(task.prompt))


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


def test_one_record_for_each_recording(release, monkeypatch):
    dataset = _convert(release, monkeypatch)
    assert {record.record_id for record in dataset.records} == {"sleep-edfx-SC4901E0", "sleep-edfx-SC4902E0"}


def test_every_record_carries_the_same_study_annotation(release, monkeypatch):
    dataset = _convert(release, monkeypatch)
    study = [_annotations(dataset, record.record_id, "study")[0] for record in dataset.records]
    assert study[0] is study[1]
    assert study[0].value == _STUDY


def test_the_night_comes_from_the_filename(release, monkeypatch):
    dataset = _convert(release, monkeypatch)
    assert _annotations(dataset, "sleep-edfx-SC4901E0", "night")[0].value == 1
    assert _annotations(dataset, "sleep-edfx-SC4902E0", "night")[0].value == 2


def test_the_sex_of_two_records_is_one_annotation(release, monkeypatch):
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
    record = next(one for one in dataset.records if one.record_id == "sleep-edfx-SC4901E0")
    assert record.start_time is None
    clock = _annotations(dataset, record.record_id, "recording_start_local")
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


def test_a_cassette_record_carries_no_condition(release, monkeypatch):
    dataset = _convert(release, monkeypatch)
    assert _annotations(dataset, "sleep-edfx-SC4901E0", "condition") == []


def test_a_second_build_names_the_same_records_and_asks_the_same_questions(release, monkeypatch):
    # An annotation and a task each take the generated id, which is a fresh uuid on every
    # build. A record id is named, and it must survive a rebuild. So must every question the
    # build asks, even though no two runs name one the same.
    first = _convert(release, monkeypatch)
    second = _convert(release, monkeypatch)
    assert {one.record_id for one in first.records} == {one.record_id for one in second.records}
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
    assert {record.record_id for record in restored.records} == {"sleep-edfx-SC4901E0", "sleep-edfx-SC4902E0"}
    record = next(one for one in restored.records if one.record_id == "sleep-edfx-SC4901E0")
    assert len(record.time_series) == len(_SIGNALS)
    lights_off = next(one for one in record.annotations if one.key == "lights_off")
    assert lights_off.value == "22:30:00"
    assert lights_off.span is not None
    assert lights_off.span.start_us == 1800 * US_PER_S
    assert len([one for one in record.annotations if one.key == "sleep_stage"]) == len(_SCORING)
    # The prefiltering annotation is the one whose value is a map, so the round trip proves that
    # the manifest carries the signal names and not only the strings.
    prefiltering = next(one for one in record.annotations if one.key == "prefiltering")
    assert prefiltering.value == dict.fromkeys(_SIGNALS, _PREFILTERING)
    # The tasks stream, so the writer reads them one time as it writes. No other test proves
    # that they survive the round trip. The writer wrote the ids of the pass it consumed, and a
    # fresh pass generates new ones, so compare what each task asks rather than its id.
    written = sorted(_asked(one) for one in _streamed(dataset))
    assert sorted(_asked(one) for one in restored.tasks) == written
    assert written


def test_the_tasks_stream_and_are_not_materialized(release, monkeypatch):
    dataset = _convert(release, monkeypatch)
    # A streamed task never reaches Record.task_ids. This difference separates a streamed task
    # from a materialized one.
    assert all(not record.task_ids for record in dataset.records)
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


def test_every_streamed_task_carries_its_own_record_ids(release, monkeypatch):
    dataset = _convert(release, monkeypatch)
    known = {record.record_id for record in dataset.records}
    for task in _streamed(dataset):
        assert task.record_ids
        assert set(task.record_ids) <= known


def test_the_vocabularies_are_registered_and_belong_to_no_record(release, monkeypatch):
    dataset = _convert(release, monkeypatch)
    registered = {one.id for one in dataset._registered_annotations.values()}
    assert registered == {
        "sleep-edfx-vocabulary-sleep_stage",
        "sleep-edfx-vocabulary-sex",
        "sleep-edfx-vocabulary-condition",
    }
    for record in dataset.records:
        assert not registered & {one.id for one in record.annotations}


def test_an_epoch_task_covers_each_scored_epoch(release, monkeypatch):
    dataset = _convert(release, monkeypatch)
    # An epoch task is the only classification task that carries a scope. Age, sex and the drug
    # condition are asked of the whole record.
    epochs = [one for one in _streamed(dataset) if isinstance(one, ClassificationTask) and one.scope is not None]
    scored = sum(
        (one.span.exclusive_end - one.span.start_us) // (30 * US_PER_S)
        for record in dataset.records
        for one in record.annotations
        if one.key == "sleep_stage" and one.span is not None
    )
    assert len(epochs) == scored


def test_a_scoring_that_cannot_be_expanded_fails_the_write(release, monkeypatch, tmp_path):
    # The writer reads the stream after convert returns. This error path exists only because
    # the tasks stream. Nothing hides this error.
    dataset = _convert(release, monkeypatch)
    record = dataset.records[0]
    ragged = Annotation(
        key=AnnotationKey.SLEEP_STAGE,
        value="Sleep stage W",
        span=TimeInterval.micros(0, 45 * US_PER_S),
        id=f"{record.record_id}-stage-ragged",
    )
    record.add_annotations([ragged])
    dataset.derive_schema()
    with pytest.raises(TimeFFormatError, match=record.record_id):
        store_dataset(dataset, tmp_path / "out")


def test_a_span_carrying_annotation_asks_for_a_region_and_not_a_value(release, monkeypatch):
    # lights_off states a fact about the recording, and it carries a span. The task asks where
    # that moment is. The task is never a whole-record question like age or sex.
    dataset = _convert(release, monkeypatch)
    moment = _annotations(dataset, "sleep-edfx-SC4901E0", "lights_off")[0].span
    assert moment is not None
    asked = [one for one in _streamed(dataset) if isinstance(one, TemporalLocalizationTask) and one.target == (moment,)]
    assert asked


def test_a_recording_missing_a_scoring_signal_fails_the_build(tmp_path: Path, monkeypatch):
    study_dir = tmp_path / _STUDY
    study_dir.mkdir()
    _write_recording(study_dir, "SC4901E0", signals=_SIGNALS[:3])
    with pytest.raises(TimeFFormatError, match="does not hold the signals"):
        _convert(tmp_path, monkeypatch)


def test_a_telemetry_record_carries_the_condition_of_its_night(tmp_path: Path, monkeypatch):
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
    # signal again with edfio and compares the values. A build that decodes every sample wrong
    # fails here.
    dataset = _convert(release, monkeypatch)
    record = next(one for one in dataset.records if one.record_id == "sleep-edfx-SC4901E0")
    series = next(one for one in record.time_series if one.signal == _SIGNALS[0])
    written = edfio.read_edf(release / _STUDY / "SC4901E0-PSG.edf").signals[0].data
    read_back = series.to_numpy()
    assert len(read_back) == len(written)
    assert float(read_back[0]) == pytest.approx(float(written[0]), rel=1e-3)
    assert float(read_back[-1]) == pytest.approx(float(written[-1]), rel=1e-3)
    assert series.spec.spec_type == "eeg"
    assert str(series.spec.unit_value) == "microvolt"


def _emg_spec_type(dataset: TimeFDataset, record_id: str) -> str:
    record = next(one for one in dataset.records if one.record_id == record_id)
    return next(one for one in record.time_series if one.signal == "EMG submental").spec.spec_type


def test_the_two_studies_write_their_emg_under_different_spec_types(release, tmp_path: Path, monkeypatch):
    # The cassette recorder rectified its submental EMG and low-passed it at 0.7 Hz, so that
    # signal holds an envelope. The telemetry recorder did neither. One spec type over both
    # would let a consumer pool an envelope with a broadband signal and never learn it did.
    cassette = _convert(release, monkeypatch)
    study_dir = tmp_path / _TELEMETRY_STUDY
    study_dir.mkdir()
    _write_recording(study_dir, "ST7911J0", patient_name="Male_62yr")
    telemetry = _convert(
        tmp_path, monkeypatch, rows=_TELEMETRY_ROWS, study=_TELEMETRY_STUDY, table=tables.TELEMETRY_TABLE_NAME
    )
    assert _emg_spec_type(cassette, "sleep-edfx-SC4901E0") == "emg_envelope"
    assert _emg_spec_type(telemetry, "sleep-edfx-ST7911J0") == "emg"


def test_a_record_carries_the_prefiltering_of_every_signal(release, monkeypatch):
    dataset = _convert(release, monkeypatch)
    carried = _annotations(dataset, "sleep-edfx-SC4901E0", AnnotationKey.PREFILTERING)
    assert [one.value for one in carried] == [dict.fromkeys(_SIGNALS, _PREFILTERING)]


def test_two_recordings_that_were_filtered_alike_share_one_prefiltering_annotation(release, monkeypatch):
    dataset = _convert(release, monkeypatch)
    first = _annotations(dataset, "sleep-edfx-SC4901E0", AnnotationKey.PREFILTERING)[0]
    second = _annotations(dataset, "sleep-edfx-SC4902E0", AnnotationKey.PREFILTERING)[0]
    assert first is second


def test_a_recording_that_states_no_prefiltering_carries_no_annotation(tmp_path: Path, monkeypatch):
    study_dir = tmp_path / _STUDY
    study_dir.mkdir()
    _write_recording(study_dir, "SC4901E0", prefiltering="")
    dataset = _convert(tmp_path, monkeypatch)
    assert _annotations(dataset, "sleep-edfx-SC4901E0", AnnotationKey.PREFILTERING) == []


def test_a_prefiltering_string_that_states_no_filter_is_kept(tmp_path: Path, monkeypatch):
    # The cassette study writes "Hold during 2 seconds" here for its event marker, which is an
    # instruction for the button and not a filter. The map keeps whatever the field holds, so
    # nothing may describe it as a filter and nothing may drop it.
    study_dir = tmp_path / _STUDY
    study_dir.mkdir()
    _write_recording(study_dir, "SC4901E0")
    path = study_dir / "SC4901E0-PSG.edf"
    psg = edfio.read_edf(path)
    psg.signals[0].prefiltering = "Hold during 2 seconds"
    psg.write(path)
    dataset = _convert(tmp_path, monkeypatch)
    carried = _annotations(dataset, "sleep-edfx-SC4901E0", AnnotationKey.PREFILTERING)[0]
    assert carried.value == {**dict.fromkeys(_SIGNALS, _PREFILTERING), _SIGNALS[0]: "Hold during 2 seconds"}


def test_a_signal_declaring_another_physical_dimension_fails_the_build(tmp_path: Path, monkeypatch):
    # The unit written into TimeF comes from the connector's signal table. This is the check
    # that keeps that table honest when a copy of the release states something else.
    study_dir = tmp_path / _STUDY
    study_dir.mkdir()
    _write_recording(study_dir, "SC4901E0")
    path = study_dir / "SC4901E0-PSG.edf"
    psg = edfio.read_edf(path)
    psg.signals[0].physical_dimension = "mV"
    psg.write(path)
    with pytest.raises(TimeFFormatError, match="declares the physical dimension 'mV'"):
        _convert(tmp_path, monkeypatch)


def test_every_signal_of_the_two_studies_declares_a_header_dimension():
    # The check skips a signal the dimension table does not name, so a signal added to a spec
    # table and not to that one would never be checked, and nothing else would fail.
    assert specs.HEADER_DIMENSIONS.keys() == specs.CASSETTE_SPECS.keys() | specs.TELEMETRY_SPECS.keys()


def test_a_cassette_recording_holding_a_telemetry_signal_fails_the_build(tmp_path: Path, monkeypatch):
    # Each study has its own signal table, so the signal named "Marker" belongs to telemetry
    # alone. Before the split, one table over both studies accepted it here.
    study_dir = tmp_path / _STUDY
    study_dir.mkdir()
    _write_recording(study_dir, "SC4901E0", signals=(*_SIGNALS, "Marker"))
    with pytest.raises(TimeFFormatError, match="does not record: 'Marker'"):
        _convert(tmp_path, monkeypatch)
