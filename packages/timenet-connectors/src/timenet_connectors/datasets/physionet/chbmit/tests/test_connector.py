from datetime import date, time
from pathlib import Path
import shutil

import edfio
import numpy as np
import pytest

from timenet.dataset import Record, TimeFDataset
from timenet.engine import store_dataset
from timenet.errors import TimeFFormatError
from timenet.reader import TimeFReader
from timenet.registry import open_local_version
from timenet.types import (
    US_PER_S,
    ClassificationTask,
    LocalizationMode,
    Split,
    TemporalLocalizationTask,
    TimeInterval,
    read_value,
)
from timenet_connectors.datasets.physionet.chbmit.connector import (
    ChbMitCase,
    ChbMitConnector,
    ChbMitSource,
    parse_checksums,
)


# A synthetic release of two cases. These tests write it, and it holds no byte of the real
# release. The two summaries in fixtures/ have the shape of the real ones; every time and seizure
# in them is invented. The cases are named chb01 and chb21 because the connector maps chb21 to the
# subject of chb01, and that mapping is under test.
_FIXTURES = Path(__file__).parent / "fixtures"
_RATE_HZ = 256
_FILE_SECONDS = 11  # five 2 s windows and one second left over

# Each file: its case, its name, its EDF channel labels, and the clock its header states.
_FILES = (
    ("chb01", "chb01_01.edf", ("FP1-F7", "-", "T8-P8", "FZ-CZ", "T8-P8"), date(1990, 1, 1), time(23, 59, 50)),
    ("chb01", "chb01_02.edf", ("FP1-F7", "-", "T8-P8", "FZ-CZ", "T8-P8"), date(1990, 1, 2), time(0, 0, 4)),
    ("chb21", "chb21_01.edf", ("FP1-F7", "T8-P8", "T8-P8"), date(1991, 7, 1), time(10, 0, 0)),
    ("chb21", "chb21_02.edf", ("FP1-F7", ".", "T8-P8", "ECG"), date(1991, 7, 1), time(12, 0, 0)),
)


def _write_edf(path: Path, labels: tuple[str, ...], start_date: date, start_time: time) -> None:
    rng = np.random.default_rng(0)
    signals = [
        edfio.EdfSignal(
            np.zeros(_RATE_HZ * _FILE_SECONDS)
            if label in {"-", "."}
            else rng.uniform(-100, 100, _RATE_HZ * _FILE_SECONDS),
            _RATE_HZ,
            label=label,
            physical_dimension="uV",
            physical_range=(-800.0, 800.0),
            digital_range=(-2048, 2047),
        )
        for label in labels
    ]
    edf = edfio.Edf(signals, starttime=start_time, data_record_duration=1)
    edf.startdate = start_date
    edf.write(path)


@pytest.fixture(scope="module")
def source(tmp_path_factory) -> ChbMitSource:
    root = tmp_path_factory.mktemp("chbmit_release")
    cases = []
    for case_id in ("chb01", "chb21"):
        case_dir = root / case_id
        case_dir.mkdir()
        summary_path = case_dir / f"{case_id}-summary.txt"
        shutil.copy(_FIXTURES / summary_path.name, summary_path)
        edf_paths = []
        for file_case, name, labels, start_date, start_time in _FILES:
            if file_case == case_id:
                _write_edf(case_dir / name, labels, start_date, start_time)
                edf_paths.append(case_dir / name)
        cases.append(ChbMitCase(case_id=case_id, summary_path=summary_path, edf_paths=tuple(edf_paths)))
    return ChbMitSource(cases=tuple(cases))


@pytest.fixture(scope="module")
def dataset(source) -> TimeFDataset:
    return ChbMitConnector().convert(source)


def _record(dataset: TimeFDataset, file_name: str) -> Record:
    return next(record for record in dataset.records if record.metadata["file_name"] == file_name)


def _annotation_value(record: Record, key: str) -> object:
    return read_value(next(one for one in record.annotations if one.key == key).value)


def _tasks_of(dataset: TimeFDataset, record: Record, task_type: type) -> list:
    return [task for task in dataset.iter_tasks() if isinstance(task, task_type) and task.inputs[0] is record]


def test_convert_gives_one_record_for_each_edf_file(dataset):
    assert [record.metadata for record in dataset.records] == [
        {"case_id": case_id, "file_name": name} for case_id, name, *_ in _FILES
    ]
    assert all(record.time_span == TimeInterval.micros(0, _FILE_SECONDS * US_PER_S) for record in dataset.records)


def test_channels_follow_each_file_drop_placeholders_and_number_repeats(dataset):
    chb01_01 = _record(dataset, "chb01_01.edf").sources[0].signals
    assert [signal.name for signal in chb01_01] == ["FP1-F7", "T8-P8", "FZ-CZ", "T8-P8 (2)"]
    chb21_01 = _record(dataset, "chb21_01.edf").sources[0].signals
    assert [signal.name for signal in chb21_01] == ["FP1-F7", "T8-P8", "T8-P8 (2)"]
    # chb21 changes its montage between its two files.
    signals = _record(dataset, "chb21_02.edf").sources[0].signals
    assert [(signal.name, signal.spec.spec_type) for signal in signals] == [
        ("FP1-F7", "eeg"),
        ("T8-P8", "eeg"),
        ("ECG", "ecg"),
    ]


def test_a_signal_keeps_the_samples_of_its_own_header_position(dataset, source):
    # chb21_02 drops the placeholder at position 1, so T8-P8 must still read position 2.
    record = _record(dataset, "chb21_02.edf")
    path = source.cases[1].edf_paths[1]
    expected = edfio.read_edf(path).signals[2].data
    (t8_p8,) = [signal for signal in record.signals if signal.name == "T8-P8"]
    np.testing.assert_allclose(t8_p8.to_numpy(), expected, atol=1e-3)


def test_chb21_is_the_subject_of_chb01_and_shares_its_split(dataset):
    for record in dataset.records:
        assert _annotation_value(record, "subject_ids") == ["chb01"]
    assert {task.split for task in dataset.iter_tasks()} == {Split.TRAIN}


def test_case_offset_counts_seconds_from_the_case_start_across_midnight(dataset):
    assert _annotation_value(_record(dataset, "chb01_01.edf"), "case_offset") == 0
    assert _annotation_value(_record(dataset, "chb01_02.edf"), "case_offset") == 14
    assert _annotation_value(_record(dataset, "chb21_02.edf"), "case_offset") == 2 * 60 * 60


def test_localization_targets_match_the_summary(dataset):
    expected = {
        "chb01_01.edf": (),
        "chb01_02.edf": (TimeInterval.seconds(3, 6),),
        "chb21_01.edf": (),
        "chb21_02.edf": (TimeInterval.seconds(2, 4), TimeInterval.seconds(8, 10)),
    }
    for file_name, targets in expected.items():
        record = _record(dataset, file_name)
        (task,) = _tasks_of(dataset, record, TemporalLocalizationTask)
        assert task.mode is LocalizationMode.EXHAUSTIVE
        assert task.targets == targets
        assert tuple(one.span for one in record.annotations if one.key == "seizure") == targets


def test_window_tasks_cover_whole_2_s_windows_and_label_any_overlap(dataset):
    record = _record(dataset, "chb21_02.edf")
    tasks = _tasks_of(dataset, record, ClassificationTask)
    # 11 s give five whole windows; the last second gets no task.
    assert [task.scope for task in tasks] == [TimeInterval.seconds(start, start + 2) for start in range(0, 10, 2)]
    # The seizures are [2, 4) and [8, 10). The window [0, 2) ends where the first one starts, and
    # [4, 6) starts where it ends, so neither overlaps it.
    assert [task.targets for task in tasks] == [
        ("no seizure",),
        ("seizure",),
        ("no seizure",),
        ("no seizure",),
        ("seizure",),
    ]


def test_a_seizure_across_two_windows_labels_both(dataset):
    tasks = _tasks_of(dataset, _record(dataset, "chb01_02.edf"), ClassificationTask)
    assert [task.targets[0] for task in tasks] == ["no seizure", "seizure", "seizure", "no seizure", "no seizure"]


def test_the_dataset_round_trips_through_the_writer(dataset, tmp_path):
    version_dir = store_dataset(dataset, tmp_path / "out")
    with TimeFReader(open_local_version(version_dir)) as reader:
        restored = reader.read()
        tasks = list(restored.iter_tasks())
    assert len(restored.records) == len(_FILES)
    assert len(tasks) == len(_FILES) * (1 + 5)
    record = _record(restored, "chb21_02.edf")
    localization = next(
        task for task in tasks if isinstance(task, TemporalLocalizationTask) and task.inputs[0].id == record.id
    )
    assert localization.targets == (TimeInterval.seconds(2, 4), TimeInterval.seconds(8, 10))


def test_convert_refuses_a_file_whose_header_disagrees_with_the_summary(source, tmp_path):
    case = source.cases[0]
    edf_path = tmp_path / "chb01_01.edf"
    _write_edf(edf_path, ("FP1-F7", "T8-P8"), date(1990, 1, 1), time(23, 59, 50))
    changed = ChbMitCase(case_id="chb01", summary_path=case.summary_path, edf_paths=(edf_path,))
    with pytest.raises(TimeFFormatError, match="the case summary lists"):
        ChbMitConnector().convert(ChbMitSource(cases=(changed,)))


def test_convert_refuses_a_file_the_summary_does_not_list(source, tmp_path):
    case = source.cases[0]
    edf_path = tmp_path / "chb01_03.edf"
    _write_edf(edf_path, ("FP1-F7", "-", "T8-P8", "FZ-CZ", "T8-P8"), date(1990, 1, 1), time(23, 59, 50))
    unlisted = ChbMitCase(case_id="chb01", summary_path=case.summary_path, edf_paths=(edf_path,))
    with pytest.raises(TimeFFormatError, match=r"does not list chb01_03\.edf"):
        ChbMitConnector().convert(ChbMitSource(cases=(unlisted,)))


def test_parse_checksums_reads_each_digest_by_release_path(tmp_path):
    path = tmp_path / "SHA256SUMS.txt"
    path.write_text("ab12 chb02/chb02_16+.edf\ncd34 RECORDS\n", encoding="utf-8")
    assert parse_checksums(path) == {"chb02/chb02_16+.edf": "ab12", "RECORDS": "cd34"}
