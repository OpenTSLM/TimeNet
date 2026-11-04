import gc
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pytest

from timenet.errors import TimeFFormatError, TimeFValidationError
from timenet.reader import TimeFReader
from timenet.registry import LocalRegistry
from timenet.types import ClassificationTask
from timenet.types.text import read_value
from timenet_connectors.datasets.zenodo.code_15 import Code15Connector, connector
from timenet_connectors.datasets.zenodo.code_15.connector import _centred_padding


_HEADER = "exam_id,age,is_male,nn_predicted_age,1dAVb,RBBB,LBBB,SB,ST,AF,patient_id,death,timey,normal_ecg,trace_file"
# The fixture's exams, as exams.csv writes them, in an order the parts don't follow. Exam 30 has ST and AF,
# which the CSV writes ST first and the authors' output order AF first.
_ROWS = {
    50: "50,30,False,33.5,True,False,False,False,False,False,8,False,0.0,False,exams_part1.hdf5",
    10: "10,61,True,58.25,False,False,False,False,False,False,7,,,True,exams_part0.hdf5",
    40: "40,45,False,49.0,False,True,False,False,False,False,9,True,1.25,False,exams_part1.hdf5",
    30: "30,60,True,71.75,False,False,False,False,True,True,7,False,3.5,False,exams_part0.hdf5",
    20: "20,30,False,28.0,False,False,False,False,False,False,8,,,True,exams_part0.hdf5",
}
# Each part's exams in row order, with the zeros of the trace before and after, or None for a trace that is
# zero everywhere. Exam 40's 100 zeros at the start alone can't be the authors' centred padding.
_PARTS = {
    0: ((30, (581, 581)), (10, (0, 0)), (20, None)),
    1: ((40, (100, 0)), (50, (47, 48))),
}
# The trace of each exam that has one, and the padding the connector takes from it. Exam 40's zeros at the
# start alone stay, as zeros of the published trace.
_ZEROS = {exam: zeros for exams in _PARTS.values() for exam, zeros in exams}
_PADDING = {30: (581, 581), 10: (0, 0), 40: (0, 0), 50: (47, 48)}
_LEADS = ["DI", "DII", "DIII", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]


@pytest.fixture(autouse=True)
def _fixture_counts(monkeypatch):
    """Pin the fixture's counts: exams 30, 10, 40 and 50 have a trace, from patients 7, 8 and 9."""
    monkeypatch.setattr(connector, "_KEPT_EXAMS", 4)
    monkeypatch.setattr(connector, "_KEPT_PATIENTS", 3)


def _trace(exam_id: int, zeros: tuple[int, int] | None) -> np.ndarray:
    """A published trace whose value names its exam, lead and sample, between ``zeros``.

    Every value is a multiple of 2**-13 below 64, so float32 and halving keep it exact.
    """
    trace = np.zeros((4096, 12), dtype=np.float32)
    if zeros is not None:
        before, after = zeros
        for sample in range(before, 4096 - after):
            trace[sample] = exam_id + np.arange(1, 13) + sample / 8192
        # One lead is zero at the first kept sample. A sample is padding only if every lead is zero.
        trace[before, 5] = 0
    return trace


def _write_part(path: Path, exams: tuple[tuple[int, tuple[int, int] | None], ...]) -> None:
    """Write a part as the release does: the exams' traces, then the all-zero filler with exam id 0."""
    ids = [exam_id for exam_id, _ in exams] + [0]
    traces = [_trace(exam_id, zeros) for exam_id, zeros in exams] + [_trace(0, None)]
    with h5py.File(path, "w") as handle:
        handle.create_dataset("exam_id", data=np.array(ids, dtype=np.int64))
        handle.create_dataset("tracings", data=np.stack(traces))


def _release(tmp_path: Path) -> Path:
    root = tmp_path / "code15"
    root.mkdir()
    (root / "exams.csv").write_text("\n".join([_HEADER, *_ROWS.values()]) + "\n", encoding="utf-8")
    for index, exams in _PARTS.items():
        _write_part(root / f"exams_part{index}.hdf5", exams)
    return root


def _exam_id(record) -> int:
    exam_id = record.metadata["exam_id"]
    assert isinstance(exam_id, int)
    return exam_id


def _expected_millivolts(exam_id: int) -> np.ndarray:
    """The kept samples of an exam in millivolts: the published values, halved, without the padding."""
    before, after = _PADDING[exam_id]
    return _trace(exam_id, _ZEROS[exam_id])[before : 4096 - after] / np.float32(2)


def _read_leads(record) -> dict[str, np.ndarray]:
    return {signal.name: signal.to_numpy() for signal in record.sources[0].signals}


def _output(dataset) -> dict[str, Any]:
    """Everything the dataset holds but the samples, as plain values, with Records keyed by exam."""
    records = {}
    for record in dataset.records:
        (source,) = record.sources
        records[_exam_id(record)] = {
            "metadata": record.metadata,
            "annotations": {a.key: (read_value(a.value), a.unit) for a in record.annotations},
            "source": source.name,
            "signals": sorted(
                (s.name, s.n_values, s.time_axis.offset_us, str(s.spec.unit_value), s.spec.dtype)
                for s in source.signals
            ),
        }
    tasks = [
        (
            type(task).__name__,
            _exam_id(task.inputs[0]),
            tuple(read_value(target) for target in task.targets),
            task.split.value,
            {a.key: (read_value(a.value), a.unit) for a in task.annotations},
            (getattr(task, "unit", None), getattr(task, "target_name", None), getattr(task, "label_set", None)),
            (task.prompt, task.scope, task.input_annotations),
        )
        for task in dataset.iter_tasks()
    ]
    return {
        "annotations": {a.key: read_value(a.value) for a in dataset.annotations},
        "records": records,
        "tasks": tasks,
    }


def _signals(n_values: int) -> list:
    return sorted((lead, n_values, 0, "millivolt", "float32") for lead in _LEADS)


def _record(exam_id, part, pads, annotations, n_values) -> dict:
    return {
        "metadata": {
            "exam_id": exam_id,
            "trace_file": f"exams_part{part}.hdf5",
            "n_pad_before": pads[0],
            "n_pad_after": pads[1],
        },
        "annotations": annotations,
        "source": "12-lead ECG",
        "signals": _signals(n_values),
    }


_NO_INPUT = (None, None, ())
# The whole converted fixture, written out by hand from the fixture rows above.
_EXPECTED: dict[str, Any] = {
    "annotations": {
        "source_url": "https://zenodo.org/records/4916206",
        "source_doi": "10.5281/zenodo.4916206",
    },
    # Exam 20's trace is zero everywhere, so it has no Record and no tasks.
    "records": {
        30: _record(
            30,
            0,
            (581, 581),
            {"subject_ids": (["7"], None), "sex": ("male", None), "death": (False, None), "follow_up": (3.5, "year")},
            2934,
        ),
        # Only the patient's first exam states mortality.
        10: _record(10, 0, (0, 0), {"subject_ids": (["7"], None), "sex": ("male", None)}, 4096),
        # The zeros at the start alone stay, as zeros of the published trace.
        40: _record(
            40,
            1,
            (0, 0),
            {"subject_ids": (["9"], None), "sex": ("female", None), "death": (True, None), "follow_up": (1.25, "year")},
            4096,
        ),
        50: _record(
            50,
            1,
            (47, 48),
            {"subject_ids": (["8"], None), "sex": ("female", None), "death": (False, None), "follow_up": (0.0, "year")},
            4001,
        ),
    },
    # Every exam's age task, then every exam's abnormality task. Exam 30's AF comes before its ST.
    "tasks": [
        (
            "ScalarPredictionTask",
            30,
            (60.0,),
            "test",
            {"nn_predicted_age": (71.75, "year")},
            ("year", "age", None),
            _NO_INPUT,
        ),
        (
            "ScalarPredictionTask",
            10,
            (61.0,),
            "test",
            {"nn_predicted_age": (58.25, "year")},
            ("year", "age", None),
            _NO_INPUT,
        ),
        (
            "ScalarPredictionTask",
            40,
            (45.0,),
            "test",
            {"nn_predicted_age": (49.0, "year")},
            ("year", "age", None),
            _NO_INPUT,
        ),
        (
            "ScalarPredictionTask",
            50,
            (30.0,),
            "test",
            {"nn_predicted_age": (33.5, "year")},
            ("year", "age", None),
            _NO_INPUT,
        ),
        (
            "ClassificationTask",
            30,
            ("AF", "ST"),
            "train",
            {"normal_ecg": (False, None)},
            (None, None, connector.ABNORMALITIES),
            _NO_INPUT,
        ),
        (
            "ClassificationTask",
            10,
            (),
            "train",
            {"normal_ecg": (True, None)},
            (None, None, connector.ABNORMALITIES),
            _NO_INPUT,
        ),
        (
            "ClassificationTask",
            40,
            ("RBBB",),
            "train",
            {"normal_ecg": (False, None)},
            (None, None, connector.ABNORMALITIES),
            _NO_INPUT,
        ),
        (
            "ClassificationTask",
            50,
            ("1dAVb",),
            "train",
            {"normal_ecg": (False, None)},
            (None, None, connector.ABNORMALITIES),
            _NO_INPUT,
        ),
    ],
}


def _check_values(dataset) -> None:
    """Check every lead of every Record holds that exam's published values, halved, without the padding."""
    for record in dataset.records:
        leads = _read_leads(record)
        expected = _expected_millivolts(_exam_id(record))
        assert sorted(leads) == sorted(_LEADS)
        for column, lead in enumerate(_LEADS):
            assert leads[lead].dtype == np.float32
            np.testing.assert_array_equal(leads[lead], expected[:, column])


# Two rows per block put exam 20 (row 2 of part 0) in a second block.
@pytest.mark.parametrize("scan_rows", [connector._SCAN_ROWS, 2])
def test_converts_the_fixture(tmp_path, monkeypatch, scan_rows):
    monkeypatch.setattr(connector, "_SCAN_ROWS", scan_rows)
    dataset = Code15Connector().convert([_release(tmp_path)])
    assert _output(dataset) == _EXPECTED
    _check_values(dataset)


def test_storage_round_trip_keeps_the_whole_conversion(tmp_path):
    original = Code15Connector().convert([_release(tmp_path)])
    registry = LocalRegistry(tmp_path / "registry")
    registry.store(original)
    with TimeFReader(registry.open_version("zenodo/code-15", "1.0.0")) as reader:
        restored = reader.read()
        assert _output(restored) == _EXPECTED
        _check_values(restored)
        # The reader gives every abnormality task one shared, multi-label set.
        label_sets = {id(task.label_set) for task in restored.iter_tasks() if isinstance(task, ClassificationTask)}
        assert len(label_sets) == 1


def test_the_abnormality_label_set_is_multi_label_in_the_authors_order():
    assert connector.ABNORMALITIES.multi_label
    assert connector.ABNORMALITIES.names == ("1dAVb", "RBBB", "LBBB", "SB", "AF", "ST")
    assert connector.ABNORMALITIES.labels[4].description == "Atrial fibrillation."


def test_the_published_trace_can_be_rebuilt_from_the_stored_values(tmp_path):
    # The Record keeps the padding counts, and the published values are twice the millivolts.
    dataset = Code15Connector().convert([_release(tmp_path)])
    for record in dataset.records:
        exam_id = _exam_id(record)
        before, after = record.metadata["n_pad_before"], record.metadata["n_pad_after"]
        assert isinstance(before, int)
        assert isinstance(after, int)
        leads = _read_leads(record)
        rebuilt = np.zeros((4096, 12), dtype=np.float32)
        rebuilt[before : 4096 - after] = np.stack([leads[lead] for lead in _LEADS], axis=1) * np.float32(2)
        np.testing.assert_array_equal(rebuilt, _trace(exam_id, _ZEROS[exam_id]))


@pytest.mark.parametrize(
    ("leading", "trailing", "padding"),
    [
        # A 2934-sample trace: (4096 - 2934) // 2 = 581 zeros each side.
        (581, 581, (581, 581)),
        # An odd length puts the extra zero after.
        (47, 48, (47, 48)),
        # A cut trace has no zeros at either end.
        (0, 0, (0, 0)),
        # Zeros at the start alone aren't padding, which puts at least as many after.
        (100, 0, (0, 0)),
        # Of the zeros the rule can't explain, the ones at the start stay.
        (200, 106, (106, 106)),
        # More zeros at the end than one past the start's: the rest stay.
        (10, 30, (10, 11)),
        (0, 1, (0, 1)),
    ],
)
def test_centred_padding_follows_preprocess_ecg(leading, trailing, padding):
    assert tuple(_centred_padding(leading, trailing)) == padding


def _edit_csv(root: Path, old: str, new: str) -> None:
    path = root / "exams.csv"
    path.write_text(path.read_text(encoding="utf-8").replace(old, new), encoding="utf-8")


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ("exam_id,age", "exam,age", "expected the header"),
        (_ROWS[50], _ROWS[50].replace("50,30,", "50,30,x,"), r"exams\.csv:2: expected 15 fields, got 16"),
        (
            _ROWS[10],
            _ROWS[10].replace(",True,58.25", ",true,58.25"),
            r"exams\.csv:3: is_male: expected True or False",
        ),
        (_ROWS[40], _ROWS[40].replace("True,1.25", ",1.25"), r"exams\.csv:4: death and timey"),
        (_ROWS[40], _ROWS[40].replace("1.25", "nan"), r"exams\.csv:4: timey: expected a decimal number"),
        (_ROWS[40], _ROWS[40].replace("1.25", "1_0"), r"exams\.csv:4: timey: expected a decimal number"),
        (_ROWS[40], _ROWS[40].replace("1.25", " 1.25"), r"exams\.csv:4: timey: expected a decimal number"),
        (_ROWS[40], _ROWS[40].replace("1.25", "-1"), r"exams\.csv:4: timey: expected a number of at least 0"),
        (_ROWS[40], _ROWS[40].replace("49.0", ""), r"exams\.csv:4: nn_predicted_age: expected a decimal number"),
        (_ROWS[30], _ROWS[30].replace("exams_part0.hdf5", "part0.hdf5"), r"exams\.csv:5: trace_file 'part0.hdf5'"),
        (_ROWS[30], _ROWS[30].replace("30,60,", "30,x,"), r"exams\.csv:5: age: expected an integer of at least 0"),
        (_ROWS[30], _ROWS[30].replace(",7,False", ",x,False"), r"exams\.csv:5: patient_id: expected an integer"),
        (_ROWS[30], _ROWS[30].replace("3.5,False", "3.5,True"), r"exams\.csv:5: normal_ecg is True"),
        (_ROWS[30], _ROWS[30].replace("False,True,True,7", "True,True,True,7"), r"exams\.csv:5: SB and ST"),
        (_ROWS[20], _ROWS[30], "exam 30 is listed twice"),
        (_ROWS[10], _ROWS[10].replace(",True,58.25", ",False,58.25"), "patient 7 is listed as both sexes"),
        (_ROWS[20], _ROWS[20].replace(",,,True", ",False,2.0,True"), "patient 8's mortality is stated twice"),
    ],
)
def test_malformed_exams_csv_fails_with_its_path_and_line(tmp_path, old, new, message):
    root = _release(tmp_path)
    _edit_csv(root, old, new)
    with pytest.raises(TimeFFormatError, match=message):
        Code15Connector().convert([root])


def test_an_exams_csv_without_rows_fails(tmp_path):
    root = _release(tmp_path)
    (root / "exams.csv").write_text(_HEADER + "\n", encoding="utf-8")
    with pytest.raises(TimeFFormatError, match="lists no exams"):
        Code15Connector().convert([root])


@pytest.mark.parametrize(
    ("part", "exams", "message"),
    [
        (0, ((30, (581, 581)), (10, (0, 0)), (20, None), (60, (0, 0))), "exam 60, which exams.csv does not list in"),
        (1, ((40, (100, 0)), (50, (47, 48)), (30, (0, 0))), "exam 30, which exams.csv does not list in exams_part1"),
        (1, ((40, (100, 0)), (50, (47, 48)), (40, (0, 0))), "exam 40 has two traces"),
        (1, ((40, (100, 0)),), "1 exams of exams.csv have no trace, such as \\[50\\]"),
    ],
)
def test_parts_that_dont_match_exams_csv_fail(tmp_path, part, exams, message):
    root = _release(tmp_path)
    _write_part(root / f"exams_part{part}.hdf5", exams)
    with pytest.raises(TimeFFormatError, match=message):
        Code15Connector().convert([root])


def _rewrite_part1(root: Path, ids, traces, dtype=np.float32) -> None:
    with h5py.File(root / "exams_part1.hdf5", "w") as handle:
        handle.create_dataset("exam_id", data=np.array(ids, dtype=np.int64))
        handle.create_dataset("tracings", data=np.stack(traces).astype(dtype))


def _drop_filler(root: Path) -> None:
    _rewrite_part1(root, [40, 50], [_trace(40, (100, 0)), _trace(50, (47, 48))])


def _filler_id_first(root: Path) -> None:
    _rewrite_part1(root, [0, 50, 0], [_trace(40, (100, 0)), _trace(50, (47, 48)), _trace(0, None)])


def _add_dataset(root: Path) -> None:
    with h5py.File(root / "exams_part1.hdf5", "a") as handle:
        handle.create_dataset("signal", data=np.zeros(1))


def _float64_traces(root: Path) -> None:
    traces = [_trace(40, (100, 0)), _trace(50, (47, 48)), _trace(0, None)]
    _rewrite_part1(root, [40, 50, 0], traces, dtype=np.float64)


def _zero_exam_50(root: Path) -> None:
    with h5py.File(root / "exams_part1.hdf5", "a") as handle:
        handle["tracings"][1] = 0


def _nan_value(root: Path) -> None:
    with h5py.File(root / "exams_part1.hdf5", "a") as handle:
        handle["tracings"][1, 2000, 3] = np.nan


@pytest.mark.parametrize(
    ("damage", "message"),
    [
        (_drop_filler, "the last row is not the all-zero filler"),
        (_filler_id_first, "a row other than the last has the filler's exam id"),
        (_add_dataset, "expected the datasets exam_id and tracings"),
        (_float64_traces, "expected int64 ids and float32 traces"),
        (lambda root: (root / "exams_part1.hdf5").unlink(), r"exams_part1\.hdf5 is missing"),
        (_nan_value, "exam 50's trace holds a value that is not finite"),
        # A trace that reads as zero everywhere loses its exam, and here its patient (8), which the counts catch.
        (_zero_exam_50, "expected 4 exams from 3 patients .*, got 3 exams from 2 patients"),
        (lambda root: (root / "exams.csv").unlink(), r"exams\.csv is missing"),
    ],
)
def test_a_damaged_release_fails(tmp_path, damage, message):
    root = _release(tmp_path)
    damage(root)
    with pytest.raises(TimeFFormatError, match=message):
        Code15Connector().convert([root])


def test_more_than_one_directory_fails(tmp_path):
    with pytest.raises(TimeFValidationError, match="converts one directory"):
        Code15Connector().convert([tmp_path, tmp_path])


def test_a_path_that_is_not_a_directory_fails(tmp_path):
    with pytest.raises(TimeFValidationError, match="converts one directory"):
        Code15Connector().convert([tmp_path / "missing"])


def test_a_changed_value_in_the_samples_fails_when_read(tmp_path):
    root = _release(tmp_path)
    dataset = Code15Connector().convert([root])
    # The zero runs and the shape stay as scanned. Only a digest of the values sees this.
    with h5py.File(root / "exams_part0.hdf5", "a") as handle:
        handle["tracings"][0, 1000, 3] += 1
    record = next(r for r in dataset.records if _exam_id(r) == 30)
    with pytest.raises(TimeFFormatError, match=r"row 0 \(exam 30\) changed since it was scanned"):
        record.sources[0].signals[0].to_numpy()


def _open_hdf5_files() -> int:
    return h5py.h5f.get_obj_count(h5py.h5f.OBJ_ALL, h5py.h5f.OBJ_FILE)


def test_a_build_keeps_one_part_open_and_closes_it(tmp_path):
    root = _release(tmp_path)
    gc.collect()  # Readers of earlier tests close when collected.
    before = _open_hdf5_files()
    dataset = Code15Connector().convert([root])
    assert _open_hdf5_files() == before
    for record in dataset.records:
        _read_leads(record)
        assert _open_hdf5_files() == before + 1
    del dataset, record
    gc.collect()
    assert _open_hdf5_files() == before
