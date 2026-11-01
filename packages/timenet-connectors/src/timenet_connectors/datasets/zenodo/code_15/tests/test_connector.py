import gc
from pathlib import Path
from typing import ClassVar

import h5py
import numpy as np
import pytest

from timenet.errors import TimeFFormatError, TimeFValidationError
from timenet.reader import TimeFReader
from timenet.registry import LocalRegistry
from timenet.types import ClassificationTask, ScalarPredictionTask, Split
from timenet.types.text import read_value
from timenet_connectors.datasets.zenodo.code_15 import Code15Connector
from timenet_connectors.datasets.zenodo.code_15.loader import Code15DatasetLoader
from timenet_connectors.datasets.zenodo.code_15.preprocessing import Padding, authors_input, centred_padding
from timenet_connectors.datasets.zenodo.code_15.records import record_metadata
from timenet_connectors.datasets.zenodo.code_15.release import PINNED_COUNTS, Part, PinnedCounts
from timenet_connectors.datasets.zenodo.code_15.tracings import ExamLocation, TraceReader
from timenet_connectors.datasets.zenodo.code_15.vocabulary import Lead


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
# Each part's exams in row order, and each trace's zeros: the padding before and after, or None for a trace
# that is zero everywhere. Exam 40's 100 zeros at the start alone can't be the authors' centred padding.
_PARTS = {
    0: ((30, Padding(581, 581)), (10, Padding(0, 0)), (20, None)),
    1: ((40, Padding(100, 0)), (50, Padding(47, 48))),
}
_FIXTURE_PARTS = (Part(0, "unused"), Part(1, "unused"))
_FIXTURE_COUNTS = PinnedCounts(exams=5, patients=3, all_zero_exam_ids=frozenset({20}))


class _FixtureConnector(Code15Connector):
    """The connector, pinned to the fixture's parts and counts instead of the release's."""

    CARD = Path(__file__).parents[1] / "dataset.yaml"
    SOURCE_PARTS: ClassVar[tuple[Part, ...]] = _FIXTURE_PARTS
    SOURCE_COUNTS: ClassVar[PinnedCounts] = _FIXTURE_COUNTS


def _trace(exam_id: int, zeros: Padding | None) -> np.ndarray:
    """A published trace: nonzero float32 values, seeded by the exam, between ``zeros``."""
    trace = np.zeros((4096, 12), dtype=np.float32)
    if zeros is not None:
        values = np.random.default_rng(exam_id).uniform(0.5, 3.0, (4096 - zeros.start - zeros.end, 12))
        trace[zeros.start : 4096 - zeros.end] = values.astype(np.float32)
    return trace


def _write_part(path: Path, exams: tuple[tuple[int, Padding | None], ...]) -> None:
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


def _loader(root: Path) -> Code15DatasetLoader:
    return Code15DatasetLoader.parse_from([root], parts=_FIXTURE_PARTS, counts=_FIXTURE_COUNTS)


def _output(dataset) -> dict:
    """Everything the dataset holds but the samples, as plain values, with Records keyed by exam."""
    records = {}
    for record in dataset.records:
        metadata = record_metadata(record)
        (source,) = record.sources
        records[metadata.exam_id] = {
            "subject_id": metadata.subject_id,
            "trace_file": metadata.trace_file,
            "padding": tuple(metadata.padding),
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
            record_metadata(task.inputs[0]).exam_id,
            tuple(read_value(target) for target in task.targets),
            task.split.value,
            {a.key: (read_value(a.value), a.unit) for a in task.annotations},
        )
        for task in dataset.iter_tasks()
    ]
    annotations = {a.key: read_value(a.value) for a in dataset.annotations}
    return {"annotations": annotations, "records": records, "tasks": tasks}


def _signals(n_values: int) -> list:
    return sorted((lead.value, n_values, 0, "millivolt", "float32") for lead in Lead)


_SEX = {"male": ("male", None), "female": ("female", None)}
# The whole converted fixture, written out by hand from the fixture rows above.
_EXPECTED = {
    "annotations": {
        "source_url": "https://zenodo.org/records/4916206",
        "source_doi": "10.5281/zenodo.4916206",
        "abnormality_vocabulary": ["1dAVb", "RBBB", "LBBB", "SB", "AF", "ST"],
    },
    # By part, then by row: exam 20's trace is zero everywhere, so it has no Record.
    "records": {
        30: {
            "subject_id": "7",
            "trace_file": "exams_part0.hdf5",
            "padding": (581, 581),
            "annotations": {"sex": _SEX["male"], "death": (False, None), "follow_up": (3.5, "year")},
            "source": "12-lead ECG",
            "signals": _signals(2934),
        },
        10: {
            "subject_id": "7",
            "trace_file": "exams_part0.hdf5",
            "padding": (0, 0),
            # Only the patient's first exam states mortality.
            "annotations": {"sex": _SEX["male"]},
            "source": "12-lead ECG",
            "signals": _signals(4096),
        },
        40: {
            "subject_id": "9",
            "trace_file": "exams_part1.hdf5",
            # The zeros at the start alone stay, as zeros of the published trace.
            "padding": (0, 0),
            "annotations": {"sex": _SEX["female"], "death": (True, None), "follow_up": (1.25, "year")},
            "source": "12-lead ECG",
            "signals": _signals(4096),
        },
        50: {
            "subject_id": "8",
            "trace_file": "exams_part1.hdf5",
            "padding": (47, 48),
            "annotations": {"sex": _SEX["female"], "death": (False, None), "follow_up": (0.0, "year")},
            "source": "12-lead ECG",
            "signals": _signals(4001),
        },
    },
    # Every exam's age task, then every exam's abnormality task. Exam 30's AF comes before its ST.
    "tasks": [
        ("ScalarPredictionTask", 30, (60.0,), "test", {"nn_predicted_age": (71.75, "year")}),
        ("ScalarPredictionTask", 10, (61.0,), "test", {"nn_predicted_age": (58.25, "year")}),
        ("ScalarPredictionTask", 40, (45.0,), "test", {"nn_predicted_age": (49.0, "year")}),
        ("ScalarPredictionTask", 50, (30.0,), "test", {"nn_predicted_age": (33.5, "year")}),
        ("ClassificationTask", 30, ("AF", "ST"), "train", {"normal_ecg": (False, None)}),
        ("ClassificationTask", 10, (), "train", {"normal_ecg": (True, None)}),
        ("ClassificationTask", 40, ("RBBB",), "train", {"normal_ecg": (False, None)}),
        ("ClassificationTask", 50, ("1dAVb",), "train", {"normal_ecg": (False, None)}),
    ],
}


def test_converts_the_fixture(tmp_path):
    dataset = _FixtureConnector().convert([_release(tmp_path)])
    assert _output(dataset) == _EXPECTED


def test_storage_round_trip_keeps_the_whole_conversion(tmp_path):
    original = _FixtureConnector().convert([_release(tmp_path)])
    registry = LocalRegistry(tmp_path / "registry")
    registry.store(original)
    with TimeFReader(registry.open_version("zenodo/code-15", "1.0.0")) as reader:
        restored = reader.read()
        assert _output(restored) == _EXPECTED
        for record in restored.records:
            metadata = record_metadata(record)
            assert _millivolts(record).tobytes() == _published(metadata.exam_id, metadata.padding).tobytes()


def _millivolts(record) -> np.ndarray:
    signals = {signal.name: signal for signal in record.sources[0].signals}
    return np.stack([signals[lead.value].to_numpy() for lead in Lead], axis=1)


def _published(exam_id: int, padding: Padding) -> np.ndarray:
    """The fixture's samples of an exam, halved, without the padding."""
    zeros = dict(exam for exams in _PARTS.values() for exam in exams)[exam_id]
    return _trace(exam_id, zeros)[padding.start : 4096 - padding.end] / np.float32(2)


def test_signals_rebuild_the_authors_input_bit_for_bit(tmp_path):
    root = _release(tmp_path)
    dataset = _FixtureConnector().convert([root])
    for record in dataset.records:
        metadata = record_metadata(record)
        millivolts = _millivolts(record)
        assert millivolts.dtype == np.float32
        zeros = dict(exam for exams in _PARTS.values() for exam in exams)[metadata.exam_id]
        np.testing.assert_array_equal(authors_input(millivolts, metadata.padding), _trace(metadata.exam_id, zeros))


@pytest.mark.parametrize(
    ("leading", "trailing", "padding"),
    [
        # A 2934-sample trace: (4096 - 2934) // 2 = 581 zeros each side.
        (581, 581, Padding(581, 581)),
        # An odd length puts the extra zero after.
        (47, 48, Padding(47, 48)),
        # A cut trace has no zeros at either end.
        (0, 0, Padding(0, 0)),
        # Zeros at the start alone aren't padding, which puts at least as many after.
        (100, 0, Padding(0, 0)),
        # Of the zeros the rule can't explain, the ones at the start stay.
        (200, 106, Padding(106, 106)),
        # More zeros at the end than one past the start's: the rest stay.
        (10, 30, Padding(10, 11)),
        (0, 1, Padding(0, 1)),
    ],
)
def test_centred_padding_follows_preprocess_ecg(leading, trailing, padding):
    assert centred_padding(leading, trailing) == padding


def test_tasks_carry_their_papers_split(tmp_path):
    loader = _loader(_release(tmp_path))
    age = list(loader.age_tasks())
    abnormality = list(loader.abnormality_tasks())
    assert len(age) == len(abnormality) == 4
    for task in age:
        assert isinstance(task, ScalarPredictionTask)
        assert task.split is Split.TEST
        assert (task.unit, task.target_name) == ("year", "age")
    for task in abnormality:
        assert isinstance(task, ClassificationTask)
        assert task.split is Split.TRAIN
        assert task.target_schema == "abnormality_vocabulary"


def test_tasks_yield_each_method_in_turn(tmp_path):
    loader = _loader(_release(tmp_path))
    stream = [(type(task), task.inputs[0], task.split) for task in loader.tasks()]
    by_method = [
        (type(task), task.inputs[0], task.split) for task in [*loader.age_tasks(), *loader.abnormality_tasks()]
    ]
    assert stream == by_method


def test_targets_stay_off_the_inputs(tmp_path):
    # The age and the labels are targets, and the published prediction and normal_ecg task annotations; no
    # task passes the model an annotation, and the Record states neither.
    dataset = _FixtureConnector().convert([_release(tmp_path)])
    for record in dataset.records:
        assert {a.key for a in record.annotations} <= {"sex", "death", "follow_up"}
    for task in dataset.iter_tasks():
        assert task.input_annotations == ()
        assert task.prompt is None
        assert task.scope is None


def test_pinned_counts_mismatch_fails(tmp_path):
    with pytest.raises(
        TimeFValidationError, match="differs from the pinned release: exams 5 where the pinned release holds 345779"
    ):
        Code15DatasetLoader.parse_from([_release(tmp_path)], parts=_FIXTURE_PARTS, counts=PINNED_COUNTS)


def _edit_csv(root: Path, old: str, new: str) -> None:
    path = root / "exams.csv"
    path.write_text(path.read_text(encoding="utf-8").replace(old, new), encoding="utf-8")


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ("exam_id,age", "exam,age", "expected the header"),
        (_ROWS[50], _ROWS[50].replace("50,30,", "50,30,x,"), r"exams\.csv:2: expected 15 fields, got 16"),
        (_ROWS[10], _ROWS[10].replace(",True,58.25", ",true,58.25"), r"exams\.csv:3: .*\nis_male"),
        (_ROWS[40], _ROWS[40].replace("True,1.25", ",1.25"), r"(?s)exams\.csv:4: .*death and timey"),
        (_ROWS[40], _ROWS[40].replace("1.25", "nan"), r"exams\.csv:4: .*\ntimey"),
        (_ROWS[30], _ROWS[30].replace("exams_part0.hdf5", "part0.hdf5"), r"exams\.csv:5: .*\ntrace_file"),
        (_ROWS[30], _ROWS[30].replace("3.5,False", "3.5,True"), r"(?s)exams\.csv:5: .*normal_ecg is True"),
        (_ROWS[30], _ROWS[30].replace("False,True,True,7", "True,True,True,7"), r"(?s)exams\.csv:5: .*SB and ST"),
        (_ROWS[20], _ROWS[30], "exam 30 is listed twice"),
        (_ROWS[10], _ROWS[10].replace(",True,58.25", ",False,58.25"), "patient 7 is listed as both sexes"),
        (_ROWS[20], _ROWS[20].replace(",,,True", ",False,2.0,True"), "patient 8's mortality is stated on two exams"),
    ],
)
def test_malformed_exams_csv_fails_with_its_path(tmp_path, old, new, message):
    root = _release(tmp_path)
    _edit_csv(root, old, new)
    with pytest.raises(TimeFFormatError, match=message):
        _loader(root)


@pytest.mark.parametrize(
    ("part", "exams", "message"),
    [
        (0, ((30, Padding(581, 581)), (10, Padding(0, 0)), (20, None), (60, Padding(0, 0))), "doesn't list"),
        (1, ((40, Padding(100, 0)), (50, Padding(47, 48)), (30, Padding(0, 0))), "doesn't list in exams_part1"),
        (1, ((40, Padding(100, 0)), (50, Padding(47, 48)), (40, Padding(0, 0))), "exam 40 has two traces"),
        (1, ((40, Padding(100, 0)),), "exams of exams.csv have no trace"),
    ],
)
def test_parts_that_dont_match_exams_csv_fail(tmp_path, part, exams, message):
    root = _release(tmp_path)
    _write_part(root / f"exams_part{part}.hdf5", exams)
    with pytest.raises(TimeFFormatError, match=message):
        _loader(root)


def test_a_part_without_its_filler_fails(tmp_path):
    root = _release(tmp_path)
    with h5py.File(root / "exams_part1.hdf5", "w") as handle:
        handle.create_dataset("exam_id", data=np.array([40, 50], dtype=np.int64))
        handle.create_dataset("tracings", data=np.stack([_trace(40, Padding(100, 0)), _trace(50, Padding(47, 48))]))
    with pytest.raises(TimeFFormatError, match="the last row isn't the all-zero filler"):
        _loader(root)


def test_a_part_with_other_datasets_fails(tmp_path):
    root = _release(tmp_path)
    with h5py.File(root / "exams_part1.hdf5", "a") as handle:
        handle.create_dataset("signal", data=np.zeros(1))
    with pytest.raises(TimeFFormatError, match="expected the datasets exam_id and tracings"):
        _loader(root)


def test_a_missing_part_fails(tmp_path):
    root = _release(tmp_path)
    (root / "exams_part1.hdf5").unlink()
    with pytest.raises(TimeFFormatError, match=r"exams_part1\.hdf5 is missing"):
        _loader(root)


def test_a_value_that_isnt_finite_fails_when_scanned(tmp_path):
    root = _release(tmp_path)
    with h5py.File(root / "exams_part1.hdf5", "a") as handle:
        handle["tracings"][1, 2000, 3] = np.nan
    with pytest.raises(TimeFFormatError, match="exam 50's trace holds a value that isn't finite"):
        _loader(root)


def test_a_trace_changed_after_the_scan_fails_when_read(tmp_path):
    root = _release(tmp_path)
    dataset = _FixtureConnector().convert([root])
    with h5py.File(root / "exams_part0.hdf5", "a") as handle:
        handle["tracings"][0, :600] = 0
    record = next(r for r in dataset.records if record_metadata(r).exam_id == 30)
    with pytest.raises(TimeFFormatError, match="row 0 changed since it was scanned"):
        record.sources[0].signals[0].to_numpy()


def _open_hdf5_files() -> int:
    return h5py.h5f.get_obj_count(h5py.h5f.OBJ_ALL, h5py.h5f.OBJ_FILE)


def test_the_trace_reader_keeps_one_part_open_and_closes_it(tmp_path):
    root = _release(tmp_path)
    # Readers of earlier tests close when collected, so collect them before counting.
    gc.collect()
    before = _open_hdf5_files()
    reader = TraceReader()
    for part in (0, 1, 0):
        reader.read(ExamLocation(path=root / f"exams_part{part}.hdf5", row=0))
        assert _open_hdf5_files() == before + 1
    reader.close()
    assert _open_hdf5_files() == before
    reader.read(ExamLocation(path=root / "exams_part1.hdf5", row=0))
    del reader
    gc.collect()
    assert _open_hdf5_files() == before
