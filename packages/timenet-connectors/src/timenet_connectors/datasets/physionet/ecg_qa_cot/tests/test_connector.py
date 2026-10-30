from collections.abc import Generator
from contextlib import contextmanager
import csv
from pathlib import Path
import shutil
from typing import cast

import numpy as np
import pytest
import wfdb

from timenet.connectors import BaseConnector
from timenet.dataset import Record, TimeFDataset
from timenet.engine import open_parents, store_dataset
from timenet.errors import TimeNetConnectorError
from timenet.reader import TimeFReader
from timenet.registry import LocalRegistry, open_local_version
from timenet.types import AnswerTask, ObjectKind, Split
from timenet.types.reference_keys import ParentAlias
from timenet.types.text import read_value
from timenet_connectors.datasets.physionet.ecg_qa_cot.connector import EcgQaCotConnector, EcgQaCotSource
from timenet_connectors.datasets.physionet.ptb_xl.connector import PtbXlConnector, PtbXlSource


_CSV_COLUMNS = ["ecg_id", "question", "answer", "rationale", "template_id", "question_type", "clinical_context"]
_LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]

# Synthetic rows, hand-written for the tests and not derived from the real ECG-QA / PTB-XL data, so the
# repo ships no dataset bytes. The distribution is what the assertions need: three questions over two
# recordings (ecg_id 1 twice, 2 once), template 0 seen twice and template 1 once, and every answer
# distinct from its rationale.
_ROWS = [
    {
        "ecg_id": 1,
        "question": "Synthetic question: does this recording show a normal rhythm?",
        "answer": "no",
        "rationale": "Synthetic rationale one: the leads were reviewed before the label was assigned.",
        "template_id": 0,
        "question_type": "single-verify",
        "clinical_context": "12-lead ECG recording.",
    },
    {
        "ecg_id": 1,
        "question": "Synthetic question: which rhythm best describes this recording?",
        "answer": "sinus rhythm",
        "rationale": "Synthetic rationale two: the morphology was compared against the options.",
        "template_id": 1,
        "question_type": "single-query",
        "clinical_context": "12-lead ECG recording.",
    },
    {
        "ecg_id": 2,
        "question": "Synthetic question: is the finding present in this recording?",
        "answer": "not sure",
        "rationale": "Synthetic rationale three: the evidence was inconclusive for this lead set.",
        "template_id": 0,
        "question_type": "single-verify",
        "clinical_context": "12-lead ECG recording.",
    },
]

_PTBXL_COLUMNS = [
    "ecg_id",
    "filename_hr",
    "strat_fold",
    "patient_id",
    "age",
    "sex",
    "device",
    "recording_date",
    "baseline_drift",
    "static_noise",
    "burst_noise",
    "electrodes_problems",
    "extra_beats",
    "pacemaker",
]
# Synthetic ptbxl_database.csv rows for the PTB-XL parent. Recording 1 is male (code 0) with no quality
# findings. Recording 2 is female (code 1) with baseline drift, burst noise, extra beats and a pacemaker,
# and a whitespace-only static-noise cell. Both are in PTB-XL's test fold, so the parent publishes a
# "test" split that the child replaces with its own.
_PTBXL_ROWS: list[dict[str, str]] = [
    {
        "ecg_id": "1",
        "strat_fold": "10",
        "patient_id": "101.0",
        "age": "56.0",
        "sex": "0",
        "device": "CS-12   E",
        "recording_date": "1984-11-09 09:17:34",
    },
    {
        "ecg_id": "2",
        "strat_fold": "10",
        "patient_id": "102.0",
        "age": "300.0",
        "sex": "1",
        "device": "AT-6 C 5.5",
        "recording_date": "1985-01-01 10:00:00",
        "baseline_drift": " , II,III,AVF",
        "static_noise": "  ",
        "burst_noise": "alles",
        "extra_beats": "1ES",
        "pacemaker": "ja, pacemaker",
    },
]
_EXPECTED_CONTEXTS = {
    1: "56-year-old male patient. 12-lead ECG. clinical recording. recorded with CS-12   E. "
    "Signal quality: adequate for analysis.",
    2: "300-year-old female patient. 12-lead ECG. clinical recording. recorded with AT-6 C 5.5. "
    "Signal quality: baseline drift noted, burst noise present. extra beats detected during recording. "
    "pacemaker present.",
}
# Each recording's sex and age, as the records of ECG-QA-CoT 1.1.0 carry them, with the same descriptions.
_SEX_AND_AGE = ((1, "male", 56), (2, "female", 300))
_SEX_DESCRIPTION = "The patient's sex, as PTB-XL states it."
_AGE_DESCRIPTION = (
    "The patient's age in whole years, as PTB-XL states it. PTB-XL states 300 for every patient older than 89."
)

# template_id -> question_type and its answer classes. Template 0's classes are asserted verbatim.
_ANSWER_TEMPLATES = [
    {
        "template_id": 0,
        "question_type": "single-verify",
        "classes": "['yes', 'no', 'not sure']",
    },
    {
        "template_id": 1,
        "question_type": "single-query",
        "classes": "['sinus rhythm', 'atrial fibrillation', 'not sure']",
    },
]


@pytest.fixture(scope="session")
def records_dir(tmp_path_factory) -> Path:
    # Write two valid format-16 WFDB records (00001_hr, 00002_hr) with synthetic signals for the PTB-XL
    # parent. wfdb.wrsamp derives the gain/baseline it stores in each .hea, so wfdb.rdsamp and PTB-XL's
    # direct int16 reader decode the same physical values. Built once per session.
    out = tmp_path_factory.mktemp("ecg_qa_cot_records")
    rng = np.random.default_rng(0)
    for ecg_id in (1, 2):
        signal = rng.standard_normal((100, len(_LEADS)))
        wfdb.wrsamp(
            f"{ecg_id:05d}_hr",
            fs=500,
            units=["mV"] * len(_LEADS),
            sig_name=list(_LEADS),
            p_signal=signal,
            fmt=["16"] * len(_LEADS),
            write_dir=str(out),
        )
    return out


def _rows():
    return _ROWS


def _write_answers(tmp_path: Path) -> Path:
    csv_path = tmp_path / "answers_for_each_template.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["template_id", "question_type", "classes"])
        writer.writeheader()
        writer.writerows(_ANSWER_TEMPLATES)
    return csv_path


def _source(tmp_path: Path, split: Split = Split.TRAIN) -> EcgQaCotSource:
    # Recreate what download() produces: an answers CSV and a CoT CSV. All fixture rows go in one split.
    # The CoT CSV writes each ecg_id bracketed, as "[1]", because every row of the real release does.
    csv_path = tmp_path / f"ecg_qa_cot_{split}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=_CSV_COLUMNS)
        writer.writeheader()
        for row in _rows():
            writer.writerow({column: row.get(column, "") for column in _CSV_COLUMNS} | {"ecg_id": f"[{row['ecg_id']}]"})
    return EcgQaCotSource(
        answers_path=_write_answers(tmp_path),
        cot_csvs=((split, csv_path),),
    )


def _store_ptbxl(tmp_path: Path, records_dir: Path, rows: list[dict[str, str]], root: Path) -> None:
    """Build the PTB-XL parent from the fixture recordings and store it under ``root``."""
    for row in rows:
        bucket = tmp_path / "records500" / "00000"
        bucket.mkdir(parents=True, exist_ok=True)
        for ext in (".hea", ".dat"):
            name = f"{int(row['ecg_id']):05d}_hr{ext}"
            shutil.copy(records_dir / name, bucket / name)
    database = tmp_path / "ptbxl_database.csv"
    with database.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=_PTBXL_COLUMNS)
        writer.writeheader()
        # Listed in reverse, so PTB-XL's record IDs differ from the ecg_ids: only the metadata finds a recording.
        for row in sorted(rows, key=lambda row: int(row["ecg_id"]), reverse=True):
            filename = f"records500/00000/{int(row['ecg_id']):05d}_hr"
            writer.writerow({column: row.get(column, "") for column in _PTBXL_COLUMNS} | {"filename_hr": filename})
    store_dataset(PtbXlConnector().convert([PtbXlSource(tmp_path, database)]), root)


def _sex_and_age(record: Record) -> tuple[object, ...]:
    """Return a record's sex and its description, and its age with the age's type, unit and description."""
    by_key = {ann.key: ann for ann in record.annotations}
    sex, age = by_key["sex"], by_key["age"]
    return read_value(sex.value), sex.description, age.value, type(age.value), age.unit, age.description


@contextmanager
def _composed(
    tmp_path: Path,
    records_dir: Path,
    registry_root: Path | None = None,
    *,
    split: Split = Split.TRAIN,
    ptbxl_rows: list[dict[str, str]] | None = None,
) -> Generator[TimeFDataset]:
    """Build the PTB-XL parent, then compose ECG-QA from it inside the parent context, as the engine does.

    The composed records keep handles to the parent's stored values, so the tests read and write the
    dataset before the context closes the parent readers.
    """
    root = registry_root or tmp_path / "registry"
    _store_ptbxl(tmp_path, records_dir, _PTBXL_ROWS if ptbxl_rows is None else ptbxl_rows, root)
    connector = EcgQaCotConnector()
    with open_parents(connector.metadata(), LocalRegistry(root)) as context:
        yield connector.compose([_source(tmp_path, split)], context)


def test_is_a_connector():
    assert isinstance(EcgQaCotConnector(), BaseConnector)


def test_metadata():
    assert EcgQaCotConnector().metadata().dataset_id == "physionet/ecg-qa-cot"
    assert str(EcgQaCotConnector().metadata().license) == "CC-BY-4.0"


def test_one_record_per_ecg_not_per_question(tmp_path, records_dir):
    with _composed(tmp_path, records_dir) as dataset:
        # 3 rows over 2 recordings -> 2 records, not one per question
        assert {record.metadata["ecg_id"] for record in dataset.records} == {1, 2}
        # Each is a copy of the PTB-XL Record with that ecg_id, linked by the import map.
        copies = {
            child_id
            for (alias, kind, _), child_id in dataset._import_state.import_map.items()
            if alias == ParentAlias("ptbxl") and kind is ObjectKind.RECORD
        }
        assert copies == {record.id for record in dataset.records}


def test_a_recording_the_parent_lacks_is_rejected(tmp_path, records_dir):
    with (
        pytest.raises(TimeNetConnectorError, match=r"no recording with ecg_id \[2\]"),
        _composed(tmp_path, records_dir, ptbxl_rows=_PTBXL_ROWS[:1]),
    ):
        pass


def test_twelve_leads_per_record(tmp_path, records_dir):
    with _composed(tmp_path, records_dir) as dataset:
        assert all(len(record.signals) == 12 for record in dataset.records)


def test_each_question_is_an_answer_task_on_its_recording(tmp_path, records_dir):
    rows = _rows()
    with _composed(tmp_path, records_dir) as dataset:
        tasks = list(dataset.iter_tasks())
        assert len(tasks) == len(rows)
        assert all(isinstance(task, AnswerTask) for task in tasks)
        assert {read_value(task.targets[0]) for task in tasks if task.targets is not None} == {
            row["answer"] for row in rows
        }
        assert {read_value(task.rationale) for task in tasks} == {row["rationale"] for row in rows}
        assert sorted((read_value(task.prompt), task.inputs[0].metadata["ecg_id"]) for task in tasks) == sorted(
            (row["question"], row["ecg_id"]) for row in rows
        )
        # The answer is the short label, not the chain of thought.
        assert all(task.targets != (task.rationale,) for task in tasks)


def test_records_and_tasks_carry_their_split(tmp_path, records_dir):
    with _composed(tmp_path, records_dir, split=Split.VALIDATION) as dataset:
        for record in dataset.records:
            assert [ann.value for ann in record.annotations if ann.key == "split"] == ["validation"]
        assert {task.split for task in dataset.iter_tasks()} == {Split.VALIDATION}


def test_question_metadata_is_deduped_dataset_annotations(tmp_path, records_dir):
    with _composed(tmp_path, records_dir) as dataset:
        keys = {ann.key for ann in dataset.annotations}
        assert {"question_type", "template_id", "answer_options", "clinical_context"} <= keys
        # template 0 appears in two rows but is attached once
        template_values = [ann.value for ann in dataset.annotations if ann.key == "template_id"]
        # Annotation values are a union with no order; template IDs are ints.
        assert sorted(cast("list[int]", template_values)) == [0, 1]


def test_task_annotation_refs_all_resolve_to_registered(tmp_path, records_dir):
    with _composed(tmp_path, records_dir) as dataset:
        registered = {id(annotation) for annotation in dataset.annotations}
        for task in dataset.iter_tasks():
            assert {id(annotation) for annotation in task.input_annotations} <= registered


def test_answer_options_come_from_template(tmp_path, records_dir):
    options = {}
    with _composed(tmp_path, records_dir) as dataset:
        for task in dataset.iter_tasks():
            by_key = {ann.key: ann.value for ann in task.input_annotations}
            options[by_key["template_id"]] = by_key.get("answer_options")
    assert options[0] == ["yes", "no", "not sure"]  # template 0 in the fixture answers CSV


def test_clinical_context_is_built_from_ptbxl(tmp_path, records_dir):
    # The CoT rows' own clinical_context is ignored. Code 0 is male, and only PTB-XL cells with a value
    # (not empty, not whitespace) state a finding.
    with _composed(tmp_path, records_dir) as dataset:
        for task in dataset.iter_tasks():
            by_key = {ann.key: ann.value for ann in task.input_annotations}
            assert by_key["clinical_context"] == _EXPECTED_CONTEXTS[cast("int", task.inputs[0].metadata["ecg_id"])]


def test_records_keep_ptbxl_sex_and_age_and_carry_the_patient(tmp_path, records_dir):
    with _composed(tmp_path, records_dir) as dataset:
        by_ecg = {record.metadata["ecg_id"]: record for record in dataset.records}
        assert by_ecg[1].metadata == {"ecg_id": 1, "patient_id": 101}
        assert by_ecg[2].metadata == {"ecg_id": 2, "patient_id": 102}
        # PTB-XL's sex and age stay as they are, without a span. Every other imported annotation is cleared,
        # and the child's own split replaces PTB-XL's.
        for ecg_id, sex, age in _SEX_AND_AGE:
            annotations = by_ecg[ecg_id].annotations
            assert sorted(ann.key for ann in annotations) == ["age", "sex", "split"]
            assert _sex_and_age(by_ecg[ecg_id]) == (sex, _SEX_DESCRIPTION, age, int, "year", _AGE_DESCRIPTION)
            by_key = {ann.key: ann for ann in annotations}
            assert by_key["split"].value == "train"
            assert all(ann.span is None for ann in annotations)


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("sex", "", "sex is empty"),
        ("age", "", "age is empty"),
        ("patient_id", "", "patient_id is empty"),
    ],
)
def test_an_incomplete_ptbxl_record_raises(tmp_path, records_dir, column, value, message):
    rows = [_PTBXL_ROWS[0], {**_PTBXL_ROWS[1], column: value}]
    with (
        pytest.raises(TimeNetConnectorError, match=f"ecg_id 2: {message}"),
        _composed(tmp_path, records_dir, ptbxl_rows=rows),
    ):
        pass


def test_series_values_match_fixture_record(tmp_path, records_dir):
    signal, _ = wfdb.rdsamp(str(records_dir / "00001_hr"), channels=[0])
    with _composed(tmp_path, records_dir) as dataset:
        record = next(r for r in dataset.records if r.metadata["ecg_id"] == 1)
        got = record.signals[0].to_numpy()
    assert len(got) == len(signal)
    assert float(got[0]) == pytest.approx(float(signal[0, 0]), rel=1e-5)


def test_compose_round_trips_through_the_writer(tmp_path, records_dir):
    root = tmp_path / "out"
    with _composed(tmp_path / "src", records_dir, registry_root=root) as dataset:
        dataset.derive_schema()
        version_dir = store_dataset(dataset, root)
    with TimeFReader(open_local_version(version_dir)) as reader:
        restored = reader.read()
    assert {record.metadata["ecg_id"] for record in restored.records} == {1, 2}
    # The written records carry the sex and age of ECG-QA-CoT 1.1.0.
    restored_by_ecg = {record.metadata["ecg_id"]: record for record in restored.records}
    for ecg_id, sex, age in _SEX_AND_AGE:
        assert _sex_and_age(restored_by_ecg[ecg_id]) == (sex, _SEX_DESCRIPTION, age, int, "year", _AGE_DESCRIPTION)
    # The child reuses its parent's signal values, so it writes no values files of its own.
    assert not list(version_dir.glob("time_series/**/*"))
    assert open_local_version(version_dir).manifest.payload_sources
    assert len(restored.tasks) == 3  # streamed to disk and back
    assert {ann.key for ann in restored.annotations} >= {
        "question_type",
        "template_id",
        "answer_options",
        "clinical_context",
    }
