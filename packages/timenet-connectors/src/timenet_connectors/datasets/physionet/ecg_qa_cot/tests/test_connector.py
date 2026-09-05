import csv
from pathlib import Path
import shutil
import sys

import numpy as np
import pytest
import wfdb

from timenet.connectors import BaseConnector
from timenet.dataset import TimeFDataset
from timenet.engine import store_dataset
from timenet.reader import TimeFReader
from timenet.registry import DatasetVersion
from timenet.types import AnswerTask
from timenet_connectors.bases.physionet import BasePhysioNetConnector
from timenet_connectors.datasets.physionet.ecg_qa_cot.connector import EcgQaCotConnector, EcgQaCotSource


_CSV_COLUMNS = ["ecg_id", "question", "answer", "rationale", "template_id", "question_type", "clinical_context"]
_CHANNELS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]

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
    # Write two valid format-16 WFDB records (00001_hr, 00002_hr) with synthetic signals. wfdb.wrsamp
    # derives the gain/baseline it stores in each .hea, so wfdb.rdsamp and the connector's direct int16
    # reader decode the same physical values. Built once per session, before any test nulls out wfdb.
    out = tmp_path_factory.mktemp("ecg_qa_cot_records")
    rng = np.random.default_rng(0)
    for ecg_id in (1, 2):
        signal = rng.standard_normal((100, len(_CHANNELS)))
        wfdb.wrsamp(
            f"{ecg_id:05d}_hr",
            fs=500,
            units=["mV"] * len(_CHANNELS),
            sig_name=list(_CHANNELS),
            p_signal=signal,
            fmt=["16"] * len(_CHANNELS),
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


def _source(tmp_path: Path, records_dir: Path, split: str = "train") -> EcgQaCotSource:
    # Recreate what download() produces: a records500/<bucket>/ tree copied from the synthesized records,
    # an answers CSV, and a CoT CSV. All fixture rows go in one split.
    records_root = tmp_path / "records500"
    for row in _rows():
        ecg_id = int(row["ecg_id"])
        bucket = records_root / f"{ecg_id // 1000 * 1000:05d}"
        bucket.mkdir(parents=True, exist_ok=True)
        for ext in (".hea", ".dat"):
            shutil.copy(records_dir / f"{ecg_id:05d}_hr{ext}", bucket / f"{ecg_id:05d}_hr{ext}")
    csv_path = tmp_path / f"ecg_qa_cot_{split}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=_CSV_COLUMNS)
        writer.writeheader()
        for row in _rows():
            writer.writerow({column: row.get(column, "") for column in _CSV_COLUMNS})
    return EcgQaCotSource(
        records_root=records_root,
        answers_path=_write_answers(tmp_path),
        cot_csvs=((split, csv_path),),
    )


def _convert(tmp_path: Path, records_dir: Path) -> TimeFDataset:
    return EcgQaCotConnector().convert([_source(tmp_path, records_dir)])


def test_is_a_connector():
    assert isinstance(EcgQaCotConnector(), BaseConnector)
    assert isinstance(EcgQaCotConnector(), BasePhysioNetConnector)


def test_metadata():
    assert EcgQaCotConnector().metadata().dataset_id == "physionet/ecg-qa-cot"
    assert str(EcgQaCotConnector().metadata().license) == "CC-BY-4.0"


def test_one_sample_per_recording_not_per_question(tmp_path, records_dir):
    dataset = _convert(tmp_path, records_dir)
    assert isinstance(dataset, TimeFDataset)
    # 3 rows over 2 recordings -> 2 records, not one per question
    assert {record.record_id for record in dataset.records} == {"ptbxl-1", "ptbxl-2"}


def test_twelve_leads_per_sample(tmp_path, records_dir):
    assert all(len(record.time_series) == 12 for record in _convert(tmp_path, records_dir).records)


def test_each_question_is_an_answer_task_on_its_recording(tmp_path, records_dir):
    dataset = _convert(tmp_path, records_dir)
    rows = _rows()
    tasks = list(dataset.iter_tasks())
    assert len(tasks) == len(rows)
    assert all(isinstance(task, AnswerTask) for task in tasks)
    assert {task.target for task in tasks} == {row["answer"] for row in rows}
    assert {task.rationale for task in tasks} == {row["rationale"] for row in rows}
    assert {task.record_ids[0] for task in tasks} == {"ptbxl-1", "ptbxl-2"}  # each task points at its recording
    assert all(task.target != task.rationale for task in tasks)  # answer is the short label, not the CoT


def test_recording_carries_its_split(tmp_path, records_dir):
    for record in _convert(tmp_path, records_dir).records:
        assert [ann.value for ann in record.annotations if ann.key == "split"] == ["train"]


def test_question_metadata_is_deduped_registered_annotations(tmp_path, records_dir):
    dataset = _convert(tmp_path, records_dir)
    keys = {ann.key for ann in dataset.registered_annotations}
    assert {"question_type", "template_id", "answer_options", "clinical_context"} <= keys
    # template 0 appears in two rows but is registered once
    template_values = sorted(ann.value for ann in dataset.registered_annotations if ann.key == "template_id")
    assert template_values == [0, 1]


def test_task_annotation_refs_all_resolve_to_registered(tmp_path, records_dir):
    dataset = _convert(tmp_path, records_dir)
    registered = {ann.id for ann in dataset.registered_annotations}
    for task in dataset.iter_tasks():
        assert set(task.input_annotation_ids) <= registered


def test_answer_options_come_from_template(tmp_path, records_dir):
    dataset = _convert(tmp_path, records_dir)
    options = {ann.id: ann.value for ann in dataset.registered_annotations if ann.key == "answer_options"}
    assert options["ecgqa-options-0"] == ["yes", "no", "not sure"]  # template 0 in the fixture answers CSV


def test_series_values_match_fixture_record(tmp_path, records_dir):
    dataset = _convert(tmp_path, records_dir)
    signal, _ = wfdb.rdsamp(str(records_dir / "00001_hr"), channels=[0])
    record = next(s for s in dataset.records if s.record_id == "ptbxl-1")
    got = record.time_series[0].to_numpy()
    assert len(got) == len(signal)
    assert float(got[0]) == pytest.approx(float(signal[0, 0]), rel=1e-5)


def test_convert_needs_no_wfdb_for_format16(tmp_path, records_dir, monkeypatch):
    # Format-16 records are read straight from the .hea/.dat, so convert and lead loading work with no
    # wfdb installed. wfdb is only a fallback for other signal formats.
    monkeypatch.setitem(sys.modules, "wfdb", None)  # `import wfdb` -> ImportError if anything reaches for it
    dataset = EcgQaCotConnector().convert([_source(tmp_path, records_dir)])
    assert dataset.records
    assert len(dataset.records[0].time_series[0].to_numpy())  # runs the direct loader, no wfdb


def test_convert_round_trips_through_the_writer(tmp_path, records_dir):
    dataset = _convert(tmp_path / "src", records_dir)
    dataset.derive_schema()
    version_dir = store_dataset(dataset, tmp_path / "out")
    with TimeFReader(DatasetVersion.open_local(version_dir)) as reader:
        restored = reader.read()
    assert {record.record_id for record in restored.records} == {"ptbxl-1", "ptbxl-2"}
    assert len(restored.tasks) == 3  # streamed to disk and back
    assert {ann.key for ann in restored.registered_annotations} >= {
        "question_type",
        "template_id",
        "answer_options",
        "clinical_context",
    }


def test_read_header_uses_the_direct_path_for_plain_single_dat_16(tmp_path):
    (tmp_path / "plain.hea").write_text(
        "plain 2 500 100\nplain.dat 16 1000(0)/mV 16 0 0 0 0 I\nplain.dat 16 1000(0)/mV 16 0 0 0 0 II\n"
    )
    assert BasePhysioNetConnector._read_header(tmp_path / "plain").direct_read is True


def test_read_header_falls_back_when_the_record_is_not_plain_16(tmp_path):
    # 16x2 (two records per frame) breaks the (-1, n_sig) reshape, so the record must route to wfdb; a
    # multi-word signal description is kept whole rather than truncated to its last token.
    (tmp_path / "framed.hea").write_text("framed 1 500 100\nframed.dat 16x2 1000(0)/mV 16 0 0 0 0 my lead II\n")
    header = BasePhysioNetConnector._read_header(tmp_path / "framed")
    assert header.direct_read is False
    assert header.sig_name == ["my lead II"]


def test_read_header_falls_back_when_signals_span_multiple_dat_files(tmp_path):
    (tmp_path / "multi.hea").write_text(
        "multi 2 500 100\na.dat 16 1000(0)/mV 16 0 0 0 0 I\nb.dat 16 1000(0)/mV 16 0 0 0 0 II\n"
    )
    assert BasePhysioNetConnector._read_header(tmp_path / "multi").direct_read is False
