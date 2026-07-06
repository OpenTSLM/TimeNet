import json
from pathlib import Path
import sys

import pytest
import wfdb

from timenet.connectors import BaseConnector
from timenet.dataset import TimeFDataset
from timenet.types import ReasoningTask
from timenet_connectors.bases.physionet import BasePhysioNetConnector
from timenet_connectors.datasets.physionet.ecg_qa_cot.connector import (
    EcgQaCotConnector,
    _build_refs,
    _load_template_answers,
)


_FIXTURES = Path(__file__).parent / "fixtures" / "ecg_qa_cot"


def _rows():
    return json.loads((_FIXTURES / "ecg_qa_cot_sample.json").read_text())


def _refs():
    # Build the same refs the real download() would, but from the checked-in fixture (flat records).
    answers = _load_template_answers(_FIXTURES / "answers_for_each_template.csv")
    return _build_refs(_rows(), _FIXTURES / "records", answers, split="train", start_index=0, flat_records=True)


def _convert() -> TimeFDataset:
    return EcgQaCotConnector().convert(_refs())


def test_is_a_connector():
    assert isinstance(EcgQaCotConnector(), BaseConnector)
    assert isinstance(EcgQaCotConnector(), BasePhysioNetConnector)


def test_metadata():
    assert EcgQaCotConnector().metadata().dataset_id == "physionet/ecg-qa-cot"
    assert str(EcgQaCotConnector().metadata().license) == "CC-BY-4.0"


def test_convert_builds_one_sample_per_row():
    dataset = _convert()
    assert isinstance(dataset, TimeFDataset)
    assert len(dataset.samples) == len(_rows())


def test_twelve_leads_per_sample():
    assert all(len(sample.time_series) == 12 for sample in _convert().samples)


def test_series_shared_across_same_ecg_id():
    # Rows 0 and 1 reference the same ecg_id, so their leads must be the same series (by id).
    dataset = _convert()
    ids0 = {ts.time_series_id for ts in dataset.samples[0].time_series}
    ids1 = {ts.time_series_id for ts in dataset.samples[1].time_series}
    assert ids0 == ids1


def test_series_values_match_fixture_record():
    dataset = _convert()
    signal, _ = wfdb.rdsamp(str(_FIXTURES / "records" / "00001_hr"), channels=[0])
    got = dataset.samples[0].time_series[0].to_numpy()
    assert len(got) == len(signal)
    assert float(got[0]) == pytest.approx(float(signal[0, 0]), rel=1e-5)


def test_reasoning_task_answer_is_label_and_rationale_is_cot():
    dataset = _convert()
    rows = _rows()
    tasks = [t for t in dataset.tasks if isinstance(t, ReasoningTask)]
    assert len(tasks) == len(rows)
    assert {t.answer for t in tasks} == {row["answer"] for row in rows}
    assert {t.rationale for t in tasks} == {row["rationale"] for row in rows}
    # answer is the short eval label, never the (much longer) chain-of-thought
    assert all(task.answer != task.rationale for task in tasks)


def test_expected_annotations_present():
    keys = {ann.key for sample in _convert().samples for ann in sample.annotations}
    assert {"split", "question_type", "template_id", "clinical_context", "answer_options"} <= keys


def test_answer_options_come_from_template():
    dataset = _convert()
    sample = dataset.samples[0]
    options = next(ann.value for ann in sample.annotations if ann.key == "answer_options")
    assert options == ["yes", "no", "not sure"]  # template_id 0 in the fixture CSV


def test_missing_wfdb_raises_helpful_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "wfdb", None)  # `import wfdb` -> ImportError
    with pytest.raises(ImportError, match="physionet"):
        EcgQaCotConnector().convert(_refs())
