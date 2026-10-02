from contextlib import contextmanager
import csv
from fractions import Fraction
from pathlib import Path

import pytest

from timenet.composition import BuildContext
from timenet.connectors import BaseConnector
from timenet.dataset import Record, RegularAxis, Signal, Source, TimeFDataset
from timenet.errors import TimeFValidationError
from timenet.registry import LocalRegistry
from timenet.types import Annotation, AnswerTask
from timenet_connectors.bases.physionet import BasePhysioNetConnector
from timenet_connectors.datasets.physionet.ecg_qa_cot.connector import (
    EcgQaCotConnector,
    EcgQaCotSource,
)
from timenet_connectors.datasets.physionet.ptb_xl.connector import ECG_SPEC, PtbXlConnector


_CSV_COLUMNS = [
    "ecg_id",
    "question",
    "answer",
    "rationale",
    "template_id",
    "question_type",
    "clinical_context",
]
_LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]

# Synthetic rows are not derived from ECG-QA. Three questions share two PTB-XL records.
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


def _write_answers(tmp_path: Path) -> Path:
    csv_path = tmp_path / "answers_for_each_template.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["template_id", "question_type", "classes"])
        writer.writeheader()
        writer.writerows(_ANSWER_TEMPLATES)
    return csv_path


def _source(tmp_path: Path, split: str = "train") -> EcgQaCotSource:
    tmp_path.mkdir(parents=True, exist_ok=True)
    csv_path = tmp_path / f"ecg_qa_cot_{split}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=_CSV_COLUMNS)
        writer.writeheader()
        for row in _ROWS:
            writer.writerow({column: row.get(column, "") for column in _CSV_COLUMNS})
    return EcgQaCotSource(
        answers_path=_write_answers(tmp_path),
        cot_csvs=((split, csv_path),),
    )


def _parent_dataset() -> TimeFDataset:
    dataset = TimeFDataset(metadata=PtbXlConnector().metadata())
    axis = RegularAxis.from_rate_hz(Fraction(500))
    for ecg_id in (1, 2):
        signals = tuple(
            Signal(
                spec=ECG_SPEC,
                name=lead,
                time_axis=axis,
                data=[float(ecg_id), float(ecg_id + 1)],
                id=f"ecg-{ecg_id}-{lead}",
            )
            for lead in _LEADS
        )
        record = Record(
            record_id=f"ptbxl-{ecg_id}",
            sources=(Source(id=f"ptbxl-{ecg_id}-source", name="PTB-XL ECG", signals=signals),),
        )
        record.annotate(Annotation(key="age", value=40 + ecg_id, unit="year", id=f"ptbxl-{ecg_id}-age"))
        dataset.add_record(record=record)
    return dataset


@contextmanager
def _converted(tmp_path: Path):
    registry = LocalRegistry(tmp_path / "registry")
    registry.store(_parent_dataset())
    connector = EcgQaCotConnector()
    with BuildContext.open(connector.metadata(), registry) as context:
        dataset = connector.convert([_source(tmp_path / "source")], context)
        dataset.set_dependencies(context.dependency_lock())
        yield dataset, registry


def test_is_a_composed_connector():
    connector = EcgQaCotConnector()

    assert isinstance(connector, BaseConnector)
    assert isinstance(connector, BasePhysioNetConnector)
    assert connector.metadata().dataset_id == "physionet/ecg-qa-cot"
    assert connector.metadata().parents[0].alias == "ptbxl"


def test_convert_requires_the_declared_parent_context(tmp_path):
    with pytest.raises(TimeFValidationError, match="requires a build context"):
        EcgQaCotConnector().convert([_source(tmp_path)])


def test_imports_one_parent_record_per_ecg_not_per_question(tmp_path):
    with _converted(tmp_path) as (dataset, _):
        assert {record.record_id for record in dataset.records} == {"ptbxl-1", "ptbxl-2"}
        assert len(dataset.record_imports) == 2
        assert all(len(record.signals) == 12 for record in dataset.records)


def test_parent_values_and_annotations_remain_available(tmp_path):
    with _converted(tmp_path) as (dataset, _):
        record = next(record for record in dataset.records if record.id == "ptbxl-1")

        assert record.signals[0].to_numpy().tolist() == [1.0, 2.0]
        assert {annotation.key: annotation.value for annotation in record.annotations}["age"] == 41


def test_each_question_is_a_split_answer_task_on_its_recording(tmp_path):
    with _converted(tmp_path) as (dataset, _):
        tasks = list(dataset.iter_tasks())

        assert len(tasks) == len(_ROWS)
        assert all(isinstance(task, AnswerTask) for task in tasks)
        assert {task.targets[0] for task in tasks if task.targets is not None} == {row["answer"] for row in _ROWS}
        assert {task.rationale for task in tasks} == {row["rationale"] for row in _ROWS}
        assert {task.inputs[0].id for task in tasks} == {"ptbxl-1", "ptbxl-2"}
        assert {str(task.split) for task in tasks} == {"train"}
        assert all(task.targets != (task.rationale,) for task in tasks)


def test_question_metadata_is_deduplicated_and_referenced(tmp_path):
    with _converted(tmp_path) as (dataset, _):
        keys = {annotation.key for annotation in dataset.annotations}
        assert {"question_type", "template_id", "answer_options", "clinical_context"} <= keys
        assert sorted(annotation.value for annotation in dataset.annotations if annotation.key == "template_id") == [
            0,
            1,
        ]
        options = {
            annotation.id: annotation.value for annotation in dataset.annotations if annotation.key == "answer_options"
        }
        assert options["ecgqa-options-0"] == ["yes", "no", "not sure"]
        registered = {annotation.occurrence_id for annotation in dataset.annotations}
        for task in dataset.iter_tasks():
            assert {annotation.occurrence_id for annotation in task.input_annotations} <= registered


def test_child_round_trip_resolves_parent_without_copying_values(tmp_path):
    with _converted(tmp_path) as (dataset, registry):
        registry.store(dataset)

    manifest = registry.get_manifest("physionet/ecg-qa-cot", "1.0.0")
    assert manifest.files.time_series == ()
    assert manifest.dependencies.direct[0].alias == "ptbxl"
    assert manifest.counts.records == 2

    with registry.open_reader("physionet/ecg-qa-cot", "1.0.0") as reader:
        restored = reader.read()

    assert {record.record_id for record in restored.records} == {"ptbxl-1", "ptbxl-2"}
    assert len(restored.tasks) == 3
    assert restored.records[0].signals[0].to_numpy().tolist() in ([1.0, 2.0], [2.0, 3.0])
    assert {annotation.key for annotation in restored.annotations} >= {
        "question_type",
        "template_id",
        "answer_options",
        "clinical_context",
    }
