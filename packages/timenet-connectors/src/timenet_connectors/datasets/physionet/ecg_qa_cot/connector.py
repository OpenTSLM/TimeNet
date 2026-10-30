"""The ECG-QA CoT connector: PTB-XL 12-lead ECGs with chain-of-thought question answering.

Each record is one PTB-XL recording: 12 leads at 500 Hz in millivolts. Every chain-of-thought row is
an :class:`~timenet.types.AnswerTask` on that recording. Its ``prompt`` is the question, ``targets``
is the short evaluation answer, and ``rationale`` is the reasoning target. So a recording carries many
QA tasks. There is no record per question. The connector stores per-question metadata (question type,
template, answer options, clinical context) once, as value-deduped annotations that the tasks
reference. Each recording carries its dataset split, and the patient's sex and age from PTB-XL. The ~230k
tasks stream to disk, so they never all live in memory.

The recordings and their patient metadata are imported from the pinned PTB-XL TimeF parent. The
per-template answer options come from the ``Jwoo5/ecg-qa`` GitHub repo. The precomputed CoT rows
(question, answer, rationale, template) come from the OpenTSLM release, the only public source for the
rationales. Build the PTB-XL parent first or pass ``--build-parents``.

The clinical context comes from the annotations of the imported PTB-XL records. It states the patient's
age and sex, the recording device, and the signal-quality issues, extra beats, and pacemaker that PTB-XL
records for the recording.
"""

import ast
import asyncio
from collections.abc import Iterator, Mapping
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

from timenet.connectors import BaseConnector
from timenet.connectors.context import BuildContext
from timenet.dataset import Record, TimeFDataset
from timenet.errors import TimeNetConnectorError
from timenet.types import (
    Annotation,
    AnswerTask,
    Split,
    read_value,
)
from timenet.types.reference_keys import ParentAlias
from timenet_connectors.download import Artifact, download_files, ensure_archive, find_dir_containing


# The per-template answer options (the multiple-choice candidates), keyed by template_id.
ECG_QA_TEMPLATE_ANSWERS_URL = (
    "https://raw.githubusercontent.com/Jwoo5/ecg-qa/master/ecgqa/ptbxl/answers_for_each_template.csv"
)
# Precomputed CoT rows (question, answer, rationale, template). OpenTSLM's release is the only
# public source. This is a swappable constant so a mirror can replace it.
ECG_QA_COT_URL = "https://polybox.ethz.ch/index.php/s/D5QaJSEw4dXkzXm/download/ecg_qa_cot_final.zip"


# PTB-XL signal-quality annotations and the phrase the clinical context states when a record has one.
_QUALITY_ISSUES = (
    ("baseline_drift", "baseline drift noted"),
    ("static_noise", "static noise present"),
    ("burst_noise", "burst noise present"),
    ("electrodes_problems", "electrode artifacts present"),
)
# PTB-XL findings outside signal quality and the phrase the clinical context states when a record has one.
_OTHER_FINDINGS = (
    ("extra_beats", "extra beats detected during recording"),
    ("pacemaker", "pacemaker present"),
)
# The imported PTB-XL annotations a record keeps. The connector clears every other one.
_KEPT_ANNOTATIONS = frozenset({"sex", "age"})


@dataclass(frozen=True)
class EcgQaCotSource:
    """A lightweight handle to the fetched sources, so ``compose`` streams rather than holding refs.

    ``download`` returns one of these instead of a row per question, so the ~230k questions are read
    lazily during ``compose`` and the task stream, never materialized as a list.
    """

    answers_path: Path
    cot_csvs: tuple[tuple[Split, Path], ...]  # (split, csv_path) for train / validation / test


def _parse_int(cell: str) -> int:
    """Parse a whole number that a source CSV writes as a float, such as ``"56.0"``.

    Args:
        cell: The CSV cell.

    Returns:
        The integer value.
    """
    return int(float(cell))


def _parse_ecg_id(raw: str) -> int:
    """Parse a PTB-XL ecg_id that arrives as ``"123"`` or ``"[123]"``.

    Args:
        raw: The ecg_id cell of a CoT row.

    Returns:
        The integer ecg_id.
    """
    return int(raw.strip().strip("[]").strip())


@dataclass(frozen=True)
class _PtbxlRecording:
    """One recording's PTB-XL values, read once from the annotations of its imported record."""

    patient_id: int
    age: int
    sex: str
    annotations: Mapping[str, object]
    """The imported annotation values by key, for the device, recording-date, signal-quality, extra-beats
    and pacemaker values."""


def _ptbxl_recording(ecg_id: int, record: Record) -> _PtbxlRecording:
    """Read a recording's PTB-XL values from the annotations of its imported record.

    Args:
        ecg_id: The recording id.
        record: The imported PTB-XL record, with its annotations.

    Returns:
        The recording's PTB-XL values.

    Raises:
        TimeNetConnectorError: If the record has no patient, age, or sex.
    """
    values = {annotation.key: read_value(annotation.value) for annotation in record.annotations}
    patients = values.get("subject_ids")
    if not isinstance(patients, list) or not patients:
        raise TimeNetConnectorError(f"PTB-XL ecg_id {ecg_id}: patient_id is empty")
    # PTB-XL states the age as an int in years and the sex as "male" or "female". Both stay as they are.
    age = values.get("age")
    if not isinstance(age, int):
        raise TimeNetConnectorError(f"PTB-XL ecg_id {ecg_id}: age is empty")
    sex = values.get("sex")
    if not isinstance(sex, str):
        raise TimeNetConnectorError(f"PTB-XL ecg_id {ecg_id}: sex is empty")
    return _PtbxlRecording(patient_id=_parse_int(patients[0]), age=age, sex=sex, annotations=values)


def _text(values: Mapping[str, object], key: str) -> str:
    """Return a text annotation value without surrounding whitespace, or ``""`` if the record has none.

    Args:
        values: The imported annotation values by key.
        key: The annotation key.

    Returns:
        The text.
    """
    value = values.get(key)
    return value.strip() if isinstance(value, str) else ""


def _clinical_context(recording: _PtbxlRecording) -> str:
    """Build a recording's clinical context from its PTB-XL values.

    The context states the patient's age and sex, the recording device, and each signal-quality issue,
    extra beats, or pacemaker that the imported record has an annotation for.

    Args:
        recording: The recording's PTB-XL values.

    Returns:
        The clinical-context text.
    """
    values = recording.annotations
    parts = [f"{recording.age}-year-old", f"{recording.sex} patient", "12-lead ECG"]
    if _text(values, "recording_date"):
        parts.append("clinical recording")
    if device := _text(values, "device"):
        parts.append(f"recorded with {device}")
    issues = [phrase for key, phrase in _QUALITY_ISSUES if _text(values, key)]
    parts.append(f"Signal quality: {', '.join(issues)}" if issues else "Signal quality: adequate for analysis")
    parts.extend(phrase for key, phrase in _OTHER_FINDINGS if _text(values, key))
    return f"{parts[0]} {parts[1]}. " + ". ".join(parts[2:]) + "."


@dataclass(frozen=True)
class _CotRow:
    """One CoT question, parsed once from its CSV row."""

    ecg_id: int
    template_id: int
    question_type: str
    question: str
    answer: str
    rationale: str


@dataclass(frozen=True)
class _TaskAnnotations:
    """The dataset annotations the QA tasks reference, one lookup per annotation key."""

    question_type: dict[str, Annotation]
    template_id: dict[int, Annotation]
    answer_options: dict[int, Annotation]
    """Keyed by template id. A template with no answer options has no entry."""
    clinical_context: dict[int, Annotation]
    """Keyed by PTB-XL ECG ID. Recordings with the same context share one annotation."""


def _load_template_answers(path: Path) -> dict[int, tuple[str, ...]]:
    """Load per-template answer options from ``answers_for_each_template.csv``.

    Args:
        path: Path to the CSV whose ``classes`` column is a Python list literal.

    Returns:
        A mapping of ``template_id`` to its ordered answer options.
    """
    answers: dict[int, tuple[str, ...]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            classes = ast.literal_eval(row["classes"])
            answers[_parse_int(row["template_id"])] = tuple(str(option) for option in classes)
    return answers


def _iter_cot_rows(csv_path: Path) -> Iterator[_CotRow]:
    """Stream one split's CoT rows, so a whole split never lives in memory at once.

    Args:
        csv_path: The split's CoT CSV.

    Yields:
        Each parsed row. Rationale and clinical-context fields span multiple physical lines.
    """
    with csv_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            yield _CotRow(
                ecg_id=_parse_ecg_id(row["ecg_id"]),
                template_id=_parse_int(row["template_id"]),
                question_type=row["question_type"],
                question=row["question"],
                answer=row["answer"],
                rationale=row["rationale"],
            )


class EcgQaCotConnector(BaseConnector[EcgQaCotSource]):
    """Connector for the ECG-QA CoT dataset (PTB-XL signals + OpenTSLM chain-of-thought QA)."""

    _COT_CSVS: ClassVar[tuple[tuple[Split, str], ...]] = (
        (Split.TRAIN, "ecg_qa_cot_train.csv"),
        (Split.VALIDATION, "ecg_qa_cot_val.csv"),
        (Split.TEST, "ecg_qa_cot_test.csv"),
    )

    async def download_async(self, cache_dir: Path) -> list[EcgQaCotSource]:
        """Fetch the template answers and CoT CSVs and return one lightweight handle.

        The template-answers CSV and the CoT archive download concurrently. No CoT
        row is read here; ``compose`` and the task stream read them lazily.

        Args:
            cache_dir: The directory that holds downloaded archives.

        Returns:
            A single-element list holding the :class:`EcgQaCotSource` handle.
        """
        answers_path = cache_dir / "answers_for_each_template.csv"
        # The two HTTP artifacts download concurrently.
        _, cot_root = await asyncio.gather(
            download_files([Artifact(ECG_QA_TEMPLATE_ANSWERS_URL, answers_path)]),
            ensure_archive(ECG_QA_COT_URL, cache_dir),
        )
        cot_csvs = tuple(
            (split, find_dir_containing(cot_root, csv_name) / csv_name) for split, csv_name in self._COT_CSVS
        )
        return [EcgQaCotSource(answers_path=answers_path, cot_csvs=cot_csvs)]

    def compose(self, raw_refs: list[EcgQaCotSource], context: BuildContext) -> TimeFDataset:
        """Build one record per recording and stream one :class:`AnswerTask` per CoT row.

        A first pass over the CoT CSVs collects each recording's split and the distinct question
        metadata. It then imports the selected parent records, found by the ``ecg_id`` in their
        metadata, builds each recording's clinical context from the imported PTB-XL annotations, and
        registers the deduped metadata annotations. The tasks themselves stream from
        :meth:`_iter_tasks`, so the ~230k questions never all live in memory.

        Args:
            raw_refs: The single-element list from :meth:`download`.
            context: The opened PTB-XL parent, under the parent alias ``ptbxl``.

        Returns:
            The dataset: recording records plus a task stream.

        Raises:
            TimeNetConnectorError: If the PTB-XL parent has no recording for an ECG the rows name, or an
                imported Record has no patient, age, or sex.
        """
        source = raw_refs[0]
        answers = _load_template_answers(source.answers_path)
        dataset = TimeFDataset(metadata=self.metadata())

        split_of_ecg: dict[int, Split] = {}
        question_types: set[str] = set()
        template_ids: set[int] = set()
        for split, csv_path in source.cot_csvs:
            for row in _iter_cot_rows(csv_path):
                split_of_ecg.setdefault(row.ecg_id, split)
                question_types.add(row.question_type)
                template_ids.add(row.template_id)

        ptbxl = context.parents[ParentAlias("ptbxl")]
        # PTB-XL keeps each recording's upstream ecg_id in the record's metadata.
        parent_ids = {
            record.metadata["ecg_id"]: record.id
            for record in ptbxl.iter_records(with_annotations=False)
            if record.metadata.get("ecg_id") in split_of_ecg
        }
        missing = sorted(split_of_ecg.keys() - parent_ids.keys())
        if missing:
            raise TimeNetConnectorError(f"the PTB-XL parent has no recording with ecg_id {missing}")
        imported = dataset.import_from(ptbxl, records=[parent_ids[ecg_id] for ecg_id in sorted(split_of_ecg)])
        records: dict[int, Record] = {}
        contexts: dict[int, str] = {}
        for ecg_id, split in sorted(split_of_ecg.items()):
            record = imported.records[parent_ids[ecg_id]]
            recording = _ptbxl_recording(ecg_id, record)
            record.metadata["patient_id"] = recording.patient_id
            # Keep PTB-XL's span-free sex and age and clear the other imported annotations, so the
            # Records match our standalone ECG-QA-CoT 1.1.0: ``sex``, ``age``, and the split.
            record.annotations = tuple(
                annotation for annotation in record.annotations if annotation.key in _KEPT_ANNOTATIONS
            )
            record.add_annotations([Annotation(key="split", value=split.value)])
            records[ecg_id] = record
            contexts[ecg_id] = _clinical_context(recording)

        task_annotations = self._annotate_task_metadata(dataset, question_types, template_ids, contexts, answers)
        dataset.set_task_stream(
            [AnswerTask],
            lambda: self._iter_tasks(source, records, task_annotations),
        )
        return dataset

    @staticmethod
    def _annotate_task_metadata(
        dataset: TimeFDataset,
        question_types: set[str],
        template_ids: set[int],
        contexts: Mapping[int, str],
        answers: dict[int, tuple[str, ...]],
    ) -> _TaskAnnotations:
        """Attach one dataset annotation per distinct question-metadata value.

        Args:
            dataset: The dataset the annotations attach to.
            question_types: The distinct question types.
            template_ids: The distinct template ids.
            contexts: Each recording's clinical context, keyed by PTB-XL ECG ID.
            answers: Per-template answer options.

        Returns:
            The attached annotations, keyed by the value that :meth:`_iter_tasks` looks them up by.
        """
        question_type_annotations = {
            question_type: dataset.annotate(Annotation(key="question_type", value=question_type))
            for question_type in sorted(question_types)
        }
        template_id_annotations: dict[int, Annotation] = {}
        answer_options_annotations: dict[int, Annotation] = {}
        for template_id in sorted(template_ids):
            template_id_annotations[template_id] = dataset.annotate(Annotation(key="template_id", value=template_id))
            options = answers.get(template_id)
            if options:
                answer_options_annotations[template_id] = dataset.annotate(
                    Annotation(key="answer_options", value=list(options))
                )
        context_annotations = {
            context: dataset.annotate(Annotation(key="clinical_context", value=context))
            for context in sorted(set(contexts.values()))
        }
        return _TaskAnnotations(
            question_type=question_type_annotations,
            template_id=template_id_annotations,
            answer_options=answer_options_annotations,
            clinical_context={ecg_id: context_annotations[context] for ecg_id, context in contexts.items()},
        )

    @staticmethod
    def _iter_tasks(
        source: EcgQaCotSource,
        records: Mapping[int, Record],
        annotations: _TaskAnnotations,
    ) -> Iterator[AnswerTask]:
        """Yield one :class:`AnswerTask` per CoT row, referencing its recording and metadata annotations.

        Args:
            source: The download handle naming the CoT CSVs.
            records: Input Records keyed by PTB-XL ECG ID.
            annotations: The dataset annotations the tasks reference.

        Yields:
            Each question as an answer task, streamed so the whole set never lives in memory.
        """
        for split, csv_path in source.cot_csvs:
            for row in _iter_cot_rows(csv_path):
                input_annotations = [
                    annotations.question_type[row.question_type],
                    annotations.template_id[row.template_id],
                ]
                if (options := annotations.answer_options.get(row.template_id)) is not None:
                    input_annotations.append(options)
                input_annotations.append(annotations.clinical_context[row.ecg_id])
                yield AnswerTask(
                    inputs=(records[row.ecg_id],),
                    prompt=row.question,
                    targets=(row.answer,),
                    rationale=row.rationale,
                    input_annotations=tuple(input_annotations),
                    split=split,
                )


CONNECTOR = EcgQaCotConnector
