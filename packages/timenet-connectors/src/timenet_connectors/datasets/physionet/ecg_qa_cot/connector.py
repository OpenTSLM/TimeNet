"""The ECG-QA CoT connector: PTB-XL 12-lead ECGs with chain-of-thought question answering.

Each record is one PTB-XL recording: 12 leads at 500 Hz in millivolts. Every chain-of-thought row is
an :class:`~timenet.types.AnswerTask` on that recording. Its ``prompt`` is the question, ``targets``
is the short evaluation answer, and ``rationale`` is the reasoning target. So a recording carries many
QA tasks. There is no record per question. The connector stores the question type, the answer options,
and the clinical context once, as value-deduped annotations that the tasks reference as their input.
The task metadata stores the question template ID for provenance. The question text already expresses
the template, so the ID is excluded from the input annotations. Each recording carries its dataset
split and the patient's subject IDs, sex, and age from PTB-XL. The ~230k tasks stream to disk, so they never all
live in memory.

The connector imports the recordings and their patient metadata from the pinned PTB-XL TimeF
parent. The per-template answer options come from the ``Jwoo5/ecg-qa`` GitHub repo. The precomputed
CoT rows (question, answer, rationale, template) come from the OpenTSLM release, the only public
source for the rationales. Build the PTB-XL parent first, or pass ``--build-parents``.

The clinical context comes from the annotations of the imported PTB-XL records. It states the patient's
age and sex, the recording device, and the signal-quality issues, extra beats, and pacemaker that PTB-XL
records for the recording.
"""

import ast
import asyncio
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Annotated, ClassVar

from pydantic import Field

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
from timenet_connectors.sources.csv import iter_csv_rows
from timenet_connectors.sources.rows import NonBlankText, SourceRow, validate_row


# The per-template answer options (the multiple-choice candidates), keyed by template_id. The URL
# names a specific commit of the ECG-QA repository, so the file cannot change under it.
ECG_QA_TEMPLATE_ANSWERS_URL = (
    "https://raw.githubusercontent.com/Jwoo5/ecg-qa/73386bb61b10d525c1e7c428bf5adee6140de750"
    "/ecgqa/ptbxl/answers_for_each_template.csv"
)
ECG_QA_TEMPLATE_ANSWERS_SHA256 = "e7d2850227793b786c49dc8d276b2e29b440123862f0a9140c0214bbc403db10"
# Precomputed CoT rows (question, answer, rationale, template). OpenTSLM's release is the only
# public source. This is a swappable constant so a mirror can replace it.
ECG_QA_COT_URL = "https://polybox.ethz.ch/index.php/s/D5QaJSEw4dXkzXm/download/ecg_qa_cot_final.zip"
# A share link can serve new bytes at the same URL, so the digest is what pins the release.
ECG_QA_COT_SHA256 = "f397786c949afc56a4c16fa31f5b01c7b0459a77a5d92a29b5d355577eb6164e"


# The PTB-XL signal-quality annotations, each with the phrase that the clinical context states if a
# record has it.
_QUALITY_ISSUES = (
    ("baseline_drift", "baseline drift noted"),
    ("static_noise", "static noise present"),
    ("burst_noise", "burst noise present"),
    ("electrodes_problems", "electrode artifacts present"),
)
# The PTB-XL findings outside signal quality, each with the phrase that the clinical context states
# if a record has it.
_OTHER_FINDINGS = (
    ("extra_beats", "extra beats detected during recording"),
    ("pacemaker", "pacemaker present"),
)
# The imported PTB-XL annotations that a record keeps. The connector clears every other one.
_KEPT_ANNOTATIONS = frozenset({"subject_ids", "sex", "age"})
# The sexes that an imported PTB-XL record can have. The connector refuses any other value.
_SEXES = frozenset({"male", "female"})


@dataclass(frozen=True)
class EcgQaCotSource:
    """A lightweight handle to the fetched sources, so ``compose`` streams rather than holding refs.

    ``download`` returns one of these instead of a row per question, so the ~230k questions are read
    lazily during ``compose`` and the task stream, never materialized as a list.
    """

    answers_path: Path
    cot_csvs: tuple[tuple[Split, Path], ...]  # (split, csv_path) for train / validation / test


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
        TimeNetConnectorError: If the record has no patient, age, or sex, or has a sex other than
            male or female.
    """
    values = {annotation.key: read_value(annotation.value) for annotation in record.annotations}
    patients = values.get("subject_ids")
    if not isinstance(patients, list) or not patients:
        raise TimeNetConnectorError(f"PTB-XL ecg_id {ecg_id}: patient_id is empty")
    # PTB-XL states the age as an int in years and the sex as "male" or "female". The connector
    # keeps both values as they are.
    age = values.get("age")
    if not isinstance(age, int):
        raise TimeNetConnectorError(f"PTB-XL ecg_id {ecg_id}: age is empty")
    sex = values.get("sex")
    if not isinstance(sex, str):
        raise TimeNetConnectorError(f"PTB-XL ecg_id {ecg_id}: sex is empty")
    if sex not in _SEXES:
        raise TimeNetConnectorError(f"PTB-XL ecg_id {ecg_id}: sex {sex!r} is not male or female")
    return _PtbxlRecording(
        patient_id=int(
            validate_row({"patient_id": patients[0]}, _PatientRow, location=f"PTB-XL ecg_id {ecg_id}").patient_id
        ),
        age=age,
        sex=sex,
        annotations=values,
    )


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
    extra beats, or pacemaker for which the imported record has an annotation.

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


_WholeNumber = Annotated[Decimal, Field(strict=False, allow_inf_nan=False, max_digits=4300, decimal_places=0)]


class _PatientRow(SourceRow):
    """A PTB-XL subject identifier written as decimal text."""

    patient_id: _WholeNumber


class _TemplateRow(SourceRow):
    """Published answer choices for one exact whole-number template ID."""

    template_id: _WholeNumber
    classes: NonBlankText
    question_type: str = ""


class _TemplateClasses(SourceRow):
    """Decoded answer options, preserving the published strings and their order."""

    classes: list[str]


class _CotCsvRow(SourceRow):
    """Published question columns, with verbatim optional clinical context."""

    ecg_id: NonBlankText
    template_id: _WholeNumber
    question_type: NonBlankText
    question: NonBlankText
    answer: NonBlankText
    rationale: str
    clinical_context: str = ""


def _load_template_answers(path: Path) -> dict[int, tuple[str, ...]]:
    """Load per-template answer options from ``answers_for_each_template.csv``.

    Args:
        path: Path to the CSV whose ``classes`` column is a Python list literal.

    Returns:
        A mapping of ``template_id`` to its ordered answer options.

    Raises:
        TimeNetConnectorError: If a classes cell is not a Python literal list of strings.
            The error names the file, physical row line, and field.
    """
    answers: dict[int, tuple[str, ...]] = {}
    for line, row in iter_csv_rows(path, _TemplateRow, unique="template_id"):
        location = f"{path}:{line}"
        try:
            decoded = ast.literal_eval(row.classes)
        except (SyntaxError, ValueError) as exc:
            raise TimeNetConnectorError(f"{location}: classes: invalid Python literal") from exc
        parsed = validate_row({"classes": decoded}, _TemplateClasses, location=location)
        answers[int(row.template_id)] = tuple(parsed.classes)
    return answers


def _iter_cot_rows(csv_path: Path) -> Iterator[_CotRow]:
    """Stream one split's CoT rows, so a whole split never lives in memory at once.

    Args:
        csv_path: The split's CoT CSV.

    Yields:
        Each parsed row. Rationale and clinical-context fields span multiple physical lines.
    """
    for _line, row in iter_csv_rows(csv_path, _CotCsvRow):
        yield _CotRow(
            ecg_id=_parse_ecg_id(row.ecg_id),
            template_id=int(row.template_id),
            question_type=row.question_type,
            question=row.question,
            answer=row.answer,
            rationale=row.rationale,
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

        The template-answers CSV and the CoT archive download concurrently. This method reads no
        CoT row. ``compose`` and the task stream read the rows lazily.

        Args:
            cache_dir: The directory that holds downloaded archives.

        Returns:
            A single-element list holding the :class:`EcgQaCotSource` handle.
        """
        answers_path = cache_dir / "answers_for_each_template.csv"
        # The two HTTP artifacts download concurrently.
        _, cot_root = await asyncio.gather(
            download_files(
                [Artifact(ECG_QA_TEMPLATE_ANSWERS_URL, answers_path, sha256=ECG_QA_TEMPLATE_ANSWERS_SHA256)]
            ),
            ensure_archive(ECG_QA_COT_URL, cache_dir, sha256=ECG_QA_COT_SHA256),
        )
        cot_csvs = tuple(
            (split, find_dir_containing(cot_root, csv_name) / csv_name) for split, csv_name in self._COT_CSVS
        )
        return [EcgQaCotSource(answers_path=answers_path, cot_csvs=cot_csvs)]

    def compose(self, raw_refs: list[EcgQaCotSource], context: BuildContext) -> TimeFDataset:
        """Build one record per recording and stream one :class:`AnswerTask` per CoT row.

        A first pass over the CoT CSVs collects each recording's split. Then the method imports the selected parent records. It finds them by the
        ``ecg_id`` in their metadata. It builds the clinical context of each recording from the
        imported PTB-XL annotations. The task stream registers shared metadata annotations on first use. The tasks
        themselves stream from :meth:`_iter_tasks`, so the ~230k questions never all live in memory.

        Args:
            raw_refs: The single-element list from :meth:`download`.
            context: The opened PTB-XL parent, under the parent alias ``ptbxl``.

        Returns:
            The dataset: recording records plus a task stream.

        Raises:
            TimeNetConnectorError: If the PTB-XL parent has no recording for an ECG that the rows
                name, or if an imported Record has no patient, age, or sex, or has a sex other than
                male or female.
        """
        source = raw_refs[0]
        answers = _load_template_answers(source.answers_path)
        dataset = TimeFDataset(metadata=self.metadata())

        split_of_ecg: dict[int, Split] = {}
        for split, csv_path in source.cot_csvs:
            for row in _iter_cot_rows(csv_path):
                split_of_ecg.setdefault(row.ecg_id, split)

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
            # Keep subject IDs, sex, and age from PTB-XL. The child adds its own split.
            record.annotations = tuple(
                annotation for annotation in record.annotations if annotation.key in _KEPT_ANNOTATIONS
            )
            record.add_annotations([Annotation(key="split", value=split.value)])
            records[ecg_id] = record
            contexts[ecg_id] = _clinical_context(recording)

        dataset.set_task_stream(
            [AnswerTask],
            lambda: self._iter_tasks(source, records, dataset, contexts, answers),
        )
        return dataset

    @staticmethod
    def _iter_tasks(
        source: EcgQaCotSource,
        records: Mapping[int, Record],
        dataset: TimeFDataset,
        contexts: Mapping[int, str],
        answers: Mapping[int, tuple[str, ...]],
    ) -> Iterator[AnswerTask]:
        """Yield one :class:`AnswerTask` per CoT row, referencing its recording and metadata annotations.

        Args:
            source: The download handle naming the CoT CSVs.
            records: Input Records keyed by PTB-XL ECG ID.
            dataset: The dataset whose shared annotations the tasks reference.
            contexts: Each recording's clinical context, keyed by PTB-XL ECG ID.
            answers: Per-template answer options.

        Yields:
            Each question as an answer task, streamed so the whole set never lives in memory.
        """
        for split, csv_path in source.cot_csvs:
            for row in _iter_cot_rows(csv_path):
                input_annotations = [
                    dataset.annotate_shared(_question_type(row.question_type)),
                ]
                if options := answers.get(row.template_id):
                    input_annotations.append(dataset.annotate_shared(_answer_options(options)))
                input_annotations.append(dataset.annotate_shared(_clinical_context_annotation(contexts[row.ecg_id])))
                yield AnswerTask(
                    inputs=(records[row.ecg_id],),
                    prompt=row.question,
                    targets=(row.answer,),
                    rationale=row.rationale,
                    input_annotations=tuple(input_annotations),
                    split=split,
                    metadata={"template_id": row.template_id},
                )


def _question_type(question_type: str) -> Annotation:
    """Return the annotation of a question type."""
    return Annotation(key="question_type", value=question_type)


def _answer_options(options: tuple[str, ...]) -> Annotation:
    """Return the annotation of a template's answer options."""
    return Annotation(key="answer_options", value=list(options))


def _clinical_context_annotation(context: str) -> Annotation:
    """Return the annotation of a recording's clinical context."""
    return Annotation(key="clinical_context", value=context)


CONNECTOR = EcgQaCotConnector
