"""The ECG-QA CoT connector: PTB-XL 12-lead ECGs with chain-of-thought question answering.

Each record is one PTB-XL recording: 12 leads at 500 Hz in millivolts. Every chain-of-thought row is
an :class:`~timenet.types.AnswerTask` on that recording. Its ``prompt`` is the question, ``targets``
is the short evaluation answer, and ``rationale`` is the reasoning target. So a recording carries many
QA tasks. There is no record per question. The connector stores per-question metadata (question type,
template, answer options, clinical context) once, as value-deduped annotations that the tasks
reference. Each recording carries its dataset split, and the patient's sex and age from PTB-XL. The ~230k
tasks stream to disk, so they never all live in memory.

Sources come from three places. The signals and the patient metadata come from PhysioNet PTB-XL. The
per-template answer options come from the ``Jwoo5/ecg-qa`` GitHub repo. The precomputed CoT rows
(question, answer, rationale, template) come from the OpenTSLM release, the only public source for the
rationales. Real build needs the network and a multi-GB PTB-XL download.

The clinical context comes from ``ptbxl_database.csv``. It states the patient's age and sex, the
recording device, and the signal-quality issues, extra beats, and pacemaker that PTB-XL records for the
recording.
"""

import ast
import asyncio
from collections.abc import Iterator, Mapping
import csv
from dataclasses import dataclass
from enum import StrEnum, unique
from fractions import Fraction
from pathlib import Path
from typing import ClassVar

from timenet.dataset import Record, Signal, Source, TimeFDataset
from timenet.dataset.axis import RegularAxis
from timenet.errors import TimeFFormatError
from timenet.types import (
    Annotation,
    AnswerTask,
    TimeSeriesSpec,
    parse_split,
    ureg,
)
from timenet_connectors.bases.physionet import BasePhysioNetConnector
from timenet_connectors.download import Artifact, download_files, ensure_archive, find_dir_containing


# PTB-XL 500 Hz records from PhysioNet's open S3 bucket. ``_hr`` means high-rate (500 Hz) recordings.
# Fetched with boto3, which resolves AWS credentials itself (anonymous access works for this open bucket).
PTBXL_ZIP_URL = "s3://physionet-open/ptb-xl/ptb-xl-1.0.3.zip"
# The per-template answer options (the multiple-choice candidates), keyed by template_id.
ECG_QA_TEMPLATE_ANSWERS_URL = (
    "https://raw.githubusercontent.com/Jwoo5/ecg-qa/master/ecgqa/ptbxl/answers_for_each_template.csv"
)
# Precomputed CoT rows (question, answer, rationale, template). OpenTSLM's release is the only
# public source. This is a swappable constant so a mirror can replace it.
ECG_QA_COT_URL = "https://polybox.ethz.ch/index.php/s/D5QaJSEw4dXkzXm/download/ecg_qa_cot_final.zip"

_ECG = TimeSeriesSpec(
    spec_type="ecg",
    name="12-lead ECG",
    unit_value=ureg.millivolt,
)


@unique
class Sex(StrEnum):
    """A patient's sex, as the clinical context and the ``sex`` annotation state it."""

    MALE = "male"
    FEMALE = "female"


# PTB-XL codes sex as 0 = male, 1 = female.
_PTBXL_SEX = {"0": Sex.MALE, "1": Sex.FEMALE}
# PTB-XL signal-quality columns and the phrase the clinical context states when a cell has a value.
_QUALITY_ISSUES = (
    ("baseline_drift", "baseline drift noted"),
    ("static_noise", "static noise present"),
    ("burst_noise", "burst noise present"),
    ("electrodes_problems", "electrode artifacts present"),
)
# PTB-XL findings outside signal quality and the phrase the clinical context states when a cell has a value.
_OTHER_FINDINGS = (
    ("extra_beats", "extra beats detected during recording"),
    ("pacemaker", "pacemaker present"),
)
# PTB-XL cells the connector parses as numbers. Each must have a value.
_REQUIRED_NUMERIC_COLUMNS = ("patient_id", "age")
# PTB-XL anonymizes the age of every patient older than 89 as 300, in compliance with HIPAA (PhysioNet
# PTB-XL page, Data Preprocessing: "ages of more than 89 years appear in the range of 300 years").
_PTBXL_AGE_OVER_89 = 300
_SEX_DESCRIPTION = "The patient's sex, as PTB-XL states it."
_AGE_DESCRIPTION = (
    f"The patient's age in whole years, as PTB-XL states it. PTB-XL states {_PTBXL_AGE_OVER_89} for every "
    "patient older than 89."
)


@dataclass(frozen=True)
class EcgQaCotSource:
    """A lightweight handle to the fetched sources, so ``convert`` streams rather than holding refs.

    ``download`` returns one of these instead of a row per question, so the ~230k questions are read
    lazily during ``convert`` and the task stream, never materialized as a list.
    """

    records_root: Path
    ptbxl_database_path: Path
    answers_path: Path
    cot_csvs: tuple[tuple[str, Path], ...]  # (split, csv_path) for train / validation / test


def _parse_ecg_id(raw: object) -> int:
    """Parse a PTB-XL ecg_id that arrives as ``123`` or ``"[123]"``.

    Args:
        raw: The raw ecg_id value from a CoT row.

    Returns:
        The integer ecg_id.
    """
    return int(str(raw).strip().strip("[]").strip())


def _record_base(records_root: Path, ecg_id: int) -> Path:
    """Resolve a recording's path in the PTB-XL ``records500/<bucket>/`` layout.

    Args:
        records_root: The ``records500`` directory.
        ecg_id: The recording id.

    Returns:
        The record path without the ``.dat`` / ``.hea`` extension.
    """
    return records_root / f"{ecg_id // 1000 * 1000:05d}" / f"{ecg_id:05d}_hr"


def _load_ptbxl_database(path: Path) -> dict[int, dict[str, str]]:
    """Load ``ptbxl_database.csv`` rows keyed by ``ecg_id``.

    Args:
        path: Path to ``ptbxl_database.csv``.

    Returns:
        Each recording's row, with every cell as its raw CSV text.
    """
    with path.open(newline="", encoding="utf-8") as handle:
        return {int(row["ecg_id"]): row for row in csv.DictReader(handle)}


@dataclass(frozen=True)
class _PtbxlRecording:
    """One recording's PTB-XL values, parsed once from its ``ptbxl_database.csv`` row."""

    patient_id: int
    age: int
    sex: Sex
    row: Mapping[str, str]
    """The raw row, for the device, recording-date, signal-quality, extra-beats and pacemaker cells."""


def _ptbxl_recording(ptbxl: Mapping[int, Mapping[str, str]], ecg_id: int) -> _PtbxlRecording:
    """Parse a recording's PTB-XL row.

    Args:
        ptbxl: PTB-XL rows keyed by ``ecg_id``.
        ecg_id: The recording id.

    Returns:
        The recording's parsed PTB-XL values.

    Raises:
        TimeFFormatError: If PTB-XL has no row for the recording, a required numeric cell is empty, or the
            row's sex is not ``0`` or ``1``.
    """
    row = ptbxl.get(ecg_id)
    if row is None:
        raise TimeFFormatError(f"ptbxl_database.csv has no row for ecg_id {ecg_id}")
    for column in _REQUIRED_NUMERIC_COLUMNS:
        if not row[column].strip():
            raise TimeFFormatError(f"ptbxl_database.csv ecg_id {ecg_id}: {column} is empty")
    sex = _PTBXL_SEX.get(row["sex"])
    if sex is None:
        raise TimeFFormatError(f"ptbxl_database.csv ecg_id {ecg_id}: sex {row['sex']!r} is not 0 or 1")
    return _PtbxlRecording(patient_id=int(float(row["patient_id"])), age=int(float(row["age"])), sex=sex, row=row)


def _clinical_context(recording: _PtbxlRecording) -> str:
    """Build a recording's clinical context from its PTB-XL values.

    The context states the patient's age and sex, the recording device, and each signal-quality issue,
    extra beats, or pacemaker whose PTB-XL cell has a value.

    Args:
        recording: The recording's parsed PTB-XL values.

    Returns:
        The clinical-context text.
    """
    row = recording.row
    parts = [f"{recording.age}-year-old", f"{recording.sex.value} patient", "12-lead ECG"]
    if row["recording_date"].strip():
        parts.append("clinical recording")
    if device := row["device"].strip():
        parts.append(f"recorded with {device}")
    issues = [phrase for column, phrase in _QUALITY_ISSUES if row[column].strip()]
    parts.append(f"Signal quality: {', '.join(issues)}" if issues else "Signal quality: adequate for analysis")
    parts.extend(phrase for column, phrase in _OTHER_FINDINGS if row[column].strip())
    return f"{parts[0]} {parts[1]}. " + ". ".join(parts[2:]) + "."


class _PatientAnnotations:
    """Give each record its sex and age annotations, one instance per distinct value.

    Many records share a sex or an age, so the annotation for a value is built once and every later record
    that states that value gets the same instance.
    """

    def __init__(self) -> None:
        self._sex: dict[Sex, Annotation] = {}
        self._age: dict[int, Annotation] = {}

    def for_recording(self, recording: _PtbxlRecording) -> list[Annotation]:
        """Give a recording's sex and age annotations.

        Args:
            recording: The recording's parsed PTB-XL values.

        Returns:
            The sex annotation and the age annotation, with no span.
        """
        sex = self._sex.get(recording.sex)
        if sex is None:
            sex = Annotation(key="sex", value=recording.sex.value, description=_SEX_DESCRIPTION)
            self._sex[recording.sex] = sex
        age = self._age.get(recording.age)
        if age is None:
            age = Annotation(key="age", value=recording.age, unit=ureg.year, description=_AGE_DESCRIPTION)
            self._age[recording.age] = age
        return [sex, age]


# The connector looks up a metadata annotation by its key and the upstream value that it comes from.
_AnnotationKey = tuple[str, str | int]


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
            answers[int(float(row["template_id"]))] = tuple(str(option) for option in classes)
    return answers


def _iter_cot_rows(csv_path: Path) -> Iterator[dict[str, str]]:
    """Stream one split's CoT rows, so a whole split never lives in memory at once.

    Args:
        csv_path: The split's CoT CSV.

    Yields:
        Each row as a dict. Rationale and clinical-context fields span multiple physical lines.
    """
    with csv_path.open(newline="", encoding="utf-8") as handle:
        yield from csv.DictReader(handle)


class EcgQaCotConnector(BasePhysioNetConnector[EcgQaCotSource]):
    """Connector for the ECG-QA CoT dataset (PTB-XL signals + OpenTSLM chain-of-thought QA)."""

    _COT_CSVS: ClassVar[tuple[tuple[str, str], ...]] = (
        ("train", "ecg_qa_cot_train.csv"),
        ("validation", "ecg_qa_cot_val.csv"),
        ("test", "ecg_qa_cot_test.csv"),
    )

    async def download_async(self, cache_dir: Path) -> list[EcgQaCotSource]:
        """Fetch PTB-XL, the template answers, and the CoT CSVs, and return one lightweight handle.

        PTB-XL is an S3 archive fetched synchronously (boto3 parallelizes the transfer internally). The
        two HTTP artifacts, the template-answers CSV and the CoT archive, download concurrently. No CoT
        row is read here; ``convert`` and the task stream read them lazily.

        Args:
            cache_dir: The directory that holds downloaded archives.

        Returns:
            A single-element list holding the :class:`EcgQaCotSource` handle.
        """
        # PTB-XL is an S3 archive. Fetch it first (boto3 blocks the loop but parallelizes the transfer).
        ptbxl_root = find_dir_containing(await ensure_archive(PTBXL_ZIP_URL, cache_dir), "ptbxl_database.csv")
        answers_path = cache_dir / "answers_for_each_template.csv"
        # The two HTTP artifacts download concurrently.
        _, cot_root = await asyncio.gather(
            download_files([Artifact(ECG_QA_TEMPLATE_ANSWERS_URL, answers_path)]),
            ensure_archive(ECG_QA_COT_URL, cache_dir),
        )
        cot_csvs = tuple(
            (split, find_dir_containing(cot_root, csv_name) / csv_name) for split, csv_name in self._COT_CSVS
        )
        return [
            EcgQaCotSource(
                records_root=ptbxl_root / "records500",
                ptbxl_database_path=ptbxl_root / "ptbxl_database.csv",
                answers_path=answers_path,
                cot_csvs=cot_csvs,
            )
        ]

    def convert(self, raw_refs: list[EcgQaCotSource]) -> TimeFDataset:
        """Build one record per recording and stream one :class:`AnswerTask` per CoT row.

        A first pass over the CoT CSVs collects each recording's split and the distinct question
        metadata. It then builds a record per recording (its 12 leads with lazy loaders, and its patient
        sex and age from PTB-XL), builds each recording's clinical context from PTB-XL, and registers
        the deduped metadata annotations. The tasks themselves stream from :meth:`_iter_tasks`, so the
        ~230k questions never all live in memory.

        Args:
            raw_refs: The single-element list from :meth:`download`.

        Returns:
            The dataset: recording records plus a task stream.
        """
        source = raw_refs[0]
        answers = _load_template_answers(source.answers_path)
        ptbxl = _load_ptbxl_database(source.ptbxl_database_path)
        dataset = TimeFDataset(metadata=self.metadata())

        split_of_ecg: dict[int, str] = {}
        question_types: set[str] = set()
        template_ids: set[int] = set()
        for split, csv_path in source.cot_csvs:
            for row in _iter_cot_rows(csv_path):
                split_of_ecg.setdefault(_parse_ecg_id(row["ecg_id"]), split)
                question_types.add(row["question_type"])
                template_ids.add(int(float(row["template_id"])))

        patient_annotations = _PatientAnnotations()
        records: dict[int, Record] = {}
        contexts: dict[int, str] = {}
        for ecg_id, split in sorted(split_of_ecg.items()):
            ptbxl_recording = _ptbxl_recording(ptbxl, ecg_id)
            record_base = _record_base(source.records_root, ecg_id)
            record = Record(
                sources=(Source(name="PTB-XL ECG", signals=self._leads_for(record_base)),),
                metadata={"ecg_id": ecg_id, "patient_id": ptbxl_recording.patient_id},
            )
            dataset.add_record(record=record)
            record.add_annotations(
                [Annotation(key="split", value=split), *patient_annotations.for_recording(ptbxl_recording)]
            )
            records[ecg_id] = record
            contexts[ecg_id] = _clinical_context(ptbxl_recording)

        task_annotations = {
            key: dataset.annotate(annotation)
            for key, annotation in self._metadata_annotations(
                question_types, template_ids, set(contexts.values()), answers
            ).items()
        }
        dataset.set_task_stream(
            [AnswerTask],
            lambda: self._iter_tasks(source, answers, records, contexts, task_annotations),
        )
        return dataset

    def _leads_for(self, record_base: Path) -> tuple[Signal, ...]:
        """Build the 12 lead :class:`Signal` for one recording with lazy per-lead loaders.

        Args:
            record_base: The record path without the ``.dat`` / ``.hea`` extension.

        Returns:
            One :class:`Signal` per lead. All leads share the ECG spec.
        """
        header = self._read_header(record_base)
        axis = RegularAxis.from_rate_hz(Fraction(str(header.fs)))
        return tuple(
            Signal.from_loader(
                spec=_ECG,
                name=name,
                time_axis=axis,
                loader=self._lead_loader(record_base, header, lead_idx),
                n_values=int(header.sig_len),
            )
            for lead_idx, name in enumerate(header.sig_name)
        )

    @staticmethod
    def _metadata_annotations(
        question_types: set[str],
        template_ids: set[int],
        contexts: set[str],
        answers: dict[int, tuple[str, ...]],
    ) -> dict[_AnnotationKey, Annotation]:
        """Build the value-deduped annotations the QA tasks reference.

        Args:
            question_types: The distinct question types.
            template_ids: The distinct template ids.
            contexts: The distinct clinical-context strings.
            answers: Per-template answer options.

        Returns:
            One annotation per distinct value. Its dict key is the annotation key and the upstream value
            that :meth:`_iter_tasks` uses to look it up.
        """
        annotations: dict[_AnnotationKey, Annotation] = {
            ("question_type", qt): Annotation(key="question_type", value=qt) for qt in sorted(question_types)
        }
        for template_id in sorted(template_ids):
            annotations["template_id", template_id] = Annotation(key="template_id", value=template_id)
            options = answers.get(template_id)
            if options:
                annotations["answer_options", template_id] = Annotation(key="answer_options", value=list(options))
        for context in sorted(contexts):
            annotations["clinical_context", context] = Annotation(key="clinical_context", value=context)
        return annotations

    @staticmethod
    def _iter_tasks(
        source: EcgQaCotSource,
        answers: dict[int, tuple[str, ...]],
        records: Mapping[int, Record],
        contexts: Mapping[int, str],
        annotations: Mapping[_AnnotationKey, Annotation],
    ) -> Iterator[AnswerTask]:
        """Yield one :class:`AnswerTask` per CoT row, referencing its recording and metadata annotations.

        Args:
            source: The download handle naming the CoT CSVs.
            answers: Per-template answer options (decides whether a task references an options annotation).
            records: Input Records keyed by PTB-XL ECG ID.
            contexts: Clinical contexts built from PTB-XL, keyed by PTB-XL ECG ID.
            annotations: Dataset annotation occurrences keyed by annotation key and upstream value.

        Yields:
            Each question as an answer task, streamed so the whole set never lives in memory.
        """
        for split, csv_path in source.cot_csvs:
            for row in _iter_cot_rows(csv_path):
                ecg_id = _parse_ecg_id(row["ecg_id"])
                template_id = int(float(row["template_id"]))
                input_keys: list[_AnnotationKey] = [
                    ("question_type", row["question_type"]),
                    ("template_id", template_id),
                ]
                if answers.get(template_id):
                    input_keys.append(("answer_options", template_id))
                input_keys.append(("clinical_context", contexts[ecg_id]))
                yield AnswerTask(
                    inputs=(records[ecg_id],),
                    prompt=str(row["question"]),
                    targets=(str(row["answer"]),),
                    rationale=str(row["rationale"]),
                    input_annotations=tuple(annotations[key] for key in input_keys),
                    split=parse_split(split),
                )


CONNECTOR = EcgQaCotConnector
