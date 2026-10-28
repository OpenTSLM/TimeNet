"""The ECG-QA CoT connector: PTB-XL 12-lead ECGs with chain-of-thought question answering.

Each record is one PTB-XL recording: 12 leads at 500 Hz in millivolts. Every chain-of-thought row is
an :class:`~timenet.types.AnswerTask` on that recording. Its ``prompt`` is the question, ``targets``
is the short evaluation answer, and ``rationale`` is the reasoning target. So a recording carries many
QA tasks. There is no record per question. The connector stores per-question metadata (question type,
template, answer options, clinical context) once, as value-deduped annotations that the tasks
reference. Each recording carries its dataset split. The ~230k tasks stream to disk, so they never all live in memory.

Signals are imported from the pinned PTB-XL TimeF parent. The per-template answer options
come from the ``Jwoo5/ecg-qa`` GitHub repo. The precomputed CoT rows (question, answer, rationale,
template) come from the OpenTSLM release, the only public source for the rationales. Build the PTB-XL
parent first or pass ``--build-parents``.
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
from timenet.errors import TimeFValidationError
from timenet.types import (
    Annotation,
    AnswerTask,
    parse_split,
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

_DEFAULT_CONTEXT = "12-lead ECG recording."


@dataclass(frozen=True)
class EcgQaCotSource:
    """A lightweight handle to the fetched sources, so ``convert`` streams rather than holding refs.

    ``download`` returns one of these instead of a row per question, so the ~230k questions are read
    lazily during ``convert`` and the task stream, never materialized as a list.
    """

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


def _clinical_context(row: Mapping[str, str]) -> str:
    """Return a row's clinical context, or the default when the row carries none.

    Args:
        row: A CoT row.

    Returns:
        The clinical-context text.
    """
    return str(row.get("clinical_context") or _DEFAULT_CONTEXT)


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


class EcgQaCotConnector(BaseConnector[EcgQaCotSource]):
    """Connector for the ECG-QA CoT dataset (PTB-XL signals + OpenTSLM chain-of-thought QA)."""

    _COT_CSVS: ClassVar[tuple[tuple[str, str], ...]] = (
        ("train", "ecg_qa_cot_train.csv"),
        ("validation", "ecg_qa_cot_val.csv"),
        ("test", "ecg_qa_cot_test.csv"),
    )

    async def download_async(self, cache_dir: Path) -> list[EcgQaCotSource]:
        """Fetch the template answers and CoT CSVs and return one lightweight handle.

        The template-answers CSV and the CoT archive download concurrently. No CoT
        row is read here; ``convert`` and the task stream read them lazily.

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

    def convert(self, raw_refs: list[EcgQaCotSource]) -> TimeFDataset:  # noqa: PLR6301, ARG002
        """Require resolved parents for this composed dataset.

        Raises:
            TimeFValidationError: Always; use the build pipeline or ``compose``.
        """
        raise TimeFValidationError("ECG-QA requires a PTB-XL parent; use compose with a BuildContext")

    def compose(self, raw_refs: list[EcgQaCotSource], context: BuildContext) -> TimeFDataset:
        """Build one record per recording and stream one :class:`AnswerTask` per CoT row.

        A first pass over the CoT CSVs collects each recording's split and the distinct question
        metadata. It then imports the selected parent records, found by the ``ecg_id`` in their
        metadata, and registers the deduped metadata annotations. The tasks themselves stream from :meth:`_iter_tasks`, so the
        ~230k questions never all live in memory.

        Args:
            raw_refs: The single-element list from :meth:`download`.
            context: The opened PTB-XL parent, under the parent alias ``ptbxl``.

        Returns:
            The dataset: recording records plus a task stream.

        Raises:
            TimeFValidationError: If the PTB-XL parent has no recording for an ECG the rows name.
        """
        source = raw_refs[0]
        answers = _load_template_answers(source.answers_path)
        dataset = TimeFDataset(metadata=self.metadata())

        split_of_ecg: dict[int, str] = {}
        question_types: set[str] = set()
        template_ids: set[int] = set()
        contexts: set[str] = set()
        for split, csv_path in source.cot_csvs:
            for row in _iter_cot_rows(csv_path):
                split_of_ecg.setdefault(_parse_ecg_id(row["ecg_id"]), split)
                question_types.add(row["question_type"])
                template_ids.add(int(float(row["template_id"])))
                contexts.add(_clinical_context(row))

        ptbxl = context.parents[ParentAlias("ptbxl")]
        # PTB-XL keeps each recording's upstream ecg_id in the record's metadata.
        parent_ids = {
            record.metadata["ecg_id"]: record.id
            for record in ptbxl.iter_records(with_annotations=False)
            if record.metadata.get("ecg_id") in split_of_ecg
        }
        missing = sorted(split_of_ecg.keys() - parent_ids.keys())
        if missing:
            raise TimeFValidationError(f"the PTB-XL parent has no recording with ecg_id {missing}")
        imported = dataset.import_from(ptbxl, records=[parent_ids[ecg_id] for ecg_id in sorted(split_of_ecg)])
        records: dict[int, Record] = {}
        for ecg_id, split in sorted(split_of_ecg.items()):
            record = imported.records[parent_ids[ecg_id]]
            record.annotations = ()
            record.add_annotations([Annotation(key="split", value=split)])
            records[ecg_id] = record

        task_annotations = {
            key: dataset.annotate(annotation)
            for key, annotation in self._metadata_annotations(question_types, template_ids, contexts, answers).items()
        }
        dataset.set_task_stream(
            [AnswerTask],
            lambda: self._iter_tasks(source, answers, records, task_annotations),
        )
        return dataset

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
        annotations: Mapping[_AnnotationKey, Annotation],
    ) -> Iterator[AnswerTask]:
        """Yield one :class:`AnswerTask` per CoT row, referencing its recording and metadata annotations.

        Args:
            source: The download handle naming the CoT CSVs.
            answers: Per-template answer options (decides whether a task references an options annotation).
            records: Input Records keyed by PTB-XL ECG ID.
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
                input_keys.append(("clinical_context", _clinical_context(row)))
                yield AnswerTask(
                    inputs=(records[ecg_id],),
                    prompt=str(row["question"]),
                    targets=(str(row["answer"]),),
                    rationale=str(row["rationale"]),
                    input_annotations=tuple(annotations[key] for key in input_keys),
                    split=parse_split(split),
                )


CONNECTOR = EcgQaCotConnector
