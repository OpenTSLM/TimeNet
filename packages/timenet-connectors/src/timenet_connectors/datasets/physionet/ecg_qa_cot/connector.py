"""Compose ECG-QA CoT questions and rationales over the reusable PTB-XL layer.

Each chain-of-thought row becomes an :class:`~timenet.types.AnswerTask` on one imported PTB-XL
record. Its ``prompt`` is the question, its ``target`` is the short evaluation answer, and its
``rationale`` is the reasoning target. Per-question metadata is stored once as value-deduplicated
annotations. The approximately 230,000 tasks stream to disk and do not all live in memory.
"""

import ast
import asyncio
from collections.abc import Iterator, Mapping
import csv
import hashlib
from pathlib import Path
from typing import Annotated, Any, ClassVar

from pydantic import BaseModel, BeforeValidator, ConfigDict
from pydantic.dataclasses import dataclass

from timenet.composition import BuildContext
from timenet.dataset import Record, TimeFDataset
from timenet.types import (
    Annotation,
    AnswerTask,
    Split,
)
from timenet_connectors.bases.physionet import BasePhysioNetConnector
from timenet_connectors.datasets.physionet.ptb_xl.connector import record_id
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


def _parse_ecg_id(value: Any) -> int:
    return int(str(value).strip().strip("[]").strip())


def _parse_literal(value: Any) -> Any:
    return ast.literal_eval(value) if isinstance(value, str) else value


_EcgId = Annotated[int, BeforeValidator(_parse_ecg_id)]
_Context = Annotated[str, BeforeValidator(lambda value: value or _DEFAULT_CONTEXT)]
_Classes = Annotated[tuple[str, ...], BeforeValidator(_parse_literal)]


class _TemplateAnswers(BaseModel):
    """One validated row of template answer options."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    template_id: int
    classes: _Classes


class _CotRow(BaseModel):
    """One validated ECG-QA chain-of-thought row."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    ecg_id: _EcgId
    question_type: str
    template_id: int
    clinical_context: _Context = _DEFAULT_CONTEXT
    question: str
    answer: str
    rationale: str


# Stable, value-derived ids so the annotation a task references is the same object every recording
# shares. Building the annotation and referencing it from a task both go through these, so they agree.
def _qtype_id(question_type: str) -> str:
    return f"ecgqa-qtype-{question_type}"


def _template_ann_id(template_id: int) -> str:
    return f"ecgqa-template-{template_id}"


def _options_id(template_id: int) -> str:
    return f"ecgqa-options-{template_id}"


def _context_id(context: str) -> str:
    return f"ecgqa-context-{hashlib.sha1(context.encode('utf-8')).hexdigest()[:12]}"  # noqa: S324 (id, not security)


def _load_template_answers(path: Path) -> dict[int, tuple[str, ...]]:
    """Load per-template answer options from ``answers_for_each_template.csv``.

    Args:
        path: Path to the CSV whose ``classes`` column is a Python list literal.

    Returns:
        A mapping of ``template_id`` to its ordered answer options.
    """
    with path.open(newline="", encoding="utf-8") as handle:
        rows = (_TemplateAnswers.model_validate(row) for row in csv.DictReader(handle))
        return {row.template_id: row.classes for row in rows}


def _iter_cot_rows(csv_path: Path) -> Iterator[_CotRow]:
    """Stream one split's CoT rows, so a whole split never lives in memory at once.

    Args:
        csv_path: The split's CoT CSV.

    Yields:
        Each validated row. Rationale and clinical-context fields span multiple physical lines.
    """
    with csv_path.open(newline="", encoding="utf-8") as handle:
        yield from (_CotRow.model_validate(row) for row in csv.DictReader(handle))


class EcgQaCotConnector(BasePhysioNetConnector[EcgQaCotSource]):
    """Connector for the ECG-QA CoT dataset (PTB-XL signals + OpenTSLM chain-of-thought QA)."""

    _COT_CSVS: ClassVar[tuple[tuple[str, str], ...]] = (
        ("train", "ecg_qa_cot_train.csv"),
        ("validation", "ecg_qa_cot_val.csv"),
        ("test", "ecg_qa_cot_test.csv"),
    )

    async def download_async(self, cache_dir: Path) -> list[EcgQaCotSource]:
        """Fetch the template answers and CoT CSVs.

        The PTB-XL parent connector owns the ECG download. This connector fetches only the metadata
        that it adds. The two HTTP artifacts download concurrently. No CoT row is read here;
        ``convert`` and the task stream read them lazily.

        Args:
            cache_dir: The directory that holds downloaded archives.

        Returns:
            A single-element list holding the :class:`EcgQaCotSource` handle.
        """
        answers_path = cache_dir / "answers_for_each_template.csv"
        _, cot_root = await asyncio.gather(
            download_files([Artifact(ECG_QA_TEMPLATE_ANSWERS_URL, answers_path)]),
            ensure_archive(ECG_QA_COT_URL, cache_dir),
        )
        cot_csvs = tuple(
            (split, find_dir_containing(cot_root, csv_name) / csv_name) for split, csv_name in self._COT_CSVS
        )
        return [EcgQaCotSource(answers_path=answers_path, cot_csvs=cot_csvs)]

    def compose(self, raw_refs: list[EcgQaCotSource], context: BuildContext) -> TimeFDataset:
        """Import the needed PTB-XL records and stream one AnswerTask per CoT row.

        A first pass over the CoT CSVs collects each recording's split and the distinct question
        metadata. It imports those records from ``physionet/ptb-xl`` and registers the
        deduplicated metadata annotations. The tasks stream from :meth:`_iter_tasks`.

        Args:
            raw_refs: The single-element list from :meth:`download`.
            context: Build context that resolves the exact PTB-XL parent.

        Returns:
            The dataset: recording records plus a task stream.

        """
        source = raw_refs[0]
        answers = _load_template_answers(source.answers_path)
        dataset = TimeFDataset(metadata=context.metadata)

        split_of_ecg: dict[int, str] = {}
        question_types: set[str] = set()
        template_ids: set[int] = set()
        contexts: set[str] = set()
        for split, csv_path in source.cot_csvs:
            for row in _iter_cot_rows(csv_path):
                split_of_ecg.setdefault(row.ecg_id, split)
                question_types.add(row.question_type)
                template_ids.add(row.template_id)
                contexts.add(row.clinical_context)

        ptbxl = context.parent("physionet/ptb-xl")
        ecg_ids = sorted(split_of_ecg)
        parent_records = ptbxl.iter_records(record_id(ecg_id) for ecg_id in ecg_ids)
        records = {
            ecg_id: dataset.import_record(parent_record, parent="physionet/ptb-xl")
            for ecg_id, parent_record in zip(ecg_ids, parent_records, strict=True)
        }

        task_annotations = {
            annotation.id: dataset.annotate(annotation)
            for annotation in self._metadata_annotations(question_types, template_ids, contexts, answers)
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
    ) -> list[Annotation]:
        """Build the value-deduped annotations the QA tasks reference.

        Args:
            question_types: The distinct question types.
            template_ids: The distinct template ids.
            contexts: The distinct clinical-context strings.
            answers: Per-template answer options.

        Returns:
            One annotation per distinct value, with the same ids :meth:`_iter_tasks` references.
        """
        annotations = [Annotation(key="question_type", value=qt, id=_qtype_id(qt)) for qt in sorted(question_types)]
        for template_id in sorted(template_ids):
            annotations.append(Annotation(key="template_id", value=template_id, id=_template_ann_id(template_id)))
            options = answers.get(template_id)
            if options:
                annotations.append(Annotation(key="answer_options", value=list(options), id=_options_id(template_id)))
        annotations.extend(
            Annotation(key="clinical_context", value=context, id=_context_id(context)) for context in sorted(contexts)
        )
        return annotations

    @staticmethod
    def _iter_tasks(
        source: EcgQaCotSource,
        answers: dict[int, tuple[str, ...]],
        records: Mapping[int, Record],
        annotations: Mapping[str, Annotation],
    ) -> Iterator[AnswerTask]:
        """Yield one :class:`AnswerTask` per CoT row, referencing its recording and metadata annotations.

        Args:
            source: The download handle naming the CoT CSVs.
            answers: Per-template answer options (decides whether a task references an options annotation).
            records: Input Records keyed by PTB-XL ECG ID.
            annotations: Dataset annotation occurrences keyed by content ID.

        Yields:
            Each question as an answer task, streamed so the whole set never lives in memory.
        """
        for split, csv_path in source.cot_csvs:
            for index, row in enumerate(_iter_cot_rows(csv_path)):
                ecg_id = row.ecg_id
                template_id = row.template_id
                input_ids = [_qtype_id(row.question_type), _template_ann_id(template_id)]
                if answers.get(template_id):
                    input_ids.append(_options_id(template_id))
                input_ids.append(_context_id(row.clinical_context))
                yield AnswerTask(
                    inputs=(records[ecg_id],),
                    prompt=row.question,
                    targets=(row.answer,),
                    split=Split(split),
                    rationale=row.rationale,
                    input_annotations=tuple(annotations[annotation_id] for annotation_id in input_ids),
                    id=f"ecgqa-{split}-{index}",
                )


CONNECTOR = EcgQaCotConnector
