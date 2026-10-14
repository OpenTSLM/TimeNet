"""Add ECG-QA CoT questions and rationales to the stored PTB-XL recording catalog.

Each record retains its original ID, 12 leads at 500 Hz, and ECG-QA split annotation.
Waveform payloads remain in the exact PTB-XL input release. Question metadata,
answers, and rationales retain the standalone connector's consumer representation.
The replayable task stream avoids holding every question in memory.
"""

import ast
import asyncio
from collections.abc import Iterator, Mapping
import csv
from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import ClassVar

from timenet.connectors import BaseConnector, BuildContext
from timenet.dataset import Record, TimeFDataset
from timenet.errors import TimeFValidationError
from timenet.types import (
    Annotation,
    AnswerTask,
    DatasetRef,
)
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
        """Fetch the template answers and CoT CSVs, and return one lightweight handle.

        The PTB-XL input owns the waveform download. This connector fetches only
        question metadata. Both HTTP artifacts download concurrently, and conversion
        reads their rows lazily.

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

    def convert(self, raw_refs: list[EcgQaCotSource], *, context: BuildContext | None = None) -> TimeFDataset:
        """Build one record per recording and stream one :class:`AnswerTask` per CoT row.

        Import the needed PTB-XL catalog records without reading waveform values.
        Retain their IDs and share stored payloads. Replace clinical annotations with
        the original ECG-QA split annotations, then add the unchanged QA task stream.

        Args:
            raw_refs: The single-element list from :meth:`download`.

        Returns:
            The dataset: recording records plus a task stream.

        Raises:
            TimeFValidationError: If the declared PTB-XL input is missing or has another release.
        """
        if context is None or "ptbxl" not in context.inputs:
            raise TimeFValidationError(
                "ECG-QA requires the opened 'ptbxl' input; build through the engine or pass BuildContext"
            )
        parent = context.inputs["ptbxl"]
        metadata = self.metadata()
        actual = DatasetRef(dataset_id=parent.metadata.dataset_id, version=parent.metadata.dataset_version)
        if actual != metadata.inputs["ptbxl"]:
            raise TimeFValidationError(f"ECG-QA requires {metadata.inputs['ptbxl']}, got {actual}")
        source = raw_refs[0]
        answers = _load_template_answers(source.answers_path)
        dataset = TimeFDataset(metadata=metadata)

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

        imported = dataset.copy_from(parent, record_ids=[f"ptbxl-{ecg_id}" for ecg_id in sorted(split_of_ecg)])
        records: dict[int, Record] = {}
        for ecg_id, split in sorted(split_of_ecg.items()):
            record = imported.records[f"ptbxl-{ecg_id}"]
            # The child preserves ECG-QA's consumer view, including its own split.
            # PTB-XL's clinical context remains available in the parent dataset.
            record.annotations = ()
            record.add_annotations([Annotation(key="split", value=split, id=f"ptbxl-{ecg_id}-split")])
            records[ecg_id] = record

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
                ecg_id = _parse_ecg_id(row["ecg_id"])
                template_id = int(float(row["template_id"]))
                input_ids = [_qtype_id(row["question_type"]), _template_ann_id(template_id)]
                if answers.get(template_id):
                    input_ids.append(_options_id(template_id))
                input_ids.append(_context_id(_clinical_context(row)))
                yield AnswerTask(
                    inputs=(records[ecg_id],),
                    prompt=str(row["question"]),
                    targets=(str(row["answer"]),),
                    rationale=str(row["rationale"]),
                    input_annotations=tuple(annotations[annotation_id] for annotation_id in input_ids),
                    id=f"ecgqa-{split}-{index}",
                )


CONNECTOR = EcgQaCotConnector
