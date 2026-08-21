"""The ECG-QA CoT connector: PTB-XL 12-lead ECGs with chain-of-thought question answering.

Each sample is one PTB-XL recording: 12 leads at 500 Hz in millivolts. Every chain-of-thought row is
an :class:`~timenet.types.AnswerTask` on that recording, whose ``prompt`` is the question, ``target``
is the short evaluation answer, and ``rationale`` is the reasoning target. So a recording carries many
QA tasks; there is no sample per question. Per-question metadata (question type, template, answer
options, clinical context) is stored once as value-deduped annotations the tasks reference, and each
recording carries its dataset split. The ~230k tasks stream to disk, so they never all live in memory.

Sources come from three places. The signals come from PhysioNet PTB-XL. The per-template answer options
come from the ``Jwoo5/ecg-qa`` GitHub repo. The precomputed CoT rows (question, answer, rationale,
template) come from the OpenTSLM release, the only public source for the rationales. Real build needs
the network and a multi-GB PTB-XL download.
"""

import ast
import asyncio
from collections.abc import Iterator, Mapping
import csv
from dataclasses import dataclass
from fractions import Fraction
import hashlib
from pathlib import Path
from typing import ClassVar

from timenet.dataset import TimeFDataset, TimeSeries
from timenet.dataset.axis import RegularAxis
from timenet.types import (
    Annotation,
    AnswerTask,
    DataSource,
    TimeSeriesSpec,
    ureg,
)
from timenet_connectors.bases.physionet import BasePhysioNetConnector
from timenet_connectors.download import Artifact, download_files, ensure_archive


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

_SOURCE = DataSource(data_source_type="physionet", name="PTB-XL", provider="PhysioNet")
_ECG = TimeSeriesSpec(
    spec_type="ecg",
    name="12-lead ECG",
    unit_value=ureg.millivolt,
    data_source=_SOURCE,
)
_DEFAULT_CONTEXT = "12-lead ECG recording."


@dataclass(frozen=True)
class EcgQaCotSource:
    """A lightweight handle to the fetched sources, so ``convert`` streams rather than holding refs.

    ``download`` returns one of these instead of a row per question, so the ~230k questions are read
    lazily during ``convert`` and the task stream, never materialized as a list.
    """

    records_root: Path
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


def _find_dir_containing(root: Path, relative: str) -> Path:
    """Find the directory under ``root`` that contains ``relative`` (archives extract nested).

    Args:
        root: The extraction root to search.
        relative: A file name expected inside the wanted directory.

    Returns:
        The parent directory of the first match.

    Raises:
        FileNotFoundError: If nothing matches.
    """
    for match in root.rglob(relative):
        return match.parent
    raise FileNotFoundError(f"{relative!r} not found under {root}")


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
        ptbxl_root = _find_dir_containing(await ensure_archive(PTBXL_ZIP_URL, cache_dir), "ptbxl_database.csv")
        answers_path = cache_dir / "answers_for_each_template.csv"
        # The two HTTP artifacts download concurrently.
        _, cot_root = await asyncio.gather(
            download_files([Artifact(ECG_QA_TEMPLATE_ANSWERS_URL, answers_path)]),
            ensure_archive(ECG_QA_COT_URL, cache_dir),
        )
        cot_csvs = tuple(
            (split, _find_dir_containing(cot_root, csv_name) / csv_name) for split, csv_name in self._COT_CSVS
        )
        return [EcgQaCotSource(records_root=ptbxl_root / "records500", answers_path=answers_path, cot_csvs=cot_csvs)]

    def convert(self, raw_refs: list[EcgQaCotSource]) -> TimeFDataset:
        """Build one sample per recording and stream one :class:`AnswerTask` per CoT row.

        A first pass over the CoT CSVs collects each recording's split and the distinct question
        metadata. It then builds a sample per recording (its 12 leads with lazy loaders) and registers
        the deduped metadata annotations. The tasks themselves stream from :meth:`_iter_tasks`, so the
        ~230k questions never all live in memory.

        Args:
            raw_refs: The single-element list from :meth:`download`.

        Returns:
            The dataset: recording samples plus a task stream.
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

        for ecg_id, split in sorted(split_of_ecg.items()):
            record_base = _record_base(source.records_root, ecg_id)
            sample = dataset.add_sample(time_series=self._leads_for(ecg_id, record_base), sample_id=f"ptbxl-{ecg_id}")
            sample.add_annotations([Annotation(key="split", value=split, id=f"ptbxl-{ecg_id}-split")])

        dataset.register_annotations(self._metadata_annotations(question_types, template_ids, contexts, answers))
        dataset.set_task_stream([AnswerTask], lambda: self._iter_tasks(source, answers))
        return dataset

    def _leads_for(self, ecg_id: int, record_base: Path) -> tuple[TimeSeries, ...]:
        """Build the 12 lead :class:`TimeSeries` for one recording with lazy per-lead loaders.

        Args:
            ecg_id: The recording id.
            record_base: The record path without the ``.dat`` / ``.hea`` extension.

        Returns:
            One :class:`TimeSeries` per lead, sharing the ECG spec and a per-recording source id.
        """
        header = self._read_header(record_base)
        axis = RegularAxis.from_rate_hz(Fraction(str(header.fs)))
        return tuple(
            TimeSeries(
                spec=_ECG,
                channel=name,
                time_axis=axis,
                loader=self._lead_loader(record_base, header, lead_idx),
                source_id=f"ptbxl-{ecg_id}",
                time_series_id=f"ecg-{ecg_id}-{name}",
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
    def _iter_tasks(source: EcgQaCotSource, answers: dict[int, tuple[str, ...]]) -> Iterator[AnswerTask]:
        """Yield one :class:`AnswerTask` per CoT row, referencing its recording and metadata annotations.

        Args:
            source: The download handle naming the CoT CSVs.
            answers: Per-template answer options (decides whether a task references an options annotation).

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
                    prompt=str(row["question"]),
                    target=str(row["answer"]),
                    rationale=str(row["rationale"]),
                    input_annotation_ids=tuple(input_ids),
                    id=f"ecgqa-{split}-{index}",
                    sample_ids=(f"ptbxl-{ecg_id}",),
                )


CONNECTOR = EcgQaCotConnector
