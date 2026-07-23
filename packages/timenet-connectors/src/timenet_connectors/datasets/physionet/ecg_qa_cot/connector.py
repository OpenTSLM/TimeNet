"""The ECG-QA CoT connector: PTB-XL 12-lead ECGs with chain-of-thought question answering.

Each sample is one PTB-XL recording (12 leads at 500 Hz, in millivolts) paired with a clinical
question, a short ground-truth answer, and a chain-of-thought rationale. The rationale is the reasoning
training target and the short answer is the evaluation label, so each sample carries a
:class:`~timenet.types.ReasoningTask` whose ``answer`` is the label and whose ``rationale`` is the CoT.

Sources: signals from PhysioNet PTB-XL; the per-template answer options from the ``Jwoo5/ecg-qa``
GitHub repo; the precomputed CoT rows (question / answer / rationale / template) from the OpenTSLM
release (the only public source for the rationales). Real curation needs the network and a multi-GB
PTB-XL download.
"""

import ast
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

from timenet.dataset import TimeFDataset, TimeSeries
from timenet.types import (
    DataSource,
    ReasoningTask,
    StaticAnnotation,
    TimeSeriesSpec,
    View,
    ureg,
)
from timenet_connectors.bases.physionet import BasePhysioNetConnector


# PTB-XL 500 Hz records from PhysioNet's open S3 bucket; ``_hr`` = high-rate (500 Hz) recordings.
# Fetched via boto3 (unsigned for this public bucket, or signed if AWS creds are in the environment).
PTBXL_ZIP_URL = "s3://physionet-open/ptb-xl/ptb-xl-1.0.3.zip"
# The per-template answer options (the multiple-choice candidates), keyed by template_id.
ECG_QA_TEMPLATE_ANSWERS_URL = (
    "https://raw.githubusercontent.com/Jwoo5/ecg-qa/master/ecgqa/ptbxl/answers_for_each_template.csv"
)
# Precomputed CoT rows (question / answer / rationale / template). OpenTSLM's release is the only
# public source; kept as a swappable constant so a mirror can replace it.
ECG_QA_COT_URL = "https://polybox.ethz.ch/index.php/s/D5QaJSEw4dXkzXm/download/ecg_qa_cot_final.zip"

_SOURCE = DataSource(data_source_type="physionet", name="PTB-XL", provider="PhysioNet")
_ECG = TimeSeriesSpec(
    spec_type="ecg",
    name="12-lead ECG",
    unit_sampling_rate=ureg.hertz,
    unit_timestamp=ureg.second,
    unit_value=ureg.millivolt,
    data_source=_SOURCE,
)


@dataclass(frozen=True)
class EcgQaCotRef:
    """A lightweight reference to one CoT sample: its text fields plus the ECG record path."""

    split: str
    index: int
    ecg_id: int
    record_base: Path
    question: str
    answer: str
    rationale: str
    template_id: int
    question_type: str
    clinical_context: str
    answer_options: tuple[str, ...]


def _parse_ecg_id(raw: object) -> int:
    """Parse a PTB-XL ecg_id that may arrive as ``123`` or ``"[123]"``.

    Args:
        raw: The raw ecg_id value from a CoT row.

    Returns:
        The integer ecg_id.
    """
    return int(str(raw).strip().strip("[]").strip())


def _load_template_answers(path: Path) -> dict[int, tuple[str, ...]]:
    """Load per-template answer options from ``answers_for_each_template.csv``.

    Args:
        path: Path to the CSV whose ``classes`` column is a Python list literal.

    Returns:
        A mapping of ``template_id`` to its ordered answer options.
    """
    answers: dict[int, tuple[str, ...]] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            classes = ast.literal_eval(row["classes"])
            answers[int(float(row["template_id"]))] = tuple(str(option) for option in classes)
    return answers


def _build_refs(
    rows: list[dict[str, object]],
    records_dir: Path,
    answers: dict[int, tuple[str, ...]],
    split: str,
    start_index: int,
    flat_records: bool,
) -> list[EcgQaCotRef]:
    """Turn parsed CoT rows into :class:`EcgQaCotRef`s, resolving each row's ECG record path.

    Args:
        rows: Parsed CoT rows (dicts with the CoT fields).
        records_dir: The directory holding ``records500`` (real) or the flat fixture records.
        answers: Per-template answer options.
        split: The split these rows belong to (``train`` / ``validation`` / ``test``).
        start_index: The running sample index to continue from (kept unique across splits).
        flat_records: Whether records sit flat in ``records_dir`` (fixture) or under the PTB-XL
            ``records500/<bucket>/`` layout.

    Returns:
        One reference per row.
    """
    refs: list[EcgQaCotRef] = []
    for offset, row in enumerate(rows):
        ecg_id = _parse_ecg_id(row["ecg_id"])
        stem = f"{ecg_id:05d}_hr"
        record_base = records_dir / stem if flat_records else records_dir / f"{ecg_id // 1000 * 1000:05d}" / stem
        template_id = int(float(str(row["template_id"])))
        refs.append(
            EcgQaCotRef(
                split=split,
                index=start_index + offset,
                ecg_id=ecg_id,
                record_base=record_base,
                question=str(row["question"]),
                answer=str(row["answer"]),
                rationale=str(row["rationale"]),
                template_id=template_id,
                question_type=str(row["question_type"]),
                clinical_context=str(row.get("clinical_context") or "12-lead ECG recording."),
                answer_options=answers.get(template_id, ()),
            )
        )
    return refs


def _find_dir_containing(root: Path, relative: str) -> Path:
    """Find the directory under ``root`` that contains ``relative`` (archives extract nested).

    Args:
        root: The extraction root to search.
        relative: A file name expected inside the wanted directory.

    Returns:
        The parent directory of the first match.

    Raises:
        FileNotFoundError: If no match is found.
    """
    for match in root.rglob(relative):
        return match.parent
    raise FileNotFoundError(f"{relative!r} not found under {root}")


class EcgQaCotConnector(BasePhysioNetConnector[EcgQaCotRef]):
    """Connector for the ECG-QA CoT dataset (PTB-XL signals + OpenTSLM chain-of-thought QA)."""

    _COT_CSVS: ClassVar[tuple[tuple[str, str], ...]] = (
        ("train", "ecg_qa_cot_train.csv"),
        ("validation", "ecg_qa_cot_val.csv"),
        ("test", "ecg_qa_cot_test.csv"),
    )

    def download(self, cache_dir: Path) -> list[EcgQaCotRef]:
        """Fetch PTB-XL, the template answers, and the CoT CSVs, and resolve references.

        Args:
            cache_dir: Directory downloaded archives are cached under.

        Returns:
            One reference per CoT row across all splits.
        """
        ptbxl_root = _find_dir_containing(
            self._ensure_archive(PTBXL_ZIP_URL, cache_dir, sentinel="ptbxl-extracted"), "ptbxl_database.csv"
        )
        answers_path = cache_dir / "answers_for_each_template.csv"
        if not answers_path.exists():
            self._stream_download(ECG_QA_TEMPLATE_ANSWERS_URL, answers_path)
        answers = _load_template_answers(answers_path)
        cot_root = self._ensure_archive(ECG_QA_COT_URL, cache_dir, sentinel="ecg-qa-cot-extracted")

        refs: list[EcgQaCotRef] = []
        for split, csv_name in self._COT_CSVS:
            csv_path = _find_dir_containing(cot_root, csv_name) / csv_name
            with csv_path.open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            refs.extend(_build_refs(rows, ptbxl_root / "records500", answers, split, len(refs), flat_records=False))
        return refs

    def convert(self, raw_refs: list[EcgQaCotRef]) -> TimeFDataset:
        """Build one sample per CoT row, sharing the 12-lead ECG across rows on the same recording.

        Args:
            raw_refs: The references from :meth:`download`.

        Returns:
            The populated :class:`~timenet.dataset.TimeFDataset`.
        """
        dataset = TimeFDataset(metadata=self.metadata())
        leads_by_ecg: dict[int, tuple[TimeSeries, ...]] = {}
        for ref in raw_refs:
            leads = leads_by_ecg.get(ref.ecg_id)
            if leads is None:
                leads = self._leads_for(ref)
                leads_by_ecg[ref.ecg_id] = leads
            sample = dataset.add_sample(time_series=leads, view=View.FULL, sample_id=f"ecgqa-{ref.split}-{ref.index}")
            sample.add_annotation(StaticAnnotation(key="split", value=ref.split, id=f"split-{ref.index}"))
            sample.add_annotation(
                StaticAnnotation(key="question_type", value=ref.question_type, id=f"qtype-{ref.index}")
            )
            sample.add_annotation(
                StaticAnnotation(key="template_id", value=ref.template_id, id=f"template-{ref.index}")
            )
            sample.add_annotation(
                StaticAnnotation(key="clinical_context", value=ref.clinical_context, id=f"context-{ref.index}")
            )
            if ref.answer_options:
                sample.add_annotation(
                    StaticAnnotation(key="answer_options", value=list(ref.answer_options), id=f"options-{ref.index}")
                )
            dataset.add_task(
                sample,
                ReasoningTask(
                    question=ref.question, rationale=ref.rationale, answer=ref.answer, id=f"reason-{ref.index}"
                ),
            )
        return dataset

    def _leads_for(self, ref: EcgQaCotRef) -> tuple[TimeSeries, ...]:
        """Build the 12 lead :class:`TimeSeries` for one recording with lazy per-lead loaders.

        Args:
            ref: The reference whose record supplies the leads.

        Returns:
            One :class:`TimeSeries` per lead, sharing the ECG spec and a stable per-recording id.
        """
        header = self._read_header(ref.record_base)
        sampling_rate_hz = float(header.fs)
        t_end_s = int(header.sig_len) / sampling_rate_hz
        return tuple(
            TimeSeries(
                spec=_ECG,
                channel=name,
                sampling_rate_hz=sampling_rate_hz,
                loader=self._lead_loader(ref.record_base, lead_idx),
                source_id=f"ptbxl-{ref.ecg_id}",
                time_series_id=f"ecg-{ref.ecg_id}-{name}",
                t_start_s=0.0,
                t_end_s=t_end_s,
            )
            for lead_idx, name in enumerate(header.sig_name)
        )


CONNECTOR = EcgQaCotConnector
