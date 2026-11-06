"""Read the ECG-QA CoT sources: the per-template answer options and the chain-of-thought CSVs."""

import ast
from collections.abc import Iterator
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Annotated

from pydantic import Field

from timenet.errors import TimeNetConnectorError
from timenet.types import (
    Split,
)
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
class _CotRow:
    """One CoT question, parsed once from its CSV row."""

    ecg_id: int
    template_id: int
    question_type: str
    question: str
    answer: str
    rationale: str


WholeNumber = Annotated[Decimal, Field(strict=False, allow_inf_nan=False, max_digits=4300, decimal_places=0)]


class _TemplateRow(SourceRow):
    """Published answer choices for one exact whole-number template ID."""

    template_id: WholeNumber
    classes: NonBlankText
    question_type: str = ""


class _TemplateClasses(SourceRow):
    """Decoded answer options, preserving the published strings and their order."""

    classes: list[str]


class _CotCsvRow(SourceRow):
    """Published question columns, with verbatim optional clinical context."""

    ecg_id: NonBlankText
    template_id: WholeNumber
    question_type: NonBlankText
    question: NonBlankText
    answer: NonBlankText
    rationale: str
    clinical_context: str = ""


def load_template_answers(path: Path) -> dict[int, tuple[str, ...]]:
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


def iter_cot_rows(csv_path: Path) -> Iterator[_CotRow]:
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


def scan_cot_rows(source: EcgQaCotSource) -> dict[int, Split]:
    """Read every CoT row once, and keep the split of each recording that the records need.

    Args:
        source: The download handle naming the CoT CSVs.

    Returns:
        The first split of each recording, keyed by PTB-XL ECG ID.
    """
    split_of_ecg: dict[int, Split] = {}
    for split, csv_path in source.cot_csvs:
        for row in iter_cot_rows(csv_path):
            split_of_ecg.setdefault(row.ecg_id, split)
    return split_of_ecg
