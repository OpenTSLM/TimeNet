"""The release as published: its Hub pin, the files the authors' code loads, and their pinned row counts.

The authors' code is behavioral-data/TSandLanguage at d4e748cb. Each config under
``configs/tasks/llms_and_ts/`` loads one kind of file through ``MultimodalMCQTask``:
``ts2desc_mcq.yaml`` the description files, ``qa_mcq.yaml`` the single-series questions, and
``qa_counterfactual.yaml`` the two-series questions.

The repository also holds ``TS_Dataset_MCQ/all.json``, which no config loads and whose 924 extra rows
are in no split, and the ``*_llava.json`` files, which pose the same description questions to the
authors' image model, with another prompt and the options in one shuffled order. The connector reads
neither.
"""

from enum import StrEnum, unique
from pathlib import Path

from timenet.errors import TimeFValidationError
from timenet.types import Split


HF_REPO = "mikeam/time-series-reasoning"
SOURCE_REVISION = "d6a1c7d48d8a454b9db0b2137383fc30847e6ec0"


@unique
class SourceFile(StrEnum):
    """A file the connector reads, named by its path in the Hub repository.

    All six are JSON Lines, ``TS_Dataset_MCQ/*.json`` included.
    """

    SERIES = "TS_Dataset.jsonl"
    DESCRIPTION_TRAIN = "TS_Dataset_MCQ/train.json"
    DESCRIPTION_VAL = "TS_Dataset_MCQ/val.json"
    DESCRIPTION_TEST = "TS_Dataset_MCQ/test.json"
    QUESTIONS = "MCQ_1_TS.jsonl"
    COUNTERFACTUALS = "MCQ_2_TS.jsonl"

    @property
    def part(self) -> str:
        """Return the file's part name, the key the Hub base downloads it under."""
        return self.name.lower()

    @property
    def timenet_split(self) -> Split | None:
        """Return the TimeNet split of the file's rows, or ``None``: only the description files have one."""
        return _SPLITS.get(self)


# The authors name the description splits by file. The question files hold no split: the configs read
# train/val/test files for them that the release doesn't include.
_SPLITS: dict[SourceFile, Split] = {
    SourceFile.DESCRIPTION_TRAIN: Split.TRAIN,
    SourceFile.DESCRIPTION_VAL: Split.VALIDATION,
    SourceFile.DESCRIPTION_TEST: Split.TEST,
}

# Rows per file at SOURCE_REVISION. The paper reports 8.7k series, about 100k single-series and over
# 130k two-series questions.
PINNED_ROWS: dict[SourceFile, int] = {
    SourceFile.SERIES: 8787,
    SourceFile.DESCRIPTION_TRAIN: 5779,
    SourceFile.DESCRIPTION_VAL: 1000,
    SourceFile.DESCRIPTION_TEST: 1036,
    SourceFile.QUESTIONS: 97330,
    SourceFile.COUNTERFACTUALS: 132689,
}


# The lines whose question or an option is blank, kept as published. The authors' loader formats a
# blank option as "C) ". MCQ_2_TS.jsonl line 45584 has a blank question and four blank options.
PINNED_BLANK_LINES: dict[SourceFile, frozenset[int]] = {
    SourceFile.QUESTIONS: frozenset({71926, 71927}),
    SourceFile.COUNTERFACTUALS: frozenset({45584, 71367, 71368, 71369, 71370, 71371, 71372, 118975, 128859, 128967}),
}


def check_row_count(file: SourceFile, path: Path, rows: int) -> None:
    """Compare a file's row count with the pinned release's.

    Raises:
        TimeFValidationError: If the count differs, as when the file isn't the pinned revision's.
    """
    if rows != PINNED_ROWS[file]:
        raise TimeFValidationError(f"{path}: expected {PINNED_ROWS[file]} rows of {file}, found {rows}")


def check_blank_lines(file: SourceFile, path: Path, lines: frozenset[int]) -> None:
    """Compare the lines with a blank question or option with the pinned release's.

    Raises:
        TimeFValidationError: If they differ, so a blank text outside the pinned lines isn't kept silently.
    """
    expected = PINNED_BLANK_LINES.get(file, frozenset())
    if lines != expected:
        raise TimeFValidationError(
            f"{path}: expected blank texts on lines {sorted(expected)} of {file}, found {sorted(lines)}"
        )
