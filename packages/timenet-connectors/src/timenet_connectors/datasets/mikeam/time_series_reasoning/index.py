"""Read the six files once, check that they agree, and index what the Records are built from.

Every row repeats its series' texts, metadata and values. The index checks each copy against the
series' row in ``TS_Dataset.jsonl``, so a question can use the series' Record. The description files
are the exception for values: they hold rounded copies of some series, so only their texts are
checked, and a copy that differs gets a Record of its own. The README gives the counts.
"""

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Self

from timenet.errors import TimeFValidationError
from timenet_connectors.datasets.mikeam.time_series_reasoning.release import (
    SourceFile,
    check_blank_lines,
    check_row_count,
)
from timenet_connectors.datasets.mikeam.time_series_reasoning.rows import (
    CounterfactualRow,
    DescriptionRow,
    QuestionRow,
    R,
    Row,
    read_rows,
)


# The texts every row repeats for its series. The metadata is compared with its key order.
_SHARED_TEXTS = ("description", "description_short", "description_tiny", "characteristics", "generator")
_DESCRIPTION_FILES = (SourceFile.DESCRIPTION_TRAIN, SourceFile.DESCRIPTION_VAL, SourceFile.DESCRIPTION_TEST)


@dataclass(frozen=True)
class ReleaseIndex:
    """The release's series, description rows, and changed series, checked against each other."""

    series: Mapping[str, Row]
    """The rows of ``TS_Dataset.jsonl`` by uuid, in file order."""
    descriptions: tuple[DescriptionRow, ...]
    """The rows of the train, val and test description files, in that order."""
    changed: Mapping[tuple[str, str], CounterfactualRow]
    """The first row of each counterfactual, by uuid and counterfactual, in file order. Every row of
    one counterfactual holds the same changed series, which its Record rereads from this row."""

    @classmethod
    def from_files(cls, paths: Mapping[SourceFile, Path], *, pinned: bool = True) -> Self:
        """Read and check each file once, in :class:`SourceFile` order.

        The question rows are checked, not kept: the tasks reread them.

        Args:
            paths: Each source file's local path.
            pinned: Whether the files are the pinned release, whose row counts and blank lines are
                checked. Test fixtures aren't.

        Returns:
            The index.

        Raises:
            TimeFValidationError: If a file is missing, a row is malformed, a row's copy of its series
                differs from ``TS_Dataset.jsonl``, a changed series differs between the rows of one
                counterfactual, or a pinned count differs.
        """
        missing = sorted(set(SourceFile) - set(paths))
        if missing:
            raise TimeFValidationError(f"the release needs each of {[file.value for file in missing]}")
        series: dict[str, Row] = {}
        for row in _read(paths, SourceFile.SERIES, Row, pinned=pinned):
            if row.uuid in series:
                raise TimeFValidationError(
                    f"{row.ref.context}: uuid {row.uuid} repeats line {series[row.uuid].ref.line}"
                )
            series[row.uuid] = row
        descriptions: list[DescriptionRow] = []
        for file in _DESCRIPTION_FILES:
            for row in _read(paths, file, DescriptionRow, pinned=pinned):
                _check_copy(row, series, values=False)
                descriptions.append(row)
        for row in _read(paths, SourceFile.QUESTIONS, QuestionRow, pinned=pinned):
            _check_copy(row, series, values=True)
        changed: dict[tuple[str, str], CounterfactualRow] = {}
        for row in _read(paths, SourceFile.COUNTERFACTUALS, CounterfactualRow, pinned=pinned):
            _check_copy(row, series, values=True)
            _add_changed(changed, row)
        return cls(series, tuple(descriptions), changed)


def _read(paths: Mapping[SourceFile, Path], file: SourceFile, row_type: type[R], *, pinned: bool) -> Iterator[R]:
    """Read one file's rows as ``row_type``, then check its row count and blank lines against the pins.

    Yields:
        Each row, in file order.
    """
    path = paths[file]
    rows = 0
    blank: set[int] = set()
    for row in read_rows(path, file, row_type):
        rows += 1
        if isinstance(row, QuestionRow) and row.has_blank_text:
            blank.add(row.ref.line)
        yield row
    if pinned:
        check_row_count(file, path, rows)
        check_blank_lines(file, path, frozenset(blank))


def _check_copy(row: Row, series: Mapping[str, Row], *, values: bool) -> None:
    """Require a row's copy of its series to equal the series' row: texts, metadata, and values if asked.

    Raises:
        TimeFValidationError: If the uuid is unknown, or a copied field differs.
    """
    main = series.get(row.uuid)
    if main is None:
        raise TimeFValidationError(f"{row.ref.context}: uuid {row.uuid} is not in {SourceFile.SERIES}")
    differs = [field for field in _SHARED_TEXTS if getattr(row, field) != getattr(main, field)]
    # Dict equality ignores key order, and the authors' prompt prints the metadata in key order.
    if list(row.metadata.items()) != list(main.metadata.items()):
        differs.append("metadata")
    if values and row.series_shape != main.series_shape:
        differs.append("series")
    if differs:
        raise TimeFValidationError(
            f"{row.ref.context}: {differs} differ from the series' row, {SourceFile.SERIES} line {main.ref.line}"
        )


def _add_changed(changed: dict[tuple[str, str], CounterfactualRow], row: CounterfactualRow) -> None:
    """Keep the first row of a counterfactual, or check a later row's changed series against it.

    Raises:
        TimeFValidationError: If the rows of one counterfactual hold different changed series.
    """
    key = (row.uuid, row.counterfactual)
    known = changed.get(key)
    if known is None:
        changed[key] = row
    elif (known.new_generator, known.new_series_shape) != (row.new_generator, row.new_series_shape):
        raise TimeFValidationError(
            f"{row.ref.context}: the changed series differs from line {known.ref.line}'s for the same counterfactual"
        )
