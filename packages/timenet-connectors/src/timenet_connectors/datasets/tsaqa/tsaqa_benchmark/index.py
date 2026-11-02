"""Read the split files once, check them against the pinned release, and find each row's source issues.

A row's issues depend on the other rows: copies of one question about one series, by
:attr:`Row.question_digest`, may repeat in one split, cross into another, or expect another answer.
So the index reads every file before any task is built.
"""

from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Self

from timenet.errors import TimeFValidationError
from timenet_connectors.datasets.tsaqa.tsaqa_benchmark.issues import SourceIssue
from timenet_connectors.datasets.tsaqa.tsaqa_benchmark.release import InventoryKey, SourceSplit, check_inventory
from timenet_connectors.datasets.tsaqa.tsaqa_benchmark.rows import Row, RowRef, read_rows


@dataclass(frozen=True)
class ReleaseIndex:
    """The rows of the split files read, each validated, and the source issues of the rows that have any."""

    paths: Mapping[SourceSplit, Path]
    """Each read file's local path, which the Records reread their series from."""
    rows: tuple[Row, ...]
    """Every row, file by file in :class:`SourceSplit` order, each file in row order."""
    issues: Mapping[RowRef, frozenset[SourceIssue]]
    """The issues of each row that has any."""

    @classmethod
    def from_files(
        cls,
        paths: Mapping[SourceSplit, Path],
        inventory: Mapping[SourceSplit, Mapping[InventoryKey, int]] | None,
    ) -> Self:
        """Read and check each file once, in :class:`SourceSplit` order.

        Args:
            paths: The local path of each split file to read, at least one.
            inventory: Each file's pinned row counts, or ``None`` to skip the check, as test fixtures do.

        Returns:
            The index.

        Raises:
            TimeFValidationError: If no file is given, a row is malformed, or a file's rows differ from
                its pinned inventory.
        """
        if not paths:
            raise TimeFValidationError("TSAQA needs at least one of its split files")
        rows: list[Row] = []
        for split in SourceSplit:
            if split not in paths:
                continue
            counts: Counter[InventoryKey] = Counter()
            for row in read_rows(paths[split], split):
                counts[row.task, row.question_type, row.format] += 1
                rows.append(row)
            check_inventory(paths[split], counts, None if inventory is None else inventory[split])
        return cls(dict(paths), tuple(rows), _issues(rows))

    def issues_of(self, row: Row) -> frozenset[SourceIssue]:
        """Return a row's source issues, none when it has none."""
        return self.issues.get(row.ref, frozenset())


def _issues(rows: list[Row]) -> dict[RowRef, frozenset[SourceIssue]]:
    """Find each row's source issues: a mislabelled question type, and copies of its question elsewhere.

    Returns:
        The issues of each row that has any.
    """
    copies: defaultdict[bytes, list[Row]] = defaultdict(list)
    for row in rows:
        copies[row.question_digest].append(row)
    issues: dict[RowRef, frozenset[SourceIssue]] = {}
    for row in rows:
        found: set[SourceIssue] = set()
        if row.question_type is not row.format.question_type:
            found.add(SourceIssue.QUESTION_TYPE_MISMATCH)
        for other in copies[row.question_digest]:
            if other is row:
                continue
            found.add(SourceIssue.REPEATED_QUESTION if other.answer == row.answer else SourceIssue.CONFLICTING_ANSWER)
            if other.ref.split is not row.ref.split:
                found.add(SourceIssue.REPEATED_ACROSS_SPLITS)
        if found:
            issues[row.ref] = frozenset(found)
    return issues
