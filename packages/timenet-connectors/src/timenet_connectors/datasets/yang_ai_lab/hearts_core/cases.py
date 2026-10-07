"""Read HEARTS cases and describe their reusable record identities."""

from collections.abc import Iterator, Mapping
import functools
import operator
from pathlib import Path
from typing import Any

import numpy as np
from pydantic.dataclasses import dataclass

from timenet.errors import TimeFFormatError
from timenet_connectors.datasets.yang_ai_lab.hearts_core.pickles import load_payload
from timenet_connectors.datasets.yang_ai_lab.hearts_core.release import (
    US_PER_MINUTE,
    TaskDef,
)


@dataclass(frozen=True)
class Case:
    """One pinned test case and its task definition."""

    directory: str
    definition: TaskDef
    index: int
    path: Path
    payload: dict[str, Any]

    @property
    def corpus(self) -> str:
        """Return the top-level source corpus name."""
        return self.directory.partition("/")[0]

    @property
    def task(self) -> str:
        """Return the task directory within the corpus."""
        return self.directory.partition("/")[2]

    @property
    def id(self) -> str:
        """Return the stable prefix used by this case's records and task."""
        return f"hearts-{self.corpus}-{self.task}-{self.index:02d}"

    def at(self, keys: tuple[str, ...]) -> Any:
        """Return the payload value at a nested key path."""
        return functools.reduce(operator.getitem, keys, self.payload)


def iter_cases(root: Path, tasks: Mapping[str, TaskDef]) -> Iterator[Case]:
    """Yield loaded cases in task-directory and numeric-file order.

    Args:
        root: Root of the pinned HEARTS snapshot.
        tasks: The connector's task definitions.

    Yields:
        Each matching case.

    Raises:
        TimeFFormatError: If an expected task directory is missing or empty.
    """
    for directory, definition in tasks.items():
        folder = root / directory
        if not folder.is_dir():
            raise TimeFFormatError(f"HEARTS task directory {folder} is missing")
        paths = sorted(folder.glob("*.pkl"), key=lambda item: int(item.stem))
        if not paths:
            raise TimeFFormatError(f"HEARTS task directory {folder} has no cases")
        for path in paths:
            yield Case(directory, definition, int(path.stem), path, load_payload(str(path)))


def frame_times_us(frame: Any, column: str) -> tuple[np.ndarray, bool]:
    """Read a frame's time column as whole microseconds.

    Returns:
        One microsecond value per row, and whether the values are naive wall-clock datetimes.

    Raises:
        TimeFFormatError: If the time column is empty or contains a missing time.
    """
    values = frame[column].to_numpy()
    if len(values) == 0:
        raise TimeFFormatError(f"HEARTS frame time column {column!r} has no values")
    if values.dtype.kind == "O":
        values = values.astype("datetime64[us]")
    if values.dtype.kind in "Mm":
        if bool(np.isnat(values).any()):
            raise TimeFFormatError(f"HEARTS frame time column {column!r} has missing values")
        unit = "datetime64[us]" if values.dtype.kind == "M" else "timedelta64[us]"
        return values.astype(unit).astype(np.int64), values.dtype.kind == "M"
    return np.rint(values * US_PER_MINUTE).astype(np.int64), False


def moment_us(value: Any) -> int:
    """Read a naive source timestamp as whole microseconds on its source calendar.

    Returns:
        The timestamp in microseconds.
    """
    return int(np.datetime64(str(value), "us").astype(np.int64))
