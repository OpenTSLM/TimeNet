"""Read HEARTS cases and describe their reusable record identities."""

from collections.abc import Iterator
from dataclasses import dataclass
import functools
import operator
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

from timenet.errors import TimeFFormatError
from timenet_connectors.datasets.yang_ai_lab._hearts.pickles import load_payload
from timenet_connectors.datasets.yang_ai_lab._hearts.release import (
    TASKS,
    TIME_COLUMNS,
    US_PER_MINUTE,
    TaskDef,
    Waveform,
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


class RecordRef(NamedTuple):
    """Identity and source-clock origin of a reusable case record."""

    record_id: str
    origin_us: int


class CaseRecordRefs(NamedTuple):
    """Input and candidate record references for one case."""

    inputs: tuple[RecordRef, ...]
    candidates: tuple[RecordRef, ...]


def iter_cases(root: Path, corpus: str | None = None) -> Iterator[Case]:
    """Yield loaded cases in task-directory and numeric-file order.

    Args:
        root: Root of the pinned HEARTS snapshot.
        corpus: Optional corpus name. When omitted, yield all available cases.

    Yields:
        Each matching case.
    """
    for directory, definition in TASKS.items():
        if corpus is not None and directory.partition("/")[0] != corpus:
            continue
        folder = root / directory
        if not folder.is_dir():
            continue
        for path in sorted(folder.glob("*.pkl"), key=lambda item: int(item.stem)):
            yield Case(directory, definition, int(path.stem), path, load_payload(str(path)))


def case_record_refs(case: Case) -> CaseRecordRefs:
    """Describe a case's parent records without constructing their signals.

    Returns:
        Input and candidate references in task-definition order.
    """
    return CaseRecordRefs(
        tuple(_input_record_refs(case, case.definition.inputs)),
        tuple(_input_record_refs(case, case.definition.candidates)),
    )


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


def _input_record_refs(case: Case, entries: tuple[str | tuple[str, ...] | Waveform, ...]) -> list[RecordRef]:
    refs: list[RecordRef] = []
    for entry in entries:
        if isinstance(entry, Waveform):
            refs.append(RecordRef(case.id, 0))
        elif isinstance(entry, tuple):
            refs.append(RecordRef(case.id, min(_frame_origin_us(case.payload[key]) for key in entry)))
        elif isinstance(case.payload[entry], dict):
            refs.extend(
                RecordRef(f"{case.id}-{name}", _frame_origin_us(frame))
                for name, frame in sorted(case.payload[entry].items())
            )
        else:
            name = entry if len(entries) > 1 else None
            record_id = case.id if name is None else f"{case.id}-{name}"
            refs.append(RecordRef(record_id, _frame_origin_us(case.payload[entry])))
    return refs


def _frame_origin_us(frame: Any) -> int:
    time_columns = [column for column in frame.columns if column in TIME_COLUMNS]
    if len(time_columns) != 1:
        raise TimeFFormatError(f"HEARTS frame with columns {list(frame.columns)} needs one time column")
    return int(frame_times_us(frame, time_columns[0])[0][0])
