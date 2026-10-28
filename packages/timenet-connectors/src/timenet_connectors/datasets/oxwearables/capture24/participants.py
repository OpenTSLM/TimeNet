"""Read CAPTURE-24's participants from ``metadata.csv``: one row per participant, paired with their recording.

Each row is validated by :class:`Participant`. :func:`read_participants` checks the rules across rows.
"""

import csv
from pathlib import Path
from typing import Annotated, Self

from pydantic import BaseModel, ConfigDict, StrictStr, StringConstraints, ValidationError

from timenet.errors import TimeFFormatError
from timenet.types import Split
from timenet_connectors.datasets.oxwearables.capture24.benchmark import split_of
from timenet_connectors.datasets.oxwearables.capture24.release import METADATA_FILE, PARTICIPANT_FILE
from timenet_connectors.datasets.oxwearables.capture24.vocabulary import AgeBand, Sex


_METADATA_COLUMNS = ["pid", "age", "sex"]


class Participant(BaseModel):
    """One participant: the row ``metadata.csv`` states for them, and their recording file."""

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    pid: Annotated[StrictStr, StringConstraints(pattern=r"^P[0-9]{3}$")]
    """The participant id, such as ``P001``, which also names the file."""
    age: AgeBand
    sex: Sex
    recording_path: Path

    @classmethod
    def parse(cls, row: dict[str, str], recording_path: Path, context: str) -> Self:
        """Validate one ``metadata.csv`` row, with the participant's recording.

        Args:
            row: The row's fields, by column.
            recording_path: The participant's recording.
            context: The file and line, for the error.

        Returns:
            The participant.

        Raises:
            TimeFFormatError: If the pid isn't ``P`` and three digits, or the age or sex isn't one the release
                writes.
        """
        try:
            return cls.model_validate_strings({**row, "recording_path": str(recording_path)})
        except ValidationError as exc:
            raise TimeFFormatError(f"{context}: {exc}") from exc

    @property
    def number(self) -> int:
        """Return the participant's number, such as 1 for ``P001``."""
        return int(self.pid.removeprefix("P"))

    @property
    def split(self) -> Split:
        """Return the participant's benchmark split, as ``benchmark.py`` runs it."""
        return split_of(self.number)


def read_participants(root: Path) -> tuple[Participant, ...]:
    """Read ``metadata.csv`` and pair each row with its recording.

    Each row is validated by :class:`Participant`. The rules across rows are checked here.

    Returns:
        One participant per recording, in id order.

    Raises:
        TimeFFormatError: If the table's header isn't ``pid,age,sex``, a row doesn't have three fields or
            fails :meth:`Participant.parse`, a pid repeats, or the table and the recordings don't name the
            same participants.
    """
    metadata_path = root / METADATA_FILE
    with metadata_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    if not rows or rows[0] != _METADATA_COLUMNS:
        raise TimeFFormatError(f"{metadata_path}: expected the header {','.join(_METADATA_COLUMNS)}")
    rows_by_pid: dict[str, tuple[int, dict[str, str]]] = {}
    for line, row in enumerate(rows[1:], 2):
        if len(row) != len(_METADATA_COLUMNS):
            raise TimeFFormatError(f"{metadata_path}:{line}: expected {len(_METADATA_COLUMNS)} fields, got {len(row)}")
        if row[0] in rows_by_pid:
            raise TimeFFormatError(f"{metadata_path}:{line}: {row[0]} is listed twice")
        rows_by_pid[row[0]] = (line, dict(zip(_METADATA_COLUMNS, row, strict=True)))
    recordings_by_pid = {
        file.name.removesuffix(".csv.gz"): file for file in root.iterdir() if PARTICIPANT_FILE.fullmatch(file.name)
    }
    if set(recordings_by_pid) != set(rows_by_pid):
        missing = sorted(set(rows_by_pid) - set(recordings_by_pid))
        extra = sorted(set(recordings_by_pid) - set(rows_by_pid))
        raise TimeFFormatError(f"{root}: recordings missing for {missing}, unlisted recordings {extra}")
    return tuple(
        Participant.parse(row, recordings_by_pid[pid], f"{metadata_path}:{line}")
        for pid, (line, row) in sorted(rows_by_pid.items())
    )
