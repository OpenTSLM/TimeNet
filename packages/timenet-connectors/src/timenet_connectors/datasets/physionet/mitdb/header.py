"""Read an MIT-BIH record's header: its signals, and the subject facts its comment lines state.

A header such as ``100.hea`` reads::

    100 2 360 650000
    100.dat 212 200 11 1024 995 -22131 0 MLII
    100.dat 212 200 11 1024 1011 20052 0 V5
    # 69 M 1085 1629 x1
    # Aldomet, Inderal

The comment lines are read as the authors' ``dbnotes.c`` reads them: the first starts with the age and the
sex, and the second is the medications. An age of -1 means the age was not recorded. The other fields of the
first line are not documented, and later lines are the authors' notes on the record; both are kept, as
written, in the Record's metadata.
"""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from timenet.errors import TimeFFormatError
from timenet_connectors.bases.physionet import WfdbHeader
from timenet_connectors.datasets.physionet.mitdb.release import SAMPLE_RATE_HZ, SIGNALS_PER_RECORD
from timenet_connectors.datasets.physionet.mitdb.vocabulary import Sex


AGE_NOT_RECORDED = -1
"""The age the first comment line writes where the age was not recorded, in records 103 and 219."""
_GAIN_FIELD = 2


@dataclass(frozen=True)
class RecordHeader:
    """One record's header: its signals, as WFDB reads them, and its comment lines, parsed."""

    name: str
    """The record name, such as ``100``."""
    base: Path
    """The record's path without an extension."""
    wfdb: WfdbHeader
    age: int | None
    """The age in years, or ``None`` where the header writes -1."""
    sex: Sex
    medications: str
    """The second comment line, as written. ``None`` is written where the subject took none."""
    comments: tuple[str, ...]
    """Every comment line, as written without its ``#``."""


def read_header(base: Path, read_wfdb_header: Callable[[Path], WfdbHeader]) -> RecordHeader:
    """Read and check one record's header.

    Args:
        base: The record's path without an extension.
        read_wfdb_header: Parses the header's record and signal lines.

    Returns:
        The header.

    Raises:
        TimeFFormatError: If the record isn't two signals at 360 Hz in WFDB's default unit, or a comment
            line the authors' ``dbnotes.c`` reads is missing or doesn't read as it does.
    """
    path = base.with_suffix(".hea")
    wfdb = read_wfdb_header(base)
    if wfdb.fs != SAMPLE_RATE_HZ or len(wfdb.sig_name) != SIGNALS_PER_RECORD:
        raise TimeFFormatError(
            f"{path}: expected {SIGNALS_PER_RECORD} signals at {SAMPLE_RATE_HZ} Hz, "
            f"got {len(wfdb.sig_name)} at {wfdb.fs} Hz"
        )
    lines = path.read_text(encoding="utf-8").splitlines()
    # A gain field with no "/<units>" is in WFDB's default unit, millivolts.
    for line in lines[1 : 1 + SIGNALS_PER_RECORD]:
        if "/" in line.split()[_GAIN_FIELD]:
            raise TimeFFormatError(f"{path}: expected signals in WFDB's default unit, got {line!r}")
    comments = tuple(line.removeprefix("#").strip() for line in lines if line.startswith("#"))
    if len(comments) < 2:  # noqa: PLR2004 - the age-and-sex line, then the medications line
        raise TimeFFormatError(f"{path}: expected an age-and-sex comment line and a medications line")
    fields = comments[0].split()
    try:
        age = int(fields[0])
        sex = Sex(fields[1])
    except (IndexError, ValueError) as exc:
        raise TimeFFormatError(
            f"{path}: the first comment line {comments[0]!r} doesn't start with age and sex"
        ) from exc
    if age <= 0 and age != AGE_NOT_RECORDED:
        raise TimeFFormatError(f"{path}: states an age of {age}")
    return RecordHeader(
        name=base.name,
        base=base,
        wfdb=wfdb,
        age=None if age == AGE_NOT_RECORDED else age,
        sex=sex,
        medications=comments[1],
        comments=comments,
    )
