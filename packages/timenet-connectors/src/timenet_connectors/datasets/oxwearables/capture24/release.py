"""The CAPTURE-24 release as published: its archive, its files and columns, its rate, and its pinned counts."""

from pathlib import Path
import re
from typing import NamedTuple

from timenet.errors import TimeFValidationError
from timenet.types import US_PER_S


ARCHIVE_URL = "https://ora.ox.ac.uk/objects/uuid:99d7c092-d865-4a19-b096-cc16440cd001/files/rpr76f381b"
"""The one zip of the Oxford Research Archive record, 6.9 GB."""
ARCHIVE_FILENAME = "capture24.zip"
ARCHIVE_SHA256 = "69740c22d3e000367988373336dc6486c10fe3f3f929811fd66e4d84861e40e2"
ARCHIVE_ROOT = "capture24"
"""The archive's one top-level directory, which holds every file below."""

METADATA_FILE = "metadata.csv"
"""One row per participant: ``pid``, ``age`` (a band, such as ``38-52``) and ``sex``."""
DICTIONARY_FILE = "annotation-label-dictionary.csv"
"""One row per raw label: the label, then its class in each of the six label schemes."""
PARTICIPANT_FILE = re.compile(r"P[0-9]{3}\.csv\.gz")
"""A participant's recording: ``time,x,y,z,annotation``, one row per sample."""
PARTICIPANT_COLUMNS = ("time", "x", "y", "z", "annotation")

SAMPLE_RATE_HZ = 100
SAMPLE_PERIOD_US = US_PER_S // SAMPLE_RATE_HZ


class PinnedCounts(NamedTuple):
    """What the pinned archive holds, which every build checks."""

    participants: int
    segments: int
    """Runs of samples exactly one period apart. Eight participants have a gap, so two runs."""
    windows: int
    """The authors' 10 s windows, which ``prepare_data.make_windows`` keeps."""

    def check(self, root: Path, found: "PinnedCounts") -> None:
        """Compare the counts parsed from ``root`` with these.

        Raises:
            TimeFValidationError: If the files are not the pinned archive's.
        """
        if found != self:
            raise TimeFValidationError(f"{root}: CAPTURE-24 holds {found}, the pinned archive holds {self}")


# The windows are what the authors' prepare_data.make_windows keeps from the pinned archive: the same
# 934,762 window starts, each with the same class in both benchmarked schemes.
PINNED_COUNTS = PinnedCounts(participants=151, segments=159, windows=934_762)
