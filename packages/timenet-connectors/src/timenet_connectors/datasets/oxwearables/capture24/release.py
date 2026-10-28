"""The CAPTURE-24 release as published: its archive, its files, its benchmark split, and its pinned counts."""

from pathlib import Path
import re
from typing import NamedTuple

from timenet.errors import TimeFValidationError
from timenet.types import US_PER_S, Split


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
PARTICIPANT_FILE = re.compile(r"P(?P<number>[0-9]{3})\.csv\.gz")
"""A participant's recording: ``time,x,y,z,annotation``, one row per sample."""
PARTICIPANT_COLUMNS = ("time", "x", "y", "z", "annotation")

SAMPLE_RATE_HZ = 100
SAMPLE_PERIOD_US = US_PER_S // SAMPLE_RATE_HZ
WINDOW_US = 10 * US_PER_S
"""The authors' window: 10 s, ``prepare_data.py --winsec 10``."""
WINDOW_SAMPLES = WINDOW_US // SAMPLE_PERIOD_US

FIRST_TEST_PARTICIPANT = 101
"""The first test participant, as ``benchmark.py``'s ``range(101, 152)`` runs it.

The README says P102 to P151. The code, which produced the published numbers, also tests P101.
"""


def split_of(participant_number: int) -> Split:
    """Return a participant's benchmark split: P001 to P100 train, P101 to P151 test.

    Returns:
        ``Split.TEST`` from :data:`FIRST_TEST_PARTICIPANT` on, else ``Split.TRAIN``.
    """
    return Split.TEST if participant_number >= FIRST_TEST_PARTICIPANT else Split.TRAIN


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
