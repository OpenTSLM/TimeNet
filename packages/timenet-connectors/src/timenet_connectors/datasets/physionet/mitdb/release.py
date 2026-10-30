"""The MIT-BIH Arrhythmia Database release as published: its archive, its files, its rate, and its pinned counts."""

from pathlib import Path
from typing import NamedTuple

from timenet.errors import TimeFValidationError


ARCHIVE_URL = "https://physionet.org/static/published-projects/mitdb/mit-bih-arrhythmia-database-1.0.0.zip"
"""The one zip of PhysioNet's ``mitdb`` 1.0.0 project, 73 MB."""
ARCHIVE_FILENAME = "mit-bih-arrhythmia-database-1.0.0.zip"
ARCHIVE_SHA256 = "47b26926927c11bd9174154d367c811afc1b186f650f8a205931ca8c520f0a87"
"""PhysioNet corrected ``102.atr`` in place in 2018 without a new version, so the digest, not the version,
names the files."""
ARCHIVE_ROOT = "mit-bih-arrhythmia-database-1.0.0"
"""The archive's one top-level directory, which holds every file below."""

RECORDS_FILE = "RECORDS"
"""The record names, one per line: ``100`` to ``124`` and ``200`` to ``234``, with some numbers missing."""
REFERENCE_ANNOTATOR = "atr"
"""The extension of the reference annotation files, the only annotator ``ANNOTATORS`` lists."""

SAMPLE_RATE_HZ = 360
"""Every signal's rate: the authors digitized at 360 Hz per signal."""
SIGNALS_PER_RECORD = 2


class PinnedCounts(NamedTuple):
    """What the pinned archive holds, which every build checks."""

    records: int
    beats: int
    """Beat labels: the annotations whose code WFDB's ``isqrs`` counts as a QRS complex."""
    rhythm_labels: int
    """Rhythm changes, each starting a rhythm episode."""
    quality_labels: int
    """Signal quality changes, each starting a signal quality episode."""
    non_beats: int
    """Blocked APCs, isolated QRS-like artifacts, and the starts and ends of ventricular flutter."""
    comments: int

    def check(self, root: Path, found: "PinnedCounts") -> None:
        """Compare the counts parsed from ``root`` with these.

        Raises:
            TimeFValidationError: If the files are not the pinned archive's.
        """
        if found != self:
            raise TimeFValidationError(f"{root}: MIT-BIH holds {found}, the pinned archive holds {self}")


# The 109,966 beats are the 16 beat columns of the authors' table of beat types, without the blocked
# APCs (`x`), which the table counts but WFDB's isqrs does not.
PINNED_COUNTS = PinnedCounts(
    records=48, beats=109_966, rhythm_labels=1_291, quality_labels=616, non_beats=337, comments=437
)
