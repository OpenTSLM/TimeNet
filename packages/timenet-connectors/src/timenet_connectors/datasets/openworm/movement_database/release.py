"""The movement database as published on Zenodo: one record per video, its features file, and the pinned counts.

Each video of the Open Worm Movement Database is its own Zenodo record, in the community
``open-worm-movement-database``. Of a record's three files, the connector reads ``<base name>_features.hdf5``,
the Tierpsy Tracker output that holds the skeletons.
"""

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import NamedTuple
from urllib.parse import quote

from timenet.errors import TimeFFormatError, TimeFValidationError


COMMUNITY_URL = "https://zenodo.org/communities/open-worm-movement-database"

FEATURES_SUFFIX = "_features.hdf5"
"""A record's features file is its video's base name plus this suffix."""

EXPERIMENT_INFO = "experiment_info"
"""A JSON string in the features file: the video's strain, sex, ventral side, timestamp and lab."""
SKELETONS = "coordinates/skeletons"
"""The features file's skeletons: one row per frame, 49 points from head to tail, then x and y."""
TIMESERIES = "features_timeseries"
"""The features file's per-frame table, whose attributes give the frame rate and the units."""

SKELETON_POINTS = 49
XY_UNITS = "micrometers"
TIME_UNITS = "seconds"


@dataclass(frozen=True)
class DownloadedRelease:
    """What ``download`` fetched: the authors' three benchmark files, and the directory of features files."""

    folds_path: Path
    classes_path: Path
    video_table_path: Path
    videos_dir: Path
    """One features file per video, named by its Zenodo record, such as ``1019082.hdf5``."""


def file_url(record_id: int, base_name: str) -> str:
    """Return the download URL of one record's features file.

    Returns:
        The Zenodo API URL of the file's content.
    """
    return f"https://zenodo.org/api/records/{record_id}/files/{quote(base_name + FEATURES_SUFFIX)}/content"


def check_md5(path: Path, md5: str) -> None:
    """Check a downloaded file against the MD5 Zenodo publishes for it.

    Raises:
        TimeFFormatError: If the file's bytes differ.
    """
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    if digest.hexdigest() != md5:
        raise TimeFFormatError(f"{path}: MD5 {digest.hexdigest()}, Zenodo publishes {md5}")


class PinnedCounts(NamedTuple):
    """What the pinned benchmark holds, which every build checks."""

    videos: int
    """The videos of the authors' folds, each a Zenodo record."""
    classes: int
    """The authors' classes: a strain, and N2 once per sex."""
    test_videos: int
    """The videos of the authors' test fold."""

    def check(self, found: "PinnedCounts") -> None:
        """Compare the counts parsed from the downloaded files with these.

        Raises:
            TimeFValidationError: If the files are not the pinned benchmark's.
        """
        if found != self:
            raise TimeFValidationError(f"the movement database holds {found}, the pinned benchmark holds {self}")


# The paper: "The dataset comprises 10476 videos of individual worms divided between 365 different classes".
PINNED_COUNTS = PinnedCounts(videos=10_476, classes=365, test_videos=3_609)
