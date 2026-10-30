"""The PLAID release as published on figshare: its files, their digests, its rate, and its pinned counts."""

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum, unique
from pathlib import Path
from typing import NamedTuple

from timenet.errors import TimeFValidationError


ARTICLE_URL = "https://doi.org/10.6084/m9.figshare.10084619.v2"
"""Version 2 of the figshare article "PLAID 2018", which the paper cites as the dataset."""

SAMPLE_RATE_HZ = 30_000
"""The rate of every file, as each metadata entry's ``sampling_frequency`` states it (``30000Hz``)."""


@dataclass(frozen=True)
class ReleaseFile:
    """One file of the article, pinned by its SHA-256.

    figshare publishes an MD5 per file; the SHA-256 is what this connector computed from the same bytes.
    """

    name: str
    url: str
    sha256: str


@unique
class Subset(StrEnum):
    """The release's two kinds of measurement, each a zip of CSV files with its own metadata file."""

    SUBMETERED = "submetered"
    """One appliance switching on, measured alone, at one of 65 locations."""
    AGGREGATED = "aggregated"
    """Two to four appliances switching on and off together, in the CMU lab."""

    @property
    def archive(self) -> ReleaseFile:
        """Return the zip of the subset's CSV files."""
        return _ARCHIVES[self]

    @property
    def metadata(self) -> ReleaseFile:
        """Return the subset's metadata file: one entry per CSV file, keyed by its number."""
        return _METADATA[self]

    @property
    def directory(self) -> str:
        """Return the zip's top-level directory, which holds ``<number>.csv`` for every entry."""
        return _DIRECTORIES[self]


_ARCHIVES = {
    Subset.SUBMETERED: ReleaseFile(
        name="submetered.zip",
        url="https://ndownloader.figshare.com/files/18183113",
        sha256="7950ae288071025f0a32f095fd4fe477f28b64250a4cc8bb11cd373e68802c95",
    ),
    Subset.AGGREGATED: ReleaseFile(
        name="aggregated.zip",
        url="https://ndownloader.figshare.com/files/18205382",
        sha256="570073cd8171a4b03397f6026d8a83189174a10fc72d490c03136327397a90e5",
    ),
}
_DIRECTORIES = {Subset.SUBMETERED: "submetered_new", Subset.AGGREGATED: "aggregated"}
_METADATA = {
    Subset.SUBMETERED: ReleaseFile(
        name="metadata_submetered.json",
        url="https://ndownloader.figshare.com/files/19812581",
        sha256="f639453a7644321dc520c49983555d1629b442bf2987463fa4340d818311f768",
    ),
    Subset.AGGREGATED: ReleaseFile(
        name="metadata_aggregated.json",
        url="https://ndownloader.figshare.com/files/19812578",
        sha256="12f87daddab90def3e366ee8428a70b10df27903e4a3875ccb8fade2b6da050a",
    ),
}


class PinnedCounts(NamedTuple):
    """What the pinned files hold, which every build checks."""

    submetered: int
    """Submetered files, numbered 1 to 1876."""
    aggregated: int
    """Aggregated files, numbered 1 to 575. The paper says 576."""
    events: int
    """Labelled switch-on and switch-off events across the aggregated files."""
    households: int
    """Distinct ``collection_time`` and ``location`` pairs of the submetered files: the paper's 65 locations."""

    def check(self, root: Path, found: "PinnedCounts") -> None:
        """Compare the counts parsed from ``root`` with these.

        Raises:
            TimeFValidationError: If the files are not the pinned article's.
        """
        if found != self:
            raise TimeFValidationError(f"{root}: PLAID holds {found}, the pinned article holds {self}")


PINNED_COUNTS = PinnedCounts(submetered=1876, aggregated=575, events=2714, households=65)


class KnownDefects(NamedTuple):
    """The pinned files' defects that the build keeps.

    Each build checks that it finds exactly these, so any other defect still fails it.
    """

    unlisted_files: Mapping[Subset, frozenset[str]]
    """Per subset, the zip's CSV files that no metadata entry describes, which the build leaves out."""
    misstated_lengths: Mapping[Subset, frozenset[int]]
    """Per subset, the entries whose rows don't last their stated ``length``, which the build keeps."""


# These aggregated entries state 10 s for a 20 s file or 20 s for a 10 s one (and one 20 s and 30 s).
# Their labelled events still match a step in their own file's current as often as the others' do, so
# the files are joined to the right entries and only the stated length is wrong.
# fmt: off
_MISSTATED_AGGREGATED_LENGTHS = frozenset({
    325, 327, 329, 333, 335, 336, 338, 340, 341, 343, 345, 346, 348, 349, 350, 352, 353, 355, 357, 359,
    360, 361, 362, 363, 365, 367, 368, 369, 370, 371, 372, 374, 376, 378, 382, 384, 385, 386, 387, 388,
    389, 390, 395, 397, 399, 401, 403, 405, 406, 409, 410, 412, 413, 414, 415, 416, 417, 418, 419, 420,
    422, 424, 427, 429, 431, 433, 434, 437, 439, 444, 445, 448, 449, 450, 453, 455, 457, 458, 459, 460,
    461, 462, 463, 464, 465, 466, 470, 471, 472, 473,
})
# fmt: on


KNOWN_DEFECTS = KnownDefects(
    # 0.csv, 10 s long, has no entry in metadata_aggregated.json, so nothing says what it holds. The paper's
    # 576 aggregated files count it; the metadata's 575 entries don't.
    unlisted_files={Subset.SUBMETERED: frozenset(), Subset.AGGREGATED: frozenset({"0.csv"})},
    misstated_lengths={
        Subset.SUBMETERED: frozenset(),
        Subset.AGGREGATED: _MISSTATED_AGGREGATED_LENGTHS,
    },
)
