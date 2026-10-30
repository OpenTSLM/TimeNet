"""The CODE-15% release as published: its Zenodo record, its files and their digests, and its pinned counts."""

from pathlib import Path
from typing import NamedTuple

from timenet.errors import TimeFValidationError


RECORD_URL = "https://zenodo.org/records/4916206"
"""Zenodo record 4916206, version 1.0.0, the record's only version."""
RECORD_DOI = "10.5281/zenodo.4916206"

CSV_FILE = "exams.csv"
"""One row per exam: ids, age, sex, the published age prediction, six labels, mortality, and its HDF5 part."""
CSV_SHA256 = "f9f82f4c07c1793e2285915c69f1113f5a7db8a0f2cf2f4b90474502f406bc35"
"""Computed on download. Zenodo publishes the md5 ``0107516d3f63864498fb77d15799cc95``."""


class Part(NamedTuple):
    """One of the 18 zips, each holding one HDF5 file of tracings."""

    index: int
    sha256: str
    """The zip's digest, computed on download. Zenodo publishes only md5s."""

    @property
    def zip_name(self) -> str:
        """Return the zip's name on Zenodo, such as ``exams_part0.zip``."""
        return f"exams_part{self.index}.zip"

    @property
    def hdf5_name(self) -> str:
        """Return the HDF5 file the zip holds, which ``exams.csv`` names in ``trace_file``."""
        return f"exams_part{self.index}.hdf5"

    @property
    def url(self) -> str:
        """Return the zip's download URL."""
        return file_url(self.zip_name)


def file_url(name: str) -> str:
    """Return the download URL of one of the record's files.

    Returns:
        The URL.
    """
    return f"{RECORD_URL}/files/{name}?download=1"


PARTS = (
    Part(0, "4941375309c741ab676071f6bbd77c9ee739e0212535122bbb7e3f72e7ff7bf5"),
    Part(1, "ae0fccfb10cfdc938d883991587f190af837b2f742bc76b1d7852044768a3e41"),
    Part(2, "bdf054b71a92b279fbf630d0ad05199577c425a4083511457705a0391ad13bf4"),
    Part(3, "ea3d0a24c67f92f01b1d91ba6ae0da26d15d17ce717a01ee727098c43032cf98"),
    Part(4, "b17532eb40eff6c7ee632b2ef55e8b89cd8a6c3418de89b858da69a0696cbef1"),
    Part(5, "fdb9cba13c4f8f5636360ae7cb455c0e97a4b26cc6899bd05d236ea81ca90e30"),
    Part(6, "33cd2b14fd156d8ada703a2c5f3cb17ce7cc70bc55939b115892752acb2027c6"),
    Part(7, "a10c91440337560583375a423395cb13cc2f6eccbc03303d7d0b646a67ce6fc0"),
    Part(8, "1d0070dda22f02bca30c04f35ca0c2cc92ca9460fb43fa1d5977a663635e4401"),
    Part(9, "7e9812179c45e1bb5adbbee5976ba0f5adf89f8b34356c9d77bdad06fc99555b"),
    Part(10, "22dba63939faf22602ed511a888bbd1a4e1c779a8d2634a6091300e5530077bd"),
    Part(11, "491ba8bfa3defc13cdc37af94dc9bfc7ddc90b03dd989288897795b2e3822053"),
    Part(12, "ce47ac6180ba15868dec6277124c0f0ef41dfad54bba564d3a204117d4f65811"),
    Part(13, "e39b90d72295a4a0923e3af89de09346e454256d24e15e2bf5a2c3e4ceec8bee"),
    Part(14, "e21fe30bf441bc09fb444719bc22ef72e504f719e71d919cc665ecc4b8f36658"),
    Part(15, "84ddf0a1ef567a3a0f308c6b2b8d240002a21be7105ba761d6d91f3299082e1a"),
    Part(16, "b66f46d9fd17162453891732039f4a62d4df66bba445610444a08d205a89b927"),
    Part(17, "af7fd3e5fa9e091ab1936e720c7fc55f116cca0722592ce2bdbff644de22e2e9"),
)
"""The record's 18 zips, ``exams_part0.zip`` to ``exams_part17.zip``, each with the SHA-256 of its bytes.

The hashes pin the release: the download fails with ``TimeFFormatError`` if a zip's bytes don't match its
hash, before the zip is extracted, so a changed, truncated or corrupted file never reaches the build. Zenodo
publishes only MD5s, so we computed these SHA-256s on the first download, whose zips matched the MD5s Zenodo
lists. Once a zip is extracted, its marker skips the download, and the hash isn't checked again
(VERIFICATION.md #3).
"""

TRACINGS_DATASET = "tracings"
"""Each HDF5 file's ``(N, 4096, 12)`` float32 tracings, in the order of its ``exam_id`` dataset."""
EXAM_ID_DATASET = "exam_id"
"""Each HDF5 file's ``(N,)`` int64 exam ids. The rows are not in ``exams.csv``'s order."""
FILLER_EXAM_ID = 0
"""The id of the last row of every HDF5 file: an all-zero trace that ``exams.csv`` doesn't list."""

SAMPLE_RATE_HZ = 400


class PinnedCounts(NamedTuple):
    """What the pinned release holds, which every build checks."""

    exams: int
    """Rows of ``exams.csv``, which is also the HDF5 rows without the filler rows."""
    patients: int
    all_zero_exam_ids: frozenset[int]
    """Exams whose published trace is zero in every lead and sample. The connector excludes them, and with
    them some patients and mortality: see :data:`ALL_ZERO_EXAM_IDS_FILE`."""

    def check(self, root: Path, found: "PinnedCounts") -> None:
        """Compare the counts parsed from ``root`` with these.

        Raises:
            TimeFValidationError: If the files are not the pinned release's.
        """
        if found == self:
            return
        differences = [
            f"{name} {getattr(found, name)} where the pinned release holds {getattr(self, name)}"
            for name in ("exams", "patients")
            if getattr(found, name) != getattr(self, name)
        ]
        unpinned = sorted(found.all_zero_exam_ids - self.all_zero_exam_ids)
        if unpinned:
            differences.append(f"all-zero traces of exams {unpinned[:5]} that aren't pinned")
        not_found = sorted(self.all_zero_exam_ids - found.all_zero_exam_ids)
        if not_found:
            differences.append(f"pinned all-zero exams {not_found[:5]} whose trace isn't all zero")
        raise TimeFValidationError(f"{root}: CODE-15% differs from the pinned release: {'; '.join(differences)}")


def read_exam_ids(path: Path) -> frozenset[int]:
    """Read a file of exam ids, one per line, skipping ``#`` comments.

    Returns:
        The ids.
    """
    lines = (line.strip() for line in path.read_text(encoding="utf-8").splitlines())
    return frozenset(int(line) for line in lines if line and not line.startswith("#"))


ALL_ZERO_EXAM_IDS_FILE = Path(__file__).with_name("all_zero_exam_ids.txt")
"""The 670 exams whose trace is zero everywhere, found by scanning every part of the pinned release.

The connector excludes them, as they hold no signal to predict from: they have no Record and no tasks. So
the dataset loses every exam of 290 patients, and the death and timey of 419 patients, which ``exams.csv``
states only on one of these exams.
"""

PINNED_COUNTS = PinnedCounts(exams=345_779, patients=233_770, all_zero_exam_ids=read_exam_ids(ALL_ZERO_EXAM_IDS_FILE))
