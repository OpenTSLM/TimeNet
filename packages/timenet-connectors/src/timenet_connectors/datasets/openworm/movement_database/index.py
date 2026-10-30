"""The pinned index of the benchmark's videos: each one's Zenodo record, its features file's MD5, and its strain.

Zenodo gives each video its own record, so no single digest pins the release. ``zenodo_index.csv.gz``, beside
this module, does instead: one row per video of the authors' folds, harvested from the community's records.
README.md says how it was built.
"""

import csv
import gzip
from pathlib import Path
from typing import Annotated, Self

from pydantic import BaseModel, ConfigDict, StrictStr, StringConstraints, ValidationError

from timenet.errors import TimeFFormatError
from timenet_connectors.datasets.openworm.movement_database.release import file_url
from timenet_connectors.datasets.openworm.movement_database.vocabulary import Sex


INDEX_FILE = Path(__file__).with_name("zenodo_index.csv.gz")
INDEX_COLUMNS = ["base_name", "record_id", "md5", "size", "strain", "sex"]

# A strict, non-blank string, as the other connectors define it: a blank field must not pass as text.
Text = Annotated[StrictStr, StringConstraints(pattern=r"\S")]


class IndexedVideo(BaseModel):
    """One video: its Zenodo record and features file, and the strain and sex the record states."""

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    base_name: Text
    """The video's name, which keys the authors' folds and names its files."""
    record_id: int
    md5: Annotated[StrictStr, StringConstraints(pattern=r"^[0-9a-f]{32}$")]
    """The MD5 of the features file, as Zenodo publishes it."""
    size: int
    """The features file's size in bytes."""
    strain: Text
    sex: Sex

    @classmethod
    def parse(cls, row: dict[str, str], context: str) -> Self:
        """Validate one index row.

        Returns:
            The video.

        Raises:
            TimeFFormatError: If a field is missing or malformed.
        """
        try:
            return cls.model_validate_strings(row)
        except ValidationError as exc:
            raise TimeFFormatError(f"{context}: {exc}") from exc

    @property
    def url(self) -> str:
        """Return the download URL of the video's features file."""
        return file_url(self.record_id, self.base_name)

    @property
    def file_name(self) -> str:
        """Return the name the features file is stored under: its record, so names stay short and unique."""
        return f"{self.record_id}.hdf5"


def read_index(path: Path = INDEX_FILE) -> tuple[IndexedVideo, ...]:
    """Read the pinned index.

    Returns:
        One entry per video, in the file's order: by base name.

    Raises:
        TimeFFormatError: If the header isn't :data:`INDEX_COLUMNS`, a row is malformed, or a base name or
            a record repeats.
    """
    with gzip.open(path, "rt", newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    if not rows or rows[0] != INDEX_COLUMNS:
        raise TimeFFormatError(f"{path}: expected the header {','.join(INDEX_COLUMNS)}")
    videos = []
    for line, row in enumerate(rows[1:], 2):
        if len(row) != len(INDEX_COLUMNS):
            raise TimeFFormatError(f"{path}:{line}: expected {len(INDEX_COLUMNS)} fields, got {len(row)}")
        videos.append(IndexedVideo.parse(dict(zip(INDEX_COLUMNS, row, strict=True)), f"{path}:{line}"))
    for field in ("base_name", "record_id"):
        values = [getattr(video, field) for video in videos]
        if len(set(values)) != len(values):
            raise TimeFFormatError(f"{path}: a {field} repeats")
    return tuple(videos)
