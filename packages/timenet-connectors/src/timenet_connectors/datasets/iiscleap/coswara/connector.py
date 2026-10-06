"""Build original Coswara recordings from a pinned upstream repository revision."""

import asyncio
import csv
from itertools import groupby
from pathlib import Path
import re
import shutil
import tarfile

import httpx
from pydantic import BaseModel, ConfigDict, Field, NonNegativeInt, StrictStr

from timenet.composition import BuildContext
from timenet.connectors import BaseConnector
from timenet.dataset import Record, Source, TimeFDataset
from timenet.errors import TimeFFormatError, TimeNetDownloadError
from timenet.types import Annotation
from timenet_connectors.download import Artifact, download_files
from timenet_connectors.sources.audio import audio_signal


REVISION = "4942c97e31de7180a93d17f2e7530a9c543cfd50"
REPOSITORY = "iiscleap/Coswara-Data"


class Participant(BaseModel):
    """Validate participant identity and age while preserving all released metadata."""

    model_config = ConfigDict(extra="allow")
    id: StrictStr = Field(pattern=r"^[A-Za-z0-9_-]+$")
    a: NonNegativeInt
    covid_status: StrictStr


class CoswaraConnector(BaseConnector[Path]):
    """Keep every original recording, rather than only the HEARTS selection."""

    values_backend = "zarr"

    def download(self, cache_dir: Path) -> list[Path]:  # noqa: PLR6301
        """Download pinned split archives, metadata, and quality annotations.

        Returns:
            The extraction root.

        Raises:
            TimeNetDownloadError: If the repository listing is incomplete or contains no archives.
        """
        response = httpx.get(
            f"https://api.github.com/repos/{REPOSITORY}/git/trees/{REVISION}", params={"recursive": "1"}, timeout=60
        )
        response.raise_for_status()
        tree = response.json()
        names = sorted(entry["path"] for entry in tree["tree"] if entry["type"] == "blob")
        parts = [name for name in names if re.fullmatch(r"\d{8}/\d{8}\.tar\.gz\.[a-z]+", name)]
        if tree["truncated"] or not parts:
            raise TimeNetDownloadError("Coswara source listing is incomplete or has no recording archives")
        paths = [
            "combined_data.csv",
            *parts,
            *(name for name in names if re.fullmatch(r"annotations/[a-z-]+_labels\.csv", name)),
        ]
        asyncio.run(
            download_files(
                Artifact(f"https://raw.githubusercontent.com/{REPOSITORY}/{REVISION}/{name}", cache_dir / name)
                for name in paths
            )
        )
        for date, group in groupby(parts, key=lambda name: name.partition("/")[0]):
            marker = cache_dir / f".{date}.extracted"
            if marker.exists():
                continue
            archive_path = cache_dir / f"{date}.tar.gz"
            with archive_path.open("wb") as output:
                for name in group:
                    with (cache_dir / name).open("rb") as part:
                        shutil.copyfileobj(part, output)
            with tarfile.open(archive_path) as archive:
                archive.extractall(cache_dir / "audio", filter="data")
            marker.touch()
            archive_path.unlink()
        return [cache_dir]

    def convert(self, raw_refs: list[Path], context: BuildContext | None = None) -> TimeFDataset:  # noqa: ARG002
        """Keep each audio file and its participant metadata without assigning tasks.

        Returns:
            The taskless original dataset.

        Raises:
            TimeFFormatError: If an audio file names an unknown participant or no audio is found.
        """
        root = raw_refs[0]
        with (root / "combined_data.csv").open(newline="", encoding="utf-8-sig") as handle:
            participants = [Participant.model_validate(row) for row in csv.DictReader(handle)]
        people = {participant.id: participant.model_dump(mode="json") for participant in participants}
        quality: dict[str, int] = {}
        for path in sorted((root / "annotations").glob("*_labels.csv")):
            with path.open(newline="", encoding="utf-8-sig") as handle:
                quality.update(
                    (row["FILENAME"], int(row["QUALITY"])) for row in csv.DictReader(handle, skipinitialspace=True)
                )
        dataset = TimeFDataset(metadata=self.metadata())
        for subject, metadata in people.items():
            dataset.add_record(
                record=Record(
                    record_id=f"coswara-{subject}",
                    subject_ids=(subject,),
                    sources=(Source(id=f"coswara-{subject}-metadata", name="Participant metadata"),),
                    metadata=metadata,
                )
            )
        recordings = sorted(path for path in (root / "audio").rglob("*.wav") if not path.name.startswith("."))
        if not recordings:
            raise TimeFFormatError("Coswara contains no waveform files")
        for path in recordings:
            subject, kind = path.parent.name, path.stem
            if subject not in people:
                raise TimeFFormatError(f"Coswara waveform {path} names an unknown participant")
            record_id = f"coswara-{subject}-{kind}"
            signal = audio_signal(path, signal_id=f"{record_id}-audio")
            record = Record(
                record_id=record_id,
                subject_ids=(subject,),
                sources=(Source(id=f"{record_id}-source", name=kind, signals=() if signal is None else (signal,)),),
                metadata={
                    **people[subject],
                    "audio_type": kind,
                    "audio_status": "empty" if signal is None else "available",
                },
            )
            rating = quality.get(f"{subject}_{kind}")
            if rating is not None:
                record.annotate(Annotation(key="audio_quality", value=rating))
            dataset.add_record(record=record)
        return dataset


CONNECTOR = CoswaraConnector
