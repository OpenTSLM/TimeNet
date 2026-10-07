"""Build the original VCTK 0.92 release without benchmark audio transformations."""

import asyncio
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, NonNegativeInt, StrictStr

from timenet.composition import BuildContext
from timenet.connectors import BaseConnector
from timenet.dataset import Record, Source, TimeFDataset
from timenet.errors import TimeFFormatError
from timenet.json import JsonMapping
from timenet_connectors.download import ensure_archive, find_dir_containing
from timenet_connectors.sources.audio import audio_signal


URL = "https://datashare.ed.ac.uk/bitstream/handle/10283/3443/VCTK-Corpus-0.92.zip"
SHA256 = "f96258be9fdc2cbff6559541aae7ea4f59df3fcaf5cf963aae5ca647357e359c"


class Speaker(BaseModel):
    """Validate speaker fields while retaining the free-form region and comments."""

    id: StrictStr = Field(pattern=r"^[A-Za-z0-9]+$")
    age: NonNegativeInt
    gender: Literal["F", "M"]
    accent: StrictStr
    notes: StrictStr = ""


class VctkConnector(BaseConnector[Path]):
    """Keep both microphones, native audio clocks, and available transcripts."""

    values_backend = "zarr"

    def download(self, cache_dir: Path) -> list[Path]:  # noqa: PLR6301
        """Fetch the checksum-pinned Edinburgh 0.92 archive.

        Returns:
            The original corpus root.
        """
        root = asyncio.run(ensure_archive(URL, cache_dir, sha256=SHA256))
        return [find_dir_containing(root, "speaker-info.txt")]

    def convert(self, raw_refs: list[Path], context: BuildContext | None = None) -> TimeFDataset:  # noqa: ARG002
        """Preserve each original microphone recording without resampling.

        Returns:
            The taskless original corpus, with speaker metadata and transcripts.

        Raises:
            TimeFFormatError: If no recordings are present or a speaker is absent from the metadata.
        """
        root = raw_refs[0]
        speakers = _read_speakers(root / "speaker-info.txt")
        recordings = sorted((root / "wav48_silence_trimmed").glob("*/*.flac"))
        if not recordings:
            raise TimeFFormatError("VCTK contains no original FLAC recordings")
        dataset = TimeFDataset(metadata=self.metadata())
        for path in recordings:
            dataset.add_record(record=_record(root, path, speakers))
        return dataset


def _read_speakers(path: Path) -> dict[str, JsonMapping]:
    """Parse the released speaker metadata.

    Returns:
        Validated metadata keyed by speaker ID.
    """
    speakers = {}
    for line in path.read_text(encoding="utf-8").splitlines()[1:]:
        if not line.strip():
            continue
        fields = dict(zip(("id", "age", "gender", "accent", "notes"), line.split(maxsplit=4), strict=False))
        speaker = Speaker.model_validate(fields)
        speakers[speaker.id] = speaker.model_dump(mode="json")
    return speakers


def _record(root: Path, path: Path, speakers: dict[str, JsonMapping]) -> Record:
    """Build one microphone recording with its speaker metadata and transcript.

    Returns:
        The original audio record.

    Raises:
        TimeFFormatError: If the recording names an unknown speaker.
    """
    speaker_id = path.parent.name
    if speaker_id not in speakers:
        raise TimeFFormatError(f"VCTK recording {path.name} names an unknown speaker")
    utterance, microphone = path.stem.rsplit("_", 1)
    record_id = f"vctk-{path.stem}"
    signal = audio_signal(path, signal_id=f"{record_id}-audio")
    transcript = root / "txt" / speaker_id / f"{utterance}.txt"
    metadata: JsonMapping = {
        "speaker": speakers[speaker_id],
        "utterance": utterance,
        "microphone": microphone,
        "audio_status": "empty" if signal is None else "available",
    }
    if transcript.is_file():
        metadata["transcript"] = transcript.read_text().strip()
    return Record(
        record_id=record_id,
        subject_ids=(speaker_id,),
        sources=(Source(id=f"{record_id}-source", name=microphone, signals=() if signal is None else (signal,)),),
        metadata=metadata,
    )


CONNECTOR = VctkConnector
