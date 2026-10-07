"""Build the public COUGHVID v3 corpus without benchmark-specific filtering."""

import asyncio
import csv
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import UUID

import numpy as np
import pyarrow as pa
from pydantic import BaseModel, ConfigDict

from timenet.composition import BuildContext
from timenet.connectors import BaseConnector
from timenet.dataset import Record, RegularAxis, Signal, Source, TimeFDataset
from timenet.errors import TimeFFormatError
from timenet.types import InputModality, TimeSeriesSpec, ureg
from timenet_connectors.download import ensure_archive, find_dir_containing


if TYPE_CHECKING:
    from av.audio.layout import AudioLayout


URL = "https://zenodo.org/api/records/7024894/files/public_dataset_v3.zip/content"
SHA256 = "37544c58ac5a7d79cb68af56fb3d0a690773b100bef8a03aac96570086fea335"


class RecordingMetadata(BaseModel):
    """Validate the released recording ID and retain all user and expert labels."""

    model_config = ConfigDict(extra="allow")
    uuid: UUID


class CoughvidConnector(BaseConnector[Path]):
    """Keep native audio and all public metadata, including unavailable recordings."""

    values_backend = "zarr"

    def download(self, cache_dir: Path) -> list[Path]:  # noqa: PLR6301
        """Fetch the checksum-pinned public v3 archive.

        Returns:
            The directory with original recordings and compiled metadata.
        """
        root = asyncio.run(ensure_archive(URL, cache_dir, filename="public_dataset_v3.zip", sha256=SHA256))
        return [find_dir_containing(root, "metadata_compiled.csv")]

    def convert(self, raw_refs: list[Path], context: BuildContext | None = None) -> TimeFDataset:  # noqa: ARG002
        """Keep each recording and its labels, decoding sample arrays only on read.

        Returns:
            The taskless original corpus. Missing, empty, or undecodable audio keeps its metadata
            and an explicit status instead of an invented waveform.

        Raises:
            TimeFFormatError: If multiple audio files share a recording ID.
        """
        root = raw_refs[0]
        with (root / "metadata_compiled.csv").open(newline="", encoding="utf-8-sig") as handle:
            metadata = [RecordingMetadata.model_validate(row) for row in csv.DictReader(handle)]
        audio_paths: dict[str, Path] = {}
        for path in root.iterdir():
            if path.suffix not in {".webm", ".ogg", ".wav"}:
                continue
            if path.stem in audio_paths:
                raise TimeFFormatError(f"COUGHVID {path.stem} has multiple original audio files")
            audio_paths[path.stem] = path
        dataset = TimeFDataset(metadata=self.metadata())
        for entry in metadata:
            dataset.add_record(record=_record(entry, audio_paths.get(str(entry.uuid))))
        return dataset


def _record(entry: RecordingMetadata, path: Path | None) -> Record:
    """Build a recording and preserve the status of unavailable audio.

    Returns:
        The original recording's record.
    """
    import av  # noqa: PLC0415 - connector dependency

    record_id = f"coughvid-{entry.uuid}"
    fields = entry.model_dump(mode="json")
    signal = None
    fields["audio_status"] = "missing"
    if path is not None:
        fields["filename"] = path.name
        try:
            signal = _audio(path, record_id)
            fields["audio_status"] = "empty" if signal is None else "available"
        except av.error.FFmpegError as error:
            fields.update(audio_status="undecodable", audio_error=str(error))
    return Record(
        record_id=record_id,
        sources=(
            Source(id=f"{record_id}-source", name="Original recording", signals=() if signal is None else (signal,)),
        ),
        metadata=fields,
    )


def _audio(path: Path, record_id: str) -> Signal | None:
    import av  # noqa: PLC0415 - connector dependency

    # Compressed containers do not reliably expose an exact decoded sample count. Count frame
    # lengths without creating sample arrays; the lazy loader decodes values only when requested.
    with av.open(str(path)) as container:
        stream = container.streams.audio[0]
        rate, layout = stream.codec_context.sample_rate, stream.codec_context.layout
        channels = len(layout.channels)
        count = sum(frame.samples for frame in container.decode(audio=0))
    if count == 0:
        return None

    def load() -> pa.Array:
        values = _decode_samples(path, layout, rate)
        if len(values) != count:
            raise TimeFFormatError(f"COUGHVID recording {path} changed after its sample count was read")
        return (
            pa.array(values[:, 0])
            if channels == 1
            else pa.FixedShapeTensorArray.from_numpy_ndarray(np.ascontiguousarray(values))
        )

    return Signal.from_loader(
        id=f"{record_id}-audio",
        name="audio",
        n_values=count,
        time_axis=RegularAxis.from_rate_hz(rate),
        spec=TimeSeriesSpec(
            spec_type=f"audio_{channels}ch",
            name="Audio waveform",
            dtype="float32",
            unit_value=ureg.dimensionless,
            modality=InputModality.AUDIO,
            value_shape=() if channels == 1 else (channels,),
            dimension_names=() if channels == 1 else ("channel",),
        ),
        loader=load,
    )


def _decode_samples(path: Path, layout: "AudioLayout", rate: int) -> np.ndarray:
    """Decode native audio samples with an explicit channel dimension.

    Returns:
        Samples arranged as rows, with one column per channel.
    """
    import av  # noqa: PLC0415 - connector dependency

    with av.open(str(path)) as container:
        converter = av.AudioResampler(format="fltp", layout=layout, rate=rate)
        chunks = []
        for frame in container.decode(audio=0):
            for converted in converter.resample(frame):
                chunks.append(converted.to_ndarray())
        chunks.extend(frame.to_ndarray() for frame in converter.resample(None))
    return np.concatenate(chunks, axis=1).T


CONNECTOR = CoughvidConnector
