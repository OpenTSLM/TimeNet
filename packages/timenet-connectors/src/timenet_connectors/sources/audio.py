"""Lazy, lossless decoding of original waveform files."""

from pathlib import Path

import numpy as np
import pyarrow as pa

from timenet.dataset import RegularAxis, Signal
from timenet.errors import TimeFFormatError
from timenet.types import InputModality, TimeSeriesSpec, ureg


def audio_signal(path: Path, *, signal_id: str, name: str = "audio") -> Signal | None:
    """Keep the file's native rate and channels, decoding samples only on read.

    Returns:
        A float32 audio signal, or ``None`` for an empty recording. Multichannel files
        have an explicit channel dimension.

    Raises:
        TimeFFormatError: If the file contains no channels.
    """
    import soundfile as sf  # noqa: PLC0415 - connector dependency

    info = sf.info(path)
    if info.channels <= 0:
        raise TimeFFormatError(f"audio file {path} has no channels")
    if info.frames == 0:
        return None
    shape = () if info.channels == 1 else (info.channels,)

    def load() -> pa.Array:
        values, rate = sf.read(path, dtype="float32", always_2d=bool(shape))
        if rate != info.samplerate or len(values) != info.frames:
            raise TimeFFormatError(f"audio file {path} changed after its header was read")
        return pa.FixedShapeTensorArray.from_numpy_ndarray(np.ascontiguousarray(values)) if shape else pa.array(values)

    return Signal.from_loader(
        id=signal_id,
        name=name,
        n_values=info.frames,
        time_axis=RegularAxis.from_rate_hz(info.samplerate),
        spec=TimeSeriesSpec(
            spec_type=f"audio_{info.channels}ch",
            name="Audio waveform",
            dtype="float32",
            unit_value=ureg.dimensionless,
            modality=InputModality.AUDIO,
            value_shape=shape,
            dimension_names=("channel",) if shape else (),
        ),
        loader=load,
    )
