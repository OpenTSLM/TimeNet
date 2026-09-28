"""Represent source images as RGB or RGBA tensor Signals for Zarr."""

from io import BytesIO
from pathlib import Path

import numpy as np
import pyarrow as pa

from timenet.dataset import OrdinalAxis, Signal
from timenet.errors import TimeFFormatError
from timenet.types import TimeSeriesSpec, ureg


_RGBA_CHANNELS = 4


def _decode_image(source: Path | bytes) -> np.ndarray:
    """Decode one image and retain its alpha channel when present.

    Returns:
        A height by width by channel RGB or RGBA array.

    Raises:
        TimeFFormatError: If the image cannot be decoded.
    """
    from PIL import Image, ImageOps  # noqa: PLC0415 - optional connector dependency

    try:
        with Image.open(source if isinstance(source, Path) else BytesIO(source)) as opened:
            image = ImageOps.exif_transpose(opened)
            mode = "RGBA" if image.mode in {"RGBA", "LA", "PA", "RGBa", "La"} or "transparency" in image.info else "RGB"
            return np.asarray(image.convert(mode), dtype=np.uint8)
    except Exception as exc:
        raise TimeFFormatError(
            f"could not decode image {source if isinstance(source, Path) else '<bytes>'}: {exc}"
        ) from exc


def image_signal(source: Path | bytes, *, signal_id: str, name: str) -> Signal:
    """Create a lazy one-frame RGB or RGBA Signal with a shape-specific spec.

    Returns:
        A Signal whose single value is the original-size image tensor.
    """
    image = _decode_image(source)
    height, width, channels = image.shape
    color = "rgba" if channels == _RGBA_CHANNELS else "rgb"
    spec = TimeSeriesSpec(
        spec_type=f"{color}_image_{height}x{width}",
        name=f"{color.upper()} image",
        unit_value=ureg.dimensionless,
        dtype="uint8",
        value_shape=(height, width, channels),
        dimension_names=("height", "width", "channel"),
    )
    return Signal.from_loader(
        id=signal_id,
        name=name,
        spec=spec,
        time_axis=OrdinalAxis(),
        n_values=1,
        loader=lambda: pa.FixedShapeTensorArray.from_numpy_ndarray(np.expand_dims(_decode_image(source), axis=0)),
    )
