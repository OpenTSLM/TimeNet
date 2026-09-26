"""Represent source images as RGB tensor Signals for the Zarr values backend."""

from io import BytesIO
from pathlib import Path

import numpy as np
import pyarrow as pa

from timenet.dataset import OrdinalAxis, Signal
from timenet.errors import TimeFFormatError
from timenet.types import TimeSeriesSpec, ureg


def _rgb(source: Path | bytes) -> np.ndarray:
    """Decode one image and composite transparency onto white.

    Returns:
        A height by width by channel RGB array.

    Raises:
        TimeFFormatError: If the image cannot be decoded.
    """
    from PIL import Image, ImageOps  # noqa: PLC0415 - optional connector dependency

    try:
        with Image.open(source if isinstance(source, Path) else BytesIO(source)) as opened:
            image = ImageOps.exif_transpose(opened)
            if image.mode in {"RGBA", "LA"} or "transparency" in image.info:
                rgba = image.convert("RGBA")
                canvas = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
                image = Image.alpha_composite(canvas, rgba)
            return np.asarray(image.convert("RGB"), dtype=np.uint8)
    except Exception as exc:
        raise TimeFFormatError(
            f"could not decode image {source if isinstance(source, Path) else '<bytes>'}: {exc}"
        ) from exc


def image_signal(source: Path | bytes, *, signal_id: str, name: str) -> Signal:
    """Create a lazy one-frame RGB Signal with a shape-specific specification.

    Returns:
        A Signal whose single value is the original-size image tensor.
    """
    image = _rgb(source)
    height, width, channels = image.shape
    spec = TimeSeriesSpec(
        spec_type=f"rgb_image_{height}x{width}",
        name="RGB image",
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
        loader=lambda: pa.FixedShapeTensorArray.from_numpy_ndarray(np.expand_dims(_rgb(source), axis=0)),
    )
