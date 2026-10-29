"""Read one video's features file: the facts its Record needs, then, on demand, its skeletons.

A features file is an HDF5 file that Tierpsy Tracker wrote. :func:`read_video` reads only its header: the
experiment's facts, the frame count and the frame rate. :func:`read_skeletons` reads the skeletons, which the
Record's signals load lazily.
"""

from datetime import datetime
from pathlib import Path
from types import ModuleType
from typing import Any, NamedTuple

import numpy as np
from pydantic import BaseModel, ConfigDict, ValidationError

from timenet.errors import TimeFFormatError
from timenet.types import US_PER_S
from timenet_connectors.datasets.openworm.movement_database.index import Text
from timenet_connectors.datasets.openworm.movement_database.release import (
    EXPERIMENT_INFO,
    SKELETON_POINTS,
    SKELETONS,
    TIME_UNITS,
    TIMESERIES,
    XY_UNITS,
)
from timenet_connectors.datasets.openworm.movement_database.vocabulary import Sex, VentralSide


# The frame rate is stored as a float, 1 over a period that is a whole number of microseconds. The float's
# own error is under a nanosecond; anything larger means the period isn't whole.
_PERIOD_TOLERANCE_US = 1e-3


def _h5py() -> ModuleType:
    """Import ``h5py`` lazily. Raise an error that states the fix if it is missing.

    Returns:
        The imported ``h5py`` module.

    Raises:
        ImportError: If ``h5py``, declared in this connector's requirements, is not installed.
    """
    try:
        import h5py  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "reading the movement database needs h5py, declared in this connector's "
            "requirements.txt. Run the build without --no-isolation, or install it yourself"
        ) from exc
    return h5py


class Experiment(BaseModel):
    """The facts of ``experiment_info`` the connector reads. The file states more, such as the lab and food."""

    model_config = ConfigDict(strict=True, frozen=True, extra="ignore")

    base_name: Text
    strain: Text
    sex: Sex
    ventral_side: VentralSide
    timestamp: datetime
    """When recording started, with its UTC offset."""


class Video(NamedTuple):
    """One features file, as the connector reads its header."""

    path: Path
    experiment: Experiment
    n_frames: int
    period_us: int
    """The time from one frame to the next, as the file's frame rate states it."""


def _attribute(attributes: Any, name: str, context: str) -> Any:
    """Return one attribute, decoding bytes.

    Returns:
        The attribute's value.

    Raises:
        TimeFFormatError: If the attribute is missing.
    """
    if name not in attributes:
        raise TimeFFormatError(f"{context}: no {name!r} attribute")
    value = attributes[name]
    return value.decode() if isinstance(value, bytes) else value


def period_us(fps: float, context: str) -> int:
    """Return the frame period of a frame rate, in whole microseconds.

    Returns:
        The period.

    Raises:
        TimeFFormatError: If the rate isn't positive, or its period isn't a whole number of microseconds.
    """
    if not fps > 0:
        raise TimeFFormatError(f"{context}: frame rate {fps}")
    exact = US_PER_S / fps
    period = round(exact)
    if abs(exact - period) > _PERIOD_TOLERANCE_US:
        raise TimeFFormatError(f"{context}: frame rate {fps} has a period of {exact} us, not a whole number")
    return period


def read_video(path: Path) -> Video:
    """Read a features file's header: its experiment, frame count and frame rate.

    Returns:
        The video.

    Raises:
        TimeFFormatError: If ``experiment_info`` is malformed, the units aren't micrometers and seconds, the
            skeletons aren't frames of 49 x and y points, or the per-frame table doesn't number the same
            frames 0 to n - 1, one worm's.
    """
    h5py = _h5py()
    with h5py.File(path, "r") as handle:
        try:
            experiment = Experiment.model_validate_json(handle[EXPERIMENT_INFO][()])
        except (KeyError, ValidationError) as exc:
            raise TimeFFormatError(f"{path}: {EXPERIMENT_INFO}: {exc}") from exc
        skeletons = handle[SKELETONS]
        n_frames = skeletons.shape[0]
        if skeletons.shape[1:] != (SKELETON_POINTS, 2):
            raise TimeFFormatError(f"{path}: skeletons of shape {skeletons.shape}, expected (n, {SKELETON_POINTS}, 2)")
        table = handle[TIMESERIES]
        context = f"{path}: {TIMESERIES}"
        for name, expected in (("xy_units", XY_UNITS), ("time_units", TIME_UNITS)):
            if (unit := _attribute(table.attrs, name, context)) != expected:
                raise TimeFFormatError(f"{context}: {name} is {unit!r}, expected {expected!r}")
        period = period_us(float(_attribute(table.attrs, "fps", context)), context)
        frames = table["timestamp"]
        if len(frames) != n_frames or not np.array_equal(frames, np.arange(n_frames)):
            raise TimeFFormatError(f"{context}: its frames aren't the skeletons' 0 to {n_frames - 1}")
        if len(np.unique(table["worm_index"])) != 1:
            raise TimeFFormatError(f"{context}: more than one worm")
    return Video(path=path, experiment=experiment, n_frames=n_frames, period_us=period)


def read_skeletons(video: Video) -> np.ndarray:
    """Read a video's skeletons: one row per frame, 49 points, then x and y, in micrometers.

    A frame without a skeleton is NaN in all its values.

    Returns:
        The float32 skeletons, of shape ``(n_frames, 49, 2)``.

    Raises:
        TimeFFormatError: If the file changed since its header was read, or a frame is only partly NaN.
    """
    with _h5py().File(video.path, "r") as handle:
        skeletons = handle[SKELETONS][()]
    if skeletons.shape != (video.n_frames, SKELETON_POINTS, 2) or skeletons.dtype != np.float32:
        raise TimeFFormatError(f"{video.path}: skeletons of {skeletons.shape} {skeletons.dtype} changed since read")
    missing = np.isnan(skeletons)
    partial = missing.any(axis=(1, 2)) & ~missing.all(axis=(1, 2))
    if partial.any():
        raise TimeFFormatError(f"{video.path}: frames {np.flatnonzero(partial)[:5].tolist()} are partly NaN")
    return skeletons
