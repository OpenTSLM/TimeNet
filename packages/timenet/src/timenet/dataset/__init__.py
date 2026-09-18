"""The in-memory TimeF model a connector populates during ``convert()``."""

from timenet.dataset.axis import AxisType, IrregularAxis, OrdinalAxis, RegularAxis, TimeAxis
from timenet.dataset.dataset import TimeFDataset
from timenet.dataset.record import Record
from timenet.dataset.source import Source
from timenet.dataset.time_series import Signal, TimeSeries


__all__ = [
    "AxisType",
    "IrregularAxis",
    "OrdinalAxis",
    "Record",
    "RegularAxis",
    "Signal",
    "Source",
    "TimeAxis",
    "TimeFDataset",
    "TimeSeries",
]
