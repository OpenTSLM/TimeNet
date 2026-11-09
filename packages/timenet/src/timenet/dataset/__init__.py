"""The in-memory TimeF model a connector populates during ``convert()``."""

from timenet.dataset.axis import AxisType, IrregularAxis, OrdinalAxis, RegularAxis, TimeAxis
from timenet.dataset.dataset import TimeFDataset
from timenet.dataset.record import Record
from timenet.dataset.selection import SignalSelection
from timenet.dataset.source import Source
from timenet.dataset.time_series import Signal, SignalChannel
from timenet.types.shared_loading import SharedChannelLoader


__all__ = [
    "AxisType",
    "IrregularAxis",
    "OrdinalAxis",
    "Record",
    "RegularAxis",
    "SharedChannelLoader",
    "Signal",
    "SignalChannel",
    "SignalSelection",
    "Source",
    "TimeAxis",
    "TimeFDataset",
]
