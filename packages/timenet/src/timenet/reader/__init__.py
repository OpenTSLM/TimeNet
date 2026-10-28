"""Deserialization of a TimeF version directory into a :class:`~timenet.dataset.TimeFDataset`."""

from timenet.format.control_reader import TASK_BATCH_SIZE
from timenet.reader.reader import TimeFReader


__all__ = ["TASK_BATCH_SIZE", "TimeFReader"]
