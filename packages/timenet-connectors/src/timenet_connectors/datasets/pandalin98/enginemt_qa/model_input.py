"""Rebuild the series the authors' model reads for a task, from the task's Records.

The authors' ``TsQaDataset`` gives the model one array shaped (600, 33) per QA pair. A single-cycle
pair reads its row whole. A multi-cycle pair reads the first :data:`~.tasks.STEPS_PER_CYCLE` steps of
each of its 10 rows and concatenates them in published order. A task holds the whole Records instead,
so :func:`authors_series` cuts and joins them the same way.
"""

import numpy as np

from timenet.dataset import Record
from timenet.errors import TimeFValidationError
from timenet.types import Task
from timenet_connectors.datasets.pandalin98.enginemt_qa.keys import channel_name
from timenet_connectors.datasets.pandalin98.enginemt_qa.release import MULTI_CYCLE_ROWS, N_CHANNELS
from timenet_connectors.datasets.pandalin98.enginemt_qa.tasks import STEPS_PER_CYCLE


_CHANNELS = tuple(channel_name(index) for index in range(N_CHANNELS))


def record_values(record: Record) -> np.ndarray:
    """Return a Record's values as the HDF5 row holds them.

    Returns:
        The values shaped (steps, channels), the channels in source order.

    Raises:
        TimeFValidationError: If the Record doesn't hold the 33 EngineMT-QA channels.
    """
    signals = {signal.name: signal for signal in record.signals}
    if tuple(sorted(signals)) != _CHANNELS:
        raise TimeFValidationError("the Record is not an EngineMT-QA Record")
    return np.stack([signals[name].to_numpy() for name in _CHANNELS], axis=1)


def authors_series(task: Task) -> np.ndarray:
    """Rebuild the array the authors' ``TsQaDataset`` feeds the model for this task.

    Returns:
        The values shaped (600, 33): one row whole, or the first 60 steps of each of 10 rows, joined.

    Raises:
        TimeFValidationError: If the task names neither one nor 10 Records.
    """
    records = task.inputs
    if len(records) == 1:
        return record_values(records[0])
    if len(records) == MULTI_CYCLE_ROWS:
        return np.concatenate([record_values(record)[:STEPS_PER_CYCLE] for record in records])
    raise TimeFValidationError(f"an EngineMT-QA task names 1 or {MULTI_CYCLE_ROWS} Records, not {len(records)}")
