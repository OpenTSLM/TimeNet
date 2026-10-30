"""One Record per HDF5 row: its 33 channels as lazy signals, and its row id in typed metadata."""

from functools import lru_cache, partial
from pathlib import Path
from typing import Annotated

import pyarrow as pa
from pydantic import Field, StrictInt

from timenet.dataset import Record, Signal, Source
from timenet.dataset.axis import RegularAxis
from timenet.types import TimeSeriesSpec
from timenet_connectors.datasets.pandalin98.enginemt_qa.keys import channel_name
from timenet_connectors.datasets.pandalin98.enginemt_qa.release import N_CHANNELS, N_STEPS, SAMPLE_RATE_HZ
from timenet_connectors.datasets.pandalin98.enginemt_qa.signals import read_row
from timenet_connectors.metadata import RecordMetadata


SOURCE_NAME = "N-CMAPSS engine"

# The authors z-score the values and publish neither the units nor the statistics.
_SPEC = TimeSeriesSpec(
    spec_type="enginemt_qa_channel", name="EngineMT-QA channel, z-scored", unit_value=None, dtype="float64"
)


class EngineMTQARecordMetadata(RecordMetadata):
    """EngineMT-QA's Record metadata: the row the Record holds.

    The release names no engine unit or flight cycle, so the shared ``subject_id`` stays unset.
    """

    row_id: Annotated[StrictInt, Field(ge=1)]
    """The row's ``data_ID``, which QAs name as their ``id``; the HDF5 row is ``row_id - 1``."""


def record_metadata(record: Record) -> EngineMTQARecordMetadata:
    """Read an EngineMT-QA Record's typed metadata back.

    Returns:
        The row the Record holds.

    Raises:
        TimeFValidationError: If the Record has no EngineMT-QA metadata.
    """  # noqa: DOC502 - raised by from_json
    return EngineMTQARecordMetadata.from_json(record.metadata)


class RowLoader:
    """Load one channel of a row lazily, reading each row once for all its channels.

    A loader serves the Records of one load. A Record's signals load one after another, so one cached
    row suffices.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._read = lru_cache(maxsize=1)(partial(read_row, path))

    def __call__(self, row_id: int, channel: int) -> pa.Array:
        """Return one channel of a row.

        Returns:
            The channel's 600 float64 values.
        """
        return pa.array(self._read(row_id)[:, channel], type=pa.float64())


def build_record(row_id: int, rows: RowLoader) -> Record:
    """Build a row's Record: one Source of 33 lazy signals at 1 Hz, and the row id as metadata.

    Args:
        row_id: The row's id.
        rows: The loader the signals read their values through.

    Returns:
        The Record, whose values stay in the HDF5 file until read.
    """
    time_axis = RegularAxis.from_rate_hz(SAMPLE_RATE_HZ)
    signals = tuple(
        Signal.from_loader(
            name=channel_name(channel),
            spec=_SPEC,
            time_axis=time_axis,
            n_values=N_STEPS,
            loader=partial(rows, row_id, channel),
        )
        for channel in range(N_CHANNELS)
    )
    return Record(
        sources=(Source(name=SOURCE_NAME, signals=signals),),
        metadata=EngineMTQARecordMetadata(row_id=row_id).to_json(),
    )
