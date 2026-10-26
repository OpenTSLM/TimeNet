"""Typed signal chunk rows for the control catalog."""

from pydantic import StrictInt, StrictStr

from timenet.format.control_schema import SIGNAL_CHUNKS
from timenet.format.table_writer import ModelTable
from timenet.types._model import TimeFModel
from timenet.types.reference_keys import OwnerKey


class SignalChunkRow(TimeFModel):
    """Locate one signal chunk using the control table's column names."""

    signal_key: StrictInt
    chunk_index: StrictInt
    value_path: StrictStr
    owner_key: OwnerKey | None
    chunk_major_index: StrictInt
    chunk_minor_index: StrictInt | None
    n_values: StrictInt


SIGNAL_CHUNK_TABLE = ModelTable[SignalChunkRow](SIGNAL_CHUNKS)
