"""Read the published ``time_series_data.h5``: check its layout once, then read one row at a time.

The file holds two HDF5 datasets and no attributes: ``seq_data``, every row's values shaped
``(118921, 600, 33)`` in float64, and ``data_ID``, every row's id. The ids are ``1, 2, ...``, so a QA's
id ``i`` is row ``i - 1``, as the authors' ``TsQaDataset`` reads it (``seq_data[int(id) - 1]``). The values
are z-scored, and the file names no channel, so neither does the connector.
"""

from pathlib import Path
from types import ModuleType

import numpy as np

from timenet.errors import TimeFFormatError, TimeFValidationError
from timenet_connectors.datasets.pandalin98.enginemt_qa.release import DATA_ID, N_CHANNELS, N_ROWS, N_STEPS, SEQ_DATA


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
            "reading EngineMT-QA needs h5py, declared in this connector's requirements.txt. Run the build "
            "without --no-isolation, or install it yourself"
        ) from exc
    return h5py


def check_layout(path: Path, *, n_rows: int = N_ROWS) -> None:
    """Check the file holds exactly ``seq_data`` and ``data_ID``, of the published shapes, with ids ``1..n``.

    Args:
        path: The downloaded ``time_series_data.h5``.
        n_rows: The rows the file must hold.

    Raises:
        TimeFValidationError: If the layout or the ids differ from the published file's.
    """
    with _h5py().File(path, "r") as file:
        if set(file) != {SEQ_DATA, DATA_ID}:
            raise TimeFValidationError(f"{path}: expected {SEQ_DATA} and {DATA_ID}, got {sorted(file)}")
        values, ids = file[SEQ_DATA], file[DATA_ID]
        if values.shape != (n_rows, N_STEPS, N_CHANNELS) or values.dtype != np.float64:
            raise TimeFValidationError(f"{path}: {SEQ_DATA} is {values.dtype} {values.shape}")
        if ids.shape != (n_rows,) or not np.array_equal(ids[:], np.arange(1, n_rows + 1)):
            raise TimeFValidationError(f"{path}: {DATA_ID} is not the row ids 1 to {n_rows}")


def read_row(path: Path, row_id: int) -> np.ndarray:
    """Read one row's values.

    Args:
        path: The ``time_series_data.h5`` that :func:`check_layout` checked.
        row_id: The row's id, which QAs name; the row is ``row_id - 1``.

    Returns:
        The row's values, shaped ``(600, 33)``: steps, then channels.

    Raises:
        TimeFFormatError: If the file no longer holds the row, as when it changed since conversion.
    """
    try:
        with _h5py().File(path, "r") as file:
            values = np.asarray(file[SEQ_DATA][row_id - 1], dtype=np.float64)
    except (OSError, KeyError, IndexError, ValueError) as exc:
        raise TimeFFormatError(f"{path}: cannot read row id {row_id}: {exc}") from exc
    if values.shape != (N_STEPS, N_CHANNELS):
        raise TimeFFormatError(f"{path}: row id {row_id} is shaped {values.shape}; the file changed")
    return values
