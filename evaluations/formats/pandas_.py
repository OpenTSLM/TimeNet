"""The pandas format: the frame PyHealth reads out of the release, written to Parquet."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from evaluations.formats.base import FormatName
from evaluations.pyhealth_loader import LABEL, PATIENT, SIGNAL, load_frame


# Columns added at write time, naming which epoch and which channel a row's values belong to.
CHANNEL = "channel"
EPOCH = "epoch"


class PandasFormat:
    """Parquet, written from the frame PyHealth hands back.

    Nothing is rebuilt here. The frame under test is the one the loader produced, reshaped only
    where Parquet requires it.
    """

    name = FormatName.PANDAS

    def write(self, source: Path, out: Path) -> Path:
        """Read the release with PyHealth and write the frame it gives back.

        Args:
            source: The directory the release was extracted into.
            out: Directory to write into.

        Returns:
            The Parquet file written.
        """
        return self._write_frame(load_frame(source), out)

    @staticmethod
    def _write_frame(frame: pd.DataFrame, out: Path) -> Path:
        """Write a loaded frame to a single Parquet file.

        PyHealth gives a two-dimensional array per row, one row per epoch. Parquet has no type
        for that, so the frame is exploded to one row per (epoch, channel) and the values become
        a ``list<float32>`` column, which Parquet stores natively.

        Args:
            frame: The frame from :func:`~evaluations.pyhealth_loader.load_frame`.
            out: Directory to write into.

        Returns:
            The Parquet file written.
        """
        out.mkdir(parents=True, exist_ok=True)
        path = out / "epochs.parquet"

        flat = pd.DataFrame(
            {
                EPOCH: np.repeat(np.arange(len(frame), dtype=np.int32), _n_channels(frame)),
                CHANNEL: np.tile(np.arange(_n_channels(frame), dtype=np.int16), len(frame)),
                LABEL: np.repeat(frame[LABEL].to_numpy(), _n_channels(frame)),
                PATIENT: np.repeat(frame[PATIENT].to_numpy(), _n_channels(frame)),
                SIGNAL: list(np.concatenate(frame[SIGNAL].to_numpy())),
            }
        )
        flat.to_parquet(path, compression="zstd", index=False)

        return path

    def read_all(self, path: Path) -> list[np.ndarray]:  # noqa: PLR6301 - implements the Format Protocol
        """Read every epoch back from Parquet.

        Args:
            path: The Parquet file written by :meth:`write`.

        Returns:
            One array per epoch, shaped ``(n_channels, n_samples)``.
        """
        flat = pd.read_parquet(path)
        n_channels = int(flat[CHANNEL].max()) + 1
        values = np.stack(flat[SIGNAL].to_numpy()).astype(np.float32, copy=False)

        return list(values.reshape(-1, n_channels, values.shape[-1]))


def _n_channels(frame: pd.DataFrame) -> int:
    """Report how many channels each epoch carries.

    Args:
        frame: The loaded frame.

    Returns:
        The channel count, read from the first epoch.
    """
    return int(np.asarray(frame[SIGNAL].iloc[0]).shape[0])
