"""The pandas format: the frame PyHealth produced, written straight to Parquet."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from evaluations.formats.base import Artifact, directory_size
from evaluations.source import LABEL, PATIENT, SIGNAL


CHANNEL = "channel"
"""Column added at write time, naming which channel a row's values belong to."""
EPOCH = "epoch"
"""Column added at write time, naming which epoch a row's values belong to."""


class PandasFormat:
    """Parquet, written from the frame PyHealth handed back.

    Nothing is rebuilt here. The frame under test is the one the loader produced, reshaped only
    where Parquet requires it.
    """

    name = "pandas"

    def write(self, frame: pd.DataFrame, out: Path) -> Artifact:
        """Write the frame to a single Parquet file.

        PyHealth gives a two-dimensional array per row, one row per epoch. Parquet has no type
        for that, so the frame is exploded to one row per (epoch, channel) and the values become
        a ``list<float32>`` column, which Parquet stores natively.

        Args:
            frame: The shared frame from ``evaluations.source.load_frame``.
            out: Directory to write into.

        Returns:
            The Parquet artifact and its size.
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

        return Artifact(format=self.name, path=path, size_bytes=directory_size(path))

    def read_all(self, path: Path) -> np.ndarray:  # noqa: PLR6301 - implements the Format Protocol
        """Read every epoch back from Parquet.

        Args:
            path: The Parquet file written by :meth:`write`.

        Returns:
            The signals, shaped ``(n_epochs, n_channels, n_samples)``.
        """
        flat = pd.read_parquet(path)
        n_channels = int(flat[CHANNEL].max()) + 1
        values = np.stack(flat[SIGNAL].to_numpy()).astype(np.float32, copy=False)

        return values.reshape(-1, n_channels, values.shape[-1])


def _n_channels(frame: pd.DataFrame) -> int:
    """Report how many channels each epoch carries.

    Args:
        frame: The shared frame.

    Returns:
        The channel count, read from the first epoch.
    """
    return int(np.asarray(frame[SIGNAL].iloc[0]).shape[0])
