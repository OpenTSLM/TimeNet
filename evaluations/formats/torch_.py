"""The torch format: the frame PyHealth reads out of the release, saved with ``torch.save``."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from evaluations.formats.base import FormatName
from evaluations.pyhealth_loader import LABEL, PATIENT, load_frame, signal_stack


class TorchFormat:
    """A ``.pt`` file, written from the frame PyHealth hands back."""

    name = FormatName.TORCH

    def write(self, source: Path, out: Path) -> Path:
        """Read the release with PyHealth and save the signals it gives back.

        Args:
            source: The directory the release was extracted into.
            out: Directory to write into.

        Returns:
            The ``.pt`` file written.
        """
        return self._write_frame(load_frame(source), out)

    @staticmethod
    def _write_frame(frame: pd.DataFrame, out: Path) -> Path:
        """Write a loaded frame's signals as one tensor.

        Args:
            frame: The frame from :func:`~evaluations.pyhealth_loader.load_frame`.
            out: Directory to write into.

        Returns:
            The ``.pt`` file written.
        """
        out.mkdir(parents=True, exist_ok=True)
        path = out / "epochs.pt"

        torch.save(
            {
                "signals": torch.from_numpy(signal_stack(frame)),
                "labels": frame[LABEL].tolist(),
                "patient_ids": frame[PATIENT].tolist(),
            },
            path,
        )

        return path

    def read_all(self, path: Path) -> list[np.ndarray]:  # noqa: PLR6301 - implements the Format Protocol
        """Read every epoch back from the ``.pt`` file.

        Args:
            path: The file written by :meth:`write`.

        Returns:
            One array per epoch, shaped ``(n_channels, n_samples)``.
        """
        loaded = torch.load(path, weights_only=False)

        return list(loaded["signals"].numpy())
