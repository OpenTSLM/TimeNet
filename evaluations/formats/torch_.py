"""The torch format: the frame PyHealth reads out of the release, saved with ``torch.save``."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from evaluations.formats.base import Artifact, directory_size
from evaluations.source import LABEL, PATIENT, load_frame, signal_stack


class TorchFormat:
    """A ``.pt`` file, written from the frame PyHealth hands back."""

    name = "torch"

    def write(self, source: Path, out: Path) -> Artifact:
        """Read the release with PyHealth and save the signals it gives back.

        Args:
            source: The directory the release was extracted into.
            out: Directory to write into.

        Returns:
            The ``.pt`` artifact and its size.
        """
        return self._write_frame(load_frame(source), out)

    def _write_frame(self, frame: pd.DataFrame, out: Path) -> Artifact:
        """Write a loaded frame's signals as one tensor.

        Args:
            frame: The frame from :func:`~evaluations.source.load_frame`.
            out: Directory to write into.

        Returns:
            The ``.pt`` artifact and its size.
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

        return Artifact(format=self.name, path=path, size_bytes=directory_size(path))

    def read_all(self, path: Path) -> list[np.ndarray]:  # noqa: PLR6301 - implements the Format Protocol
        """Read every epoch back from the ``.pt`` file.

        Args:
            path: The file written by :meth:`write`.

        Returns:
            One array per epoch, shaped ``(n_channels, n_samples)``.
        """
        loaded = torch.load(path, weights_only=False)

        return list(loaded["signals"].numpy())
