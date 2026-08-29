"""The torch format: tensors saved with ``torch.save``."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from evaluations.formats.base import Artifact, directory_size
from evaluations.source import LABEL, PATIENT, signal_stack


class TorchFormat:
    """A ``.pt`` file, derived from the same frame the pandas format writes."""

    name = "torch"

    def write(self, frame: pd.DataFrame, out: Path) -> Artifact:
        """Write the frame's signals as one tensor.

        Args:
            frame: The shared frame from ``evaluations.source.load_frame``.
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

    def read_all(self, path: Path) -> np.ndarray:  # noqa: PLR6301 - implements the Format Protocol
        """Read every epoch back from the ``.pt`` file.

        Args:
            path: The file written by :meth:`write`.

        Returns:
            The signals, shaped ``(n_epochs, n_channels, n_samples)``.
        """
        loaded = torch.load(path, weights_only=False)

        return loaded["signals"].numpy()
