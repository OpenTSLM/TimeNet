"""Load the dataset into memory once, as the one frame every format writes from."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pyhealth.datasets import SleepEDFDataset
from pyhealth.datasets.utils import get_dataloader
from pyhealth.tasks import SleepStagingSleepEDF

from evaluations.errors import EvaluationError


SIGNAL = "signal"
"""Column holding one epoch's values, shaped ``(n_channels, n_samples)``."""
LABEL = "label"
"""Column holding the scored sleep stage."""
PATIENT = "patient_id"
"""Column holding the subject the epoch came from."""

RATE_HZ = 100
"""The rate PyHealth resamples every channel to before it cuts epochs."""


def load_frame(source: Path, *, batch_size: int = 64) -> pd.DataFrame:
    """Load Sleep-EDF into one pandas DataFrame, through PyHealth.

    This frame is the single in-memory object of the whole run. ``PandasFormat`` writes it
    directly; ``TorchFormat`` and ``TimeFFormat`` derive their artifacts from it. Loading once is
    what lets the three artifacts hold the same content without a check.

    Three steps happen here. ``SleepEDFDataset`` reads the metadata table, which carries a path
    per recording rather than any values. ``set_task`` parses each EDF and cuts it into
    30-second epochs, dropping those outside the six scored stages. Draining ``get_dataloader``
    is what pulls the signals into memory.

    Args:
        source: Directory holding the Sleep-EDF recordings.
        batch_size: How many epochs the dataloader yields at a time. It affects only how the
            values are drained, never what ends up in the frame.

    Returns:
        One row per epoch, with the columns ``signal``, ``label`` and ``patient_id``.

    Raises:
        EvaluationError: If ``source`` does not exist.
    """
    if not source.is_dir():
        raise EvaluationError(f"source directory does not exist: {source}")

    dataset = SleepEDFDataset(root=str(source))
    samples = dataset.set_task(SleepStagingSleepEDF())
    loader = get_dataloader(samples, batch_size=batch_size, shuffle=False)

    rows: list[dict[str, Any]] = []
    for batch in loader:
        rows.extend(unbatch(batch))

    return pd.DataFrame(rows)


def unbatch(batch: dict[str, Any]) -> list[dict[str, Any]]:
    """Split one collated batch back into per-epoch rows.

    The dataloader collates a batch into one dict of stacked columns. The frame wants one row
    per epoch, so the columns are zipped back apart.

    Args:
        batch: One batch as the dataloader yields it.

    Returns:
        One dict per epoch in the batch, holding only the columns the frame keeps.

    Raises:
        EvaluationError: If the batch's columns disagree on how many epochs they hold.
    """
    signals = _to_numpy(batch[SIGNAL])
    labels = _as_list(batch[LABEL])
    patients = _as_list(batch.get(PATIENT, [""] * len(labels)))

    if not len(signals) == len(labels) == len(patients):
        raise EvaluationError(
            f"batch columns disagree on epoch count: "
            f"{SIGNAL}={len(signals)}, {LABEL}={len(labels)}, {PATIENT}={len(patients)}"
        )

    return [
        {SIGNAL: np.asarray(signal, dtype=np.float32), LABEL: str(label), PATIENT: str(patient)}
        for signal, label, patient in zip(signals, labels, patients, strict=True)
    ]


def signal_stack(frame: pd.DataFrame) -> np.ndarray:
    """Stack the frame's signal column into one array.

    Args:
        frame: A frame from :func:`load_frame`.

    Returns:
        The signals, shaped ``(n_epochs, n_channels, n_samples)``, as ``float32``.
    """
    return np.stack(frame[SIGNAL].to_numpy()).astype(np.float32, copy=False)


def _to_numpy(values: Any) -> np.ndarray:
    """Convert a batch column to numpy, whether it arrived as a tensor or an array.

    Args:
        values: One collated column.

    Returns:
        The column as a numpy array.
    """
    if hasattr(values, "detach"):
        return values.detach().cpu().numpy()

    return np.asarray(values)


def _as_list(values: Any) -> list[Any]:
    """Convert a batch column to a plain list.

    Args:
        values: One collated column.

    Returns:
        The column as a list, one entry per epoch.
    """
    if hasattr(values, "tolist"):
        return list(values.tolist())

    return list(values)
