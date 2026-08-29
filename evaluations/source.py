"""Read the release into a pandas DataFrame, the way a pandas or a torch user would."""

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

CASSETTE_STUDY = "sleep-cassette"
"""A directory of the release. Its parent is the root PyHealth reads."""


def load_frame(source: Path, *, batch_size: int = 64) -> pd.DataFrame:
    """Read Sleep-EDF into one pandas DataFrame, through PyHealth.

    Neither pandas nor torch can open an EDF file, so a reference loader stands between the
    release and the frame. This is that loader, and both formats call it inside their own
    ``write``, because what it costs is part of what those formats cost.

    Three steps happen here. ``SleepEDFDataset`` reads the metadata table, which carries a path
    per recording rather than any values. ``set_task`` parses each EDF and cuts it into
    30-second epochs, dropping those outside the six scored stages. Draining ``get_dataloader``
    is what pulls the signals into memory.

    Args:
        source: The directory the release was extracted into. The archive unpacks into a
            directory of its own beneath it, which this function finds.
        batch_size: How many epochs the dataloader yields at a time. It affects only how the
            values are drained, never what ends up in the frame.

    Returns:
        One row per epoch, with the columns ``signal``, ``label`` and ``patient_id``.

    Raises:
        EvaluationError: If ``source`` does not exist, or holds no release.
    """
    if not source.is_dir():
        raise EvaluationError(f"source directory does not exist: {source}")

    # The archive unpacks into a directory of its own, whose name the release states and this
    # suite does not. Find the study directory instead, and read the release from its parent.
    root = next((match.parent for match in source.rglob(CASSETTE_STUDY) if match.is_dir()), None)
    if root is None:
        raise EvaluationError(f"no {CASSETTE_STUDY!r} directory under {source}, so it holds no release")

    dataset = SleepEDFDataset(root=str(root))
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
