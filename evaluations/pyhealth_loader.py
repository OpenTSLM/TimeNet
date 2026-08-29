"""Read the release with PyHealth, the loader that stands in for pandas and torch."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from pyhealth.datasets import SleepEDFDataset
from pyhealth.tasks import SleepStagingSleepEDF

from evaluations.errors import EvaluationError


SIGNAL = "signal"
"""Column holding one epoch's values, shaped ``(n_channels, n_samples)``."""
LABEL = "label"
"""Column holding the scored sleep stage."""
PATIENT = "patient_id"
"""Column holding the subject the epoch came from."""

SUBSETS = ("cassette", "telemetry")
"""Both studies of the release. PyHealth reads one per instance, so the loader reads each."""

SUBJECT_TABLES = ("SC-subjects.xls", "ST-subjects.xls")
"""One table per study. The directory holding both of them is the root PyHealth reads."""


def load_frame(source: Path) -> pd.DataFrame:
    """Read every recording of Sleep-EDF into one pandas DataFrame, through PyHealth.

    Neither pandas nor torch can open an EDF file, so PyHealth stands between the release and the
    frame. Both formats call this inside their own ``write``, because what it costs is part of
    what those formats cost.

    ``SleepEDFDataset`` reads a subject table, which names a file per recording and holds no
    values. ``set_task`` then parses each EDF and cuts it into 30-second epochs, dropping those
    outside the six scored stages.

    Args:
        source: The directory the release was extracted into. The archive unpacks into a
            directory of its own beneath it, which this function finds.

    Returns:
        One row per epoch, with the columns ``signal``, ``label`` and ``patient_id``.

    Raises:
        EvaluationError: If ``source`` does not exist, or holds no release.
    """
    if not source.is_dir():
        raise EvaluationError(f"source directory does not exist: {source}")

    root = _release_root(source)
    rows = [row for subset in SUBSETS for row in _read_subset(root, subset)]

    return pd.DataFrame(rows)


def signal_stack(frame: pd.DataFrame) -> np.ndarray:
    """Stack the frame's signal column into one array.

    Args:
        frame: A frame from :func:`load_frame`.

    Returns:
        The signals, shaped ``(n_epochs, n_channels, n_samples)``, as ``float32``.
    """
    return np.stack(frame[SIGNAL].to_numpy()).astype(np.float32, copy=False)


def _read_subset(root: Path, subset: str) -> list[dict[str, object]]:
    """Read one study of the release into per-epoch rows.

    The epochs are read one at a time, as ``set_task`` produced them. Nothing batches them,
    because a batch would be split apart again here, and collating pads a short tensor to the
    length of the longest one in its batch. This suite stores what the release holds.

    Args:
        root: The directory holding both subject tables.
        subset: Which study to read, ``cassette`` or ``telemetry``.

    Returns:
        One dict per epoch, holding only the columns the frame keeps.
    """
    samples = SleepEDFDataset(root=str(root), subset=subset).set_task(SleepStagingSleepEDF())

    return [
        {
            SIGNAL: np.asarray(sample[SIGNAL], dtype=np.float32),
            LABEL: str(sample[LABEL]),
            PATIENT: str(sample.get(PATIENT, "")),
        }
        for sample in samples
    ]


def _release_root(source: Path) -> Path:
    """Find the directory the release unpacked into.

    Args:
        source: The directory the release was extracted into.

    Returns:
        The directory holding both subject tables.

    Raises:
        EvaluationError: If no directory beneath ``source`` holds both of them.
    """
    first, *rest = SUBJECT_TABLES
    for match in sorted(source.rglob(first)):
        if all((match.parent / table).is_file() for table in rest):
            return match.parent

    raise EvaluationError(f"no directory under {source} holds {' and '.join(SUBJECT_TABLES)}, so it holds no release")
