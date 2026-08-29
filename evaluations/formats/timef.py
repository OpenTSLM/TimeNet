"""The TimeF format: a version directory written through ``TimeFWriter``."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from evaluations.formats.base import Artifact, directory_size
from evaluations.source import LABEL, PATIENT, RATE_HZ, SIGNAL
from timenet.dataset import TimeFDataset, TimeSeries
from timenet.dataset.axis import RegularAxis
from timenet.reader import TimeFReader
from timenet.registry import DatasetVersion
from timenet.types import (
    ClassificationTask,
    DatasetMetadata,
    License,
    TimeSeriesSpec,
    Version,
    ureg,
)
from timenet.writer import TimeFWriter


DATASET_ID = "evaluations/sleep-edfx-epochs"
"""Id of the fixture this format writes. It is a benchmark artifact, never published."""
DATASET_VERSION = Version(1, 0, 0)
"""Fixed, so a rebuild lands in the same version directory."""
SPEC_TYPE = "eeg"
"""One modality covers every channel: they share a rate, a dtype and a unit."""


class TimeFFormat:
    """A TimeF version, derived from the same frame the pandas format writes."""

    name = "timef"

    def write(self, frame: pd.DataFrame, out: Path) -> Artifact:
        """Write the frame as a TimeF version.

        One sample per epoch, one ``TimeSeries`` per channel, and the scored stage as a
        ``ClassificationTask`` on that sample.

        Args:
            frame: The shared frame from ``evaluations.source.load_frame``.
            out: Directory to write into. The version lands beneath it.

        Returns:
            The version directory and its size.
        """
        out.mkdir(parents=True, exist_ok=True)
        spec = TimeSeriesSpec(spec_type=SPEC_TYPE, name="EEG", unit_value=ureg.microvolt)
        axis = RegularAxis.from_rate_hz(RATE_HZ)
        dataset = TimeFDataset(metadata=_metadata())

        for position, row in enumerate(frame.itertuples(index=False)):
            signal = np.asarray(getattr(row, SIGNAL), dtype=np.float32)
            series = tuple(
                TimeSeries.from_values(
                    signal[channel],
                    spec=spec,
                    channel=f"ch{channel}",
                    time_axis=axis,
                )
                for channel in range(signal.shape[0])
            )
            sample = dataset.add_sample(
                time_series=series,
                subject_ids=(str(getattr(row, PATIENT)),),
                sample_id=f"epoch-{position:08d}",
            )
            dataset.add_task(
                sample,
                ClassificationTask(target=str(getattr(row, LABEL)), target_schema="sleep_stage"),
            )

        dataset.derive_schema()
        with TimeFWriter(out, dataset) as writer:
            writer.write()

        version = out / DATASET_ID / str(DATASET_VERSION)

        return Artifact(format=self.name, path=version, size_bytes=directory_size(version))

    def read_all(self, path: Path) -> np.ndarray:  # noqa: PLR6301 - implements the Format Protocol
        """Read every epoch back from the TimeF version.

        Args:
            path: The version directory written by :meth:`write`.

        Returns:
            The signals, shaped ``(n_epochs, n_channels, n_samples)``.
        """
        with TimeFReader(DatasetVersion.open_local(path)) as reader:
            dataset = reader.read()

            return np.stack(
                [np.stack([series.to_numpy() for series in sample.time_series]) for sample in dataset.samples]
            )


def _metadata() -> DatasetMetadata:
    """Describe the fixture this format writes.

    Returns:
        Metadata for the benchmark artifact.
    """
    return DatasetMetadata(
        dataset_id=DATASET_ID,
        dataset_version=DATASET_VERSION,
        name="Sleep-EDF epochs (benchmark fixture)",
        description="PyHealth's 30-second epoch projection of Sleep-EDF, written as TimeF.",
        license=License.ODBL_1_0,
    )
