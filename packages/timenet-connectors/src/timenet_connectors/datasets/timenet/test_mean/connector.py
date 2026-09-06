"""Synthetic offline connector for the ``timenet/test-mean`` classification demo.

Each record is one noisy signal. The signal is a constant offset plus Gaussian noise. The label is
``above_zero`` or ``below_zero`` by the sign of the offset. A classifier can recover the sign from
simple summary features. Offsets alternate by index, so the two-class set is exactly balanced. Every
value is seeded, so the dataset is deterministic and needs no network. The dataset has no raw source.
``download`` returns nothing and ``convert`` builds every record. This connector backs the end-to-end
training example.
"""

from pathlib import Path

import numpy as np

from timenet.connectors import BaseConnector
from timenet.dataset import TimeFDataset, TimeSeries
from timenet.dataset.axis import RegularAxis
from timenet.types import ClassificationTask, DataSource, TimeSeriesSpec, ureg


_N_RECORDS = 1000
_LENGTH = 64
_SAMPLING_RATE_HZ = 16.0
_NOISE_STD = 0.5
_SEED = 20260715  # fixed base seed keeps every build byte-for-byte identical

_SOURCE = DataSource(data_source_type="synthetic", name="Synthetic Generator", provider="TimeNet")
_SIGNAL = TimeSeriesSpec(
    spec_type="signal",
    name="Signal",
    unit_value=ureg.dimensionless,
    data_source=_SOURCE,
)


class TestMeanConnector(BaseConnector[None]):
    """A deterministic, offline connector for the ``timenet/test-mean`` classification demo.

    Synthetic: :meth:`download` returns no references and :meth:`convert` generates every record directly.
    """

    def download(self, cache_dir: Path) -> list[None]:  # noqa: ARG002, PLR6301 (synthetic: nothing to download)
        """Return no references; the data is synthetic and built entirely in :meth:`convert`.

        Args:
            cache_dir: Unused; nothing is fetched.

        Returns:
            An empty list.
        """
        return []

    def convert(self, raw_refs: list[None]) -> TimeFDataset:  # noqa: ARG002 (synthetic: nothing to convert)
        """Generate the balanced set of labeled, single-signal records.

        Args:
            raw_refs: Unused; the data is synthetic (``download`` returns nothing).

        Returns:
            The populated dataset: one single-signal series and one ``ClassificationTask`` per record.
        """
        dataset = TimeFDataset(metadata=self.metadata())
        rng = np.random.default_rng(_SEED)
        for index in range(_N_RECORDS):
            offset = float(rng.uniform(0.3, 1.5)) * (1.0 if index % 2 == 0 else -1.0)
            values = offset + rng.normal(0.0, _NOISE_STD, _LENGTH)
            series = TimeSeries.from_values(
                values,
                spec=_SIGNAL,
                signal="signal",
                time_axis=RegularAxis.from_rate_hz(16),
                source_id=f"rec-{index}",
                time_series_id=f"ts-{index}",
            )
            record = dataset.add_record(time_series=(series,), record_id=f"record-{index}")
            label = "above_zero" if offset > 0 else "below_zero"
            dataset.add_task(record, ClassificationTask(target=label, id=f"task-{index}"))
        return dataset


CONNECTOR = TestMeanConnector
