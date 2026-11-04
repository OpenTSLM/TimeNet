"""Synthetic offline connector for the ``timenet/test-mean`` classification demo.

Each record is one noisy signal. The signal is a constant offset plus Gaussian noise. The label is
``above_zero`` or ``below_zero`` by the sign of the offset. A classifier can recover the sign from
simple summary features. Offsets alternate by index, so the two-class set is exactly balanced. Every
value is seeded, so the dataset is deterministic and needs no network. The dataset has no raw source.
``download`` returns nothing and ``convert`` builds every record. This connector backs the end-to-end
training example.
"""

from enum import StrEnum
from pathlib import Path

import numpy as np

from timenet.connectors import BaseConnector
from timenet.dataset import Record, Signal, Source, TimeFDataset
from timenet.dataset.axis import RegularAxis
from timenet.types import ClassificationTask, LabelSet, TimeSeriesSpec, ureg


_N_RECORDS = 1000
_LENGTH = 64
_SAMPLING_RATE_HZ = 16.0
_NOISE_STD = 0.5
_SEED = 20260715  # fixed base seed keeps every build's values identical


class _MeanSign(StrEnum):
    """Whether the mean of a signal is above or below zero."""

    ABOVE_ZERO = "above_zero"
    BELOW_ZERO = "below_zero"


_MEAN_SIGN = LabelSet.from_enum(
    _MeanSign,
    name="mean_sign",
    descriptions={
        _MeanSign.ABOVE_ZERO: "The mean of the signal is above zero.",
        _MeanSign.BELOW_ZERO: "The mean of the signal is below zero.",
    },
)

_SIGNAL = TimeSeriesSpec(
    spec_type="signal",
    name="Signal",
    unit_value=ureg.dimensionless,
)


class TestMeanConnector(BaseConnector[None]):
    """A deterministic, offline connector for the ``timenet/test-mean`` classification demo.

    Synthetic: :meth:`download` returns no references and :meth:`convert` generates every record directly.
    """

    # ruff: ignore[unused-method-argument, no-self-use] (synthetic: nothing to download)
    def download(self, cache_dir: Path) -> list[None]:
        """Return no references; the data is synthetic and built entirely in :meth:`convert`.

        Args:
            cache_dir: Unused; nothing is fetched.

        Returns:
            An empty list.
        """
        return []

    # ruff: ignore[unused-method-argument] (synthetic: nothing to convert)
    def convert(self, raw_refs: list[None]) -> TimeFDataset:
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
            series = Signal.from_values(
                values,
                spec=_SIGNAL,
                name="signal",
                time_axis=RegularAxis.from_rate_hz(16),
            )
            record = Record(sources=(Source(name="Synthetic generator", signals=(series,)),))
            dataset.add_record(record=record)
            label = _MeanSign.ABOVE_ZERO if offset > 0 else _MeanSign.BELOW_ZERO
            dataset.add_task(
                task=ClassificationTask(
                    inputs=(record,),
                    targets=(label,),
                    label_set=_MEAN_SIGN,
                )
            )
        return dataset


CONNECTOR = TestMeanConnector
