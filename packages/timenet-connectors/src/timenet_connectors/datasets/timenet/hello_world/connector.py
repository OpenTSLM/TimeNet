"""A synthetic, offline connector that exercises every TimeF feature.

``HelloWorldConnector`` needs no network and produces a fully deterministic dataset, so it doubles as the
fixture the writer and reader test suites round-trip against. It covers two modalities, a shared data
source, a series shared across samples, a windowed sample, a longer series (which the writer tests split
into chunks under a small chunk cap), all three annotation shapes (including one shared across samples),
and a task chain.
"""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa

from timenet.connectors import BaseConnector
from timenet.dataset import TimeFDataset, TimeSeries
from timenet.types import (
    ClassificationTask,
    DataSource,
    IntervalAnnotation,
    LabelingTask,
    PointAnnotation,
    QATask,
    StaticAnnotation,
    TimeSeriesSpec,
    View,
    ureg,
)


_SAMPLING_RATE_HZ = 16.0
_SOURCE = DataSource(data_source_type="synthetic", name="Synthetic Generator", provider="TimeNet")
_SINE = TimeSeriesSpec(
    spec_type="sine",
    name="Sine",
    unit_sampling_rate=ureg.hertz,
    unit_timestamp=ureg.second,
    unit_value=ureg.dimensionless,
    data_source=_SOURCE,
)
_COSINE = TimeSeriesSpec(
    spec_type="cosine",
    name="Cosine",
    unit_sampling_rate=ureg.hertz,
    unit_timestamp=ureg.second,
    unit_value=ureg.dimensionless,
    data_source=_SOURCE,
)


@dataclass(frozen=True)
class HelloWorldRecording:
    """A lightweight, deterministic description of one synthetic recording."""

    index: int
    n_values: int


def _wave(fn: Callable[[np.ndarray], np.ndarray], n: int, phase: float) -> Callable[[], pa.Array]:
    """Build a deterministic loader for a closed-form wave (no RNG, no I/O).

    Args:
        fn: The wave function applied to the angular time base (e.g. ``np.sin``).
        n: Number of samples.
        phase: Phase offset in radians.

    Returns:
        A no-argument loader returning the wave as a float32 Arrow array.
    """

    def load() -> pa.Array:
        t = np.arange(n, dtype=np.float64) / _SAMPLING_RATE_HZ
        return pa.array(fn(2.0 * np.pi * t + phase).astype(np.float32))

    return load


class HelloWorldConnector(BaseConnector[HelloWorldRecording]):
    """A deterministic, offline demo connector for the ``timenet/hello-world`` dataset."""

    def download(self, cache_dir: Path) -> list[HelloWorldRecording]:  # noqa: ARG002 (synthetic: no cache needed)
        """Return deterministic recording descriptions (no network, ``cache_dir`` unused).

        Args:
            cache_dir: Ignored; the data is synthetic.

        Returns:
            One short recording and one longer one (the writer splits the longer one under a small chunk cap).
        """
        return [HelloWorldRecording(index=0, n_values=16), HelloWorldRecording(index=1, n_values=512)]

    def convert(self, raw_refs: list[HelloWorldRecording]) -> TimeFDataset:
        """Build the feature-complete dataset from the recording descriptions.

        Args:
            raw_refs: The recordings from :meth:`download`.

        Returns:
            The populated :class:`~timenet.dataset.TimeFDataset`.
        """
        dataset = TimeFDataset(metadata=self.metadata())
        short, long = raw_refs[0], raw_refs[1]

        # A series shared across two samples (dedupe-by-id path).
        shared = TimeSeries(
            spec=_SINE,
            channel="a",
            sampling_rate_hz=_SAMPLING_RATE_HZ,
            loader=_wave(np.sin, short.n_values, phase=0.0),
            source_id="rec-0",
            time_series_id="ts-shared",
            t_start_s=0.0,
            t_end_s=short.n_values / _SAMPLING_RATE_HZ,
        )
        # An annotation shared across two samples (dedupe-by-id path).
        cohort = StaticAnnotation(key="cohort", value="A", id="cohort-shared")

        # Sample 0: full recording, two modalities, all annotation shapes, a task chain.
        cosine = TimeSeries(
            spec=_COSINE,
            channel="b",
            sampling_rate_hz=_SAMPLING_RATE_HZ,
            loader=_wave(np.cos, short.n_values, phase=0.0),
            source_id="rec-0",
            time_series_id="ts-cos-0",
            t_start_s=0.0,
            t_end_s=short.n_values / _SAMPLING_RATE_HZ,
        )
        sample0 = dataset.add_sample(
            time_series=(shared, cosine), view=View.FULL, subject_ids=("subj-0",), sample_id="sample-0"
        )
        sample0.add_annotation(StaticAnnotation(key="age", value=64, unit="years", id="age-0"))
        sample0.add_annotation(cohort)
        sample0.add_annotation(PointAnnotation(key="stimulus", start_time_s=0.5, id="stim-0"))
        sample0.add_annotation(
            IntervalAnnotation(
                key="artifact", start_time_s=0.0, end_time_s=0.25, time_series_ids=(shared.time_series_id,), id="art-0"
            )
        )
        classification = dataset.add_task(sample0, ClassificationTask(target="normal", id="task-cls-0"))
        dataset.add_task(
            sample0,
            QATask(question="What rhythm?", target="Normal.", id="task-qa-0"),
            from_tasks=(classification,),
        )

        # Sample 1: reuses the shared series plus a longer series (the writer tests split it under a tiny cap).
        long_series = TimeSeries(
            spec=_SINE,
            channel="a",
            sampling_rate_hz=_SAMPLING_RATE_HZ,
            loader=_wave(np.sin, long.n_values, phase=1.0),
            source_id="rec-1",
            time_series_id="ts-long-1",
            t_start_s=0.0,
            t_end_s=long.n_values / _SAMPLING_RATE_HZ,
        )
        sample1 = dataset.add_sample(
            time_series=(shared, long_series), view=View.FULL, subject_ids=("subj-1",), sample_id="sample-1"
        )
        sample1.add_annotation(cohort)  # same instance/id => shared

        # Sample 2: a windowed slice with a labeling task. It covers the *second* half of rec-0, so it
        # is a genuine offset window rather than a byte-identical prefix of `ts-shared`. The phase
        # offset continues the same wave, so the values match rec-0 over [t_start_s, t_end_s).
        window_start = short.n_values // 2
        window = TimeSeries(
            spec=_SINE,
            channel="a",
            sampling_rate_hz=_SAMPLING_RATE_HZ,
            loader=_wave(np.sin, short.n_values - window_start, phase=2.0 * np.pi * window_start / _SAMPLING_RATE_HZ),
            source_id="rec-0",
            time_series_id="ts-window-2",
            t_start_s=window_start / _SAMPLING_RATE_HZ,
            t_end_s=short.n_values / _SAMPLING_RATE_HZ,
        )
        sample2 = dataset.add_sample(
            time_series=(window,), view=View.WINDOW, subject_ids=("subj-0",), sample_id="sample-2"
        )
        dataset.add_task(
            sample2,
            LabelingTask(
                target="onset",
                time_series_ids=(window.time_series_id,),
                # windows_s is in the source recording timeline, so it sits inside the window's span.
                windows_s=((0.5, 0.75),),
                id="task-lbl-2",
            ),
        )
        return dataset


CONNECTOR = HelloWorldConnector
