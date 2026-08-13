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

from jaxtyping import Float, Float64
import numpy as np
import pyarrow as pa

from timenet.connectors import BaseConnector
from timenet.dataset import TimeFDataset, TimeSeries
from timenet.dataset.axis import RegularAxis
from timenet.types import (
    Annotation,
    AnswerTask,
    ClassificationTask,
    DataSource,
    LocalizationMode,
    ScalarPredictionTask,
    TemporalLocalizationTask,
    TimeInterval,
    TimePoint,
    TimeSeriesSpec,
    ureg,
)


_SAMPLING_RATE_HZ = 16.0
_AXIS = RegularAxis.from_rate_hz(16)
_SOURCE = DataSource(data_source_type="synthetic", name="Synthetic Generator", provider="TimeNet")
_SINE = TimeSeriesSpec(
    spec_type="sine",
    name="Sine",
    unit_value=ureg.dimensionless,
    data_source=_SOURCE,
)
_COSINE = TimeSeriesSpec(
    spec_type="cosine",
    name="Cosine",
    unit_value=ureg.dimensionless,
    data_source=_SOURCE,
)


@dataclass(frozen=True)
class HelloWorldRecording:
    """A lightweight, deterministic description of one synthetic recording."""

    index: int
    n_values: int


def _wave_values(
    fn: Callable[[Float64[np.ndarray, " time"]], Float[np.ndarray, " time"]], n: int, phase: float
) -> Float[np.ndarray, " time"]:
    """Compute a closed-form wave as an array (no RNG, no I/O).

    Args:
        fn: The wave function applied to the angular time base (e.g. ``np.sin``).
        n: Number of samples.
        phase: Phase offset in radians.

    Returns:
        The wave values as a float32 ``np.ndarray``.
    """
    t = np.arange(n, dtype=np.float64) / _SAMPLING_RATE_HZ
    return fn(2.0 * np.pi * t + phase).astype(np.float32)


def _wave(
    fn: Callable[[Float64[np.ndarray, " time"]], Float[np.ndarray, " time"]], n: int, phase: float
) -> Callable[[], pa.Array]:
    """Build a deterministic lazy loader for a closed-form wave (used for the chunk-split long series).

    Args:
        fn: The wave function applied to the angular time base (e.g. ``np.sin``).
        n: Number of samples.
        phase: Phase offset in radians.

    Returns:
        A no-argument loader returning the wave as a float32 Arrow array.
    """
    return lambda: pa.array(_wave_values(fn, n, phase))


class HelloWorldConnector(BaseConnector[HelloWorldRecording]):
    """A deterministic, offline demo connector for the ``timenet/hello-world`` dataset."""

    def download(self, cache_dir: Path) -> list[HelloWorldRecording]:  # noqa: ARG002, PLR6301 (override; synthetic: no cache)
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
        shared = TimeSeries.from_values(
            _wave_values(np.sin, short.n_values, phase=0.0),
            spec=_SINE,
            channel="a",
            time_axis=_AXIS,
            source_id="rec-0",
            time_series_id="ts-shared",
        )
        # An annotation shared across two samples (dedupe-by-id path).
        cohort = Annotation(key="cohort", value="A", id="cohort-shared")

        # Sample 0: full recording, two modalities, all annotation shapes, a task chain.
        cosine = TimeSeries.from_values(
            _wave_values(np.cos, short.n_values, phase=0.0),
            spec=_COSINE,
            channel="b",
            time_axis=_AXIS,
            source_id="rec-0",
            time_series_id="ts-cos-0",
        )
        sample0 = dataset.add_sample(time_series=(shared, cosine), subject_ids=("subj-0",), sample_id="sample-0")
        sample0.add_annotation(Annotation(key="age", value=64, unit="years", id="age-0"))
        sample0.add_annotation(cohort)
        sample0.add_annotation(Annotation(key="stimulus", span=TimePoint.seconds(0.5), id="stim-0"))
        sample0.add_annotation(
            Annotation(
                key="artifact",
                span=TimeInterval.seconds(0.0, 0.25, time_series_ids=(shared.time_series_id,)),
                id="art-0",
            )
        )
        classification = dataset.add_task(sample0, ClassificationTask(target="normal", id="task-cls-0"))
        dataset.add_task(
            sample0,
            AnswerTask(
                prompt="What rhythm?",
                target="Normal.",
                # Any task may carry a chain of thought; an answer task with one is the old reasoning task.
                rationale="The peaks repeat once per cycle at a constant interval.",
                # The cohort annotation is context the model reads, not something it has to produce.
                input_annotation_ids=("cohort-shared",),
                id="task-answer-0",
                from_tasks=(classification,),
            ),
        )
        dataset.add_task(
            sample0,
            ScalarPredictionTask(target=60.0, unit="bpm", target_name="mean_rate", id="task-scalar-0"),
        )
        # Localization runs a scope backwards: the query goes in and the regions come out. Here the
        # answer is stored by reference, so the target *is* the two temporal annotations above.
        dataset.add_task(
            sample0,
            TemporalLocalizationTask(
                prompt="Locate the stimulus and the artifact.",
                mode=LocalizationMode.SPARSE,
                target_annotation_ids=("stim-0", "art-0"),
                id="task-localize-0",
            ),
        )

        # Sample 1: reuses the shared series plus a longer series (the writer tests split it under a tiny cap).
        long_series = TimeSeries(
            spec=_SINE,
            channel="a",
            time_axis=_AXIS,
            loader=_wave(np.sin, long.n_values, phase=1.0),
            source_id="rec-1",
            time_series_id="ts-long-1",
            n_values=long.n_values,
        )
        sample1 = dataset.add_sample(time_series=(shared, long_series), subject_ids=("subj-1",), sample_id="sample-1")
        sample1.add_annotation(cohort)  # same instance/id => shared

        # Sample 2: a windowed slice with a scoped classification task. It covers the *second* half of
        # rec-0, so it is a genuine offset window rather than a byte-identical prefix of `ts-shared`. The
        # phase offset continues the same wave, so the values match rec-0 over the window.
        window_start = short.n_values // 2
        window = TimeSeries.from_values(
            _wave_values(np.sin, short.n_values - window_start, phase=2.0 * np.pi * window_start / _SAMPLING_RATE_HZ),
            spec=_SINE,
            channel="a",
            time_axis=_AXIS.at_index(window_start),
            source_id="rec-0",
            time_series_id="ts-window-2",
        )
        sample2 = dataset.add_sample(time_series=(window,), subject_ids=("subj-0",), sample_id="sample-2")
        # A scope narrows the input to a region: same task type as the whole-sample label above, with the
        # window supplied. Span times are in the source recording timeline, so this sits inside the
        # window's span.
        dataset.add_task(
            sample2,
            ClassificationTask(
                target="onset",
                id="task-cls-2",
                scope=TimeInterval.seconds(0.5, 0.75, time_series_ids=(window.time_series_id,)),
            ),
        )
        return dataset


CONNECTOR = HelloWorldConnector
