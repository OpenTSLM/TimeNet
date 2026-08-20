"""A synthetic, offline connector that tests every TimeF feature.

``HelloWorldConnector`` needs no network. It creates a fully deterministic dataset. The writer and
reader test suites use this dataset as a fixture for round-trip tests. The connector covers two
modalities, a shared data source, a series shared across samples, and a windowed sample. It also
covers a longer series. The writer tests split this longer series into chunks under a small chunk
cap. The connector covers all three annotation shapes, including one shared across samples, and a
task chain.
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
    """Compute a closed-form wave as an array.

    The function does not use a random number generator or perform I/O.

    Args:
        fn: The wave function to apply to the angular time base, for example ``np.sin``.
        n: The number of samples.
        phase: The phase offset in radians.

    Returns:
        The wave values as a float32 ``np.ndarray``.
    """
    t = np.arange(n, dtype=np.float64) / _SAMPLING_RATE_HZ
    return fn(2.0 * np.pi * t + phase).astype(np.float32)


def _wave(
    fn: Callable[[Float64[np.ndarray, " time"]], Float[np.ndarray, " time"]], n: int, phase: float
) -> Callable[[], pa.Array]:
    """Build a deterministic lazy loader for a closed-form wave.

    The connector uses this loader for the long series that the writer splits into chunks.

    Args:
        fn: The wave function to apply to the angular time base, for example ``np.sin``.
        n: The number of samples.
        phase: The phase offset in radians.

    Returns:
        A loader with no arguments. The loader returns the wave as a float32 Arrow array.
    """
    return lambda: pa.array(_wave_values(fn, n, phase))


class HelloWorldConnector(BaseConnector[HelloWorldRecording]):
    """A deterministic, offline demo connector for the ``timenet/hello-world`` dataset."""

    def download(self, cache_dir: Path) -> list[HelloWorldRecording]:  # noqa: ARG002, PLR6301 (override: the data is synthetic, so this method needs no cache)
        """Return deterministic recording descriptions.

        This method uses no network and does not use ``cache_dir``.

        Args:
            cache_dir: Not used. The data is synthetic.

        Returns:
            One short recording and one longer recording. The writer splits the longer recording into
            chunks under a small chunk cap.
        """
        return [HelloWorldRecording(index=0, n_values=16), HelloWorldRecording(index=1, n_values=512)]

    def convert(self, raw_refs: list[HelloWorldRecording]) -> TimeFDataset:
        """Build the complete dataset from the recording descriptions.

        The dataset includes every TimeF feature.

        Args:
            raw_refs: The recordings from :meth:`download`.

        Returns:
            The populated :class:`~timenet.dataset.TimeFDataset`.
        """
        dataset = TimeFDataset(metadata=self.metadata())
        short, long = raw_refs[0], raw_refs[1]

        # A series shared across two samples. This uses the dedupe-by-id path.
        shared = TimeSeries.from_values(
            _wave_values(np.sin, short.n_values, phase=0.0),
            spec=_SINE,
            channel="a",
            time_axis=_AXIS,
            source_id="rec-0",
            time_series_id="ts-shared",
        )
        # An annotation shared across two samples. This uses the dedupe-by-id path.
        cohort = Annotation(key="cohort", value="A", id="cohort-shared")

        # Sample 0: the full recording, with two modalities, all annotation shapes, and a task chain.
        cosine = TimeSeries.from_values(
            _wave_values(np.cos, short.n_values, phase=0.0),
            spec=_COSINE,
            channel="b",
            time_axis=_AXIS,
            source_id="rec-0",
            time_series_id="ts-cos-0",
        )
        sample0 = dataset.add_sample(time_series=(shared, cosine), subject_ids=("subj-0",), sample_id="sample-0")
        # These annotations have names. The localization task below can reference them by id instead
        # of repeating the literal values.
        stimulus = Annotation(key="stimulus", span=TimePoint.seconds(0.5), id="stim-0")
        artifact = Annotation(
            key="artifact",
            span=TimeInterval.seconds(0.0, 0.25, time_series_ids=(shared.time_series_id,)),
            id="art-0",
        )
        sample0.add_annotations(
            [
                Annotation(key="age", value=64, unit="years", id="age-0"),
                cohort,
                stimulus,
                artifact,
            ]
        )
        classification = ClassificationTask(target="normal", id="task-cls-0")
        dataset.add_tasks(
            sample0,
            [
                classification,
                AnswerTask(
                    prompt="What rhythm?",
                    target="Normal.",
                    # Any task can carry a chain of thought. An answer task with a chain of thought is
                    # the old reasoning task.
                    rationale="The peaks repeat once per cycle at a constant interval.",
                    # The cohort annotation gives context to the model. The model does not need to
                    # produce this annotation.
                    input_annotation_ids=(cohort.id,),
                    from_tasks=(classification,),
                    id="task-answer-0",
                ),
                ScalarPredictionTask(target=60.0, unit="bpm", target_name="mean_rate", id="task-scalar-0"),
                # Localization works backward from a normal task. The query is the input, and the
                # regions are the output. Here, the task stores the answer by reference, so the target
                # is the two temporal annotations above.
                TemporalLocalizationTask(
                    prompt="Locate the stimulus and the artifact.",
                    mode=LocalizationMode.SPARSE,
                    target_annotation_ids=(stimulus.id, artifact.id),
                    id="task-localize-0",
                ),
            ],
        )

        # Sample 1: this reuses the shared series and adds a longer series. The writer tests split the
        # longer series into chunks under a small chunk cap.
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
        sample1.add_annotation(cohort)  # same instance and id, so the annotation is shared

        # Sample 2: a windowed slice with a scoped classification task. This window covers the second
        # half of rec-0. This makes the window a genuine offset window, not a byte-identical prefix of
        # `ts-shared`. The phase offset continues the same wave, so the values match rec-0 over the
        # window.
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
        # A scope narrows the input to a region. This task has the same task type as the whole-sample
        # label above, but it supplies the window. Span times use the source recording timeline, so
        # this task's span sits inside the window's span.
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
