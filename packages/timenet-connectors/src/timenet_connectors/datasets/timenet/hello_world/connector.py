"""A synthetic, offline connector that tests every TimeF feature.

``HelloWorldConnector`` needs no network. It creates a fully deterministic dataset. The writer and
reader test suites use this dataset as a fixture for round-trip tests. The connector covers two
modalities, explicit Sources, independent signals, and a windowed record. It also
covers a longer series. The writer tests split this longer series into chunks under a small chunk
cap. The connector covers all three annotation shapes, including one shared across records, and a
task chain.
"""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from jaxtyping import Float, Float64
import numpy as np
import pyarrow as pa

from timenet.connectors import BaseConnector
from timenet.dataset import Record, Signal, Source, TimeFDataset
from timenet.dataset.axis import RegularAxis
from timenet.types import (
    Annotation,
    AnswerTask,
    ClassificationTask,
    InputModality,
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
_SINE = TimeSeriesSpec(
    spec_type="sine",
    name="Sine",
    unit_value=ureg.dimensionless,
)
_COSINE = TimeSeriesSpec(
    spec_type="cosine",
    name="Cosine",
    unit_value=ureg.dimensionless,
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

        # Record 0's sine signal.
        shared = Signal.from_values(
            _wave_values(np.sin, raw_refs[0].n_values, phase=0.0),
            spec=_SINE,
            name="a",
            time_axis=_AXIS,
            source_id="rec-0",
            id="ts-shared",
        )
        # An annotation shared across two records. This uses the dedupe-by-id path.
        cohort = Annotation(key="cohort", value="A", id="cohort-shared")

        # Record 0: the full recording, with two modalities, all annotation shapes, and a task chain.
        cosine = Signal.from_values(
            _wave_values(np.cos, raw_refs[0].n_values, phase=0.0),
            spec=_COSINE,
            name="b",
            time_axis=_AXIS,
            source_id="rec-0",
            id="ts-cos-0",
        )
        record0 = Record(
            record_id="record-0",
            sources=(
                Source(
                    id="source-record-0",
                    name="Synthetic generator",
                    signals=(shared, cosine),
                ),
            ),
        )
        dataset.add_record(record=record0)
        # These annotations have names. The localization task below can reference them by id instead
        # of repeating the literal values.
        stimulus = Annotation(key="stimulus", span=TimePoint.seconds(0.5), id="stim-0")
        artifact = Annotation(
            key="artifact",
            span=TimeInterval.seconds(0.0, 0.25, time_series_ids=(shared.id,)),
            id="art-0",
        )
        _, record0_cohort, stimulus, artifact = record0.add_annotations(
            [
                Annotation(key="age", value=64, unit="years", id="age-0"),
                cohort,
                stimulus,
                artifact,
            ]
        )
        classification = ClassificationTask(
            input_modalities=frozenset({InputModality.TIME_SERIES}),
            inputs=(record0,),
            targets=("normal",),
            id="task-cls-0",
        )
        dataset.add_tasks(
            tasks=[
                classification,
                AnswerTask(
                    input_modalities=frozenset({InputModality.TEXT, InputModality.TIME_SERIES}),
                    inputs=(record0,),
                    prompt="What rhythm?",
                    targets=("Normal.",),
                    # Any task can carry a chain of thought. An answer task with a chain of thought is
                    # the old reasoning task.
                    rationale="The peaks repeat once per cycle at a constant interval.",
                    # The cohort annotation gives context to the model. The model does not need to
                    # produce this annotation.
                    input_annotations=(record0_cohort,),
                    from_tasks=(classification,),
                    id="task-answer-0",
                ),
                ScalarPredictionTask(
                    input_modalities=frozenset({InputModality.TIME_SERIES}),
                    inputs=(record0,),
                    targets=(60.0,),
                    unit="bpm",
                    target_name="mean_rate",
                    id="task-scalar-0",
                ),
                # Localization works backward from a normal task. The query is the input, and the
                # regions are the output. Here, the task stores the answer by reference, so the target
                # is the two temporal annotations above.
                TemporalLocalizationTask(
                    input_modalities=frozenset({InputModality.TEXT, InputModality.TIME_SERIES}),
                    inputs=(record0,),
                    prompt="Locate the stimulus and the artifact.",
                    mode=LocalizationMode.SPARSE,
                    target_annotations=(stimulus, artifact),
                    id="task-localize-0",
                ),
            ],
        )

        # Record 1: a short signal and a longer signal. The writer tests split the longer signal into
        # chunks under a small chunk cap.
        long_series = Signal.from_loader(
            spec=_SINE,
            name="a",
            time_axis=_AXIS,
            loader=_wave(np.sin, raw_refs[1].n_values, phase=1.0),
            source_id="rec-1",
            id="ts-long-1",
            n_values=raw_refs[1].n_values,
        )
        record1 = Record(
            record_id="record-1",
            sources=(
                Source(
                    id="source-record-1",
                    name="Synthetic generator",
                    signals=(
                        Signal.from_values(
                            _wave_values(np.sin, raw_refs[0].n_values, phase=0.0),
                            spec=_SINE,
                            name="a",
                            time_axis=_AXIS,
                            source_id="rec-1",
                            id="ts-short-1",
                        ),
                        long_series,
                    ),
                ),
            ),
        )
        dataset.add_record(record=record1)
        record1.annotate(cohort)  # same instance and id, so the annotation is shared

        # Record 2: a windowed slice with a scoped classification task. This window covers the second
        # half of rec-0. This makes the window a genuine offset window, not a byte-identical prefix of
        # `ts-shared`. The phase offset continues the same wave, so the values match rec-0 over the
        # window.
        window_start = raw_refs[0].n_values // 2
        window = Signal.from_values(
            _wave_values(
                np.sin,
                raw_refs[0].n_values - window_start,
                phase=2.0 * np.pi * window_start / _SAMPLING_RATE_HZ,
            ),
            spec=_SINE,
            name="a",
            time_axis=_AXIS.at_index(window_start),
            source_id="rec-0",
            id="ts-window-2",
        )
        record2 = Record(
            record_id="record-2",
            sources=(
                Source(
                    id="source-record-2",
                    name="Synthetic generator",
                    signals=(window,),
                ),
            ),
        )
        dataset.add_record(record=record2)
        # A scope narrows the input to a region. This task has the same task type as the whole-record
        # label above, but it supplies the window. Span times use the source recording timeline, so
        # this task's span sits inside the window's span.
        dataset.add_task(
            task=ClassificationTask(
                input_modalities=frozenset({InputModality.TIME_SERIES}),
                inputs=(record2,),
                targets=("onset",),
                id="task-cls-2",
                scope=TimeInterval.seconds(0.5, 0.75, time_series_ids=(window.id,)),
            ),
        )
        return dataset


CONNECTOR = HelloWorldConnector
