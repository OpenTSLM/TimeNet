"""Build a deterministic, diverse synthetic TimeF benchmark corpus."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pyarrow as pa

from timenet.dataset import Sample, TimeFDataset, TimeSeries
from timenet.types import (
    AnswerTask,
    ClassificationTask,
    DatasetMetadata,
    DataSource,
    Domain,
    ForecastingTask,
    IntervalAnnotation,
    License,
    PointAnnotation,
    Span,
    StaticAnnotation,
    TimeSeriesSpec,
    Version,
    View,
    ureg,
)


@dataclass(frozen=True)
class Scenario:
    """One deterministic workload in the synthetic corpus."""

    name: str
    channels: tuple[str, ...]
    sampling_rate_hz: float
    steps: int
    unit: str


SCENARIOS = (
    Scenario("vibration", ("radial",), 12_800.0, 16_384, "meter / second ** 2"),
    Scenario(
        "ecg",
        tuple(f"lead-{lead}" for lead in ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")),
        500.0,
        5_000,
        "millivolt",
    ),
    Scenario("sleep", ("eeg", "eog-left", "eog-right", "emg", "ppg"), 128.0, 15_360, "microvolt"),
    Scenario("accelerometer", ("x", "y", "z"), 100.0, 4_096, "meter / second ** 2"),
    Scenario("finance", ("open", "high", "low", "close", "volume"), 1.0, 2_048, "dimensionless"),
    Scenario("workout", ("heart-rate", "pace", "cadence", "power"), 1.0, 7_200, "dimensionless"),
    Scenario("energy", ("load", "temperature", "solar"), 0.25, 4_096, "dimensionless"),
    Scenario("automotive", ("rpm", "torque", "coolant", "vibration"), 200.0, 8_192, "dimensionless"),
)
"""The eight portable workloads. All use scalar float32 values for cross-backend runs."""


_SOURCE = DataSource(data_source_type="synthetic-benchmark", name="Closed-form generator", provider="TimeNet")


def _loader(values: np.ndarray) -> Callable[[], pa.Array]:
    """Wrap numeric values in a stable float32 Arrow loader.

    Returns:
        A no-argument loader returning the cached array.
    """
    array = pa.array(values, type=pa.float32())
    return lambda: array


def _values(scenario_index: int, channel_index: int, steps: int, scale: int) -> np.ndarray:
    """Return deterministic, nontrivial float32 values for one channel."""
    count = steps * scale
    t = np.arange(count, dtype=np.float64)
    slow = np.sin((scenario_index + 1) * t / 97.0 + channel_index / 3.0)
    fast = np.cos((channel_index + 2) * t / 17.0)
    trend = ((t % 1_009) / 1_009.0) * (scenario_index + 1) / 10.0
    impulses = ((t.astype(np.int64) + 31 * channel_index) % (211 + scenario_index * 17) == 0) * 0.5
    return (slow + 0.2 * fast + trend + impulses).astype(np.float32)


def _scalar_series(
    scenario: Scenario,
    scenario_index: int,
    channel: str,
    channel_index: int,
    scale: int,
) -> TimeSeries:
    """Construct one portable scalar float32 series.

    Returns:
        The fully described series.
    """
    spec = TimeSeriesSpec(
        spec_type=scenario.name,
        name=scenario.name.replace("-", " ").title(),
        unit_sampling_rate=ureg.hertz,
        unit_timestamp=ureg.second,
        unit_value=ureg.Unit(scenario.unit),
        data_source=_SOURCE,
    )
    values = pa.array(_values(scenario_index, channel_index, scenario.steps, scale), type=pa.float32())
    return TimeSeries(
        loader=lambda: values,
        spec=spec,
        channel=channel,
        sampling_rate_hz=scenario.sampling_rate_hz,
        source_id=f"{scenario.name}-recording",
        time_series_id=f"{scenario.name}-{channel}",
        t_end_s=len(values) / scenario.sampling_rate_hz,
    )


def _add_tasks(dataset: TimeFDataset, samples: dict[str, Sample]) -> None:
    """Attach every currently supported task payload to representative scenarios."""
    vibration = samples["vibration"]
    ecg = samples["ecg"]
    sleep = samples["sleep"]
    accelerometer = samples["accelerometer"]
    finance = samples["finance"]
    workout = samples["workout"]
    energy = samples["energy"]
    automotive = samples["automotive"]

    dataset.add_task(
        vibration,
        ClassificationTask(
            target="outer-race-fault",
            target_schema="condition",
            id="task-vibration-class",
        ),
    )
    dataset.add_task(
        ecg,
        ClassificationTask(
            target="atrial-fibrillation",
            target_schema="rhythm",
            id="task-ecg-class",
        ),
    )
    dataset.add_task(
        sleep,
        ClassificationTask(
            target="N2",
            target_schema="sleep-stage",
            scope=Span(start_s=30.0, end_s=60.0),
            id="task-sleep-label",
        ),
    )
    dataset.add_task(
        accelerometer,
        AnswerTask(
            target="A trace with rising amplitude and a periodic impact after five seconds.",
            id="task-accelerometer-caption",
        ),
    )
    dataset.add_task(
        finance,
        AnswerTask(
            prompt="Summarize the session.",
            target="Choppy open, midday rally, positive close.",
            id="task-finance-qa",
        ),
    )
    dataset.add_task(
        workout,
        AnswerTask(
            prompt="Assess this workout.",
            target="Aerobic base session",
            rationale="Heart-rate drift appears late while pace remains stable.",
            id="task-workout-reasoning",
        ),
    )
    dataset.add_task(
        energy,
        ForecastingTask(
            context_sample_ids=(energy.sample_id,),
            target_sample_id=energy.sample_id,
            id="task-energy-forecast",
        ),
    )
    dataset.add_task(
        automotive,
        AnswerTask(
            prompt="Estimate remaining useful life.",
            target="74 cycles",
            rationale="Vibration rises while torque efficiency falls.",
            id="task-automotive-reasoning",
        ),
    )


def _add_connector_patterns(dataset: TimeFDataset, samples: dict[str, Sample], scale: int) -> None:
    """Mirror the cardinality patterns that distinguish the real connectors.

    ECG-QA reuses one long 12-lead recording across several text tasks, TSQA contains many short
    independently stored series, and test-mean contains many tiny labeled series. These patterns have
    measurably different control-plane and random-access costs even when their total value bytes match.
    """
    ecg = samples["ecg"]
    for index in range(1, 8 * scale):
        sample = dataset.add_sample(
            time_series=ecg.time_series,
            sample_id=f"sample-ecg-question-{index:03d}",
            subject_ids=("subject-ecg",),
            view=View.FULL,
        )
        sample.add_annotation(StaticAnnotation(key="scenario", value="ecg", id=f"annotation-ecg-{index:03d}"))
        dataset.add_task(
            sample,
            AnswerTask(
                prompt=f"Is rhythm abnormal in view {index}?",
                target="atrial-fibrillation",
                rationale="The synthetic rhythm has repeatable irregular intervals.",
                id=f"task-ecg-reasoning-{index:03d}",
            ),
        )

    finance_spec = samples["finance"].time_series[0].spec
    for index in range(64 * scale):
        channel_count = 1 + index % 3
        length = 64 + (index % 8) * 32
        series = tuple(
            TimeSeries(
                loader=_loader(_values(4, channel, length, 1)),
                spec=finance_spec,
                channel=f"c{channel}",
                sampling_rate_hz=1.0,
                source_id=f"tsqa-row-{index:04d}",
                time_series_id=f"tsqa-row-{index:04d}-c{channel}",
                t_end_s=float(length),
            )
            for channel in range(channel_count)
        )
        sample = dataset.add_sample(time_series=series, sample_id=f"sample-tsqa-{index:04d}", view=View.FULL)
        sample.add_annotation(StaticAnnotation(key="scenario", value="tsqa", id=f"annotation-tsqa-{index:04d}"))
        dataset.add_task(
            sample,
            AnswerTask(
                prompt=f"What pattern appears in series {index}?",
                target="A deterministic trend with periodic variation.",
                id=f"task-tsqa-{index:04d}",
            ),
        )

    vibration_spec = samples["vibration"].time_series[0].spec
    for index in range(64 * scale):
        offset = 0.75 if index % 2 == 0 else -0.75
        values = (_values(0, index, 64, 1) * 0.1 + offset).astype(np.float32)
        series = TimeSeries(
            loader=_loader(values),
            spec=vibration_spec,
            channel="signal",
            sampling_rate_hz=16.0,
            source_id=f"mean-recording-{index:04d}",
            time_series_id=f"mean-series-{index:04d}",
            t_end_s=len(values) / 16.0,
        )
        sample = dataset.add_sample(time_series=(series,), sample_id=f"sample-mean-{index:04d}", view=View.FULL)
        sample.add_annotation(StaticAnnotation(key="scenario", value="test-mean", id=f"annotation-mean-{index:04d}"))
        dataset.add_task(
            sample,
            ClassificationTask(
                target="above-zero" if offset > 0 else "below-zero",
                target_schema="mean-sign",
                id=f"task-mean-{index:04d}",
            ),
        )


def _add_rich_series(dataset: TimeFDataset, scale: int) -> None:
    """Add Zarr-only N-D and non-float32 workloads."""

    def tensor(values: np.ndarray, spec: TimeSeriesSpec, channel: str) -> TimeSeries:
        array = pa.FixedShapeTensorArray.from_numpy_ndarray(values, dim_names=dimensions_by_spec[spec.spec_type])
        return TimeSeries(
            spec=spec,
            channel=channel,
            sampling_rate_hz=50.0,
            loader=lambda: array,
            time_series_id=f"rich-{channel}",
            t_end_s=len(values) / 50.0,
        )

    rich_cases: tuple[tuple[str, np.ndarray, tuple[str, ...]], ...] = (
        ("ecg-frame", _values(1, 0, 2_048, scale).astype(np.float64).reshape(-1, 1).repeat(12, axis=1), ("lead",)),
        ("spectrogram", np.arange(512 * scale * 32, dtype=np.uint16).reshape(512 * scale, 32), ("frequency",)),
        ("engine-grid", np.arange(1_024 * scale * 4, dtype=np.int16).reshape(1_024 * scale, 2, 2), ("row", "column")),
    )
    dimensions_by_spec = {name: dimensions for name, _, dimensions in rich_cases}
    for name, values, dimensions in rich_cases:
        spec = cast("Any", TimeSeriesSpec)(
            spec_type=name,
            name=name.replace("-", " ").title(),
            unit_sampling_rate=ureg.hertz,
            unit_timestamp=ureg.second,
            unit_value=ureg.dimensionless,
            data_source=_SOURCE,
            dtype=values.dtype.name,
            value_shape=values.shape[1:],
            dimension_names=dimensions,
        )
        sample = dataset.add_sample(
            time_series=(tensor(values, spec, name),),
            sample_id=f"sample-rich-{name}",
            subject_ids=(f"subject-rich-{name}",),
            view=View.FULL,
        )
        sample.add_annotation(StaticAnnotation(key="rich-profile", value=True, id=f"annotation-rich-{name}"))


def build_corpus(*, profile: str = "portable", scale: int = 1) -> TimeFDataset:
    """Build the benchmark corpus.

    Args:
        profile: ``"portable"`` for scalar float32 data accepted by both backends, or ``"rich"``
            to add Zarr-only N-D and varied-dtype series.
        scale: Positive multiplier for each scenario's number of temporal steps.

    Returns:
        A deterministic dataset with a derived schema.

    Raises:
        ValueError: If ``profile`` or ``scale`` is invalid.
    """
    if profile not in {"portable", "rich"}:
        raise ValueError(f"profile must be 'portable' or 'rich', got {profile!r}")
    if scale < 1:
        raise ValueError(f"scale must be >= 1, got {scale}")
    dataset = TimeFDataset(
        metadata=DatasetMetadata(
            dataset_id="timenet/end-to-end-benchmark",
            dataset_version=Version(1, 0, 0),
            name="TimeNet end-to-end benchmark",
            description="Deterministic synthetic workloads spanning common time-series domains.",
            license=License.CC_BY_4_0,
            domains=(Domain.GENERAL,),
            tags=("benchmark", "synthetic", profile),
        )
    )
    samples: dict[str, Sample] = {}
    for scenario_index, scenario in enumerate(SCENARIOS):
        series = tuple(
            _scalar_series(scenario, scenario_index, channel, channel_index, scale)
            for channel_index, channel in enumerate(scenario.channels)
        )
        sample = dataset.add_sample(
            time_series=series,
            sample_id=f"sample-{scenario.name}",
            subject_ids=(f"subject-{scenario.name}",),
            view=View.FULL,
        )
        sample.add_annotations(
            [
                StaticAnnotation(key="scenario", value=scenario.name, id=f"annotation-{scenario.name}-scenario"),
                PointAnnotation(key="event", start_time_s=1.0, id=f"annotation-{scenario.name}-event"),
            ]
        )
        if scenario.steps / scenario.sampling_rate_hz > 2:  # noqa: PLR2004, RUF100 - minimum interval duration
            sample.add_annotation(
                IntervalAnnotation(
                    key="quality-window",
                    start_time_s=1.0,
                    end_time_s=2.0,
                    id=f"annotation-{scenario.name}-window",
                )
            )
        samples[scenario.name] = sample
    _add_tasks(dataset, samples)
    _add_connector_patterns(dataset, samples, scale)
    if profile == "rich":
        _add_rich_series(dataset, scale)
    dataset.derive_schema()
    return dataset
