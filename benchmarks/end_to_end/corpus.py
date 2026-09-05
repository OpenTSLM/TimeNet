"""Build a deterministic, diverse synthetic TimeF benchmark corpus."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, cast

import numpy as np
import pyarrow as pa

from timenet.dataset import Record, TimeFDataset, TimeSeries
from timenet.dataset.axis import OrdinalAxis, RegularAxis
from timenet.types import (
    Annotation,
    AnswerTask,
    ClassificationTask,
    DatasetMetadata,
    DataSource,
    Domain,
    ForecastingTask,
    License,
    TimeInterval,
    TimePoint,
    TimeSeriesSpec,
    Version,
    ureg,
)


@dataclass(frozen=True)
class Scenario:
    """One deterministic workload in the synthetic corpus."""

    name: str
    signals: tuple[str, ...]
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
    """Return deterministic, nontrivial float32 values for one signal."""
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
    signal: str,
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
        unit_value=ureg.Unit(scenario.unit),
        data_source=_SOURCE,
    )
    values = pa.array(_values(scenario_index, channel_index, scenario.steps, scale), type=pa.float32())
    return TimeSeries(
        loader=lambda: values,
        spec=spec,
        signal=signal,
        time_axis=RegularAxis.from_rate_hz(Fraction(scenario.sampling_rate_hz)),
        source_id=f"{scenario.name}-recording",
        time_series_id=f"{scenario.name}-{signal}",
        n_values=len(values),
    )


def _nonfloat_series(
    name: str,
    dtype: str,
    values: tuple[str, ...] | np.ndarray,
    scale: int,
    categories: tuple[str, ...] = (),
) -> TimeSeries:
    """Construct one portable scalar non-float series, shareable by both backends.

    Args:
        name: The modality and signal tag.
        dtype: A scalar dtype both values backends can store (int/bool/float64/str/enum).
        values: Deterministic per-series values. A ``str`` or ``enum`` dtype takes a tuple of
            label strings; every other dtype takes a NumPy array already in the target dtype.
        scale: Positive step multiplier, repeating each value ``scale`` times.
        categories: Ordered codebook for ``dtype="enum"``.

    Returns:
        The fully described series.
    """
    spec = TimeSeriesSpec(
        spec_type=f"portable-{name}",
        name=name.replace("-", " ").title(),
        unit_value=ureg.dimensionless,
        data_source=_SOURCE,
        dtype=dtype,
        categories=categories,
    )
    if dtype in {"str", "enum"}:
        labels = [label for label in values for _ in range(scale)]
        array = pa.array(labels, type=pa.string())
        if dtype == "enum":
            array = array.dictionary_encode()
    else:
        array = pa.array(np.repeat(np.asarray(values), scale, axis=0))
    return TimeSeries(
        loader=lambda: array,
        spec=spec,
        signal=name,
        time_axis=RegularAxis.from_rate_hz(Fraction(1)),
        source_id=f"{name}-recording",
        time_series_id=f"portable-{name}",
        n_values=len(array),
    )


_NONFLOAT_CHANNELS: tuple[tuple[str, str, tuple[str, ...] | np.ndarray, tuple[str, ...]], ...] = (
    ("machine-mode", "int16", np.array([0, 1, 2, 1, 0, 1, 2, 2], dtype=np.int16), ()),
    ("alarm", "bool", np.array([False, False, True, False, True, False, False, True]), ()),
    ("precise", "float64", np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0], dtype=np.float64), ()),
    ("rhythm", "str", ("normal", "afib", "vt", "normal"), ()),
    ("activity", "enum", ("walk", "run", "sit", "walk"), ("walk", "run", "sit", "stand")),
)
"""Portable scalar non-float signals, appended to a dedicated record."""


def _add_tasks(dataset: TimeFDataset, records: dict[str, Record]) -> None:
    """Attach every currently supported task payload to representative scenarios."""
    vibration = records["vibration"]
    ecg = records["ecg"]
    sleep = records["sleep"]
    accelerometer = records["accelerometer"]
    finance = records["finance"]
    workout = records["workout"]
    energy = records["energy"]
    automotive = records["automotive"]

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
            scope=TimeInterval.seconds(30.0, 60.0),
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
    # A single unsplit recording carrying its own horizon: predict the tail from the head. The energy
    # scenario is 4096 steps at 0.25 Hz, a [0, 16384 s) window at the smallest scale, so this span is
    # inside every scale.
    dataset.add_task(
        energy,
        ForecastingTask(
            scope=TimeInterval.seconds(0.0, 16000.0),
            target_span=TimeInterval.seconds(16000.0, 16380.0),
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


def _add_connector_patterns(dataset: TimeFDataset, records: dict[str, Record], scale: int) -> None:
    """Mirror the cardinality patterns that distinguish the real connectors.

    ECG-QA reuses one long 12-lead recording across several text tasks, TSQA contains many short
    independently stored series, and test-mean contains many tiny labeled series. These patterns have
    measurably different control-plane and random-access costs even when their total value bytes match.
    """
    ecg = records["ecg"]
    for index in range(1, 8 * scale):
        record = dataset.add_record(
            time_series=ecg.time_series,
            record_id=f"record-ecg-question-{index:03d}",
            subject_ids=("subject-ecg",),
        )
        record.add_annotation(Annotation(key="scenario", value="ecg", id=f"annotation-ecg-{index:03d}"))
        dataset.add_task(
            record,
            AnswerTask(
                prompt=f"Is rhythm abnormal in view {index}?",
                target="atrial-fibrillation",
                rationale="The synthetic rhythm has repeatable irregular intervals.",
                id=f"task-ecg-reasoning-{index:03d}",
            ),
        )

    finance_spec = records["finance"].time_series[0].spec
    for index in range(64 * scale):
        channel_count = 1 + index % 3
        length = 64 + (index % 8) * 32
        series = tuple(
            TimeSeries(
                loader=_loader(_values(4, signal, length, 1)),
                spec=finance_spec,
                signal=f"c{signal}",
                time_axis=OrdinalAxis(),
                source_id=f"tsqa-row-{index:04d}",
                time_series_id=f"tsqa-row-{index:04d}-c{signal}",
                n_values=length,
            )
            for signal in range(channel_count)
        )
        record = dataset.add_record(time_series=series, record_id=f"record-tsqa-{index:04d}")
        record.add_annotation(Annotation(key="scenario", value="tsqa", id=f"annotation-tsqa-{index:04d}"))
        dataset.add_task(
            record,
            AnswerTask(
                prompt=f"What pattern appears in series {index}?",
                target="A deterministic trend with periodic variation.",
                id=f"task-tsqa-{index:04d}",
            ),
        )

    vibration_spec = records["vibration"].time_series[0].spec
    for index in range(64 * scale):
        offset = 0.75 if index % 2 == 0 else -0.75
        values = (_values(0, index, 64, 1) * 0.1 + offset).astype(np.float32)
        series = TimeSeries(
            loader=_loader(values),
            spec=vibration_spec,
            signal="signal",
            time_axis=RegularAxis.from_rate_hz(16),
            source_id=f"mean-recording-{index:04d}",
            time_series_id=f"mean-series-{index:04d}",
            n_values=len(values),
        )
        record = dataset.add_record(time_series=(series,), record_id=f"record-mean-{index:04d}")
        record.add_annotation(Annotation(key="scenario", value="test-mean", id=f"annotation-mean-{index:04d}"))
        dataset.add_task(
            record,
            ClassificationTask(
                target="above-zero" if offset > 0 else "below-zero",
                target_schema="mean-sign",
                id=f"task-mean-{index:04d}",
            ),
        )


def _add_rich_series(dataset: TimeFDataset, scale: int) -> None:
    """Add Zarr-only N-D and non-float32 workloads."""

    def tensor(values: np.ndarray, spec: TimeSeriesSpec, signal: str) -> TimeSeries:
        array = pa.FixedShapeTensorArray.from_numpy_ndarray(values, dim_names=dimensions_by_spec[spec.spec_type])
        return TimeSeries(
            spec=spec,
            signal=signal,
            time_axis=RegularAxis.from_rate_hz(50),
            loader=lambda: array,
            time_series_id=f"rich-{signal}",
            n_values=len(values),
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
            unit_value=ureg.dimensionless,
            data_source=_SOURCE,
            dtype=values.dtype.name,
            value_shape=values.shape[1:],
            dimension_names=dimensions,
        )
        record = dataset.add_record(
            time_series=(tensor(values, spec, name),),
            record_id=f"record-rich-{name}",
            subject_ids=(f"subject-rich-{name}",),
        )
        record.add_annotation(Annotation(key="rich-profile", value=True, id=f"annotation-rich-{name}"))


def _add_nonfloat_sample(dataset: TimeFDataset, scale: int) -> None:
    """Add a record whose signals come in varied scalar dtypes (int16, bool, float64, str).

    Args:
        dataset: The dataset to add the record to.
        scale: Positive step multiplier.
    """
    signals = _NONFLOAT_CHANNELS
    series = tuple(
        _nonfloat_series(name, dtype, values, scale, categories=cats) for name, dtype, values, cats in signals
    )
    record = dataset.add_record(
        time_series=series,
        record_id="record-nonfloat",
        subject_ids=("subject-nonfloat",),
    )
    record.add_annotation(Annotation(key="scenario", value="nonfloat", id="annotation-nonfloat"))


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
    records: dict[str, Record] = {}
    for scenario_index, scenario in enumerate(SCENARIOS):
        series = tuple(
            _scalar_series(scenario, scenario_index, signal, channel_index, scale)
            for channel_index, signal in enumerate(scenario.signals)
        )
        record = dataset.add_record(
            time_series=series,
            record_id=f"record-{scenario.name}",
            subject_ids=(f"subject-{scenario.name}",),
        )
        annotations = [
            Annotation(key="scenario", value=scenario.name, id=f"annotation-{scenario.name}-scenario"),
            Annotation(key="event", span=TimePoint.seconds(1.0), id=f"annotation-{scenario.name}-event"),
        ]
        if scenario.steps / scenario.sampling_rate_hz > 2:  # noqa: PLR2004, RUF100 - minimum interval duration
            annotations.append(
                Annotation(
                    key="quality-window",
                    span=TimeInterval.seconds(1.0, 2.0),
                    id=f"annotation-{scenario.name}-window",
                )
            )
        record.add_annotations(annotations)
        records[scenario.name] = record
    _add_tasks(dataset, records)
    _add_connector_patterns(dataset, records, scale)
    _add_nonfloat_sample(dataset, scale)
    if profile == "rich":
        _add_rich_series(dataset, scale)
    dataset.derive_schema()
    return dataset
