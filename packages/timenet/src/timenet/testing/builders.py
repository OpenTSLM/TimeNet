"""Deterministic dataset builders, a counting loader, and a logical dataset comparison."""

from collections.abc import Callable, Sequence

import numpy as np
import pyarrow as pa

from timenet.dataset import TimeFDataset, TimeSeries
from timenet.types import (
    ClassificationTask,
    DatasetMetadata,
    Domain,
    IntervalAnnotation,
    License,
    PointAnnotation,
    StaticAnnotation,
    TimeSeriesSpec,
    Version,
    View,
    ureg,
)


class CountingLoader:
    """A loader that records how many times it was called, for lazy-read assertions."""

    def __init__(self, values: Sequence[float]) -> None:
        """Store the values to return.

        Args:
            values: The float values the loader yields on each call.
        """
        self._values = [float(v) for v in values]
        self.calls = 0

    def __call__(self) -> pa.Array:
        """Return the values as a float32 Arrow array and increment the call count."""
        self.calls += 1
        return pa.array(self._values, type=pa.float32())


def sine_loader(
    *, n: int, freq_hz: float = 1.0, sampling_rate_hz: float = 100.0, phase: float = 0.0
) -> Callable[[], pa.Array]:
    """Build a loader returning a deterministic float32 sine wave (closed-form, no RNG).

    Args:
        n: Number of samples.
        freq_hz: Wave frequency in Hz.
        sampling_rate_hz: Sampling rate in Hz.
        phase: Phase offset in radians.

    Returns:
        A no-argument loader returning the wave as a float32 Arrow array.
    """

    def load() -> pa.Array:
        t = np.arange(n, dtype=np.float64) / sampling_rate_hz
        return pa.array(np.sin(2.0 * np.pi * freq_hz * t + phase).astype(np.float32))

    return load


def make_dataset() -> TimeFDataset:
    """Build a small, fully deterministic dataset exercising the common TimeF features.

    Two samples over one shared modality, with static / point / interval annotations and a task. All ids
    are fixed, so two calls produce equal datasets.

    Returns:
        The populated :class:`TimeFDataset`.
    """
    dataset = TimeFDataset(
        metadata=DatasetMetadata(
            dataset_id="hello_world",
            dataset_version=Version(1, 0, 0),
            name="Hello World",
            description="A synthetic demo dataset.",
            license=License.CC_BY_4_0,
            domains=(Domain.GENERAL,),
        )
    )
    spec = TimeSeriesSpec(
        spec_type="sine",
        name="Sine",
        unit_sampling_rate=ureg.hertz,
        unit_timestamp=ureg.second,
        unit_value=ureg.dimensionless,
    )
    for index in range(2):
        series = TimeSeries(
            spec=spec,
            channel="a",
            sampling_rate_hz=16.0,
            loader=sine_loader(n=16, freq_hz=1.0, sampling_rate_hz=16.0, phase=index),
            source_id=f"rec-{index}",
            time_series_id=f"ts-{index}",
            t_start_s=0.0,
            t_end_s=1.0,
        )
        sample = dataset.add_sample(
            time_series=(series,),
            view=View.FULL,
            subject_ids=(f"subj-{index}",),
            sample_id=f"sample-{index}",
        )
        sample.add_annotation(StaticAnnotation(key="age", value=40 + index, unit="years", id=f"age-{index}"))
        sample.add_annotation(PointAnnotation(key="stimulus", start_time_s=0.5, id=f"stim-{index}"))
        sample.add_annotation(
            IntervalAnnotation(key="artifact", start_time_s=0.0, end_time_s=1.0, id=f"artifact-{index}")
        )
        dataset.add_task(sample, ClassificationTask(label="normal", id=f"task-{index}"))
    return dataset


def assert_datasets_equal(expected: TimeFDataset, actual: TimeFDataset) -> None:
    """Assert two datasets are logically equal per the round-trip preserved-field contract.

    Compares metadata, and every sample (matched by ``sample_id``) and task (matched by ``id``),
    including each series' fields and its materialized values. Raises ``AssertionError`` (via ``assert``)
    if any compared field differs.

    Args:
        expected: The reference dataset.
        actual: The dataset to check against it.
    """
    assert expected.metadata == actual.metadata, "metadata differs"

    exp_samples = {s.sample_id: s for s in expected.samples}
    act_samples = {s.sample_id: s for s in actual.samples}
    assert exp_samples.keys() == act_samples.keys(), "sample ids differ"
    for sample_id, exp in exp_samples.items():
        act = act_samples[sample_id]
        assert exp.view == act.view, f"view differs for {sample_id}"
        assert exp.subject_ids == act.subject_ids, f"subject_ids differ for {sample_id}"
        assert tuple(sorted(exp.task_ids)) == tuple(sorted(act.task_ids)), f"task_ids differ for {sample_id}"
        assert sorted(exp.annotations, key=lambda a: a.id) == sorted(act.annotations, key=lambda a: a.id), (
            f"annotations differ for {sample_id}"
        )
        _assert_series_equal(sample_id, exp.time_series, act.time_series)

    exp_tasks = {t.id: t for t in expected.tasks}
    act_tasks = {t.id: t for t in actual.tasks}
    assert exp_tasks.keys() == act_tasks.keys(), "task ids differ"
    for task_id, exp_task in exp_tasks.items():
        assert exp_task == act_tasks[task_id], f"task {task_id} differs"


def _assert_series_equal(sample_id: str, expected: tuple[TimeSeries, ...], actual: tuple[TimeSeries, ...]) -> None:
    """Assert two tuples of series (matched by ``time_series_id``) are equal, values included."""
    exp = {ts.time_series_id: ts for ts in expected}
    act = {ts.time_series_id: ts for ts in actual}
    assert exp.keys() == act.keys(), f"time_series ids differ for {sample_id}"
    for series_id, exp_ts in exp.items():
        act_ts = act[series_id]
        assert exp_ts.spec == act_ts.spec, f"spec differs for {series_id}"
        assert exp_ts.channel == act_ts.channel, f"channel differs for {series_id}"
        assert exp_ts.source_id == act_ts.source_id, f"source_id differs for {series_id}"
        assert exp_ts.sampling_rate_hz == act_ts.sampling_rate_hz, f"sampling_rate differs for {series_id}"
        assert exp_ts.t_start_s == act_ts.t_start_s, f"t_start_s differs for {series_id}"
        assert exp_ts.t_end_s == act_ts.t_end_s, f"t_end_s differs for {series_id}"
        assert exp_ts.to_arrow().equals(act_ts.to_arrow()), f"values differ for {series_id}"
