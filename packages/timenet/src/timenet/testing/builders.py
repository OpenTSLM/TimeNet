"""Deterministic dataset builders, a counting loader, and a logical dataset comparison."""

from collections.abc import Callable, Sequence

import numpy as np
import pyarrow as pa

from timenet.dataset import TimeFDataset, TimeSeries
from timenet.dataset.axis import RegularAxis
from timenet.types import (
    Annotation,
    AnswerTask,
    ClassificationTask,
    DatasetMetadata,
    DataSource,
    Domain,
    IntervalSpan,
    License,
    LocalizationMode,
    PointSpan,
    ScalarPredictionTask,
    TemporalLocalizationTask,
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


_RATE_HZ = 16.0
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


def _series(spec, channel, n, time_series_id, source_id, phase=0.0):  # noqa: PLR0913, PLR0917
    return TimeSeries(
        spec=spec,
        channel=channel,
        time_axis=RegularAxis.from_rate_hz(int(_RATE_HZ)),
        loader=sine_loader(n=n, freq_hz=1.0, sampling_rate_hz=_RATE_HZ, phase=phase),
        source_id=source_id,
        time_series_id=time_series_id,
        n_values=n,
    )


def make_dataset() -> TimeFDataset:
    """Build a fully deterministic dataset exercising every TimeF feature.

    Two modalities over a shared data source; a series shared across two samples; a long series (to
    exercise chunk splitting); a windowed sample; all three annotation shapes including one shared
    across samples; and a classification -> answer task chain (the answer carrying a rationale and an
    input annotation) plus a scoped classification, a scalar prediction, and a temporal localization
    whose target is a point and an interval. All ids are fixed, so two calls produce equal datasets,
    making this the canonical writer/reader round-trip fixture.

    Returns:
        The populated :class:`TimeFDataset`.
    """
    dataset = TimeFDataset(
        metadata=DatasetMetadata(
            dataset_id="timenet/hello-world",
            dataset_version=Version(1, 0, 0),
            name="Hello World",
            description="A synthetic demo dataset.",
            license=License.CC_BY_4_0,
            domains=(Domain.GENERAL,),
        )
    )
    shared = _series(_SINE, "a", 16, "ts-shared", "rec-0")
    cohort = Annotation(key="cohort", value="A", id="cohort-shared")

    sample0 = dataset.add_sample(
        time_series=(shared, _series(_COSINE, "b", 16, "ts-cos-0", "rec-0")),
        view=View.FULL,
        subject_ids=("subj-0",),
        sample_id="sample-0",
        start_time=9_007_199_254_740_993,  # anchored sample; the other samples stay unanchored
    )
    sample0.add_annotation(Annotation(key="age", value=64, unit="years", id="age-0"))
    sample0.add_annotation(cohort)
    sample0.add_annotation(Annotation(key="stimulus", span=PointSpan.seconds(0.5), id="stim-0"))
    sample0.add_annotation(
        Annotation(
            key="artifact", span=IntervalSpan.seconds(0.0, 0.25, time_series_ids=(shared.time_series_id,)), id="art-0"
        )
    )
    classification = dataset.add_task(sample0, ClassificationTask(target="normal", id="task-cls-0"))
    dataset.add_task(
        sample0,
        AnswerTask(
            prompt="What rhythm?",
            target="Normal.",
            rationale="Regular intervals with one peak per cycle.",
            input_annotation_ids=("cohort-shared",),
            id="task-answer-0",
        ),
        from_tasks=(classification,),
    )
    dataset.add_task(
        sample0,
        ScalarPredictionTask(target=62.0, unit="bpm", target_name="mean_rate", id="task-scalar-0"),
    )
    dataset.add_task(
        sample0,
        TemporalLocalizationTask(
            prompt="Locate the stimulus and the artifact.",
            mode=LocalizationMode.SPARSE,
            target=(
                PointSpan.seconds(0.5),
                IntervalSpan.seconds(0.0, 0.25, time_series_ids=(shared.time_series_id,)),
            ),
            id="task-localize-0",
        ),
    )

    sample1 = dataset.add_sample(
        time_series=(shared, _series(_SINE, "a", 512, "ts-long-1", "rec-1", phase=1.0)),
        view=View.FULL,
        subject_ids=("subj-1",),
        sample_id="sample-1",
    )
    sample1.add_annotation(cohort)  # same instance/id => shared across samples

    window = _series(_SINE, "a", 8, "ts-window-2", "rec-0")
    sample2 = dataset.add_sample(time_series=(window,), view=View.WINDOW, subject_ids=("subj-0",), sample_id="sample-2")
    dataset.add_task(
        sample2,
        ClassificationTask(target="onset", id="task-cls-2"),
        scope=IntervalSpan.seconds(0.0, 0.25, time_series_ids=(window.time_series_id,)),
    )
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
        assert exp.start_time == act.start_time, f"start_time differs for {sample_id}"
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
        assert exp_ts.time_axis == act_ts.time_axis, f"time axis differs for {series_id}"
        assert exp_ts.n_values == act_ts.n_values, f"n_values differs for {series_id}"
        assert exp_ts.to_arrow().equals(act_ts.to_arrow()), f"values differ for {series_id}"
