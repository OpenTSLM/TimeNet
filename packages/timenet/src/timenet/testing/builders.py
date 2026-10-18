"""Deterministic dataset builders, a counting loader, and a logical dataset comparison."""

from collections.abc import Callable, Sequence
from dataclasses import replace
from typing import TypeVar, cast

import numpy as np
import pyarrow as pa

from timenet.dataset import Record, Signal, Source, TimeFDataset
from timenet.dataset.axis import RegularAxis
from timenet.format.final_ids import FinalIds, IdKind
from timenet.format.task_codec import encode_target
from timenet.types import (
    Annotation,
    AnswerTask,
    ClassificationTask,
    DatasetMetadata,
    Domain,
    License,
    LocalizationMode,
    ScalarPredictionTask,
    Span,
    StepSpan,
    Task,
    TemporalLocalizationTask,
    TimeInterval,
    TimeOrigin,
    TimePoint,
    TimeSeriesSpec,
    TimeSpan,
    TSCorrespondenceTask,
    Version,
    ureg,
)


_S = TypeVar("_S", bound=Span)


class CountingLoader:
    """A loader that counts its own calls, for lazy-read assertions."""

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
    """Build a loader that returns a deterministic float32 sine wave.

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


def _series(spec, signal, n, phase=0.0):
    return Signal.from_loader(
        spec=spec,
        name=signal,
        time_axis=RegularAxis.from_rate_hz(int(_RATE_HZ)),
        loader=sine_loader(n=n, freq_hz=1.0, sampling_rate_hz=_RATE_HZ, phase=phase),
        n_values=n,
    )


def make_dataset() -> TimeFDataset:
    """Build a deterministic dataset that exercises every TimeF feature.

    The dataset covers two modalities over a shared data source. Every signal has one owning source.
    One long series exercises chunk splitting. One record uses a windowed series. It uses all three
    annotation shapes, and one annotation appears in two records. It chains a classification task
    to an answer task, and the answer carries a rationale and an input annotation. It also adds a
    scoped classification, a scalar prediction, and a temporal localization whose target is a point
    and an interval. TimeNet generates every id, so two calls differ only in their ids. This is
    the canonical writer and reader round-trip fixture.

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
    shared = _series(_SINE, "a", 16)
    cohort = Annotation(key="cohort", value="A")

    record0 = dataset.add_record(
        record=Record(
            start_time=TimeOrigin(9_007_199_254_740_993),
            sources=(
                Source(
                    name="Synthetic monitor",
                    signals=(shared, _series(_COSINE, "b", 16)),
                ),
            ),
        )
    )
    _, record0_cohort, _ = record0.add_annotations(
        [
            Annotation(key="age", value=64, unit="years"),
            cohort,
            Annotation(key="stimulus", span=TimePoint.seconds(0.5)),
        ]
    )
    shared.annotate(Annotation(key="artifact", span=TimeInterval.seconds(0.0, 0.25)))
    classification = ClassificationTask(inputs=(record0,), targets=("normal",))
    dataset.add_tasks(
        tasks=[
            classification,
            AnswerTask(
                inputs=(record0,),
                prompt="What rhythm?",
                targets=("Normal.",),
                rationale="Regular intervals with one peak per cycle.",
                input_annotations=(record0_cohort,),
                from_tasks=(classification,),
            ),
            ScalarPredictionTask(
                inputs=(record0,),
                targets=(62.0,),
                unit="bpm",
                target_name="mean_rate",
            ),
            TemporalLocalizationTask(
                inputs=(record0,),
                prompt="Locate the stimulus and the artifact.",
                mode=LocalizationMode.SPARSE,
                targets=(
                    TimePoint.seconds(0.5),
                    TimeInterval.seconds(0.0, 0.25, time_series_ids=(shared.id,)),
                ),
            ),
        ],
    )

    record1 = dataset.add_record(
        record=Record(
            sources=(
                Source(
                    name="Synthetic monitor",
                    signals=(
                        _series(_SINE, "a", 16),
                        _series(_SINE, "a", 512, phase=1.0),
                    ),
                ),
            ),
        )
    )
    record1.annotate(cohort)  # A new occurrence shares the cohort content across both records.

    window = _series(_SINE, "a", 8)
    record2 = dataset.add_record(
        record=Record(
            sources=(Source(name="Synthetic monitor", signals=(window,)),),
        )
    )
    dataset.add_task(
        task=ClassificationTask(
            inputs=(record2,),
            targets=("onset",),
            scope=TimeInterval.seconds(0.0, 0.25, time_series_ids=(window.id,)),
        ),
    )
    return dataset


class _Translate:
    """Translate the ids of an in-memory dataset to the ids one write stored."""

    def __init__(self, final_ids: FinalIds | None) -> None:
        self.final_ids = final_ids

    def __call__(self, kind: IdKind, value: int) -> int:
        return value if self.final_ids is None else self.final_ids[kind][value]

    def annotation(self, annotation: Annotation) -> Annotation:
        occurrence_id = annotation.occurrence_id
        return replace(
            annotation,
            id=self("content", annotation.id),
            occurrence_id=None if occurrence_id is None else self("occurrence", occurrence_id),
        )

    def annotations(self, annotations: Sequence[Annotation]) -> list[Annotation]:
        return [self.annotation(annotation) for annotation in annotations]

    def span(self, span: _S) -> _S:
        if isinstance(span, TimeSpan) and span.time_series_ids is not None:
            return replace(span, time_series_ids=tuple(self("signal", item) for item in span.time_series_ids))
        if isinstance(span, StepSpan):
            return replace(span, time_series_id=self("signal", span.time_series_id))
        return span

    def target(self, target: object) -> object:
        row = encode_target(target)
        if row["record_id"] is not None:
            row["record_id"] = self("record", cast("int", row["record_id"]))
        if row["signal_id"] is not None:
            row["signal_id"] = self("signal", cast("int", row["signal_id"]))
        if row["span_signal_ids"] is not None:
            row["span_signal_ids"] = tuple(
                self("signal", item) for item in cast("tuple[int, ...]", row["span_signal_ids"])
            )
        return row

    def task_references(self, task: Task) -> tuple[object, ...]:
        """Return the references a task's equality skips, as stored ids."""
        candidates = task.candidate_records if isinstance(task, TSCorrespondenceTask) else ()
        return (
            tuple(self("record", record.id) for record in task.inputs),
            tuple(self("record", record.id) for record in candidates),
            None if task.targets is None else tuple(self.target(item) for item in task.targets),
            tuple(self("task", parent.id) for parent in task.from_tasks),
            tuple(self.annotation(item).occurrence_id for item in task.input_annotations),
            tuple(self.annotation(item).occurrence_id for item in task.target_annotations),
        )


def assert_datasets_equal(expected: TimeFDataset, actual: TimeFDataset, *, final_ids: FinalIds | None = None) -> None:
    """Assert two datasets are logically equal per the round-trip preserved-field contract.

    This compares the metadata, every record matched by ``record_id``, and every task matched by
    ``id``. It compares each series' fields and its materialized values, and each task's references.
    It raises ``AssertionError`` if any compared field differs.

    Args:
        expected: The reference dataset.
        actual: The dataset to check against it.
        final_ids: The mapping a write of ``expected`` returned, when ``actual`` was read from that
            write. The ids of ``expected`` are translated through it before they are compared.
    """
    ids = _Translate(final_ids)
    assert expected.metadata == actual.metadata, "metadata differs"
    assert ids.annotations(expected.annotations) == list(actual.annotations), "dataset annotations differ"

    exp_records = {ids("record", s.id): s for s in expected.records}
    act_records = {s.id: s for s in actual.records}
    assert exp_records.keys() == act_records.keys(), "record ids differ"
    for record_id, exp in exp_records.items():
        act = act_records[record_id]
        assert exp.start_time.timestamp == act.start_time.timestamp, f"start_time differs for {record_id}"
        exp_task_ids = sorted(ids("task", task_id) for task_id in exp.task_ids)
        assert exp_task_ids == sorted(act.task_ids), f"task_ids differ for {record_id}"
        assert exp.metadata == act.metadata, f"metadata differs for {record_id}"
        assert sorted(ids.annotations(exp.annotations), key=lambda a: a.id) == sorted(
            act.annotations, key=lambda a: a.id
        ), f"annotations differ for {record_id}"
        _assert_sources_equal(record_id, exp.sources, act.sources, ids)
        _assert_series_equal(record_id, exp.signals, act.signals, ids)

    exp_tasks = {ids("task", t.id): t for t in expected.tasks}
    act_tasks = {t.id: t for t in actual.tasks}
    assert exp_tasks.keys() == act_tasks.keys(), "task ids differ"
    for task_id, exp_task in exp_tasks.items():
        act_task = act_tasks[task_id]
        translated = replace(
            exp_task,
            id=task_id,
            input_modalities=exp_task.resolved_input_modalities,
            scope=None if exp_task.scope is None else ids.span(exp_task.scope),
            annotations=tuple(ids.annotations(exp_task.annotations)),
        )
        assert translated == replace(act_task, input_modalities=act_task.resolved_input_modalities), (
            f"task {task_id} differs"
        )
        assert ids.task_references(exp_task) == _Translate(None).task_references(act_task), (
            f"task {task_id} references differ"
        )


def _assert_sources_equal(
    record_id: int, expected: tuple[Source, ...], actual: tuple[Source, ...], ids: _Translate
) -> None:
    """Assert two recursive source hierarchies are logically equal."""
    exp = {ids("source", source.id): source for root in expected for source in root.walk_sources()}
    act = {source.id: source for root in actual for source in root.walk_sources()}
    assert exp.keys() == act.keys(), f"source ids differ for {record_id}"
    for source_id, exp_source in exp.items():
        act_source = act[source_id]
        assert exp_source.name == act_source.name, f"source name differs for {source_id}"
        assert exp_source.metadata == act_source.metadata, f"source metadata differs for {source_id}"
        assert ids.annotations(exp_source.annotations) == list(act_source.annotations), (
            f"source annotations differ for {source_id}"
        )
        assert {ids("source", source.id) for source in exp_source.sources} == {
            source.id for source in act_source.sources
        }, f"child sources differ for {source_id}"
        assert {ids("signal", signal.id) for signal in exp_source.signals} == {
            signal.id for signal in act_source.signals
        }, f"source signals differ for {source_id}"


def _assert_series_equal(
    record_id: int, expected: tuple[Signal, ...], actual: tuple[Signal, ...], ids: _Translate
) -> None:
    """Assert two tuples of series (matched by ``time_series_id``) are equal, values included."""
    exp = {ids("signal", ts.id): ts for ts in expected}
    act = {ts.id: ts for ts in actual}
    assert exp.keys() == act.keys(), f"time_series ids differ for {record_id}"
    for series_id, exp_ts in exp.items():
        act_ts = act[series_id]
        assert exp_ts.spec == act_ts.spec, f"spec differs for {series_id}"
        assert exp_ts.name == act_ts.name, f"signal differs for {series_id}"
        assert exp_ts.time_axis == act_ts.time_axis, f"time axis differs for {series_id}"
        if ids.final_ids is not None:
            assert ids("axis", exp_ts.time_axis.id) == act_ts.time_axis.id, f"time axis id differs for {series_id}"
        assert exp_ts.metadata == act_ts.metadata, f"metadata differs for {series_id}"
        assert ids.annotations(exp_ts.annotations) == list(act_ts.annotations), f"annotations differ for {series_id}"
        assert exp_ts.n_values == act_ts.n_values, f"n_values differs for {series_id}"
        assert exp_ts.to_arrow().equals(act_ts.to_arrow()), f"values differ for {series_id}"
        # An irregular axis carries only its endpoints, so two streams differing in the middle compare
        # equal above. This function compares the stream as data, like the values.
        exp_time_offsets, act_time_offsets = exp_ts.time_offsets_loader, act_ts.time_offsets_loader
        assert (exp_time_offsets is None) == (act_time_offsets is None), (
            f"one side stores time offsets and the other does not for {series_id}"
        )
        if exp_time_offsets is not None and act_time_offsets is not None:
            assert exp_time_offsets().equals(act_time_offsets()), f"time offsets differ for {series_id}"
