"""Deterministic dataset builders, a counting loader, and a logical dataset comparison."""

from collections.abc import Callable, Iterable, Sequence
from dataclasses import replace
import json

import numpy as np
import pyarrow as pa

from timenet.dataset import Record, Signal, Source, TimeFDataset
from timenet.dataset.axis import RegularAxis
from timenet.format.task_codec import encode_span
from timenet.types import (
    Annotation,
    AnswerTask,
    ClassificationTask,
    DatasetMetadata,
    Domain,
    EntityId,
    Label,
    LabelSet,
    License,
    LocalizationMode,
    ScalarPredictionTask,
    StepSpan,
    Task,
    TemporalLocalizationTask,
    TimeInterval,
    TimeOrigin,
    TimePoint,
    TimeSeriesSpec,
    TimeSpan,
    Version,
    read_value,
    ureg,
    value_type_of,
)


_RHYTHM_LABELS = LabelSet(name="rhythm", labels=(Label(name="normal"), Label(name="abnormal")))
"""The label set of the fixture's rhythm classification task."""
_EVENT_LABELS = LabelSet(name="event", labels=(Label(name="onset"), Label(name="offset")))
"""The label set of the fixture's scoped event classification task."""


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
    and an interval. The dataset numbers every id, so two calls build equal datasets. This is the
    canonical writer and reader round-trip fixture.

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
            Annotation(key="cohort", value="A"),
            Annotation(key="stimulus", span=TimePoint.seconds(0.5)),
        ]
    )
    shared.annotate(Annotation(key="artifact", span=TimeInterval.seconds(0.0, 0.25)))
    classification = ClassificationTask(inputs=(record0,), targets=("normal",), label_set=_RHYTHM_LABELS)
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
                    TimeInterval.seconds(0.0, 0.25, time_series=(shared,)),
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
    record1.annotate(record0_cohort)  # A new occurrence shares the cohort content across both records.

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
            label_set=_EVENT_LABELS,
            scope=TimeInterval.seconds(0.0, 0.25, time_series=(window,)),
        ),
    )
    return dataset


def assert_datasets_equal(expected: TimeFDataset, actual: TimeFDataset) -> None:
    """Assert that two datasets agree on every field that writing and reading must preserve.

    This compares the metadata, the parent cards and lineage, the dataset annotations, every record
    matched by its id, and every task matched by its id. It compares each series' fields and its
    materialized values. A span holds Signal objects, so it compares task scopes by the ids of their
    Signals. For a dataset with a task stream, it reads the stream using the ids assigned during writing,
    and compares those tasks. If any compared field differs, it raises ``AssertionError``.

    Args:
        expected: The reference dataset.
        actual: The dataset to check against it.
    """
    assert expected.metadata == actual.metadata, "metadata differs"
    assert dict(expected.parent_cards) == dict(actual.parent_cards), "parent cards differ"
    assert dict(expected.lineage) == dict(actual.lineage), "lineage differs"
    assert _occurrences(expected.annotations) == _occurrences(actual.annotations), "dataset annotations differ"
    _assert_records_equal(expected.records, actual.records)
    _assert_tasks_equal(
        _tasks_to_compare(expected), _tasks_to_compare(actual), expected_dataset=expected, actual_dataset=actual
    )


def _tasks_to_compare(dataset: TimeFDataset) -> Sequence[Task]:
    """Return a dataset's tasks, read from its stream using the ids assigned during writing, if it has one."""
    return list(dataset.iter_streamed_tasks_validated()) if dataset.has_task_stream else dataset.tasks


def _assert_records_equal(expected: Sequence[Record], actual: Sequence[Record]) -> None:
    """Assert that two sets of records are logically equal, matched by their ids.

    This compares each record's start time, time span, metadata, annotations in order, sources and
    series, values included. If any compared field differs, it raises ``AssertionError``.

    Args:
        expected: The reference records.
        actual: The records to check against them.
    """
    exp_records = {s.id: s for s in expected}
    act_records = {s.id: s for s in actual}
    assert exp_records.keys() == act_records.keys(), "record ids differ"
    for record_id, exp in exp_records.items():
        act = act_records[record_id]
        assert exp.start_time.timestamp == act.start_time.timestamp, f"start_time differs for {record_id}"
        assert encode_span(exp.time_span) == encode_span(act.time_span), f"time_span differs for {record_id}"
        assert exp.metadata == act.metadata, f"metadata differs for {record_id}"
        assert _occurrences(exp.annotations) == _occurrences(act.annotations), f"annotations differ for {record_id}"
        _assert_sources_equal(record_id, exp, act)
        _assert_series_equal(record_id, exp.signals, act.signals)


def _assert_tasks_equal(
    expected: Sequence[Task],
    actual: Sequence[Task],
    *,
    expected_dataset: TimeFDataset | None = None,
    actual_dataset: TimeFDataset | None = None,
) -> None:
    """Assert that two sets of tasks are logically equal, matched by their ids.

    ``Task`` equality leaves out the fields that refer to other objects: inputs, targets, the input
    and target annotations, and the source tasks. This function compares them too. It compares a
    referenced record, signal, or task by its id, a span by the ids of its signals, and an
    annotation with its occurrence fields. With both datasets, it also checks where each referenced
    annotation is attached and its position within that object's annotations: the dataset, a
    record, a source by its path of names, or a signal by its id. If any compared field differs, it raises
    ``AssertionError``.

    Args:
        expected: The reference tasks.
        actual: The tasks to check against them.
        expected_dataset: The dataset that holds ``expected``, to compare annotation owners.
        actual_dataset: The dataset that holds ``actual``, to compare annotation owners.
    """
    exp_owners = _annotation_owners(expected_dataset) if expected_dataset is not None else None
    act_owners = _annotation_owners(actual_dataset) if actual_dataset is not None else None
    exp_tasks = {t.id: t for t in expected}
    act_tasks = {t.id: t for t in actual}
    assert exp_tasks.keys() == act_tasks.keys(), "task ids differ"
    for task_id, exp_task in exp_tasks.items():
        act_task = act_tasks[task_id]
        assert [record.id for record in exp_task.inputs] == [record.id for record in act_task.inputs], (
            f"task {task_id} inputs differ"
        )
        assert encode_span(exp_task.scope) == encode_span(act_task.scope), f"task {task_id} scope differs"
        assert _comparable_targets(exp_task) == _comparable_targets(act_task), f"task {task_id} targets differ"
        for name in ("annotations", "input_annotations", "target_annotations"):
            exp_annotations, act_annotations = getattr(exp_task, name), getattr(act_task, name)
            assert _occurrences(exp_annotations) == _occurrences(act_annotations), f"task {task_id} {name} differ"
            if exp_owners is not None and act_owners is not None and name != "annotations":
                where = f"task {task_id} {name}"
                exp_places = [_owner(exp_owners, annotation, where) for annotation in exp_annotations]
                act_places = [_owner(act_owners, annotation, where) for annotation in act_annotations]
                assert exp_places == act_places, f"task {task_id} {name} point to other owners"
        assert [task.id for task in exp_task.from_tasks] == [task.id for task in act_task.from_tasks], (
            f"task {task_id} from_tasks differ"
        )
        exp_candidates = [record.id for record in getattr(exp_task, "candidate_records", ())]
        act_candidates = [record.id for record in getattr(act_task, "candidate_records", ())]
        assert exp_candidates == act_candidates, f"task {task_id} candidate_records differ"
        assert replace(
            exp_task, scope=None, annotations=(), input_modalities=exp_task.resolved_input_modalities
        ) == replace(act_task, scope=None, annotations=(), input_modalities=act_task.resolved_input_modalities), (
            f"task {task_id} differs"
        )


def _occurrences(annotations: Iterable[Annotation]) -> list[str]:
    """Return canonical annotation contents in order, including type and occurrence fields."""
    result = []
    for annotation in annotations:
        value = read_value(annotation.value)
        result.append(
            json.dumps(
                {
                    "key": str(annotation.key),
                    "value_type": value_type_of(annotation.value),
                    "value": float.hex(value) if isinstance(value, float) else value,
                    "span": encode_span(annotation.span),
                    "unit": annotation.unit,
                    "description": annotation.description,
                    "source": annotation.source,
                    "metadata": annotation.metadata,
                    "confidence": None if annotation.confidence is None else float.hex(float(annotation.confidence)),
                    "occurrence_metadata": annotation.occurrence_metadata,
                },
                sort_keys=True,
            )
        )
    return result


def _annotation_owners(dataset: TimeFDataset) -> dict[int, tuple[object, ...]]:
    """Map each annotation occurrence of a dataset to its owner and its position in that owner.

    The owner is named by what a release keeps stable: the dataset, a record id, a source's path of
    names in its record, or a signal id. A position counts only within one owner, because a release
    can store the signals of a source in another order.

    Returns:
        One owner tuple per occurrence, keyed by the occurrence's ``id()``.
    """
    owners: dict[int, tuple[object, ...]] = {}
    for index, annotation in enumerate(dataset.annotations):
        owners[id(annotation)] = ("dataset", index)
    for record in dataset.records:
        for index, annotation in enumerate(record.annotations):
            owners[id(annotation)] = ("record", record.id, index)
        pending: list[tuple[tuple[str, ...], Source]] = [((source.name,), source) for source in record.sources]
        while pending:
            path, source = pending.pop()
            for index, annotation in enumerate(source.annotations):
                owners[id(annotation)] = ("source", record.id, path, index)
            for signal in source.signals:
                for index, annotation in enumerate(signal.annotations):
                    owners[id(annotation)] = ("signal", signal.id, index)
            pending.extend(((*path, child.name), child) for child in source.sources)
    return owners


def _owner(owners: dict[int, tuple[object, ...]], annotation: Annotation, where: str) -> tuple[object, ...]:
    """Return the owner of an annotation that a task references.

    Raises:
        AssertionError: If the annotation is not attached to any object in the dataset.
    """
    owner = owners.get(id(annotation))
    if owner is None:
        raise AssertionError(f"{where} reference an annotation that is not attached to any object in the dataset")
    return owner


def _comparable_targets(task: Task) -> tuple[object, ...] | None:
    """Return a task's targets in a form that compares referenced objects by id.

    Returns:
        One item for each target, or ``None`` for a task without targets.
    """
    if task.targets is None:
        return None
    items: list[object] = []
    for target in task.targets:
        if isinstance(target, Record | Signal):
            items.append((type(target).__name__, target.id))
        elif isinstance(target, TimeSpan | StepSpan):
            items.append(("span", encode_span(target)))
        elif isinstance(target, bool | int):
            items.append((type(target).__name__, target))
        elif isinstance(target, float):
            # float.hex keeps NaN equal to NaN and tells 0.0 from -0.0.
            items.append(("float", float.hex(target)))
        else:
            items.append(("text", read_value(target)))
    return tuple(items)


def _assert_sources_equal(record_id: EntityId, expected: Record, actual: Record) -> None:
    """Assert two recursive source hierarchies are logically equal, walked in the same order."""
    exp, act = list(expected.walk_sources()), list(actual.walk_sources())
    assert len(exp) == len(act), f"sources differ for {record_id}"
    for exp_source, act_source in zip(exp, act, strict=True):
        name = exp_source.name
        assert name == act_source.name, f"source name differs for {record_id}"
        assert exp_source.metadata == act_source.metadata, f"source metadata differs for {name}"
        assert _occurrences(exp_source.annotations) == _occurrences(act_source.annotations), (
            f"source annotations differ for {name}"
        )
        assert sorted(source.name for source in exp_source.sources) == sorted(
            source.name for source in act_source.sources
        ), f"child sources differ for {name}"
        assert {signal.id for signal in exp_source.signals} == {signal.id for signal in act_source.signals}, (
            f"source signals differ for {name}"
        )


def _assert_series_equal(record_id: EntityId, expected: tuple[Signal, ...], actual: tuple[Signal, ...]) -> None:
    """Assert two tuples of series (matched by ``time_series_id``) are equal, values included."""
    exp = {ts.id: ts for ts in expected}
    act = {ts.id: ts for ts in actual}
    assert exp.keys() == act.keys(), f"time_series ids differ for {record_id}"
    for series_id, exp_ts in exp.items():
        act_ts = act[series_id]
        assert exp_ts.spec == act_ts.spec, f"spec differs for {series_id}"
        assert exp_ts.name == act_ts.name, f"signal differs for {series_id}"
        assert exp_ts.time_axis == act_ts.time_axis, f"time axis differs for {series_id}"
        assert exp_ts.metadata == act_ts.metadata, f"metadata differs for {series_id}"
        assert _occurrences(exp_ts.annotations) == _occurrences(act_ts.annotations), (
            f"annotations differ for {series_id}"
        )
        assert exp_ts.n_values == act_ts.n_values, f"n_values differs for {series_id}"
        assert _arrays_equal(exp_ts.to_arrow(), act_ts.to_arrow()), f"values differ for {series_id}"
        # An irregular axis carries only its endpoints, so two streams differing in the middle compare
        # equal above. This function compares the stream as data, like the values.
        exp_time_offsets, act_time_offsets = exp_ts._time_offsets, act_ts._time_offsets
        assert (exp_time_offsets is None) == (act_time_offsets is None), (
            f"one side stores time offsets and the other does not for {series_id}"
        )
        if exp_time_offsets is not None and act_time_offsets is not None:
            assert exp_time_offsets.read().equals(act_time_offsets.read()), f"time offsets differ for {series_id}"


def _arrays_equal(expected: pa.Array, actual: pa.Array) -> bool:
    """Compare signal arrays exactly, allowing paired NaNs but keeping nulls distinct.

    Returns:
        Whether the types, null masks and present values match.
    """
    if expected.equals(actual):
        return True
    if expected.type != actual.type or len(expected) != len(actual) or not expected.is_null().equals(actual.is_null()):
        return False
    if isinstance(expected, pa.FixedShapeTensorArray) and isinstance(actual, pa.FixedShapeTensorArray):
        return _arrays_equal(expected.storage, actual.storage)
    if isinstance(expected, pa.FixedSizeListArray) and isinstance(actual, pa.FixedSizeListArray):
        return _arrays_equal(expected.flatten(), actual.flatten())
    if pa.types.is_floating(expected.type):
        # Null masks already match, so NumPy's conversion of nulls to NaN is safe here.
        return np.array_equal(
            expected.to_numpy(zero_copy_only=False), actual.to_numpy(zero_copy_only=False), equal_nan=True
        )
    return False
