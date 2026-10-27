from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar, Self

from timenet.dataset import Record, Signal, Source, TimeFDataset
from timenet.dataset.axis import OrdinalAxis
from timenet.types import (
    Annotation,
    AnswerTask,
    ClassificationTask,
    DatasetMetadata,
    License,
    Split,
    Task,
    TimeSeriesSpec,
    Version,
)
from timenet_connectors.bases.loader import DatasetLoader, DatasetTask


_SPEC = TimeSeriesSpec(spec_type="toy", name="Toy", unit_value=None, dtype="float64")
_METADATA = DatasetMetadata(
    dataset_id="timenet/toy",
    dataset_version=Version(1, 0, 0),
    name="Toy",
    description="demo",
    license=License.CC_BY_4_0,
)


@dataclass(frozen=True)
class _ToyTask(DatasetTask):
    """A toy source task on one Record."""

    record: Record
    toy_split: Split

    @property
    def split(self) -> Split:
        return self.toy_split


@dataclass(frozen=True)
class _Label(_ToyTask):
    TIMENET_TYPE: ClassVar[type[Task]] = ClassificationTask

    def to_timenet_task(self) -> ClassificationTask:
        return ClassificationTask(inputs=(self.record,), targets=("spike",), split=self.split)


@dataclass(frozen=True)
class _Severity(_ToyTask):
    # A second kind that becomes the same TimeNet type as _Label.
    TIMENET_TYPE: ClassVar[type[Task]] = ClassificationTask

    def to_timenet_task(self) -> ClassificationTask:
        return ClassificationTask(inputs=(self.record,), targets=("high",), split=self.split)


@dataclass(frozen=True)
class _Caption(_ToyTask):
    TIMENET_TYPE: ClassVar[type[Task]] = AnswerTask

    def to_timenet_task(self) -> AnswerTask:
        return AnswerTask(inputs=(self.record,), prompt="Describe it.", targets=("A spike.",), split=self.split)


class _ToyLoader(DatasetLoader[_ToyTask]):
    """Toy tasks: a label and a severity per Record, and a caption on the train Record."""

    TASK_KINDS = (_Label, _Severity, _Caption)

    def __init__(self, records: tuple[Record, ...]) -> None:
        self._records = records

    @classmethod
    def parse_from(cls, refs: Iterable[Any]) -> Self:
        records = tuple(Record(sources=(Source(name=name, signals=(_signal(values),)),)) for name, values in refs)
        return cls(records)

    def records(self) -> Iterator[Record]:
        return iter(self._records)

    def tasks(self) -> Iterator[_ToyTask]:
        train, test = self._records
        yield _Label(train, Split.TRAIN)
        yield _Severity(train, Split.TRAIN)
        yield _Caption(train, Split.TRAIN)
        yield _Label(test, Split.TEST)
        yield _Severity(test, Split.TEST)

    def dataset_annotations(self) -> Sequence[Annotation]:
        return (Annotation(key="label_vocabulary", value=["spike", "flat"]),)


def _signal(values: list[float]) -> Signal:
    return Signal.from_values(values, spec=_SPEC, name="series", time_axis=OrdinalAxis())


def _loader() -> _ToyLoader:
    return _ToyLoader.parse_from([("train", [1.0, 2.0]), ("test", [3.0])])


def test_tasks_of_selects_a_source_kind_and_a_split():
    loader = _loader()
    assert [task.split for task in loader.tasks_of(_Label, split=Split.TEST)] == [Split.TEST]
    # The two classification kinds stay apart, which a TimeNet task type can't tell.
    assert [type(task) for task in loader.tasks_of(_Severity, split=Split.TRAIN)] == [_Severity]
    # A base class selects every kind derived from it, in the order of tasks().
    assert list(loader.tasks_of(_ToyTask, split=Split.TRAIN)) == list(loader.tasks())[:3]


def test_populate_fills_the_dataset_and_streams_the_tasks():
    loader = _loader()
    dataset = TimeFDataset(metadata=_METADATA)
    assert loader.populate(dataset) is dataset
    assert [(a.key, a.value) for a in dataset.annotations] == [("label_vocabulary", ["spike", "flat"])]
    assert [record.sources[0].name for record in dataset.records] == ["train", "test"]
    # The stream starts again on each read, in the order of tasks().
    first, second = list(dataset.iter_tasks()), list(dataset.iter_tasks())
    expected = [(task.TIMENET_TYPE, task.split) for task in loader.tasks()]
    assert [(type(task), task.split) for task in first] == expected
    assert [(type(task), task.split) for task in second] == expected
    # Each TimeNet type is declared once, in TASK_KINDS order.
    assert dataset.derive_schema().tasks == (ClassificationTask, AnswerTask)
