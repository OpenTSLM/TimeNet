from collections.abc import Iterable, Iterator, Sequence
from typing import Any, Self

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
from timenet_connectors.bases.loader import DatasetLoader


_SPEC = TimeSeriesSpec(spec_type="toy", name="Toy", unit_value=None, dtype="float64")
_METADATA = DatasetMetadata(
    dataset_id="timenet/toy",
    dataset_version=Version(1, 0, 0),
    name="Toy",
    description="demo",
    license=License.CC_BY_4_0,
)


def _signal() -> Signal:
    return Signal.from_values([1.0, 2.0], spec=_SPEC, name="series", time_axis=OrdinalAxis())


class _ToyLoader(DatasetLoader):
    """Two classification tasks of the authors for each Record, and a caption without a split."""

    TASK_TYPES = (ClassificationTask, AnswerTask)

    def __init__(self, records: tuple[Record, ...]) -> None:
        self._records = records

    @classmethod
    def parse_from(cls, refs: Iterable[Any]) -> Self:
        return cls(tuple(Record(sources=(Source(name=name, signals=(_signal(),)),)) for name in refs))

    def records(self) -> Iterator[Record]:
        return iter(self._records)

    def label_tasks(self) -> Iterator[ClassificationTask]:
        """Is the series a spike or flat?"""
        train, test = self._records
        yield ClassificationTask(inputs=(train,), targets=("spike",), split=Split.TRAIN)
        yield ClassificationTask(inputs=(test,), targets=("flat",), split=Split.TEST)

    def severity_tasks(self) -> Iterator[ClassificationTask]:
        """Is the spike high or low? The same TimeNet type as the label tasks."""
        train, _ = self._records
        yield ClassificationTask(inputs=(train,), targets=("high",), split=Split.TRAIN)

    def tasks(self) -> Iterator[Task]:
        yield from self.label_tasks()
        yield from self.severity_tasks()
        train, _ = self._records
        yield AnswerTask(inputs=(train,), prompt="Describe it.", targets=("A spike.",))

    def dataset_annotations(self) -> Sequence[Annotation]:
        return (Annotation(key="label_vocabulary", value=["spike", "flat"]),)


def _loader() -> _ToyLoader:
    return _ToyLoader.parse_from(["train", "test"])


def _targets(tasks: Iterable[Task]) -> list[tuple[Any, ...] | None]:
    return [task.targets for task in tasks]


def test_populate_fills_the_dataset_and_streams_the_tasks():
    loader = _loader()
    dataset = TimeFDataset(metadata=_METADATA)
    assert loader.populate(dataset) is dataset
    assert [(a.key, a.value) for a in dataset.annotations] == [("label_vocabulary", ["spike", "flat"])]
    assert [record.sources[0].name for record in dataset.records] == ["train", "test"]
    # The stream starts again on each read, in the order of tasks().
    first, second = list(dataset.iter_streamed_tasks_validated()), list(dataset.iter_streamed_tasks_validated())
    assert _targets(first) == _targets(second) == _targets(loader.tasks())
    # The caption keeps having no split.
    assert [task.split for task in first] == [Split.TRAIN, Split.TEST, Split.TRAIN, None]
    assert dataset.derive_schema().tasks == (ClassificationTask, AnswerTask)
