from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Self

import pytest

from timenet.dataset import Record, Signal, Source, TimeFDataset
from timenet.dataset.axis import OrdinalAxis
from timenet.engine import store_dataset
from timenet.errors import TimeFValidationError
from timenet.reader import TimeFReader
from timenet.registry import open_local_version
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
_PROVENANCE = Annotation(key="source_revision", value="v1")


def _signal() -> Signal:
    return Signal.from_values([1.0, 2.0], spec=_SPEC, name="series", time_axis=OrdinalAxis())


class _ToyLoader(DatasetLoader[str]):
    """A label task for each Record, a severity task for the train Record, and a caption without a split.

    Each Record has a ``site`` annotation. The caption refers to the ``site`` of its Record and to the
    label vocabulary of the dataset.
    """

    TASK_TYPES = (ClassificationTask, AnswerTask)

    def __init__(self, records: tuple[Record, ...]) -> None:
        self._records = records
        self._vocabulary: Annotation | None = None

    @classmethod
    def parse_from(cls, refs: Iterable[str]) -> Self:
        records = []
        for name in refs:
            record = Record(sources=(Source(name=name, signals=(_signal(),)),))
            record.annotate(Annotation(key="site", value=name))
            records.append(record)
        return cls(tuple(records))

    def records(self) -> Iterator[Record]:
        return iter(self._records)

    def annotate_dataset(self, dataset: TimeFDataset) -> None:
        self._vocabulary = dataset.annotate(Annotation(key="label_vocabulary", value=["spike", "flat"]))

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
        assert self._vocabulary is not None
        yield from self.label_tasks()
        yield from self.severity_tasks()
        train, _ = self._records
        yield AnswerTask(
            inputs=(train,),
            prompt="Describe it.",
            targets=("A spike.",),
            input_annotations=(*train.annotations, self._vocabulary),
            annotations=(Annotation(key="difficulty", value="easy"),),
        )


def _loader() -> _ToyLoader:
    return _ToyLoader.parse_from(["train", "test"])


def _dataset() -> TimeFDataset:
    """A dataset with the annotation that a connector adds before it calls populate."""
    dataset = TimeFDataset(metadata=_METADATA)
    dataset.annotate(_PROVENANCE)
    return dataset


def test_populate_adds_the_records_and_the_annotations():
    dataset = _dataset()
    assert _loader().populate(dataset) is dataset
    assert [a.key for a in dataset.annotations] == ["source_revision", "label_vocabulary"]
    assert [record.sources[0].name for record in dataset.records] == ["train", "test"]
    assert dataset.derive_schema().tasks == (ClassificationTask, AnswerTask)


def test_the_task_stream_starts_again_on_each_read():
    dataset = _loader().populate(_dataset())
    first, second = list(dataset.iter_streamed_tasks_validated()), list(dataset.iter_streamed_tasks_validated())
    expected = [("spike",), ("flat",), ("high",), ("A spike.",)]
    assert [task.targets for task in first] == [task.targets for task in second] == expected


def test_populate_rejects_a_dataset_with_a_task_stream():
    dataset = _loader().populate(_dataset())
    with pytest.raises(TimeFValidationError, match="without tasks"):
        _loader().populate(dataset)
    # The second loader added nothing.
    assert len(dataset.records) == 2
    assert len(dataset.annotations) == 2


def test_populate_rejects_a_dataset_with_added_tasks():
    dataset = _dataset()
    record = dataset.add_record(record=Record(sources=(Source(name="other", signals=(_signal(),)),)))
    dataset.add_task(task=AnswerTask(inputs=(record,), prompt="Describe it.", targets=("Flat.",)))
    with pytest.raises(TimeFValidationError, match="without tasks"):
        _loader().populate(dataset)
    assert len(dataset.records) == 1
    assert len(dataset.annotations) == 1


class _EmptyTypes(_ToyLoader):
    TASK_TYPES = ()


class _MissingTypes(DatasetLoader[str]):
    """A loader that does not set ``TASK_TYPES``."""

    @classmethod
    def parse_from(cls, refs: Iterable[str]) -> Self:
        return cls()

    def records(self) -> Iterator[Record]:
        return iter(())

    def tasks(self) -> Iterator[Task]:
        return iter(())


@pytest.mark.parametrize("loader", [_EmptyTypes.parse_from(["train", "test"]), _MissingTypes.parse_from([])])
def test_populate_rejects_a_loader_without_task_types(loader: DatasetLoader[str]):
    dataset = _dataset()
    with pytest.raises(TimeFValidationError, match="TASK_TYPES"):
        loader.populate(dataset)
    assert dataset.records == ()
    assert len(dataset.annotations) == 1


def test_a_loader_that_fills_a_second_dataset_leaves_the_first_one_writable(tmp_path: Path):
    loader = _loader()
    first = loader.populate(_dataset())
    # The Records already belong to the first dataset, so the second populate fails.
    with pytest.raises(TimeFValidationError):
        loader.populate(_dataset())
    store_dataset(first, tmp_path / "out")


def test_a_task_refers_to_annotations_of_the_dataset_and_of_its_record_after_a_write(tmp_path: Path):
    dataset = _loader().populate(_dataset())
    with TimeFReader(open_local_version(store_dataset(dataset, tmp_path / "out"))) as reader:
        restored = reader.read()
    provenance, vocabulary = restored.annotations
    assert (provenance.key, vocabulary.key) == ("source_revision", "label_vocabulary")
    assert [[a.value for a in record.annotations] for record in restored.records] == [["train"], ["test"]]
    (caption,) = restored.tasks_of(AnswerTask)
    site, task_vocabulary = caption.input_annotations
    assert site is restored.records[0].annotations[0]
    assert task_vocabulary is vocabulary
    assert [(a.key, a.value) for a in caption.annotations] == [("difficulty", "easy")]
