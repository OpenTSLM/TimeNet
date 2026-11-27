from pathlib import Path

import pytest

from timenet import TimeNet
from timenet.connectors.context import BuildContext
from timenet.errors import TimeNetConnectorError
from timenet.registry import LocalRegistry
from timenet.types import TextHandle
from timenet_connectors.datasets.tsaqa.tsaqa_benchmark import TSAQAConnector
from timenet_connectors.datasets.tsaqa.tsaqa_clean import CONNECTOR, TSAQACleanConnector
from timenet_connectors.discovery import resolve


FIXTURES = Path(__file__).parents[2] / "tsaqa_benchmark" / "tests" / "fixtures"


@pytest.fixture
def parent_dataset():
    files = {
        "train": FIXTURES / "train.parquet",
        "validation": FIXTURES / "val.parquet",
        "test": FIXTURES / "test.parquet",
    }
    return TSAQAConnector().convert(files)


def identity(task):
    return task.metadata["file"], task.metadata["row"]


def task_content(task):
    return (
        task.prompt.read(),
        task.targets[0].read(),
        task.split,
        task.metadata,
        [record.metadata for record in task.inputs],
        [(annotation.key, annotation.value.read()) for annotation in task.input_annotations],
    )


def test_inheritance_excludes_only_actual_ordering_and_reuses_parent_values(tmp_path, parent_dataset):
    registry = LocalRegistry(tmp_path / "registry")
    registry.put_dataset(parent_dataset)
    client = TimeNet(sources=[registry])
    original = {identity(task): task for task in parent_dataset.iter_tasks()}
    expected = {key: task for key, task in original.items() if "\nPatches:\n" not in task.metadata["original_question"]}
    assert len(expected) < len(original)

    connector = TSAQACleanConnector()
    with client.open("tsaqa/tsaqa-benchmark@1.0.0") as parent:
        context = BuildContext(parents={connector.parent_alias: parent})
        child = connector.compose(None, context)
        assert connector.parent_alias in child.parent_cards
        registry.put_dataset(child)

    manifest = client.get_manifest("tsaqa/tsaqa-clean@1.0.0")
    assert not manifest.files.time_series
    assert {str(source.release) for source in manifest.payload_sources.values()} == {"tsaqa/tsaqa-benchmark@1.0.0"}

    with client.open("tsaqa/tsaqa-clean@1.0.0") as reader:
        reader.verify()
        restored = reader.read()
        actual = {identity(task): task for task in restored.iter_tasks()}
        assert list(actual) == list(expected)
        for key, task in actual.items():
            assert task_content(task) == task_content(expected[key])
            for record in task.inputs:
                for annotation in record.walk_annotations():
                    assert any(annotation is selected for selected in task.input_annotations)
        # These are continuation questions whose published question_type says ordering.
        assert actual["train.parquet", 12].metadata["question_format"] == "true_or_false"
        assert actual["train.parquet", 13].metadata["question_format"] == "multiple_choices"
        assert len(restored.records) == len(parent_dataset.records)
        for before, after in zip(parent_dataset.records, restored.records, strict=True):
            assert before.metadata == after.metadata
            for before_signal, after_signal in zip(before.signals, after.signals, strict=True):
                assert before_signal.metadata == after_signal.metadata
                assert before_signal.to_numpy().tobytes() == after_signal.to_numpy().tobytes()
            for source, inherited in zip(before.walk_annotations(), after.walk_annotations(), strict=True):
                assert isinstance(source.value, TextHandle)
                assert isinstance(inherited.value, TextHandle)
                assert (source.key, source.value.read()) == (inherited.key, inherited.value.read())

    # The complete parent remains available, including ordering tasks.
    with client.open("tsaqa/tsaqa-benchmark@1.0.0") as reader:
        assert {identity(task): task_content(task) for task in reader.iter_tasks()} == {
            key: task_content(task) for key, task in original.items()
        }


def test_missing_resolved_format_fails_instead_of_guessing_from_published_type(tmp_path, parent_dataset):
    first = next(parent_dataset.iter_tasks())
    first.metadata.pop("question_format")
    registry = LocalRegistry(tmp_path / "registry")
    registry.put_dataset(parent_dataset)
    connector = TSAQACleanConnector()
    with (
        TimeNet(sources=[registry]).open("tsaqa/tsaqa-benchmark") as parent,
        pytest.raises(TimeNetConnectorError, match=r"(?s)parent task .*question_format"),
    ):
        connector.compose(None, BuildContext(parents={connector.parent_alias: parent}))


def test_discovery_declares_the_exact_parent_release():
    assert CONNECTOR is TSAQACleanConnector
    assert resolve("tsaqa/tsaqa-clean") is TSAQACleanConnector
    connector = TSAQACleanConnector()
    assert str(connector.metadata().parents[connector.parent_alias]) == "tsaqa/tsaqa-benchmark@1.0.0"
