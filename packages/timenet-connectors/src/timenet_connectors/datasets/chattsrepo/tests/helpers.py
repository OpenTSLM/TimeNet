"""Reusable fixture assertions for the ChatTS configuration connectors."""

from timenet.dataset import TimeFDataset
from timenet.types import AnswerTask
from timenet_connectors.datasets.chattsrepo.chatts_core.connector import ChatTSConnector


ROWS = [
    {
        "input": "Compare the two signals.",
        "timeseries": [[1.0, 2.0, 3.0], [3.0, 2.0]],
        "output": "The first rises while the second falls.",
    }
]


def assert_configuration_conversion(connector: ChatTSConnector, configuration: str) -> None:
    """Assert the shared conversion contract for one ChatTS configuration."""
    dataset = connector.convert(ROWS)
    assert isinstance(dataset, TimeFDataset)
    assert len(dataset.records) == 1
    (record,) = dataset.records
    assert record.metadata["chatts_configuration"] == configuration
    assert [signal.to_numpy().tolist() for signal in record.signals] == [[1.0, 2.0, 3.0], [3.0, 2.0]]
    (task,) = dataset.tasks
    assert isinstance(task, AnswerTask)
    assert task.prompt == ROWS[0]["input"]
    assert task.targets == (ROWS[0]["output"],)
    assert task.metadata["chatts_configuration"] == configuration
    assert record.id == f"{configuration}-0"
    assert task.id == f"{configuration}-0-answer"
    assert task.inputs == (record,)
    assert task.targets is not None
