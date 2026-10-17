"""Real, isolated TimeF fixtures shared by viewer integration tests."""

from collections.abc import Callable, Iterator
from contextlib import ExitStack
from pathlib import Path

from fastapi.testclient import TestClient
import pytest

from timenet.dataset import TimeFDataset
from timenet.reader import TimeFReader
from timenet.registry import LocalRegistry
from timenet.testing import make_dataset
from timenet.viewer.app import create_app, create_registry_app
from timenet.viewer.launch import open_reader
from timenet.writer import TimeFWriter


@pytest.fixture
def stored(tmp_path: Path) -> tuple[Path, str]:
    dataset = make_dataset()
    with TimeFWriter(tmp_path, dataset) as writer:
        writer.write()
    return tmp_path, dataset.metadata.dataset_id


@pytest.fixture
def reader(stored: tuple[Path, str]) -> Iterator[TimeFReader]:
    directory, dataset_id = stored
    with open_reader(dataset_id, None, str(directory), None) as opened:
        yield opened


@pytest.fixture
def client(reader: TimeFReader) -> Iterator[TestClient]:
    with TestClient(
        create_app(reader, token="test", port=8000),
        base_url="http://127.0.0.1:8000",
        headers={"Authorization": "Bearer test"},
    ) as client:
        yield client


@pytest.fixture
def registry_client(stored: tuple[Path, str]) -> Iterator[TestClient]:
    directory, _ = stored
    with TestClient(
        create_registry_app(LocalRegistry(directory), token="test", port=8000, registry_name="local"),
        base_url="http://127.0.0.1:8000",
        headers={"Authorization": "Bearer test"},
    ) as client:
        yield client


@pytest.fixture
def open_dataset(tmp_path: Path) -> Iterator[Callable[[TimeFDataset], TimeFReader]]:
    with ExitStack() as stack:

        def open_written(dataset: TimeFDataset) -> TimeFReader:
            with TimeFWriter(tmp_path, dataset, values_backend="zarr") as writer:
                writer.write()
            return stack.enter_context(open_reader(dataset.metadata.dataset_id, None, str(tmp_path), None))

        yield open_written
