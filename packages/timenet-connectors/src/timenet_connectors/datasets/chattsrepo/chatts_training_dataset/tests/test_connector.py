"""Tests for ChatTS composition and configuration-specific downloads."""

from pathlib import Path
from typing import cast

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from timenet.composition import BuildContext, ParentDatasetView
from timenet.dataset import TimeFDataset
from timenet.errors import TimeFFormatError
from timenet.reader import TimeFReader
from timenet.registry import DatasetVersion
from timenet.types import Split
from timenet.writer import TimeFWriter
from timenet_connectors.datasets.chattsrepo.align_256 import ChatTSAlign256Connector
from timenet_connectors.datasets.chattsrepo.align_random import ChatTSAlignRandomConnector
from timenet_connectors.datasets.chattsrepo.chatts_training_dataset import ChatTSTrainingDatasetConnector
from timenet_connectors.datasets.chattsrepo.dev import ChatTSDevConnector
from timenet_connectors.datasets.chattsrepo.ift import ChatTSIftConnector
from timenet_connectors.datasets.chattsrepo.sft import ChatTSSftConnector


_ROWS = [{"input": "<ts><ts/>", "timeseries": [[1.0, 2.0]], "output": "a"}]


def _raw_ref(
    tmp_path: Path,
    configuration: str,
    rows: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    """Write one configuration fixture and return its lightweight raw reference."""
    parquet_path = tmp_path / f"{configuration}.parquet"
    pq.write_table(pa.Table.from_pylist(rows if rows is not None else _ROWS), parquet_path)
    return {"path": parquet_path, "__chatts_configuration": configuration}


def _write_parent(root: Path, dataset: TimeFDataset) -> Path:
    """Write a converted parent dataset and return its version directory."""
    with TimeFWriter(root, dataset) as writer:
        writer.write()
    return root / dataset.metadata.dataset_id / str(dataset.metadata.dataset_version)


def test_complete_connector_imports_each_configuration(tmp_path):
    """Compose reader-hydrated parent records and tasks into the complete dataset."""
    connector = ChatTSTrainingDatasetConnector()
    configurations = (
        ChatTSAlign256Connector(),
        ChatTSAlignRandomConnector(),
        ChatTSSftConnector(),
        ChatTSIftConnector(),
        ChatTSDevConnector(),
    )
    readers: list[TimeFReader] = []
    try:
        parents = {}
        for parent in configurations:
            version_dir = _write_parent(tmp_path, parent.convert([_raw_ref(tmp_path, parent.HF_CONFIG)]))
            reader = TimeFReader(DatasetVersion.open_local(version_dir))
            readers.append(reader)
            parents[parent.metadata().dataset_id] = reader
        context = BuildContext(
            connector.metadata(),
            cast("dict[str, ParentDatasetView]", parents),
        )
        dataset = connector.compose([], context)
    finally:
        for reader in readers:
            reader.close()

    assert len(dataset.records) == len(parents)
    assert {task.metadata["chatts_configuration"] for task in dataset.tasks} == {
        "align_256",
        "align_random",
        "sft",
        "ift",
        "dev",
    }
    assert {task.split for task in dataset.tasks if task.metadata["chatts_configuration"] == "dev"} == {
        Split.VALIDATION
    }
    assert {record_import.parent_dataset_id for record_import in dataset.record_imports.values()} == set(parents)
    assert {id(task.inputs[0]) for task in dataset.tasks} == {id(record) for record in dataset.records}
    assert all(signal.to_arrow().type == pa.float64() for record in dataset.records for signal in record.signals)


def test_configuration_download_uses_only_its_parquet_prefix(monkeypatch, tmp_path):
    """Select only the requested Hub configuration rather than all ChatTS rows."""
    huggingface_hub = pytest.importorskip("huggingface_hub")
    parquet_path = tmp_path / "align_256" / "train" / "0000.parquet"
    parquet_path.parent.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(_ROWS), parquet_path)
    seen: dict[str, object] = {}

    def fake_list(repo_id, repo_type=None, revision=None):
        seen["revision"] = revision
        return ["align_256/train/0000.parquet", "sft/train/0000.parquet"]

    def fake_download(repo_id, filename, repo_type=None, revision=None, cache_dir=None):
        seen["filename"] = filename
        return str(parquet_path)

    monkeypatch.setattr(huggingface_hub, "list_repo_files", fake_list)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)

    assert ChatTSAlign256Connector().download(tmp_path / "cache") == [
        {"path": parquet_path, "__chatts_configuration": "align_256"}
    ]
    assert seen == {
        "revision": "30e0c2813f390bb47f4a1619aa9f742bf6a21282",
        "filename": "align_256/train/0000.parquet",
    }


def test_task_stream_rejects_mismatched_timeseries_markers(tmp_path):
    """Reject a prompt whose time-series markers do not match its record signals."""
    dataset = ChatTSAlign256Connector().convert(
        [
            _raw_ref(
                tmp_path,
                "align_256",
                [{"input": "<ts><ts/>", "timeseries": [[1.0], [2.0]], "output": "a"}],
            )
        ]
    )

    with pytest.raises(TimeFFormatError, match=r"align_256 record 'align_256-0'.*1.*2 time series"):
        tuple(dataset.iter_tasks())
