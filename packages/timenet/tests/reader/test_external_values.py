from copy import deepcopy

import pytest

from timenet.dataset import Record, Signal, Source, TimeFDataset
from timenet.format.duckdb import connect_control
from timenet.format.external import ValueHandle
from timenet.reader import TimeFReader
from timenet.registry import DatasetVersion
from timenet.testing import make_dataset
from timenet.types import Annotation, AnswerTask
from timenet.writer import TimeFWriter


def test_external_values_and_offsets_round_trip_without_decoding_on_copy(tmp_path, monkeypatch):
    sample = make_dataset()
    dataset = TimeFDataset(metadata=sample.metadata)
    signal = Signal.from_irregular(
        [1.0, 2.0, 3.0], time_offsets_us=[0, 7, 19], spec=sample.records[0].signals[0].spec, name="irregular"
    )
    record = dataset.add_record(record=Record(record_id="r", sources=(Source(name="sensor", signals=(signal,)),)))
    long_text = "A detailed clinical report. " * 300
    record.annotate(Annotation(key="report", value=long_text))
    dataset.add_task(
        task=AnswerTask(id="q", inputs=(record,), prompt=long_text, targets=(long_text,), rationale=long_text)
    )
    with TimeFWriter(tmp_path, dataset) as writer:
        writer.write()
    root = tmp_path / dataset.metadata.dataset_id / str(dataset.metadata.dataset_version)
    reader = TimeFReader(DatasetVersion.open_local(root))
    restored = reader.get_handle().reader._read_catalog()
    value = restored.records[0].annotations[0].value
    assert isinstance(value, ValueHandle)
    assert value.read() == long_text
    assert restored.tasks[0].prompt.read() == long_text
    assert restored.tasks[0].targets[0].read() == long_text
    assert restored.records[0].signals[0].time_offsets_us().tolist() == [0, 7, 19]
    with connect_control(root / "control.duckdb", read_only=True) as connection:
        assert connection.execute("SELECT count(*) FROM axis_offsets").fetchone() == (0,)
        assert connection.execute("SELECT count(*) FROM external_values").fetchone() == (5,)

    def unexpected_read(*args):
        pytest.fail("copying stored payloads must not decode them")

    child = TimeFDataset(metadata=dataset.metadata.model_copy(update={"dataset_id": "test/external-child"}))
    child.add_record(record=deepcopy(restored.records[0]))
    monkeypatch.setattr(ValueHandle, "read", unexpected_read)
    with TimeFWriter(tmp_path, child) as writer:
        writer.write()
    assert not (tmp_path / "test/external-child/1.0.0/payloads").exists()
    reader.close()


def test_eager_read_retains_values_after_storage_is_unavailable(tmp_path):
    dataset = make_dataset()
    long_text = "stored clinical context " * 300
    dataset.records[0].annotate(Annotation(key="report", value=long_text))
    with TimeFWriter(tmp_path, dataset) as writer:
        writer.write()
    root = tmp_path / dataset.metadata.dataset_id / str(dataset.metadata.dataset_version)
    with TimeFReader(DatasetVersion.open_local(root)) as reader:
        restored = reader.read()
    (root / "time_series").rename(root / "unavailable-values")
    (root / "payloads").rename(root / "unavailable-text")
    assert next(ann.value for ann in restored.records[0].annotations if ann.key == "report") == long_text
    assert restored.records[0].signals[0].to_arrow().equals(dataset.records[0].signals[0].to_arrow())
