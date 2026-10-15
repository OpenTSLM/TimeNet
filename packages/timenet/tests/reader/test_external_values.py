import pytest

from timenet.dataset import Record, Signal, Source, TimeFDataset
from timenet.format.duckdb import connect_control
from timenet.format.external import ValueHandle
from timenet.reader import TimeFReader
from timenet.registry import open_local_version
from timenet.testing import make_dataset
from timenet.types import Annotation, AnswerTask
from timenet.writer import TimeFWriter


def _dataset():
    sample = make_dataset()
    dataset = TimeFDataset(metadata=sample.metadata)
    signal = Signal.from_irregular(
        [1.0, 2.0, 3.0], time_offsets_us=[0, 7, 19], spec=sample.records[0].signals[0].spec, name="irregular"
    )
    record = dataset.add_record(record=Record(record_id="r", sources=(Source(name="sensor", signals=(signal,)),)))
    long_text = "A detailed clinical report. " * 300
    labels = [f"diagnosis-{index:04d}" for index in range(512)]
    record.annotate(Annotation(key="report", value=long_text))
    record.annotate(Annotation(key="labels", value=labels))
    record.annotate(Annotation(key="note", value="normal"))
    dataset.add_task(
        task=AnswerTask(id="q", inputs=(record,), prompt=long_text, targets=(long_text,), rationale=long_text)
    )
    return dataset, long_text, labels


def _write(root, dataset):
    with TimeFWriter(root, dataset) as writer:
        writer.write()
    return root / dataset.metadata.dataset_id / str(dataset.metadata.dataset_version)


def test_external_values_and_offsets_round_trip(tmp_path):
    dataset, long_text, labels = _dataset()
    root = _write(tmp_path, dataset)
    with TimeFReader(open_local_version(root)) as reader:
        restored = reader.read()
        annotations = {annotation.key: annotation.value for annotation in restored.records[0].annotations}
        assert isinstance(annotations["report"], ValueHandle)
        assert annotations["report"].read() == long_text
        assert isinstance(annotations["labels"], ValueHandle)
        assert annotations["labels"].read() == labels
        assert annotations["note"] == "normal"
        assert isinstance(restored.tasks[0].prompt, ValueHandle)
        assert restored.tasks[0].prompt.read() == long_text
        assert isinstance(restored.tasks[0].rationale, ValueHandle)
        assert restored.tasks[0].rationale.read() == long_text
        assert restored.tasks[0].targets is not None
        assert isinstance(restored.tasks[0].targets[0], ValueHandle)
        assert restored.tasks[0].targets[0].read() == long_text
        assert reader.task_table()["prompt"].to_pylist() == [long_text]
        assert reader.target_table()["text_value"].to_pylist() == [long_text]
        annotation_table = reader.annotation_table()
        assert long_text in annotation_table["text_value"].to_pylist()
        assert labels in annotation_table["text_list_value"].to_pylist()
        assert restored.records[0].signals[0].time_offsets_us().tolist() == [0, 7, 19]
        with connect_control(root / "control.duckdb", read_only=True) as connection:
            assert connection.execute("SELECT count(*) FROM axis_offsets").fetchone() == (0,)
            assert connection.execute("SELECT count(*) FROM external_values").fetchone() == (6,)
    assert not reader._external_reader.cache


def test_inherited_external_values_are_reused_without_decoding(tmp_path, monkeypatch):
    dataset, long_text, labels = _dataset()
    original = _write(tmp_path, dataset)
    with TimeFReader(open_local_version(original)) as reader:
        restored = reader.read()
        child = TimeFDataset(metadata=restored.metadata.model_copy(update={"dataset_id": "test/child"}))
        for record in restored.records:
            child.add_record(record=record)
        child.add_tasks(tasks=restored.tasks)

        def unexpected_read(*args):
            pytest.fail("copying stored payloads must not decode them")

        with monkeypatch.context() as patch:
            patch.setattr(ValueHandle, "read", unexpected_read)
            rewritten = _write(tmp_path, child)
        assert not (rewritten / "payloads").exists()
        assert not (rewritten / "time_series").exists()

    with TimeFReader(open_local_version(rewritten)) as reader:
        restored = reader.read()
        annotations = {annotation.key: annotation.value for annotation in restored.records[0].annotations}
        assert annotations["report"].read() == long_text
        assert annotations["labels"].read() == labels
        assert isinstance(restored.tasks[0].prompt, ValueHandle)
        assert restored.tasks[0].prompt.read() == long_text
        signal = restored.records[0].signals[0]
        assert signal.time_offsets_us().tolist() == [0, 7, 19]
        assert signal.to_numpy().tolist() == [1.0, 2.0, 3.0]
