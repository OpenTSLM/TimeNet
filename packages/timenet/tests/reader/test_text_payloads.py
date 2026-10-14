import pickle

import duckdb
import pytest

from timenet.errors import TimeFValidationError
from timenet.payloads.reader import PayloadResolver
from timenet.reader import TimeFReader
from timenet.registry.version import DatasetVersion
from timenet.testing import make_dataset
from timenet.types import Annotation, AnswerTask, Payload, PayloadDescriptor
from timenet.writer import TimeFWriter


def test_large_content_and_task_text_have_lazy_portable_references(tmp_path, monkeypatch):
    dataset = make_dataset()
    manual = "Maintenance manual: 測定\n" * 2000
    annotation = Annotation(id="manual", key="manual", value=manual)
    occurrences = [record.annotate(annotation) for record in dataset.records[:2]]
    external_list = Annotation(key="logs", value=Payload.from_value(["maintenance"] * 1000))
    dataset.annotate(external_list)
    dataset.annotate(Annotation(key="short-external", value=Payload.from_value("small", storage="external")))
    task = AnswerTask(
        id="long-answer", inputs=(dataset.records[0],), prompt=manual, rationale=manual, targets=(manual,)
    )
    dataset.add_task(task=task)
    with TimeFWriter(tmp_path, dataset) as writer:
        writer.write()
    path = tmp_path / dataset.metadata.dataset_id / str(dataset.metadata.dataset_version)
    with duckdb.connect(str(path / "control.duckdb"), read_only=True) as control:
        assert control.execute(
            "SELECT count(*) FROM payloads WHERE object_type='AnnotationContent' AND object_id='manual'"
        ).fetchone() == (1,)
        assert control.execute("SELECT text_value FROM annotation_contents WHERE content_id='manual'").fetchone() == (
            None,
        )
    reads = []
    original = PayloadResolver.text

    def tracked(self, reference, descriptor):
        reads.append(reference)
        return original(self, reference, descriptor)

    monkeypatch.setattr(PayloadResolver, "text", tracked)
    with TimeFReader(DatasetVersion.open_local(path)) as reader:
        loaded = reader.read()
        loaded_task = next(item for item in loaded.iter_tasks() if item.id == task.id)
        refs = reader.annotation_table().to_pylist()
        assert sum(row["value_reference"] is not None for row in refs if row["content_id"] == "manual") == 2
        reader.task_table()
        reader.target_table()
        assert reads == []
        assert loaded_task.prompt is not None
        assert loaded_task.prompt.reference is not None
        materialized = reader.annotation_table(payloads="materialize").to_pylist()
        assert all(row["text_value"] == manual for row in materialized if row["content_id"] == "manual")
        assert next(row for row in materialized if row["key"] == "logs")["text_list_value"] == ["maintenance"] * 1000
        assert (
            next(row for row in reader.target_table(payloads="materialize").to_pylist() if row["task_id"] == task.id)[
                "text_value"
            ]
            == manual
        )
        assert (
            next(row for row in reader.task_table(payloads="materialize").to_pylist() if row["task_id"] == task.id)[
                "prompt"
            ]
            == manual
        )
    (path / "control.duckdb").unlink()
    assert pickle.loads(pickle.dumps(loaded_task.prompt)).read() == manual
    assert loaded_task.targets is not None
    assert pickle.loads(pickle.dumps(loaded_task.targets[0])).read() == manual
    assert len({item.occurrence_id for item in occurrences}) == 2


def test_computed_text_descriptor_mismatch_aborts_the_artifact(tmp_path):
    dataset = make_dataset()
    dataset.annotate(
        Annotation(
            key="manual", value=Payload.from_loader(lambda: "wrong", descriptor=PayloadDescriptor("str", length=10000))
        )
    )
    with pytest.raises(TimeFValidationError, match="descriptor"), TimeFWriter(tmp_path, dataset) as writer:
        writer.write()
    assert not (tmp_path / dataset.metadata.dataset_id / str(dataset.metadata.dataset_version)).exists()
