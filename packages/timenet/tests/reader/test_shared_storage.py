from copy import deepcopy

import pytest

from timenet.dataset import TimeFDataset
from timenet.errors import TimeFValidationError
from timenet.format.stored_values import ValueHandle
from timenet.manifest import Manifest
from timenet.reader import TimeFReader
from timenet.registry import open_local_version
from timenet.testing import make_dataset
from timenet.types import Annotation, AnswerTask
from timenet.writer import TimeFWriter


def store(root, dataset, **kwargs):
    dataset.derive_schema()
    with TimeFWriter(root, dataset, **kwargs) as writer:
        writer.write()
    return root / dataset.metadata.dataset_id / str(dataset.metadata.dataset_version)


@pytest.mark.parametrize("backend", ["parquet", "zarr"])
def test_prepared_shared_payloads_do_not_open_parent_catalogs(tmp_path, monkeypatch, backend):
    original = make_dataset()
    text = "A detailed clinical report. " * 300
    original.tasks[0].prompt = text
    report = original.records[0].annotate(Annotation(key="report", value=text))
    original.records[0].signals[0].annotate(report)
    expected = original.records[0].signals[0].to_arrow()
    parent_dir = store(tmp_path, original, values_backend=backend)
    reader = TimeFReader(open_local_version(parent_dir))
    parent = reader.get_handle().reader._read_catalog()
    child = TimeFDataset(metadata=original.metadata.model_copy(update={"dataset_id": "test/child"}))
    child.add_record(record=deepcopy(parent.records[0]))
    prompt = next(task.prompt for task in parent.tasks if task.id == original.tasks[0].id)
    assert prompt is not None
    child.add_task(task=AnswerTask(id="question", inputs=(child.records[0],), prompt=prompt, targets=(prompt,)))

    def unexpected_read(*args):
        pytest.fail("inheriting packed values must not decode or copy them")

    with monkeypatch.context() as patch:
        patch.setattr(ValueHandle, "read", unexpected_read)
        child_dir = store(tmp_path, child)
    reader.close()
    manifest = Manifest.model_validate_json((child_dir / "manifest.json").read_text())
    assert not manifest.files.time_series
    assert not manifest.files.payloads
    assert len(manifest.payload_sources) == 1
    # ID/version is the pin; descriptive metadata does not change the reference.
    source = Manifest.model_validate_json((parent_dir / "manifest.json").read_text())
    metadata = source.metadata.model_copy(update={"description": "Updated description"})
    source = source.model_copy(update={"metadata": metadata})
    (parent_dir / "manifest.json").write_text(source.to_json())
    prepared = open_local_version(child_dir)
    # Preparation requires the full release, but processing never opens parent catalogs.
    (parent_dir / "control.duckdb").rename(parent_dir / "unavailable.duckdb")
    (parent_dir / "manifest.json").unlink()
    with TimeFReader(prepared) as child_reader:
        assert child_reader.read().read().records[0].signals[0].to_arrow().equals(expected)
        assert child_reader.create_task_table()["prompt"].to_pylist() == [text]
        assert child_reader.create_target_table()["text_value"].to_pylist() == [text]
        assert text in child_reader.create_annotation_table()["text_value"].to_pylist()


def test_shared_payloads_cannot_reference_the_output_release(tmp_path):
    path = store(tmp_path, make_dataset())
    original_manifest = (path / "manifest.json").read_bytes()
    with TimeFReader(open_local_version(path)) as reader:
        loaded = reader._read_catalog()
        with pytest.raises(TimeFValidationError, match="also the output release"):
            store(tmp_path / "copy", loaded)
    assert (path / "manifest.json").read_bytes() == original_manifest
    with TimeFReader(open_local_version(path)) as reader:
        assert reader.read().read().records
