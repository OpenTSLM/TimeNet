import shutil

import pytest

from timenet.dataset import Record, Signal, Source, TimeFDataset
from timenet.payloads.reader import StoredPayloadLoader
from timenet.reader import TimeFReader
from timenet.registry.version import DatasetVersion
from timenet.testing import make_dataset
from timenet.types import Annotation, TimeSeriesSpec, ureg
from timenet.writer import TimeFWriter


@pytest.mark.parametrize("backend", ["parquet", "zarr"])
def test_standalone_copy_preserves_codecs_and_never_decodes_stored_payloads(tmp_path, monkeypatch, backend):
    source = tmp_path / "source"
    parent = make_dataset()
    spec = TimeSeriesSpec(
        spec_type="nullable", name="Nullable monitor", dtype="float32", nullable=True, unit_value=ureg.dimensionless
    )
    signal = Signal.from_irregular([1.0, None, 3.0], time_offsets_us=[0, 17, 200], name="irregular", spec=spec)
    parent.add_record(record=Record(record_id="nullable", sources=(Source(name="monitor", signals=(signal,)),)))
    parent.annotate(Annotation(id="manual", key="manual", value="maintenance" * 10000))
    parent_path = parent.write(path=source, values_backend=backend)
    child = TimeFDataset(metadata=parent.metadata.model_copy(update={"dataset_id": "timenet/standalone"}))

    def forbidden(self):
        pytest.fail("copy policy decoded a stored payload")

    with TimeFReader(DatasetVersion.open_local(parent_path)) as reader, monkeypatch.context() as guard:
        guard.setattr(StoredPayloadLoader, "__call__", forbidden)
        child.copy_from(reader, namespace="base", include_tasks=True)
        output = tmp_path / "output"
        with TimeFWriter(output, child, payload_policy="copy", values_backend="parquet") as writer:
            writer.write()
    path = output / child.metadata.dataset_id / "1.0.0"
    manifest = DatasetVersion.open_local(path).manifest
    assert manifest.payload_sources == ()
    copied = manifest.files.payloads
    assert len({part.path for part in copied}) == len(copied)
    parent_parts = {part.path: part for part in DatasetVersion.open_local(parent_path).manifest.files.all_files()}
    for part in copied:
        original_path = part.path.split("/", 3)[3]
        assert part.checksum == parent_parts[original_path].checksum
    shutil.rmtree(source)
    with TimeFReader(DatasetVersion.open_local(path)) as reader:
        reader.verify()
        loaded = reader.read()
    copied_signal = next(signal for record in loaded.records for signal in record.signals if signal.name == "irregular")
    assert copied_signal.values.read().to_pylist() == [1.0, None, 3.0]
    assert copied_signal.values.reference is not None
    assert copied_signal.values.reference.codec == backend
    assert copied_signal.time_offsets is not None
    assert copied_signal.time_offsets.read().to_pylist() == [0, 17, 200]
    assert loaded.annotations[0].value() == "maintenance" * 10000
    assert loaded.annotations[0].content_origin is not None
    assert loaded.annotations[0].content_origin.id == "manual"
