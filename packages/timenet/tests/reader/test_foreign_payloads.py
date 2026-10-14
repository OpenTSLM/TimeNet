from copy import deepcopy
import pickle

import pytest

from timenet.dataset import Record, Signal, Source, TimeFDataset
from timenet.errors import TimeFFormatError
from timenet.manifest import Manifest
from timenet.payloads.reader import StoredPayloadLoader
from timenet.reader import TimeFReader
from timenet.registry.version import DatasetVersion
from timenet.testing import make_dataset
from timenet.types import Annotation
from timenet.writer import TimeFWriter


def test_child_and_grandchild_share_pinned_payloads_without_body_reads(tmp_path, monkeypatch):
    parent = make_dataset()
    spec = parent.records[0].signals[0].spec
    parent.add_record(
        record=Record(
            record_id="irregular",
            sources=(
                Source(
                    id="irregular-source",
                    name="monitor",
                    signals=(
                        Signal.from_irregular(
                            [1.0, 2.0, 3.0], time_offsets_us=[0, 17, 200], spec=spec, name="irregular"
                        ),
                    ),
                ),
            ),
        )
    )
    parent.annotate(Annotation(key="manual", value="maintenance" * 10000))
    with TimeFWriter(tmp_path, parent, values_backend="zarr") as writer:
        writer.write()
    parent_path = tmp_path / parent.metadata.dataset_id / str(parent.metadata.dataset_version)
    with TimeFReader(DatasetVersion.open_local(parent_path)) as reader:
        loaded = reader.read()
        expected = loaded.records[0].signals[0].values.read()

    def forbidden(self):
        pytest.fail("writing a descendant read a stored payload body")

    with monkeypatch.context() as guard:
        guard.setattr(StoredPayloadLoader, "__call__", forbidden)
        for name in ("child", "grandchild"):
            child = TimeFDataset(
                metadata=parent.metadata.model_copy(update={"dataset_id": f"timenet/{name}", "name": name})
            )
            # Metadata graph copy is implemented separately; the physical handles are already immutable.
            for record in deepcopy(loaded.records):
                child.add_record(record=record)
            for annotation in deepcopy(loaded.annotations):
                child.annotate(annotation)
            with TimeFWriter(tmp_path, child, values_backend="parquet") as writer:
                writer.write()
            path = tmp_path / child.metadata.dataset_id / "1.0.0"
            manifest = Manifest.from_artifact((path / "manifest.json").read_bytes())
            assert len(manifest.payload_sources) == 1
            assert manifest.payload_sources[0].dataset.dataset_id == parent.metadata.dataset_id
            assert manifest.files.time_series == ()
            assert manifest.files.payloads == ()
            with TimeFReader(DatasetVersion.open_local(path)) as reader:
                loaded = reader.read()
    (parent_path / "control.duckdb").unlink()
    (tmp_path / "timenet/child/1.0.0/control.duckdb").unlink()
    assert pickle.loads(pickle.dumps(loaded.records[0].signals[0].values)).read().equals(expected)
    irregular = next(signal for record in loaded.records for signal in record.signals if signal.name == "irregular")
    assert irregular.time_offsets is not None
    assert pickle.loads(pickle.dumps(irregular.time_offsets)).read().to_pylist() == [0, 17, 200]
    assert loaded.annotations[0].read_value() == "maintenance" * 10000
    reference = loaded.records[0].signals[0].values.reference
    assert reference is not None
    source = reference.source
    owner_manifest = parent_path / "manifest.json"
    owner_manifest.write_bytes(owner_manifest.read_bytes() + b"\n")
    with TimeFReader(DatasetVersion.open_local(tmp_path / "timenet/grandchild/1.0.0")) as reader:
        signal = next(reader.iter_records()).signals[0]
    with pytest.raises(TimeFFormatError, match="checksum mismatch"):
        signal.values.read()
    assert source is not None
