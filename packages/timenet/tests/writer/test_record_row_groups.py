import numpy as np
import pyarrow.parquet as pq
import pytest

from timenet.dataset import Record, RegularAxis, Signal, Source, TimeFDataset
from timenet.reader import TimeFReader
from timenet.registry import DatasetVersion
from timenet.testing import make_dataset
from timenet.writer import TimeFWriter


def _parent():
    parent = TimeFDataset(metadata=make_dataset().metadata)
    spec = make_dataset().records[0].signals[0].spec
    expected = {}
    rng = np.random.default_rng(1)
    for name, samples, leads in (("a", 5000, 12), ("b", 5000, 12), ("small-1", 8, 1), ("small-2", 8, 1)):
        signals = []
        for lead in range(leads):
            values = rng.standard_normal(samples, dtype=np.float32)
            identity = f"{name}-{lead}"
            expected[identity] = values
            signals.append(
                Signal.from_values(
                    values, spec=spec, name=str(lead), id=identity, time_axis=RegularAxis.from_rate_hz(500)
                )
            )
        parent.add_record(record=Record(record_id=name, sources=(Source(name="sensor", signals=tuple(signals)),)))
    return parent, expected


@pytest.mark.parametrize("layout", ["packed", "record", "auto"])
def test_child_record_reads_only_its_parent_group_when_records_are_separated(tmp_path, monkeypatch, layout):
    parent, expected = _parent()
    reads = []
    original_values = Signal.to_arrow

    def tracked_values(signal):
        reads.append(signal.id)
        return original_values(signal)

    with monkeypatch.context() as guard, TimeFWriter(tmp_path, parent, row_group_layout=layout) as writer:
        guard.setattr(Signal, "to_arrow", tracked_values)
        writer.write()
    assert sorted(reads) == sorted(expected)
    parent_path = tmp_path / parent.metadata.dataset_id / "1.0.0"
    manifest = DatasetVersion.open_local(parent_path).manifest
    assert manifest.value_encoding[parent.records[0].signals[0].spec.spec_type] == "byte_stream_split"
    child = TimeFDataset(metadata=parent.metadata.model_copy(update={"dataset_id": "test/child"}))
    with TimeFReader(DatasetVersion.open_local(parent_path)) as reader:
        child.copy_from(reader, record_ids=["b"])
    child_path = child.write(path=tmp_path)
    original = pq.ParquetFile.read_row_group
    decoded = []

    def tracked(file, index, *args, **kwargs):
        decoded.append(index)
        return original(file, index, *args, **kwargs)

    with monkeypatch.context() as guard:
        guard.setattr(pq.ParquetFile, "read_row_group", tracked)
        with TimeFReader(DatasetVersion.open_local(child_path)) as reader:
            record = next(reader.iter_records())
            for signal in record.signals:
                np.testing.assert_array_equal(signal.to_numpy(), expected[signal.id])
            np.testing.assert_array_equal(
                record.signals[0].values.read_steps(20, 30).to_numpy(), expected[record.signals[0].id][20:30]
            )
    assert len(decoded) == 1
    reference = record.signals[0].values.reference
    assert reference is not None
    chunk = reference.chunks[0]
    with pq.ParquetFile(parent_path / chunk.path) as file:
        ids = file.read_row_group(chunk.major, columns=["time_series_id"]).column(0).to_pylist()
    if layout == "packed":
        assert "a-0" in ids
    else:
        assert set(ids) == {f"b-{lead}" for lead in range(12)}
    if layout == "auto":
        with TimeFReader(DatasetVersion.open_local(parent_path)) as reader:
            small = [next(reader.iter_records([name])).signals[0].values.reference for name in ("small-1", "small-2")]
        assert small[0] is not None and small[1] is not None
        assert small[0].chunks[0].major == small[1].chunks[0].major
