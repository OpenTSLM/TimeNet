from timenet.dataset import Record, RegularAxis, Signal, Source, TimeFDataset
from timenet.format.duckdb import connect_control
from timenet.reader import TimeFReader
from timenet.registry import DatasetVersion
from timenet.testing import make_dataset
from timenet.types import TimeSeriesSpec, ureg


def test_shared_contracts_and_release_ids_survive_catalog_normalization(tmp_path):
    parent = TimeFDataset(metadata=make_dataset().metadata)
    spec = TimeSeriesSpec(spec_type="ecg", name="ECG", unit_value=ureg.millivolt, dtype="float32")
    parent.add_record(
        record=Record(
            record_id="record",
            sources=(
                Source(
                    name="ECG",
                    id="source",
                    signals=tuple(
                        Signal.from_values(
                            [1.0, 2.0, 3.0], spec=spec, name=name, time_axis=RegularAxis.from_rate_hz(500), id=name
                        )
                        for name in ("I", "II")
                    ),
                ),
            ),
        )
    )
    parent_path = parent.write(path=tmp_path)
    child = TimeFDataset(metadata=parent.metadata.model_copy(update={"dataset_id": "test/child"}))
    with TimeFReader(DatasetVersion.open_local(parent_path)) as reader:
        child.copy_from(reader)
    child_path = child.write(path=tmp_path)
    with connect_control(child_path / "control.duckdb", read_only=True) as connection:
        assert connection.execute("SELECT count(*) FROM dataset_refs").fetchone() == (1,)
        assert connection.execute("SELECT count(*) FROM payload_descriptors").fetchone() == (1,)
        assert connection.execute("SELECT count(*) FROM payload_files").fetchone() == (1,)
        assert connection.execute("SELECT count(*) FROM payload_chunks").fetchone() == (2,)
    with TimeFReader(DatasetVersion.open_local(child_path)) as reader:
        record = next(reader.iter_records())
    first, second = record.signals
    assert first.values.descriptor is second.values.descriptor
    assert first.values.reference is not None
    assert second.values.reference is not None
    assert first.origin is not None
    assert second.origin is not None
    assert first.values.reference.source is second.values.reference.source
    assert first.origin.dataset is second.origin.dataset
    assert first.origin.id == "I"
    assert second.origin.id == "II"
    assert first.to_arrow().to_pylist() == second.to_arrow().to_pylist() == [1.0, 2.0, 3.0]
