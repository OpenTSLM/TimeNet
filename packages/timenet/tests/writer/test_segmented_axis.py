import duckdb
import numpy as np
import pytest

from timenet.dataset import Record, RegularAxis, RegularRun, SegmentedRegularAxis, Signal, Source, TimeFDataset
from timenet.errors import TimeFFormatError
from timenet.format.control_audit import audit_control_database
from timenet.reader import TimeFReader
from timenet.registry import open_local_version
from timenet.types import DatasetMetadata, License, TimeInterval, TimeSeriesSpec, Version
from timenet.types.references import DatasetRef
from timenet.writer import TimeFWriter


def write_dataset(tmp_path, backend="parquet", segmented=True):
    axis = (
        SegmentedRegularAxis(
            runs=(
                RegularRun(axis=RegularAxis.from_rate_hz(100), n_values=30),
                RegularRun(axis=RegularAxis.from_rate_hz(100, offset_us=500_000), n_values=20),
            )
        )
        if segmented
        else RegularAxis.from_rate_hz(100)
    )
    dataset = TimeFDataset(
        metadata=DatasetMetadata(
            dataset_id="test/segmented",
            dataset_version=Version(1, 0, 0),
            name="Segmented",
            description="Test",
            license=License.MIT,
        )
    )
    vector = backend == "zarr"
    values = np.arange(150).reshape(50, 3) if vector else np.arange(50)
    signals = tuple(
        Signal(
            name=name,
            spec=TimeSeriesSpec(
                spec_type="acceleration",
                name="Acceleration",
                unit_value=None,
                value_shape=(3,) if vector else (),
            ),
            time_axis=axis,
            data=values,
        )
        for name in ("a", "b")
    )
    dataset.add_record(record=Record(sources=(Source(name="wrist", signals=signals),)))
    dataset.derive_schema()
    with TimeFWriter(tmp_path, dataset, values_backend=backend, chunk_max_bytes=120) as writer:
        writer.write()
    return next(tmp_path.rglob("manifest.json")).parent, axis


@pytest.mark.parametrize("backend", ["parquet", "zarr"])
def test_segmented_roundtrip_and_shared_axis(tmp_path, backend):
    path, axis = write_dataset(tmp_path, backend)
    with duckdb.connect(str(path / "control.duckdb"), read_only=True) as connection:
        audit_control_database(connection)
        assert connection.execute("SELECT count(*) FROM axes").fetchone() == (1,)
        assert connection.execute("SELECT count(*) FROM axis_runs").fetchone() == (2,)
        assert connection.execute("SELECT count(*) FROM external_values WHERE object_kind = 'Axis'").fetchone() == (0,)
    with TimeFReader(open_local_version(path)) as reader:
        a, b = next(reader.iter_records()).signals
        assert a.time_axis == b.time_axis == axis
        assert a._time_offsets is None
        part = a.slice(TimeInterval.micros(200_000, 600_000))
        expected = np.arange(150).reshape(50, 3) if backend == "zarr" else np.arange(50)
        np.testing.assert_array_equal(part.to_numpy(), expected[20:40])
        assert isinstance(part.time_axis, SegmentedRegularAxis)
        assert not part.is_contiguous(TimeInterval.micros(200_000, 600_000))


@pytest.mark.parametrize(
    "mutation",
    [
        "UPDATE axis_runs SET n_values = 0 WHERE run_index = 0",
        "UPDATE axis_runs SET offset_us = 1 WHERE run_index = 1",
        "UPDATE axis_runs SET run_index = 3 WHERE run_index = 1",
        "DELETE FROM axis_runs WHERE run_index = 1",
        "UPDATE axes SET period_denominator = 0",
    ],
)
def test_corrupt_runs_rejected_by_audit_and_reader(tmp_path, mutation):
    path, _ = write_dataset(tmp_path)
    version = open_local_version(path)
    with duckdb.connect(str(path / "control.duckdb")) as connection:
        connection.execute(mutation)
        with pytest.raises(TimeFFormatError):
            audit_control_database(connection)
    with TimeFReader(version) as reader, pytest.raises(TimeFFormatError):
        next(reader.iter_records())


def test_schema_v1_without_run_table_remains_readable(tmp_path):
    path, axis = write_dataset(tmp_path, segmented=False)
    version = open_local_version(path)
    with duckdb.connect(str(path / "control.duckdb")) as connection:
        connection.execute("DROP TABLE axis_runs")
        connection.execute("UPDATE control_metadata SET value = '1' WHERE key = 'schema_version'")
        audit_control_database(connection)
    with TimeFReader(version) as reader:
        assert next(reader.iter_records()).signals[0].time_axis == axis


@pytest.mark.parametrize("backend", ["parquet", "zarr"])
def test_import_preserves_runs_and_reuses_values(tmp_path, backend, monkeypatch):
    path, axis = write_dataset(tmp_path, backend)
    with TimeFReader(open_local_version(path)) as reader:
        parent = next(reader.iter_records())
        child = TimeFDataset(
            metadata=reader.metadata.model_copy(
                update={
                    "dataset_id": "test/child",
                    "parents": {
                        "base": DatasetRef(
                            dataset_id=reader.metadata.dataset_id, version=reader.metadata.dataset_version
                        )
                    },
                }
            )
        )
        child.import_from(reader, records=[parent.id])
        child.derive_schema()

        def unexpected_read(*args, **kwargs):
            pytest.fail("importing segmented axes must not read stored samples")

        with monkeypatch.context() as patch:
            patch.setattr(TimeFReader, "_read_array", unexpected_read)
            patch.setattr(TimeFReader, "_read_array_steps", unexpected_read)
            with TimeFWriter(tmp_path, child, values_backend=backend) as writer:
                writer.write()
    child_path = tmp_path / "test/child/1.0.0"
    with TimeFReader(open_local_version(child_path)) as reader:
        back = next(reader.iter_records()).signals[0]
        assert back.time_axis == axis
        assert back.to_numpy().shape == ((50, 3) if backend == "zarr" else (50,))
