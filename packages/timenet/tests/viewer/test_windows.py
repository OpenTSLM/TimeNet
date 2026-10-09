"""Regression coverage for viewer signal windows."""

from fastapi.testclient import TestClient
import pyarrow as pa
import pytest

from timenet.client import TimeNet
from timenet.dataset import Record, Signal, Source, TimeFDataset
from timenet.dataset.axis import RegularAxis
from timenet.testing import make_dataset
from timenet.types import TimeSeriesSpec
from timenet.viewer.app import create_app
from timenet.writer import TimeFWriter


@pytest.mark.parametrize("mode", ["raw", "plot"])
@pytest.mark.parametrize("component", [0, 1])
def test_tensor_window_preserves_missing_timesteps(tmp_path, mode, component):
    storage = pa.array(
        [[9.0, 90.0], [1.0, 10.0], None, [0.0, 0.0], [3.0, 30.0], [float("nan"), float("inf")]],
        type=pa.list_(pa.float32(), 2),
    )
    values = pa.FixedShapeTensorArray.from_storage(pa.fixed_shape_tensor(pa.float32(), [2]), storage)
    signal = Signal(
        id="tensor",
        name="Tensor",
        spec=TimeSeriesSpec(spec_type="tensor", name="Tensor", unit_value=None, value_shape=(2,), nullable=True),
        time_axis=RegularAxis.from_rate_hz(1),
        data=values,
    )
    dataset = TimeFDataset(metadata=make_dataset().metadata)
    dataset.add_record(record=Record(record_id="record", sources=(Source(name="Sensor", signals=(signal,)),)))
    with TimeFWriter(tmp_path, dataset, values_backend="zarr") as writer:
        writer.write()

    with TimeNet(tmp_path).open_reader(dataset.metadata.dataset_id) as reader:
        app = create_app(reader, token="test-token", port=8000)
        with TestClient(app, base_url="http://127.0.0.1:8000") as client:
            response = client.post(
                "/api/v1/windows",
                json={
                    "record_id": "record",
                    "signal_id": "tensor",
                    "component": [component],
                    "start": 1,
                    "stop": 6,
                    "mode": mode,
                },
                headers={"Authorization": "Bearer test-token"},
            )

    assert response.status_code == 200
    result = response.json()
    expected_values = ["1.0", None, "0.0", "3.0", None] if component == 0 else ["10.0", None, "0.0", "30.0", None]
    assert [item["index"] for item in result["items"]] == ["1", "2", "3", "4", "5"]
    assert [item["value"] for item in result["items"]] == expected_values
    assert result["coverage"] == {
        "null": "1",
        "nan": "1" if component == 0 else "0",
        "positive_infinity": "0" if component == 0 else "1",
        "negative_infinity": "0",
        "finite": "3",
    }
