"""Serve deterministic local fixtures for real-browser inspector regressions."""

from pathlib import Path
from tempfile import TemporaryDirectory

import pyarrow as pa
import uvicorn

from timenet.client import TimeNet
from timenet.dataset import Record, Signal, Source, TimeFDataset
from timenet.dataset.axis import RegularAxis
from timenet.testing import make_dataset
from timenet.types import Annotation, AnswerTask, TimeSeriesSpec
from timenet.viewer.app import create_app
from timenet.writer import TimeFWriter


def main() -> None:
    """Launch a fixture-only server with a fixed test credential."""
    dataset = TimeFDataset(metadata=make_dataset().metadata)
    values = pa.array([1.0, None, 3.0, float("nan"), 5.0] + [float(i) for i in range(99995)])
    signal = Signal(
        id="signal",
        name="<b>Untrusted</b>",
        spec=TimeSeriesSpec(spec_type="test", name="Test", dtype="float64", unit_value=None, nullable=True),
        time_axis=RegularAxis.from_rate_hz(100),
        data=values,
    )
    tensor = Signal(
        id="tensor",
        name="Tensor",
        spec=TimeSeriesSpec(spec_type="tensor", name="Tensor", unit_value=None, nullable=True, value_shape=(2,)),
        time_axis=RegularAxis.from_rate_hz(1),
        data=pa.FixedShapeTensorArray.from_storage(
            pa.fixed_shape_tensor(pa.float32(), [2]),
            pa.array([[1.0, 10.0], None, [3.0, 30.0]], type=pa.list_(pa.float32(), 2)),
        ),
    )
    record = Record(record_id="record", sources=(Source(id="source", name="Sensor", signals=(signal, tensor)),))
    record.annotate(Annotation(key="reviewed", value=True))
    dataset.add_record(record=record)
    dataset.add_task(task=AnswerTask(id="many", inputs=(record,), targets=tuple(f"answer-{i}" for i in range(205))))
    with TemporaryDirectory() as directory:
        with TimeFWriter(Path(directory), dataset, values_backend="zarr") as writer:
            writer.write()
        with TimeNet(directory).open_reader(dataset.metadata.dataset_id) as reader:
            uvicorn.run(
                create_app(reader, token="browser-test", port=8765),
                host="127.0.0.1",
                port=8765,
                access_log=False,
                proxy_headers=False,
            )


if __name__ == "__main__":
    main()
