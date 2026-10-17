"""Serve deterministic local fixtures for real-browser inspector regressions."""

from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import pyarrow as pa
import uvicorn

from timenet.dataset import Record, Signal, Source, TimeFDataset
from timenet.dataset.axis import OrdinalAxis, RegularAxis
from timenet.reader import TimeFReader
from timenet.registry import open_registry
from timenet.testing import make_dataset
from timenet.types import Annotation, AnswerTask, TemporalLocalizationTask, TimeInterval, TimePoint, TimeSeriesSpec
from timenet.viewer.app import create_app, create_registry_app
from timenet.writer import TimeFWriter


def main() -> None:
    """Launch a fixture-only server with a fixed test credential."""
    dataset = TimeFDataset(metadata=make_dataset().metadata)
    values = pa.array([1.0, None, 3.0, float("nan"), 5.0] + [float(i) for i in range(19996)])
    signal = Signal(
        id="signal",
        name="<b>Untrusted</b>",
        spec=TimeSeriesSpec(spec_type="test", name="Test", dtype="float64", unit_value=None, nullable=True),
        time_axis=RegularAxis.from_rate_hz(100),
        data=values,
    )
    signal.annotate(Annotation(key="event", span=TimePoint.seconds(0.5)))
    signal.annotate(Annotation(key="interval", span=TimeInterval.seconds(0.25, 0.75)))
    ordinal = Signal(
        id="ordinal",
        name="Ordinal",
        spec=TimeSeriesSpec(spec_type="ordinal", name="Ordinal", dtype="float64", unit_value=None),
        time_axis=OrdinalAxis(),
        data=pa.array([1.0, 2.0, 3.0]),
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
    other = Signal(
        id="z-signal",
        name="Other source signal",
        spec=TimeSeriesSpec(spec_type="other", name="Other", dtype="float64", unit_value=None),
        time_axis=RegularAxis.from_rate_hz(1),
        data=pa.array([float(i) for i in range(200)]),
    )
    source = Source(
        id="z-source", name="Other source", sources=(Source(id="z-child", name="Nested source", signals=(other,)),)
    )
    source.annotate(Annotation(key="source-event", span=TimePoint.seconds(100)))
    record = Record(
        record_id="record", sources=(Source(id="source", name="Sensor", signals=(signal, tensor, ordinal)), source)
    )
    record.annotate(Annotation(key="reviewed", value=True))
    record.annotate(Annotation(key="record-event", span=TimePoint.seconds(0.125)))
    dataset.add_record(record=record)
    dataset.add_task(task=AnswerTask(id="many", inputs=(record,), targets=tuple(f"answer-{i}" for i in range(205))))
    with TemporaryDirectory() as directory:
        with TimeFWriter(Path(directory), dataset, values_backend="zarr") as writer:
            writer.write()
        registry = open_registry(directory)
        with TimeFReader(registry.open_version(dataset.metadata.dataset_id)) as reader:
            uvicorn.run(
                create_app(reader, token="browser-test", port=8765),
                host="127.0.0.1",
                port=8765,
                access_log=False,
                proxy_headers=False,
            )


def registry_main() -> None:
    """Serve two disjoint datasets through the real registry switching endpoints."""
    with TemporaryDirectory() as directory:
        for suffix, value in (("a", 11.0), ("b", 22.0)):
            dataset = TimeFDataset(
                metadata=make_dataset().metadata.model_copy(
                    update={"dataset_id": f"test/{suffix}", "name": f"Dataset {suffix.upper()}"}
                )
            )
            signal = Signal(
                id=f"signal-{suffix}",
                name=f"Signal {suffix}",
                spec=TimeSeriesSpec(spec_type="test", name="Test", dtype="float64", unit_value=None),
                time_axis=RegularAxis.from_rate_hz(1),
                data=pa.array([value, value + 1]),
            )
            record = dataset.add_record(
                record=Record(
                    record_id=f"record-{suffix}",
                    sources=(Source(id=f"source-{suffix}", name="Sensor", signals=(signal,)),),
                )
            )
            dataset.add_task(
                task=TemporalLocalizationTask(
                    id=f"task-{suffix}",
                    inputs=(record,),
                    targets=(TimePoint.seconds(0, time_series_ids=(signal.id,)),),
                )
            )
            with TimeFWriter(Path(directory), dataset) as writer:
                writer.write()
        registry = open_registry(directory)
        template = registry.get_manifest("test/a")
        for index in range(55):
            identity = f"test/z-{index:02}"
            metadata = template.metadata.model_copy(
                update={"dataset_id": identity, "name": "Hidden needle" if index == 54 else f"Catalog fixture {index}"}
            )
            manifest = template.model_copy(update={"dataset_id": identity, "metadata": metadata})
            path = Path(directory) / identity / str(metadata.dataset_version) / "manifest.json"
            path.parent.mkdir(parents=True)
            path.write_text(manifest.model_dump_json())
        uvicorn.run(
            create_registry_app(open_registry(directory), token="browser-test", port=8766, registry_name="fixtures"),
            host="127.0.0.1",
            port=8766,
            access_log=False,
            proxy_headers=False,
        )


if __name__ == "__main__":
    if "--registry" in sys.argv:
        registry_main()
    else:
        main()
