import importlib

import numpy as np
import pytest

from timenet.composition import BuildContext
from timenet.dataset import Record, RegularAxis, Signal, Source, TimeFDataset
from timenet.registry import LocalRegistry
from timenet.types import Annotation, AnswerTask, Split, TimeSeriesSpec, TSEditingTask
from timenet_connectors.datasets.yang_ai_lab.hearts.connector import HeartsConnector


def record(identity):
    return Record(
        record_id=identity,
        sources=(
            Source(
                id=f"{identity}-source",
                name="Original",
                signals=(
                    Signal(
                        id=f"{identity}-signal",
                        name="signal",
                        data=np.array([1.0, 2.0]),
                        time_axis=RegularAxis.from_rate_hz(1),
                        spec=TimeSeriesSpec(spec_type="example", name="Example", dtype="float64", unit_value=None),
                    ),
                ),
            ),
        ),
    )


def test_aggregate_reuses_children_including_targets_and_inputless_tasks(tmp_path):
    registry = LocalRegistry(tmp_path)
    for name, module in (
        ("cgmacros", "physionet.cgmacros"),
        ("coswara", "iiscleap.coswara"),
        ("coughvid", "epfl.coughvid"),
        ("harespod", "oca_john.harespod"),
        ("vctk", "cstr.vctk"),
    ):
        base = importlib.import_module(f"timenet_connectors.datasets.{module}.connector").CONNECTOR()
        parent = TimeFDataset(metadata=base.metadata())
        parent.add_record(record=record(f"original-{name}"))
        registry.store(parent)
        child_connector = importlib.import_module(
            f"timenet_connectors.datasets.yang_ai_lab.hearts_{name}.connector"
        ).CONNECTOR()
        with BuildContext.open(child_connector.metadata(), registry) as context:
            child = TimeFDataset(metadata=context.metadata)
            input_record = child.add_record(record=record(f"child-{name}"))
            target = child.add_record(record=record(f"target-{name}"))
            child.add_tasks(
                tasks=[TSEditingTask(id=f"{name}-editing", inputs=(input_record,), targets=(target,), split=Split.TEST)]
            )
            extra = AnswerTask(id=f"{name}-no-input", targets=("answer",), split=Split.TEST)
            extra.annotate(Annotation(key="symptom", value=True))
            child.add_tasks(tasks=[extra])
            child.set_dependencies(context.dependency_lock())
            registry.store(child)
    connector = HeartsConnector()
    assert connector.download(tmp_path / "unused") == []
    with BuildContext.open(connector.metadata(), registry) as context:
        combined = connector.compose([], context)
        combined.set_dependencies(context.dependency_lock())
        assert len(combined.tasks) == 10
        assert len(combined.record_imports) == 10
        assert not combined.owned_records
        assert all(len(task.inputs[0].task_ids) == 1 for task in combined.tasks if task.inputs)
        registry.store(combined)
    manifest = registry.get_manifest("yang-ai-lab/hearts", "1.0.0")
    assert not manifest.files.time_series
    assert len(manifest.dependencies) == 10
    with registry.open_reader("yang-ai-lab/hearts", "1.0.0") as reader:
        tasks = tuple(reader.iter_tasks())
        assert len(tasks) == 10
        assert sum(not task.inputs for task in tasks) == 5
        assert all(task.annotations[0].value is True for task in tasks if not task.inputs)
        task = next(task for task in tasks if task.inputs)
        assert task.inputs[0].signals[0].to_numpy().tolist() == [1, 2]


def test_convert_requires_compose():
    with pytest.raises(NotImplementedError, match="compose"):
        HeartsConnector().convert([])
