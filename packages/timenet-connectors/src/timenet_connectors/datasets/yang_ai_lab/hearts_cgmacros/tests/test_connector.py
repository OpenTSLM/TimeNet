import pickle

import numpy as np
import pandas as pd
import pytest

from timenet.composition import BuildContext
from timenet.dataset import Record
from timenet.errors import TimeFFormatError
from timenet.registry import LocalRegistry
from timenet_connectors.datasets.physionet.cgmacros.connector import CGMacrosConnector
from timenet_connectors.datasets.yang_ai_lab.hearts_cgmacros.connector import HeartsCgmacrosConnector


@pytest.fixture
def original_and_case(tmp_path):
    raw = tmp_path / "original"
    subject = raw / "CGMacros-001"
    subject.mkdir(parents=True)
    frame = pd.DataFrame(
        {
            "Timestamp": pd.date_range("2024-01-25", periods=120, freq="min").astype(str),
            "Libre GL": np.arange(120, dtype=float) + 100,
            "Dexcom GL": np.arange(120, dtype=float) + 101,
            "HR": [70.0] * 120,
            "Calories (Activity)": [1.0] * 120,
            "METs": [10.0] * 120,
            "Meal Type": [""] * 120,
        }
    )
    frame.to_csv(subject / "CGMacros-001.csv", index=False)
    for name in ("bio", "microbes", "gut_health_test"):
        (raw / f"{name}.csv").write_text("subject,label\n1,example\n")
    registry = LocalRegistry(tmp_path / "registry")
    connector = CGMacrosConnector()
    registry.store(connector.convert([raw]), values_backend=connector.values_backend)
    window = frame[["Timestamp", "Libre GL"]].copy()
    window.loc[30:59, "Libre GL"] = 0
    case = {
        "subject_id": "CGMacros-001",
        "window_df": window,
        "mask_indices": pd.Index(range(30, 60)),
        "mask_start": frame.Timestamp.iloc[30],
        "mask_end": frame.Timestamp.iloc[59],
        "GT": list(range(130, 160)),
    }
    folder = tmp_path / "cases/cgmacros/non_meal_imputation_cgm_only"
    folder.mkdir(parents=True)
    (folder / "0.pkl").write_bytes(pickle.dumps(case))
    return registry, tmp_path / "cases", case


def test_child_owns_masked_copy_and_parent_supplies_target(original_and_case):
    registry, cases, _ = original_and_case
    connector = HeartsCgmacrosConnector()
    with BuildContext.open(connector.metadata(), registry) as context:
        child = connector.compose([cases], context)
        child.set_dependencies(context.dependency_lock())
        (task,) = child.tasks
        assert task.targets is not None and isinstance(task.targets[0], Record)
        assert task.inputs[0].signals[0].to_numpy().tolist()[30:60] == [0] * 30
        assert task.targets[0].signals[0].to_numpy().tolist() == list(range(130, 160))
        assert task.inputs[0].signals[0].metadata["parent_dataset"] == "physionet/cgmacros@1.0.0"
        assert not child.record_imports
        registry.store(child, values_backend=connector.values_backend)
    with registry.open_reader("physionet/cgmacros", "1.0.0") as reader:
        (parent,) = reader.iter_records()
        glucose = next(signal for signal in parent.signals if signal.name == "Libre GL")
        assert glucose.to_numpy().tolist() == list(range(100, 220))
    with registry.open_reader("yang-ai-lab/hearts-cgmacros", "1.0.0") as reader:
        restored = tuple(reader.iter_tasks())
        assert len(restored) == 1
        assert restored[0].inputs[0].signals[0].to_numpy().tolist()[30:60] == [0] * 30


def test_convert_requires_compose(tmp_path):
    with pytest.raises(NotImplementedError, match="compose"):
        HeartsCgmacrosConnector().convert([tmp_path])


def test_mismatched_benchmark_values_fail_instead_of_using_frozen_values(original_and_case):
    registry, cases, payload = original_and_case
    payload["window_df"].loc[0, "Libre GL"] = 999
    (cases / "cgmacros/non_meal_imputation_cgm_only/0.pkl").write_bytes(pickle.dumps(payload))
    connector = HeartsCgmacrosConnector()
    with BuildContext.open(connector.metadata(), registry) as context:
        child = connector.compose([cases], context)
        with pytest.raises(TimeFFormatError, match="do not reproduce"):
            child.tasks[0].inputs[0].signals[0].to_arrow()


def test_missing_parent_timestamp_is_rejected(original_and_case):
    registry, cases, payload = original_and_case
    payload["window_df"].loc[0, "Timestamp"] = "2024-01-24 00:00:00"
    (cases / "cgmacros/non_meal_imputation_cgm_only/0.pkl").write_bytes(pickle.dumps(payload))
    connector = HeartsCgmacrosConnector()
    with BuildContext.open(connector.metadata(), registry) as context, pytest.raises(TimeFFormatError, match="absent"):
        connector.compose([cases], context)
