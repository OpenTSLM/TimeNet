import pickle

import numpy as np
import pytest

from timenet.composition import BuildContext
from timenet.dataset import Record, RegularAxis, Signal, Source, TimeFDataset
from timenet.errors import TimeFFormatError
from timenet.registry import LocalRegistry
from timenet.types import InputModality, TimeSeriesSpec
from timenet_connectors.datasets.epfl.coughvid.connector import CoughvidConnector
from timenet_connectors.datasets.yang_ai_lab.hearts_coughvid.connector import HeartsCoughvidConnector
from timenet_connectors.datasets.yang_ai_lab.hearts_coughvid.release import TASKS


@pytest.fixture(params=["webm", "ogg"])
def parent_and_case(tmp_path, request, monkeypatch):
    directory = "coughvid/cough_detection_good_qual"
    monkeypatch.setattr(
        "timenet_connectors.datasets.yang_ai_lab.hearts_coughvid.connector.TASKS", {directory: TASKS[directory]}
    )
    signal = Signal(
        id="original-audio",
        name="audio",
        data=np.array([0.25, -0.5, 1.01], dtype=np.float32),
        time_axis=RegularAxis.from_rate_hz(48000),
        spec=TimeSeriesSpec(
            spec_type="audio", name="Audio", dtype="float32", modality=InputModality.AUDIO, unit_value=None
        ),
    )
    parent = TimeFDataset(metadata=CoughvidConnector().metadata())
    parent.add_record(
        record=Record(
            record_id="coughvid-recording",
            sources=(Source(id="original-source", name="Audio", signals=(signal,)),),
            metadata={"filename": f"recording.{request.param}"},
        )
    )
    registry = LocalRegistry(tmp_path / "registry")
    registry.store(parent)
    case = {
        "subject_id": "recording",
        "sr": 48000,
        "audio": np.array([0.25, -0.5, 32767 / 32768], dtype=np.float32),
        "GT": True,
    }
    directory = tmp_path / "cases/coughvid/cough_detection_good_qual"
    directory.mkdir(parents=True)
    (directory / "0.pkl").write_bytes(pickle.dumps(case))
    return registry, tmp_path / "cases", case


def test_child_quantizes_without_changing_parent(parent_and_case):
    registry, cases, _ = parent_and_case
    connector = HeartsCoughvidConnector()
    with BuildContext.open(connector.metadata(), registry) as context:
        child = connector.compose([cases], context)
        child.set_dependencies(context.dependency_lock())
        assert child.tasks[0].inputs[0].signals[0].to_numpy().tolist() == [0.25, -0.5, 32767 / 32768]
        assert child.tasks[0].targets == ("true",)
        registry.store(child)
    with registry.open_reader("epfl/coughvid", "1.0.0") as reader:
        assert next(reader.iter_records()).signals[0].to_numpy()[-1] > 1


def test_different_waveform_is_rejected(parent_and_case):
    registry, cases, payload = parent_and_case
    payload["audio"][0] = 0.5
    (cases / "coughvid/cough_detection_good_qual/0.pkl").write_bytes(pickle.dumps(payload))
    connector = HeartsCoughvidConnector()
    with BuildContext.open(connector.metadata(), registry) as context:
        child = connector.compose([cases], context)
        with pytest.raises(TimeFFormatError, match="do not reproduce"):
            child.tasks[0].inputs[0].signals[0].to_arrow()


def test_missing_context(tmp_path):
    with pytest.raises(NotImplementedError, match="compose"):
        HeartsCoughvidConnector().convert([tmp_path])
