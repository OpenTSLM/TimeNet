import pickle

import numpy as np
import pytest
import soundfile as sf

from timenet.composition import BuildContext
from timenet.errors import TimeFFormatError
from timenet.registry import LocalRegistry
from timenet_connectors.datasets.iiscleap.coswara.connector import CoswaraConnector
from timenet_connectors.datasets.yang_ai_lab.hearts_coswara.connector import HeartsCoswaraConnector
from timenet_connectors.datasets.yang_ai_lab.hearts_coswara.release import TASKS


@pytest.fixture
def parent_and_cases(tmp_path, monkeypatch):
    directories = (
        "coswara/cough_covid_status_classification_with_symptoms",
        "coswara/cough_covid_status_classification_symptoms_only",
    )
    monkeypatch.setattr(
        "timenet_connectors.datasets.yang_ai_lab.hearts_coswara.connector.TASKS",
        {directory: TASKS[directory] for directory in directories},
    )
    original = tmp_path / "original"
    audio = original / "audio/20200413/participant"
    audio.mkdir(parents=True)
    (original / "combined_data.csv").write_text("id,a,covid_status,cough\nparticipant,24,healthy,True\n")
    values = np.array([[0.25, -0.5], [0.5, 0.25], [0, 0]], dtype=np.float32)
    sf.write(audio / "cough-heavy.wav", values, 48000, subtype="FLOAT")
    registry = LocalRegistry(tmp_path / "registry")
    registry.store(CoswaraConnector().convert([original]), values_backend="zarr")
    case = {
        "subject_id": "participant",
        "audio_type": "cough-heavy",
        "data": {"signal": values.mean(axis=1), "sr": 48000, "quality": 2},
        "symptoms": {"cough": True},
        "GT": "healthy",
    }
    folder = tmp_path / "cases/coswara/cough_covid_status_classification_with_symptoms"
    folder.mkdir(parents=True)
    (folder / "0.pkl").write_bytes(pickle.dumps(case))
    symptoms = tmp_path / "cases/coswara/cough_covid_status_classification_symptoms_only"
    symptoms.mkdir(parents=True)
    (symptoms / "0.pkl").write_bytes(
        pickle.dumps({"subject_id": "participant", "symptoms": {"cough": True}, "GT": "healthy"})
    )
    return registry, tmp_path / "cases", case


def test_mono_child_and_symptoms_only_roundtrip(parent_and_cases):
    registry, cases, _ = parent_and_cases
    connector = HeartsCoswaraConnector()
    with BuildContext.open(connector.metadata(), registry) as context:
        dataset = connector.compose([cases], context)
        dataset.set_dependencies(context.dependency_lock())
        assert len(dataset.tasks) == 2
        audio = next(task for task in dataset.tasks if task.inputs)
        np.testing.assert_array_equal(audio.inputs[0].signals[0].to_numpy(), [-0.125, 0.375, 0])
        symptoms = next(task for task in dataset.tasks if not task.inputs)
        assert symptoms.metadata["parent_participant"] == "coswara-participant"
        registry.store(dataset)
    with registry.open_reader("yang-ai-lab/hearts-coswara", "1.0.0") as reader:
        assert len(tuple(reader.iter_tasks())) == 2
    with registry.open_reader("iiscleap/coswara", "1.0.0") as reader:
        (original,) = reader.iter_records(["coswara-participant-cough-heavy"])
        assert original.signals[0].to_numpy().shape == (3, 2)


def test_mismatched_symptoms_are_rejected(parent_and_cases):
    registry, cases, payload = parent_and_cases
    payload["symptoms"]["cough"] = False
    (cases / "coswara/cough_covid_status_classification_with_symptoms/0.pkl").write_bytes(pickle.dumps(payload))
    connector = HeartsCoswaraConnector()
    with BuildContext.open(connector.metadata(), registry) as context, pytest.raises(TimeFFormatError, match="symptom"):
        connector.compose([cases], context)


def test_missing_context(tmp_path):
    with pytest.raises(NotImplementedError, match="compose"):
        HeartsCoswaraConnector().convert([tmp_path])


def test_empty_parent_audio_is_rejected(parent_and_cases):
    _, cases, _ = parent_and_cases
    registry = LocalRegistry(cases.parent / "empty-registry")
    original = cases.parent / "original"
    sf.write(original / "audio/20200413/participant/cough-heavy.wav", np.empty(0), 48000)
    registry.store(CoswaraConnector().convert([original]), values_backend="zarr")
    connector = HeartsCoswaraConnector()
    with (
        BuildContext.open(connector.metadata(), registry) as context,
        pytest.raises(TimeFFormatError, match="no audio"),
    ):
        connector.compose([cases], context)
