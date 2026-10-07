from fractions import Fraction

import numpy as np
from pydantic import ValidationError
import pytest
import soundfile as sf

from timenet.dataset import RegularAxis
from timenet.errors import TimeFFormatError
from timenet.registry import LocalRegistry
from timenet_connectors.datasets.iiscleap.coswara.connector import CoswaraConnector, Participant
from timenet_connectors.sources.audio import audio_signal


@pytest.fixture
def recordings(tmp_path):
    (tmp_path / "combined_data.csv").write_text("id,a,covid_status,cough\nsubject1,24,healthy,True\n")
    audio = tmp_path / "audio/20200413/subject1"
    audio.mkdir(parents=True)
    sf.write(audio / "cough-heavy.wav", np.array([0.25, -0.5, 0]), 48000, subtype="FLOAT")
    sf.write(audio / "breathing-deep.wav", np.array([[0.25, -0.5], [0, 0.5]]), 44100, subtype="FLOAT")
    annotations = tmp_path / "annotations"
    annotations.mkdir()
    (annotations / "cough-heavy_labels.csv").write_text("FILENAME, QUALITY\nsubject1_cough-heavy, 2\n")
    return tmp_path


def test_preserves_all_recordings_channels_metadata_and_rate(recordings, tmp_path):
    connector = CoswaraConnector()
    dataset = connector.convert([recordings])
    registry = LocalRegistry(tmp_path / "registry")
    registry.store(dataset, values_backend=connector.values_backend)
    with registry.open_reader("iiscleap/coswara", "1.0.0") as reader:
        records = {record.id: record for record in reader.iter_records()}
        assert len(records) == 3
        assert records["coswara-subject1"].metadata["cough"] == "True"
        cough = records["coswara-subject1-cough-heavy"].signals[0]
        assert cough.to_numpy().tolist() == [0.25, -0.5, 0]
        assert isinstance(cough.time_axis, RegularAxis)
        assert cough.time_axis.period_us == Fraction(1_000_000, 48000)
        stereo = records["coswara-subject1-breathing-deep"].signals[0]
        assert stereo.spec.value_shape == (2,)
        np.testing.assert_array_equal(stereo.to_numpy(), [[0.25, -0.5], [0, 0.5]])
        assert records["coswara-subject1-cough-heavy"].annotations[0].value == 2
        assert not tuple(reader.iter_tasks())


def test_unknown_participant_is_rejected(recordings):
    (recordings / "combined_data.csv").write_text("id,a,covid_status\nother,24,healthy\n")
    with pytest.raises(TimeFFormatError, match="unknown participant"):
        CoswaraConnector().convert([recordings])


def test_invalid_participant_is_rejected():
    with pytest.raises(ValidationError):
        Participant.model_validate({"id": "../escape", "a": -1, "covid_status": "healthy"})


def test_audio_samples_are_lazy(recordings, monkeypatch):
    signal = audio_signal(recordings / "audio/20200413/subject1/cough-heavy.wav", signal_id="audio")
    assert signal is not None

    def changed_file(*args, **kwargs):
        return np.array([0], dtype=np.float32), 16000

    monkeypatch.setattr(sf, "read", changed_file)
    with pytest.raises(TimeFFormatError, match="changed"):
        signal.to_arrow()


def test_empty_recording_keeps_metadata_without_inventing_samples(recordings):
    sf.write(recordings / "audio/20200413/subject1/cough-heavy.wav", np.empty(0), 48000)
    dataset = CoswaraConnector().convert([recordings])
    record = next(record for record in dataset.records if record.id.endswith("cough-heavy"))
    assert record.metadata["audio_status"] == "empty"
    assert not record.signals
