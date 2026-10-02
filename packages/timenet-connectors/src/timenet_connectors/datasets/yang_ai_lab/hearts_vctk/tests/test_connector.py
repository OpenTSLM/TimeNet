"""Check that the VCTK parent builds without the HEARTS child."""

import pickle

import numpy as np

from timenet.dataset import RegularAxis
from timenet_connectors.datasets.yang_ai_lab.hearts_vctk.connector import HeartsVctkConnector


def test_builds_taskless_speech_records_independently(tmp_path):
    directory = tmp_path / "vctk" / "waveform_temporal_direction_detection"
    directory.mkdir(parents=True)
    payload = {
        "speaker_id": "p361",
        "recording_id": "p361_001",
        "waveform": np.asarray([0.1, 0.2, 0.3], dtype=np.float32),
        "GT": 0,
    }
    (directory / "0.pkl").write_bytes(pickle.dumps(payload))

    dataset = HeartsVctkConnector().convert([tmp_path])

    assert dataset.tasks == ()
    assert dataset.records[0].signals[0].time_axis == RegularAxis.from_rate_hz(16_000)
    assert dataset.records[0].metadata["recording_id"] == "p361_001"
