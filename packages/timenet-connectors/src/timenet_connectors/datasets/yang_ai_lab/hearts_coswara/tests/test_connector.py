"""Check that the Coswara parent builds without the HEARTS child."""

import pickle

import numpy as np
import pytest

from timenet.types import InputModality
from timenet_connectors.datasets.yang_ai_lab.hearts_coswara.connector import HeartsCoswaraConnector


def test_builds_taskless_audio_records_independently(tmp_path):
    directory = tmp_path / "coswara" / "audio_classification"
    directory.mkdir(parents=True)
    signal = np.asarray([0.0, 0.1, -0.1], dtype=np.float32)
    payload = {
        "subject_id": "subject-1",
        "audio_type": "breathing-deep",
        "data": {"signal": signal, "sr": 48_000, "quality": 2},
        "GT": "breathing",
    }
    (directory / "0.pkl").write_bytes(pickle.dumps(payload))

    dataset = HeartsCoswaraConnector().convert([tmp_path])

    assert dataset.tasks == ()
    assert len(dataset.records) == 1
    assert dataset.records[0].signals[0].spec.modality is InputModality.AUDIO
    assert dataset.records[0].signals[0].to_arrow().to_pylist() == pytest.approx([0.0, 0.1, -0.1])
