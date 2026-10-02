"""Check that the COUGHVID parent builds without the HEARTS child."""

import pickle

import numpy as np

from timenet_connectors.datasets.yang_ai_lab.hearts_coughvid.connector import HeartsCoughvidConnector


def test_builds_taskless_cough_records_independently(tmp_path):
    directory = tmp_path / "coughvid" / "diagnosis_classification"
    directory.mkdir(parents=True)
    payload = {
        "subject_id": "subject-1",
        "audio": np.asarray([0.5, -0.5, 0.25], dtype=np.float32),
        "sr": 48_000,
        "GT": "upper_infection",
    }
    (directory / "0.pkl").write_bytes(pickle.dumps(payload))

    dataset = HeartsCoughvidConnector().convert([tmp_path])

    assert dataset.tasks == ()
    assert dataset.records[0].signals[0].n_values == 3
    assert dataset.records[0].signals[0].to_arrow().to_pylist() == [0.5, -0.5, 0.25]
