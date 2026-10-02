"""Check that the CGMacros parent builds without the HEARTS child."""

from datetime import datetime, timedelta
import pickle

import pandas as pd

from timenet_connectors.datasets.yang_ai_lab.hearts_cgmacros.connector import HeartsCgmacrosConnector


def test_builds_taskless_cgm_records_independently(tmp_path):
    directory = tmp_path / "cgmacros" / "a1c_classification"
    directory.mkdir(parents=True)
    start = datetime(2025, 1, 1)
    frame = pd.DataFrame(
        {
            "Timestamp": [str(start + timedelta(minutes=index)) for index in range(3)],
            "Libre GL": [90.0, 91.0, 92.0],
        }
    )
    payload = {"subject_id": "CGMacros-001", "window_df": frame, "GT": "normal"}
    (directory / "0.pkl").write_bytes(pickle.dumps(payload))

    dataset = HeartsCgmacrosConnector().convert([tmp_path])

    assert dataset.tasks == ()
    assert [record.id for record in dataset.records] == ["hearts-cgmacros-a1c_classification-00"]
    assert dataset.records[0].signals[0].to_arrow().to_pylist() == [90.0, 91.0, 92.0]
