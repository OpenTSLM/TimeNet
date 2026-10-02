"""Check that the HARESPOD parent builds without the HEARTS child."""

import pickle

import pandas as pd

from timenet_connectors.datasets.yang_ai_lab.hearts_harespod.connector import HeartsHarespodConnector


def test_builds_taskless_segment_records_independently(tmp_path):
    directory = tmp_path / "harespod" / "altitude_ranking_respiration"
    directory.mkdir(parents=True)
    segments = {
        name: pd.DataFrame(
            {
                "timestamp": pd.date_range(f"2025-01-01 00:0{index}:00", periods=3, freq="10ms"),
                "rsp": [0.1, 0.2, 0.3],
            }
        )
        for index, name in enumerate("ABC")
    }
    payload = {"subject_id": "subject-1", "segment_dfs": segments, "GT": ["A", "B", "C"]}
    (directory / "0.pkl").write_bytes(pickle.dumps(payload))

    dataset = HeartsHarespodConnector().convert([tmp_path])

    assert dataset.tasks == ()
    assert [record.id.rsplit("-", 1)[1] for record in dataset.records] == ["A", "B", "C"]
    assert all(record.signals[0].spec.spec_type == "respiration_norm" for record in dataset.records)
