"""Check the HARESPOD cases: segments as records of their own, rankings as text, and pairings as matches."""

import pickle

import pandas as pd
import pytest

from timenet.client import TimeNet
from timenet.dataset import RegularAxis
from timenet.types import AnswerTask, InputModality, TSCorrespondenceTask
from timenet.writer import TimeFWriter
from timenet_connectors.datasets.yang_ai_lab.hearts.connector import HeartsConnector


def _segment(column, start, values, *, freq="10ms"):
    """A HARESPOD segment: calendar timestamps at 100 Hz, or durations from zero when start is None."""
    if start is None:
        stamps = pd.timedelta_range(0, periods=len(values), freq=freq)
    else:
        stamps = pd.date_range(start, periods=len(values), freq=freq)
    return pd.DataFrame({"timestamp": stamps, column: values}, index=range(137_336, 137_336 + len(values)))


CASES = {
    "altitude_ranking_respiration": {
        "subject_id": "219a",
        "segment_dfs": {
            "A": _segment("rsp", "2023-02-19 09:19:27.360", [0.1, 0.2, 0.3, 0.4]),
            "B": _segment("rsp", "2023-02-19 09:02:01.570", [0.5, 0.4, 0.3, 0.2]),
            "C": _segment("rsp", "2023-02-19 09:08:44.600", [0.4, 0.5, 0.6, 0.7]),
        },
        "GT": ["A", "C", "B"],
    },
    "spo2_resp_pairing": {
        "subject_id": "314b",
        "respiration_dfs": {
            "respiration_A": _segment("rsp", None, [0.44, 0.45, 0.46]),
            "respiration_B": _segment("rsp", None, [0.15, 0.16, 0.17]),
        },
        "spo_dfs": {
            "spo_2": _segment("spo", None, [0.49, 0.50, 0.51]),
            "spo_1": _segment("spo", None, [0.48, 0.47, 0.46]),
        },
        "GT": {"A": "2", "B": "1"},
    },
    "hr_resp_pairing": {
        "subject_id": "223b",
        "respiration_dfs": {
            "respiration_A": _segment("rsp", None, [0.47, 0.48]),
            "respiration_B": _segment("rsp", None, [0.49, 0.50]),
        },
        "hr_dfs": {
            "hr_1": _segment("hr", None, [0.59, 0.60], freq="1s"),
            "hr_2": _segment("hr", None, [0.64, 0.65], freq="1s"),
        },
        "GT": {"A": "1", "B": "2"},
    },
}


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    root = tmp_path_factory.mktemp("hearts")
    for task, payload in CASES.items():
        directory = root / "harespod" / task
        directory.mkdir(parents=True)
        (directory / "0.pkl").write_bytes(pickle.dumps(payload))
    return HeartsConnector().convert([root])


def _task(dataset, task):
    return next(item for item in dataset.tasks if item.id == f"hearts-harespod-{task}-00-task")


def _by_key(annotations):
    return {annotation.key: annotation for annotation in annotations}


def test_ranked_segments_are_records_on_their_own_calendar_clocks(dataset):
    task = _task(dataset, "altitude_ranking_respiration")
    assert isinstance(task, AnswerTask)
    assert task.resolved_input_modalities == {InputModality.TEXT, InputModality.TIME_SERIES}
    assert [record.id for record in task.inputs] == [
        f"hearts-harespod-altitude_ranking_respiration-00-{name}" for name in "ABC"
    ]
    first, second, third = task.inputs
    assert {record.start_time.timestamp for record in task.inputs} == {None}
    assert [_by_key(record.annotations)["recording_start_local"].value for record in (first, second, third)] == [
        "2023-02-19T09:19:27.360000",
        "2023-02-19T09:02:01.570000",
        "2023-02-19T09:08:44.600000",
    ]
    assert {_by_key(record.annotations)["subject_id"].value for record in task.inputs} == {"harespod:219a"}
    assert [source.name for source in first.sources] == ["segment_dfs.A"]
    (signal,) = first.signals
    assert (signal.name, signal.spec.spec_type, signal.n_values) == ("rsp", "respiration_norm", 4)
    assert signal.time_axis == RegularAxis.from_rate_hz(100)
    assert task.targets == ('["A", "C", "B"]',)
    assert task.prompt.startswith(
        "You are given 5-minute segments of respiration signals as the records A, B and C, each holding the signal 'rsp'"
    )


def test_pairing_matches_each_respiration_segment_to_its_candidate(dataset):
    task = _task(dataset, "spo2_resp_pairing")
    assert isinstance(task, TSCorrespondenceTask)
    assert [record.id.rsplit("-", 1)[1] for record in task.inputs] == ["respiration_A", "respiration_B"]
    assert [record.id.rsplit("-", 1)[1] for record in task.candidate_records] == ["spo_1", "spo_2"]
    spo_1, spo_2 = task.candidate_records
    assert task.targets == (spo_2, spo_1)
    assert {record.start_time.timestamp for record in (*task.inputs, *task.candidate_records)} == {None}
    assert {_by_key(record.annotations)["subject_id"].value for record in (*task.inputs, *task.candidate_records)} == {
        "harespod:314b"
    }
    assert spo_1.signals[0].spec.spec_type == "spo2_norm"
    assert all(record in dataset.records for record in task.candidate_records)
    assert "The SpO2 signals are the records spo_1 and spo_2, each holding the signal 'spo'" in task.prompt


def test_heart_rate_candidates_keep_their_one_hertz_cadence(dataset):
    task = _task(dataset, "hr_resp_pairing")
    hr_1, hr_2 = task.candidate_records
    assert task.targets == (hr_1, hr_2)
    assert hr_1.signals[0].time_axis == RegularAxis.from_rate_hz(1)
    assert hr_1.signals[0].spec.spec_type == "heart_rate_norm"
    assert task.inputs[0].signals[0].time_axis == RegularAxis.from_rate_hz(100)


def test_pairing_round_trips_through_the_writer(dataset, tmp_path):
    with TimeFWriter(tmp_path / "registry", dataset, values_backend="zarr") as writer:
        writer.write()
    read = TimeNet(registry=tmp_path / "registry").load("yang-ai-lab/hearts", auto_build=False)
    task = _task(read, "spo2_resp_pairing")
    assert {_by_key(record.annotations)["subject_id"].value for record in (*task.inputs, *task.candidate_records)} == {
        "harespod:314b"
    }
    assert [record.id.rsplit("-", 1)[1] for record in task.targets] == ["spo_2", "spo_1"]
    assert task.targets[0].signals[0].to_arrow().to_pylist() == pytest.approx([0.49, 0.50, 0.51])
