"""Check OpenSQA deduplication, lazy signals, tasks, and storage."""

import json
from pathlib import Path

import pytest

from timenet.dataset import RegularAxis, TimeFDataset
from timenet.errors import TimeFFormatError
from timenet.types import InputModality, Split
from timenet_connectors.datasets.bash_lab.opensqa import connector as connector_module
from timenet_connectors.datasets.bash_lab.opensqa.connector import OpenSqaConnector
from timenet_connectors.datasets.bash_lab.opensqa.parser import OpenSqaFile, OpenSqaSource, qa_pairs
from timenet_connectors.datasets.bash_lab.opensqa.release import (
    ANSWER_TASKS,
    CAPTION_TASKS,
    FILES,
    QA_TASKS,
    RECORDS,
    REVISION,
    ReleaseFile,
)


def _messages(
    *,
    summary: str,
    assistant: str,
    gyroscope_value: float = 1.0,
    accelerometer_value: float = 2.0,
) -> dict[str, object]:
    gyroscope = [[gyroscope_value + axis for axis in range(3)] for _sample in range(60)]
    accelerometer = [[accelerometer_value + axis for axis in range(3)] for _sample in range(60)]
    user = (
        f"Summary: {summary}\n"
        f"Gyroscope: {json.dumps(gyroscope)}\n"
        f"Accelerometer: {json.dumps(accelerometer)}\n"
        "Features: synthetic fixture features\nNarration: synthetic fixture caption"
    )
    return {
        "messages": [
            {"role": "system", "content": "synthetic fixture"},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ]
    }


def _write(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(f"{json.dumps(row)}\n" for row in rows),
        encoding="utf-8",
    )


def _source(tmp_path: Path, *, mismatch: bool = False) -> OpenSqaSource:
    v1_path = tmp_path / "hhar-v1.jsonl"
    v2_path = tmp_path / "hhar-v2.jsonl"
    _write(
        v1_path,
        [
            _messages(
                summary="v1 standing",
                assistant=(
                    "1. Q: What does the first signal show?\n"
                    "A: It shows a stable synthetic fixture.\n\n"
                    "2. **Q:** Is the second axis larger?\n"
                    "**Answer:** Yes, it is larger.\n\n"
                    "3. Is this deliberately incomplete?"
                ),
            )
        ],
    )
    _write(
        v2_path,
        [
            _messages(
                summary="v2 standing",
                assistant=(
                    "**Q1:** Why is this window stable?\n"
                    "The values repeat at every sample.\n\n"
                    "**Q2:** How many samples are present?\n"
                    "A2: There are sixty samples."
                ),
                gyroscope_value=1.5 if mismatch else 1.0,
            )
        ],
    )
    return OpenSqaSource(
        files=(
            OpenSqaFile(
                ReleaseFile("fixture/hhar-v1.jsonl", "hhar", "v1", 1, 2),
                v1_path,
            ),
            OpenSqaFile(
                ReleaseFile("fixture/hhar-v2.jsonl", "hhar", "v2", 1, 2),
                v2_path,
            ),
        ),
        revision=REVISION,
    )


def test_convert_shares_one_record_across_versions_and_keeps_all_tasks(tmp_path):
    dataset = OpenSqaConnector().convert([_source(tmp_path)])

    assert len(dataset.records) == 1
    record = dataset.records[0]
    assert record.id == "opensqa-hhar-000000"
    assert [signal.name for signal in record.signals] == [
        "accelerometer_x",
        "accelerometer_y",
        "accelerometer_z",
        "gyroscope_x",
        "gyroscope_y",
        "gyroscope_z",
    ]
    assert all(signal.n_values == 60 for signal in record.signals)
    assert all(signal.spec.unit_value is None for signal in record.signals)
    assert all(signal.spec.dtype == "float64" for signal in record.signals)
    assert all(isinstance(signal.time_axis, RegularAxis) for signal in record.signals)
    assert {signal.time_axis.axis_id for signal in record.signals} == {record.signals[0].time_axis.axis_id}
    axis = record.signals[0].time_axis
    assert isinstance(axis, RegularAxis)
    assert axis.period_us == 100_000
    assert record.signals[0].to_arrow().to_pylist() == [2.0] * 60
    assert record.signals[4].to_arrow().to_pylist() == [2.0] * 60

    tasks = dataset.get_all()
    assert len(tasks) == 6
    assert [task.id for task in tasks] == [
        "opensqa-hhar-v1-000000-caption",
        "opensqa-hhar-v1-000000-qa-01",
        "opensqa-hhar-v1-000000-qa-02",
        "opensqa-hhar-v2-000000-caption",
        "opensqa-hhar-v2-000000-qa-01",
        "opensqa-hhar-v2-000000-qa-02",
    ]
    assert all(task.inputs == (record,) and task.split is Split.TRAIN for task in tasks)
    assert tasks[0].prompt is None
    assert tasks[0].targets == ("Features: synthetic fixture features\nNarration: synthetic fixture caption",)
    assert tasks[0].input_modalities == frozenset({InputModality.TIME_SERIES})
    assert tasks[1].prompt == "What does the first signal show?"
    assert tasks[1].targets == ("It shows a stable synthetic fixture.",)
    assert tasks[1].input_modalities == frozenset({InputModality.TIME_SERIES, InputModality.TEXT})
    assert tasks[1].metadata["activity_summary"] == "v1 standing"
    assert tasks[4].metadata["activity_summary"] == "v2 standing"


def test_fixture_round_trips_shared_record_values_and_tasks(tmp_path):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    dataset = OpenSqaConnector().convert([_source(source_dir)])
    version_dir = dataset.write(path=tmp_path / "timef")
    loaded = TimeFDataset.open(path=version_dir)

    assert len(loaded.records) == 1
    assert len(loaded.get_train()) == 6
    assert loaded.records[0].signals[2].to_arrow().to_pylist() == [4.0] * 60
    assert loaded.get_train()[3].targets == (
        "Features: synthetic fixture features\nNarration: synthetic fixture caption",
    )


def test_convert_rejects_disagreeing_v1_v2_sensor_windows(tmp_path):
    with pytest.raises(TimeFFormatError, match="sensor window differs"):
        OpenSqaConnector().convert([_source(tmp_path, mismatch=True)])


def test_qa_parser_drops_truncated_questions_and_accepts_source_markdown():
    text = (
        "**Q1:** First question?\n"
        "- **A1:** First answer.\n\n"
        "2. **Q: Second question?**\n"
        "Second answer without a label.\n\n"
        "3. Third question?"
    )
    assert qa_pairs(text, maximum=5) == (
        ("First question?", "First answer."),
        ("Second question?", "Second answer without a label."),
    )

    questions_only = "\n".join(
        ["1. Duplicate first question?", *(f"{index}. Question {index}?" for index in range(1, 11))]
    )
    assert qa_pairs(questions_only, maximum=10) == ()


def test_release_counts_answer_tasks_over_unique_records():
    assert sum(source.rows for source in FILES) == CAPTION_TASKS == 62_248
    assert sum(source.qa_tasks for source in FILES) == QA_TASKS == 435_178
    assert ANSWER_TASKS == 497_426
    assert RECORDS == 35_960


def test_download_uses_pinned_revision_and_all_release_paths(monkeypatch, tmp_path):
    snapshot = tmp_path / "snapshot"
    seen: dict[str, object] = {}

    def fake_snapshot(repo: str, revision: str, cache_dir: Path, patterns: tuple[str, ...]) -> Path:
        seen.update(repo=repo, revision=revision, cache_dir=cache_dir, patterns=patterns)
        for relative in patterns:
            path = snapshot / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
        return snapshot

    monkeypatch.setattr(connector_module, "hub_snapshot", fake_snapshot)
    sources = OpenSqaConnector().download(tmp_path / "cache")

    assert len(sources) == 1
    assert seen["revision"] == REVISION
    assert seen["patterns"] == tuple(source.path for source in FILES)
    assert tuple(source.release for source in sources[0].files) == FILES
