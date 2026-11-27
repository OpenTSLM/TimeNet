"""Offline contract tests for the MHEALTH connector."""

from __future__ import annotations

from pathlib import Path
import zipfile

import pytest

from timenet.dataset.axis import RegularAxis
from timenet.errors import TimeFFormatError
from timenet.types import ClassificationTask, TimeInterval
from timenet_connectors.datasets.uci.mhealth.connector import MHealthConnector, MHealthSource, _extract_members


def _row(offset: float, label: int) -> str:
    """Return one source-shaped row with 23 precise signals and its integer label."""

    return "\t".join([*(repr(offset + index / 10) for index in range(23)), str(label)]) + "\n"


@pytest.fixture
def source(tmp_path: Path) -> MHealthSource:
    """Write two small subject-native MHEALTH log files."""

    root = tmp_path / "MHEALTHDATASET"
    root.mkdir()
    (root / "mHealth_subject1.log").write_text(_row(1.0, 0) + _row(2.0, 4) + _row(3.0, 4), encoding="ascii")
    (root / "mHealth_subject2.log").write_text(_row(10.0, 1), encoding="ascii")
    return MHealthSource(dataset_root=root)


def test_convert_preserves_subject_files_all_signals_and_relative_50hz_axis(source: MHealthSource) -> None:
    """Each native log is one record with all 23 source channels and no fabricated timestamp column."""

    records = MHealthConnector().convert([source]).records
    assert [record.record_id for record in records] == ["mhealth-subject-1", "mhealth-subject-2"]
    assert records[0].subject_ids == ("1",)
    assert len(records[0].time_series) == 23
    assert all(series.n_values == 3 for series in records[0].time_series)
    assert all(series.time_axis == RegularAxis.from_rate_hz(50) for series in records[0].time_series)
    assert records[0].time_series[0].to_arrow().to_pylist() == [1.0, 2.0, 3.0]
    assert records[0].time_series[3].to_arrow().to_pylist() == [1.3, 2.3, 3.3]
    assert records[0].time_series[22].to_arrow().to_pylist() == [3.2, 4.2, 5.2]


def test_convert_preserves_null_and_activity_runs_as_timed_annotations_and_scoped_tasks(source: MHealthSource) -> None:
    """Source label zero and consecutive nonzero labels remain separately time-scoped."""

    dataset = MHealthConnector().convert([source])
    record = dataset.records[0]
    labels = [annotation for annotation in record.annotations if annotation.key == "activity_id"]
    assert [annotation.value for annotation in labels] == [0, 4]
    spans = [annotation.span for annotation in labels]
    intervals = [span for span in spans if isinstance(span, TimeInterval)]
    assert len(intervals) == len(spans)
    assert [(span.start_us, span.end_us) for span in intervals] == [
        (0, 20_000),
        (20_000, 60_000),
    ]
    tasks = [task for task in dataset.iter_tasks() if isinstance(task, ClassificationTask)]
    assert [task.target for task in tasks] == ["null_class", "walking", "standing_still"]
    scopes = [task.scope for task in tasks]
    task_intervals = [scope for scope in scopes if isinstance(scope, TimeInterval)]
    assert len(task_intervals) == len(scopes)
    assert [(scope.start_us, scope.end_us) for scope in task_intervals] == [(0, 20_000), (20_000, 60_000), (0, 20_000)]


def test_extract_members_returns_the_published_source_root(tmp_path: Path) -> None:
    """A freshly extracted fixture returns its usable source root rather than ``None``."""
    archive = tmp_path / "official.zip"
    with zipfile.ZipFile(archive, "w") as source:
        source.writestr("MHEALTHDATASET/README.txt", "fixture")
        for subject in range(1, 11):
            source.writestr(f"MHEALTHDATASET/mHealth_subject{subject}.log", _row(float(subject), 0))
    root = _extract_members(archive, tmp_path / "cache")
    assert root == tmp_path / "cache" / "MHEALTHDATASET"
    assert (root / "mHealth_subject1.log").is_file()


@pytest.mark.parametrize("corruption", ("changed", "extra", "symlink"))
def test_extract_members_rejects_changed_extra_or_symlinked_cached_source(tmp_path: Path, corruption: str) -> None:
    """A cache is reusable only when every file remains exactly the pinned ZIP member."""
    archive = tmp_path / "official.zip"
    with zipfile.ZipFile(archive, "w") as source:
        source.writestr("MHEALTHDATASET/README.txt", "fixture")
        for subject in range(1, 11):
            source.writestr(f"MHEALTHDATASET/mHealth_subject{subject}.log", _row(float(subject), 0))
    root = _extract_members(archive, tmp_path / "cache")
    if corruption == "changed":
        (root / "mHealth_subject1.log").write_text(_row(99.0, 0), encoding="ascii")
    elif corruption == "extra":
        (root / "unexpected.log").write_text("unexpected", encoding="ascii")
    else:
        (root / "mHealth_subject1.log").unlink()
        (root / "mHealth_subject1.log").symlink_to(root / "mHealth_subject2.log")
    with pytest.raises(TimeFFormatError, match=r"cache root|cached member"):
        _extract_members(archive, tmp_path / "cache")
