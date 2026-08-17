"""Tests for the timenet-curate download-progress rendering."""

import sys

from rich.progress import Progress

from timenet.cli.ui import console
from timenet_connectors.curate.cli import _bar_reporter, _download_progress
from timenet_connectors.download.progress import DownloadProgress, report_progress


def test_bar_reporter_tracks_one_task_per_url():
    progress = Progress()
    report = _bar_reporter(progress)
    report(DownloadProgress("https://host/records.zip", 10, 100))
    report(DownloadProgress("https://host/records.zip", 60, 100))  # same URL -> same row, advanced
    report(DownloadProgress("https://host/labels.csv", 5, 20))  # new URL -> its own row

    tasks = {task.fields["name"]: task for task in progress.tasks}
    assert set(tasks) == {"records.zip", "labels.csv"}
    assert (tasks["records.zip"].completed, tasks["records.zip"].total) == (60, 100)
    assert (tasks["labels.csv"].completed, tasks["labels.csv"].total) == (5, 20)


def test_bar_reporter_handles_unknown_total():
    progress = Progress()
    _bar_reporter(progress)(DownloadProgress("https://host/stream.bin", 4096, None))
    (task,) = progress.tasks
    assert task.total is None  # indeterminate bar, still tracked
    assert task.completed == 4096


def test_download_progress_is_silent_when_quiet(monkeypatch):
    monkeypatch.setattr(console, "quiet", True)
    with _download_progress() as progress:
        assert progress is None  # no display installed


def test_download_progress_falls_back_to_text_off_a_tty(monkeypatch):
    monkeypatch.setattr(console, "quiet", False)
    monkeypatch.setattr(sys.stderr, "isatty", lambda: False, raising=False)
    with _download_progress() as progress:
        assert progress is None  # throttled text lines, no bars


def test_download_progress_renders_bars_on_a_tty(monkeypatch):
    monkeypatch.setattr(console, "quiet", False)
    monkeypatch.setattr(sys.stderr, "isatty", lambda: True, raising=False)
    with _download_progress() as progress:
        assert isinstance(progress, Progress)
        report_progress(DownloadProgress("https://host/a.zip", 3, 9))  # routed through the ambient sink
        (task,) = progress.tasks
        assert task.fields["name"] == "a.zip"
        assert task.completed == 3
