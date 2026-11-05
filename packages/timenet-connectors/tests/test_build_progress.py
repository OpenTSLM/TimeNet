"""Tests for the timenet-build download-progress rendering."""

import io
import sys

from rich.console import Console

from timenet.cli.ui import console
from timenet_connectors.builder.cli import _download_progress
from timenet_connectors.download.progress import DownloadProgress, report_progress


def test_download_progress_shows_the_downloads_of_a_connector(monkeypatch):
    captured = io.StringIO()
    monkeypatch.setattr(console, "rich", Console(file=captured, width=200))
    monkeypatch.setattr(console, "quiet", False)
    monkeypatch.setattr(sys.stderr, "isatty", lambda: False, raising=False)
    with _download_progress():
        report_progress(DownloadProgress("https://host/a.zip", 3, 9))  # routed through the ambient sink
    assert captured.getvalue().splitlines() == ["⬇️ a.zip 33% (9 B)"]
