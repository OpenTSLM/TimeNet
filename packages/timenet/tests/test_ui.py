import io
import sys

import pytest
from rich.console import Console
from rich.progress import Progress

from timenet.cli import ui
from timenet.cli.ui import _bar_reporter, _Console, progress_renderer
from timenet.progress import BytesMoved, ShardWritten, StepFinished, StepStarted
from timenet.types.references import DatasetRef


HELLO = DatasetRef.parse("timenet/hello-world@1.0.0")


@pytest.fixture
def console():
    return _Console()


@pytest.fixture
def lines(monkeypatch):
    """Capture the status lines of the shared console, which is not quiet."""
    captured = io.StringIO()
    monkeypatch.setattr(ui.console, "rich", Console(file=captured, width=200))
    monkeypatch.setattr(ui.console, "quiet", False)
    return captured


@pytest.fixture
def bars(monkeypatch):
    """Record each live display of download bars that the renderer opens."""
    opened = []
    make = ui._download_bars

    def record():
        opened.append(make())
        return opened[-1]

    monkeypatch.setattr(ui, "_download_bars", record)
    return opened


def test_status_and_success_go_to_stderr(console, capsys):
    console.status("📥", "downloading")
    console.success("built hello_world")
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "📥 downloading" in captured.err
    assert "✅ built hello_world" in captured.err


def test_quiet_suppresses_status_but_not_errors(console, capsys):
    console.quiet = True
    console.status("📥", "downloading")
    console.success("done")
    console.detail("a plan line")
    console.warn("heads up")
    console.error("boom")
    captured = capsys.readouterr()
    assert "downloading" not in captured.err
    assert "done" not in captured.err
    assert "a plan line" not in captured.err
    assert "heads up" in captured.err  # warnings/errors ignore quiet
    assert "boom" in captured.err


def test_bar_reporter_tracks_one_task_per_url():
    progress = Progress()
    report = _bar_reporter(progress)
    report(BytesMoved("https://host/records.zip", 10, 100))
    report(BytesMoved("https://host/records.zip", 60, 100))  # same URL -> same row, advanced
    report(BytesMoved("https://host/labels.csv", 5, 20))  # new URL -> its own row

    tasks = {task.fields["name"]: task for task in progress.tasks}
    assert set(tasks) == {"records.zip", "labels.csv"}
    assert (tasks["records.zip"].completed, tasks["records.zip"].total) == (60, 100)
    assert (tasks["labels.csv"].completed, tasks["labels.csv"].total) == (5, 20)


def test_bar_reporter_handles_unknown_total():
    progress = Progress()
    _bar_reporter(progress)(BytesMoved("https://host/stream.bin", 4096, None))
    (task,) = progress.tasks
    assert task.total is None  # indeterminate bar, still tracked
    assert task.completed == 4096


def test_the_renderer_is_silent_when_quiet(monkeypatch, bars):
    monkeypatch.setattr(ui.console, "quiet", True)
    with progress_renderer() as progress:
        assert progress is None
    assert bars == []


def test_the_renderer_falls_back_to_text_off_a_tty(monkeypatch, lines, bars):
    monkeypatch.setattr(sys.stderr, "isatty", lambda: False, raising=False)
    with progress_renderer() as progress:
        assert progress is not None
        progress(BytesMoved("https://host/a.zip", 3, 9))
        progress(ShardWritten(HELLO, 2))
    assert lines.getvalue().splitlines() == ["⬇️ a.zip 33% (9 B)", "💾 wrote shard 2"]
    assert bars == []


def test_the_renderer_shows_downloads_as_bars_on_a_tty(monkeypatch, bars):
    monkeypatch.setattr(ui.console, "quiet", False)
    monkeypatch.setattr(sys.stderr, "isatty", lambda: True, raising=False)
    with progress_renderer() as progress:
        assert progress is not None
        progress(BytesMoved("https://host/a.zip", 3, 9))
        ((task,),) = [display.tasks for display in bars]
        assert (task.fields["name"], task.completed) == ("a.zip", 3)


def test_the_renderer_shows_copy_and_build_steps(monkeypatch, lines, bars):
    monkeypatch.setattr(sys.stderr, "isatty", lambda: True, raising=False)
    with progress_renderer() as progress:
        assert progress is not None
        for action in ("copy", "build"):
            progress(StepStarted(HELLO, action))
            progress(StepFinished(HELLO, action))

    # The live display ends with an empty line.
    assert [line for line in lines.getvalue().splitlines() if line] == [
        "📥 Copying 'timenet/hello-world'…",
        "✅ Copied 'timenet/hello-world' → 1.0.0",
        "🔧 Building 'timenet/hello-world'…",
        "✅ Built 'timenet/hello-world' → 1.0.0",
    ]
    assert len(bars) == 1


def test_detail_prints_the_line_as_it_is(console, capsys):
    # A line of a plan can name a long path. The console keeps the line whole, without an emoji.
    line = "└── test/parent@1.0.0: copy from local registry " + "/directory" * 12
    console.detail(line)
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == f"{line}\n"
