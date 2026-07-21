import pytest

from timenet.cli.ui import _Console


@pytest.fixture
def console():
    return _Console()


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
    console.warn("heads up")
    console.error("boom")
    captured = capsys.readouterr()
    assert "downloading" not in captured.err
    assert "done" not in captured.err
    assert "heads up" in captured.err  # warnings/errors ignore quiet
    assert "boom" in captured.err
