"""Tests for the download-progress sink."""

from timenet_connectors.download.progress import (
    DownloadProgress,
    current_sink,
    progress_sink,
    report_progress,
)


def test_report_progress_delivers_to_installed_sink():
    seen: list[DownloadProgress] = []
    with progress_sink(seen.append):
        report_progress(DownloadProgress("u", 1, 2))
    assert seen == [DownloadProgress("u", 1, 2)]


def test_report_progress_is_a_noop_without_a_sink():
    report_progress(DownloadProgress("u", 1, 2))  # no sink installed: must not raise


def test_current_sink_returns_the_installed_sink():
    def sink(event: DownloadProgress) -> None:
        return None

    assert current_sink() is None
    with progress_sink(sink):
        assert current_sink() is sink
    assert current_sink() is None  # reset on exit


def test_sink_is_restored_after_the_block():
    outer: list[DownloadProgress] = []
    inner: list[DownloadProgress] = []
    with progress_sink(outer.append):
        with progress_sink(inner.append):
            report_progress(DownloadProgress("inner", 0, None))
        report_progress(DownloadProgress("outer", 0, None))
    assert [e.url for e in inner] == ["inner"]
    assert [e.url for e in outer] == ["outer"]
