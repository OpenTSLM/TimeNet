"""Download-progress reporting for connectors.

Downloads report progress through an ambient sink rather than a callback threaded down every layer. A
caller (the ``timenet-curate`` CLI, or a test) installs a sink with :func:`progress_sink`; the download
helpers deep in the stack call :func:`report_progress`, and the event surfaces without the connector,
the engine, or any intermediate function growing a ``progress`` parameter. The sink is a
:class:`~contextvars.ContextVar`, so it propagates correctly across ``asyncio.run`` and into gathered
tasks, and concurrent downloads on the single event-loop thread deliver their events serially.
"""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass


@dataclass(frozen=True)
class DownloadProgress:
    """Progress for one download: bytes transferred so far and the total, if known."""

    url: str
    downloaded: int
    total: int | None


ProgressCallback = Callable[[DownloadProgress], None]

_sink: ContextVar[ProgressCallback | None] = ContextVar("download_progress_sink", default=None)


@contextmanager
def progress_sink(callback: ProgressCallback | None) -> Iterator[None]:
    """Install ``callback`` as the download-progress sink for the duration of the block.

    Args:
        callback: The sink to receive :class:`DownloadProgress` events, or ``None`` to report nothing.

    Yields:
        Nothing; the sink is active within the ``with`` block and restored on exit.
    """
    token = _sink.set(callback)
    try:
        yield
    finally:
        _sink.reset(token)


def report_progress(event: DownloadProgress) -> None:
    """Deliver a progress event to the installed sink, or do nothing if none is installed.

    Args:
        event: The progress event to report.
    """
    sink = _sink.get()
    if sink is not None:
        sink(event)


def current_sink() -> ProgressCallback | None:
    """Return the installed download-progress sink, or ``None`` if none is installed.

    A downloader captures this on its own thread before handing a callback to a library that reports
    from worker threads (e.g. boto3), where the :class:`~contextvars.ContextVar` would not propagate,
    and calls the returned sink directly. It also lets a downloader skip work it only needs for
    reporting (an S3 ``HEAD`` for the total size) when the sink is ``None``.

    Returns:
        The current sink callback, or ``None`` if none is installed.
    """
    return _sink.get()
