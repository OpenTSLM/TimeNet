"""Shared CLI output: emoji-prefixed status lines on stderr, with a quiet mode.

The ``timenet`` CLI and the ``timenet-build`` CLI both report progress through the shared
:data:`console`, so their output looks the same. Status messages go to stderr. Stdout stays free
for the machine-readable result, for example a path or an id, that a script can capture or pipe.
``--quiet`` hides status messages but still shows warnings and errors. The console wraps a Rich
:class:`~rich.console.Console` object, available as :attr:`~_Console.rich`. Because of this, a live
display on the console, for example a download progress bar, can show these status lines above it.
This avoids corruption of the display area.
"""

from collections.abc import Callable, Generator
from contextlib import contextmanager
from functools import partial
import sys

from rich.console import Console
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    TaskID,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)

from timenet.cache import human_bytes
from timenet.progress import BytesMoved, ProgressCallback, ProgressEvent, ShardWritten, StepStarted


class _Console:
    """Writes emoji status lines to stderr. Stdout stays reserved for machine-readable output."""

    def __init__(self) -> None:
        self.quiet = False
        self.rich = Console(stderr=True)

    def status(self, emoji: str, message: str) -> None:
        """Prints an ``<emoji> <message>`` status line. Quiet mode suppresses this line."""
        self._emit(emoji, message)

    def success(self, message: str) -> None:
        """Prints a success line. Quiet mode suppresses this line."""
        self._emit("✅", message, style="green")

    def detail(self, message: str) -> None:
        """Prints a line without an emoji. Quiet mode suppresses this line.

        The console does not wrap the line, so a long line stays whole.
        """
        if not self.quiet:
            self.rich.print(message, markup=False, highlight=False, soft_wrap=True)

    def warn(self, message: str) -> None:
        """Prints a warning line. Quiet mode does not suppress this line."""
        self._emit("⚠️", message, style="yellow", force=True)

    def error(self, message: str) -> None:
        """Prints an error line. Quiet mode does not suppress this line."""
        self._emit("❌", message, style="red", force=True)

    def _emit(self, emoji: str, message: str, *, style: str | None = None, force: bool = False) -> None:
        if self.quiet and not force:
            return
        self.rich.print(f"{emoji} {message}", style=style, markup=False, highlight=False)


console = _Console()


@contextmanager
def progress_renderer() -> Generator[ProgressCallback | None]:
    """Show progress events on the console for the block.

    Transfers appear as bars in a terminal or periodic status lines elsewhere.
    Steps and completed shards appear as status lines.

    Yields:
        The callback for the events, or ``None`` if the console is quiet.
    """
    if console.quiet:
        yield None
        return
    if not sys.stderr.isatty():
        yield partial(_render, _text_download_reporter())
        return
    with _download_bars() as progress:
        yield partial(_render, _bar_reporter(progress))


_STEP_LINES = {"copy": ("📥", "Copying", "Copied"), "build": ("🔧", "Building", "Built")}
"""The start emoji, start verb, and completion verb for each action."""


def _render(report_download: Callable[[BytesMoved], None], event: ProgressEvent) -> None:
    """Show one progress event on the console.

    Args:
        report_download: The callback that shows transfer progress.
        event: The event to show.
    """
    if isinstance(event, BytesMoved):
        report_download(event)
    elif isinstance(event, ShardWritten):
        console.status("💾", f"wrote shard {event.count}")
    else:
        emoji, doing, done = _STEP_LINES[event.action]
        if isinstance(event, StepStarted):
            console.status(emoji, f"{doing} '{event.release.dataset_id}'…")
        else:
            console.success(f"{done} '{event.release.dataset_id}' → {event.release.version}")


def _download_bars() -> Progress:
    """Create a live display of download progress bars.

    Returns:
        The progress display to activate with a ``with`` block.
    """
    return Progress(
        TextColumn("[bold blue]{task.fields[name]}", justify="right"),
        BarColumn(bar_width=None),
        "[progress.percentage]{task.percentage:>3.1f}%",
        "•",
        DownloadColumn(),
        "•",
        TransferSpeedColumn(),
        "•",
        TimeRemainingColumn(),
        console=console.rich,
    )


def _bar_reporter(progress: Progress) -> Callable[[BytesMoved], None]:
    """Create a callback that displays a progress bar for each transfer URL.

    Args:
        progress: The display that holds the bars.

    Returns:
        A callback that adds or updates a bar for each event.
    """
    tasks: dict[str, TaskID] = {}

    def report(event: BytesMoved) -> None:
        task_id = tasks.get(event.url)
        if task_id is None:
            task_id = progress.add_task("download", name=event.url.rsplit("/", 1)[-1], total=event.total)
            tasks[event.url] = task_id
        progress.update(task_id, completed=event.done, total=event.total)

    return report


_UNKNOWN_TOTAL_STEP_BYTES = 50 * 1024 * 1024  # progress cadence when the download size is unknown


def _text_download_reporter() -> Callable[[BytesMoved], None]:
    """Build a callback that shows throttled status lines for each transfer.

    It shows each transfer at about every 20%, or every 50 MB when the size is unknown.

    Returns:
        The callback.
    """
    last: dict[str, int] = {}

    def report(event: BytesMoved) -> None:
        # ~20% steps when the size is known, else one line every 50 MB.
        step = event.done * 5 // event.total if event.total else event.done // _UNKNOWN_TOTAL_STEP_BYTES
        if last.get(event.url) == step:
            return
        last[event.url] = step
        name = event.url.rsplit("/", 1)[-1]
        if event.total:
            console.status("⬇️", f"{name} {event.done * 100 // event.total}% ({human_bytes(event.total)})")
        else:
            console.status("⬇️", f"{name} {human_bytes(event.done)}")

    return report
