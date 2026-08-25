"""Shared CLI output: emoji-prefixed status lines on stderr, with a quiet mode.

The ``timenet`` CLI and the ``timenet-build`` CLI both report progress through the shared
:data:`console`, so their output looks the same. Status messages go to stderr. Stdout stays free
for the machine-readable result, for example a path or an id, that a script can capture or pipe.
``--quiet`` hides status messages but still shows warnings and errors. The console wraps a Rich
:class:`~rich.console.Console` object, available as :attr:`~_Console.rich`. Because of this, a live
display on the console, for example a download progress bar, can show these status lines above it.
This avoids corruption of the display area.
"""

from rich.console import Console


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
