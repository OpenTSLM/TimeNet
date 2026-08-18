"""Shared CLI presentation: emoji-prefixed status lines on stderr, with a quiet toggle.

Both the consumer ``timenet`` CLI and the producer ``timenet-curate`` CLI report progress through the
shared :data:`console` so their output reads the same. Status goes to stderr, leaving stdout for the
machine-readable result (a path or id) that a script may capture or pipe. ``--quiet`` silences status
while still surfacing warnings and errors. The console wraps a Rich :class:`~rich.console.Console`
(exposed as :attr:`~_Console.rich`), so a live display mounted on it, such as download progress bars,
coordinates with these status lines and renders them above the bars instead of corrupting the region.
"""

from rich.console import Console


class _Console:
    """Writes emoji status lines to stderr; stdout stays reserved for machine-readable output."""

    def __init__(self) -> None:
        self.quiet = False
        self.rich = Console(stderr=True)

    def status(self, emoji: str, message: str) -> None:
        """Print an ``<emoji> <message>`` status line (suppressed when quiet)."""
        self._emit(emoji, message)

    def success(self, message: str) -> None:
        """Print a success line (suppressed when quiet)."""
        self._emit("✅", message, style="green")

    def warn(self, message: str) -> None:
        """Print a warning (shown even when quiet)."""
        self._emit("⚠️", message, style="yellow", force=True)

    def error(self, message: str) -> None:
        """Print an error (shown even when quiet)."""
        self._emit("❌", message, style="red", force=True)

    def _emit(self, emoji: str, message: str, *, style: str | None = None, force: bool = False) -> None:
        if self.quiet and not force:
            return
        self.rich.print(f"{emoji} {message}", style=style, markup=False, highlight=False)


console = _Console()
