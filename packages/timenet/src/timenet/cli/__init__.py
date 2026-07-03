"""Command-line interface for TimeNet (needs the ``cli`` extra: ``pip install 'timenet[cli]'``).

Importing this package is cheap and pulls in no third-party deps; the Rich/Typer app in
:mod:`timenet.cli.app` is loaded lazily by :func:`main`. That keeps producers that only use the shared
:mod:`timenet.cli.runner` / :mod:`timenet.cli.ui` (Typer only) from dragging in Rich, and makes a base
install without the ``cli`` extra fail with a clear message instead of an opaque import error.
"""

from timenet.cli.entry import main


__all__ = ["main"]
