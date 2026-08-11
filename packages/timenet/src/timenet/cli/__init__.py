"""Command-line interface for TimeNet (needs the ``cli`` extra: ``pip install 'timenet[cli]'``).

Importing this package is cheap and pulls in no third-party deps. :func:`main` loads the Rich/Typer app
in :mod:`timenet.cli.app` lazily. This keeps producers that use only the shared
:mod:`timenet.cli.runner` and :mod:`timenet.cli.ui` (Typer only) from pulling in Rich. A base install
without the ``cli`` extra then fails with a clear message instead of an opaque import error.
"""

from timenet.cli.entry import main


__all__ = ["main"]
