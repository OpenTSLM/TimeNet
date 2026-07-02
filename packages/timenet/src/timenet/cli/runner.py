"""Shared console-script entry-point wrapper for the TimeNet CLIs."""

import sys

import typer

from timenet.errors import TimeNetError


def run_cli(app: typer.Typer) -> None:
    """Run a Typer app as a console-script entry point.

    Expected failures (a raised :class:`~timenet.errors.TimeNetError`) print a one-line message and
    exit non-zero; only unexpected errors surface a traceback.

    Args:
        app: The Typer application to invoke.
    """
    try:
        app()
    except TimeNetError as exc:
        typer.secho(f"Error: {exc}", fg=typer.colors.RED, err=True)
        sys.exit(1)
