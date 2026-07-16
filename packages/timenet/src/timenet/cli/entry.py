"""Guarded console-script entry point for the ``timenet`` CLI.

Imports nothing third-party at module load, so it stays importable in a base install; the Rich/Typer app
is loaded lazily by :func:`main`.
"""


def main() -> None:
    """Run the ``timenet`` CLI, exiting with a clear message if the ``cli`` extra is missing.

    Raises:
        SystemExit: If the ``cli`` extra (Typer + Rich) is not installed.
    """
    try:
        from timenet.cli.app import main as run
    except ModuleNotFoundError as exc:
        raise SystemExit("the timenet CLI needs the 'cli' extra: pip install 'timenet[cli]'") from exc
    run()
