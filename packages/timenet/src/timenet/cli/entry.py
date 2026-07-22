"""Guarded console-script entry point for the ``timenet`` CLI.

Imports nothing third-party at module load, so it stays importable in a base install; the Rich/Typer app
is loaded lazily by :func:`main`.
"""


def main() -> None:
    """Run the ``timenet`` CLI, exiting with a clear message if the ``cli`` extra is missing.

    Raises:
        SystemExit: If the ``cli`` extra (Typer + Rich) is not installed.
        ModuleNotFoundError: Re-raised if the CLI fails to import for any other reason, so a real
            traceback surfaces instead of a misleading install hint.
    """
    try:
        from timenet.cli.app import main as run
    except ModuleNotFoundError as exc:
        # Only a genuinely missing cli extra (Typer/Rich) should trigger the install hint. A typo in an
        # internal import raises ModuleNotFoundError too; that must surface as a real traceback rather
        # than misleading the user into installing a package.
        if (exc.name or "").split(".")[0] not in {"typer", "rich"}:
            raise
        raise SystemExit("the timenet CLI needs the 'cli' extra: pip install 'timenet[cli]'") from exc
    run()
