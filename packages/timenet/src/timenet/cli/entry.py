"""Guarded console-script entry point for the ``timenet`` CLI.

This module imports nothing third-party at load time, so it stays importable in a base install.
:func:`main` loads the Rich/Typer app lazily.
"""


def main() -> None:
    """Run the ``timenet`` CLI, exiting with a clear message if the ``cli`` extra is missing.

    Raises:
        SystemExit: If the ``cli`` extra (Typer + Rich) is not installed.
        ModuleNotFoundError: Re-raised if the CLI fails to import for any other reason, so a real
            traceback surfaces instead of a misleading install hint.
    """
    try:
        from timenet.cli.app import main as run  # noqa: PLC0415
    except ModuleNotFoundError as exc:
        # Only a real missing cli extra (Typer/Rich) can trigger the install hint. A typo in an
        # internal import raises ModuleNotFoundError too. That case must surface as a real
        # traceback instead of a misleading install hint.
        if (exc.name or "").split(".")[0] not in {"typer", "rich"}:
            raise
        raise SystemExit("the timenet CLI needs the 'cli' extra: pip install 'timenet[cli]'") from exc
    run()
