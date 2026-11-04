"""Replace a directory with a complete staging directory."""

from pathlib import Path
import shutil
import uuid


def publish_directory(staging: Path, final: Path) -> None:
    """Rename ``staging`` to ``final``, replacing any existing directory.

    If publishing fails, this function tries to restore the old directory. After a hard kill during
    replacement, rename the sibling ``<name>.old-<uuid>`` directory back to ``final`` to recover it.

    Args:
        staging: The completed directory, with the same parent as ``final``.
        final: The destination directory, whose parent must exist.

    Raises:
        OSError: If a rename fails, with the backup path if the old directory cannot be restored.
    """
    old = None
    if final.exists():
        old = final.with_name(f"{final.name}.old-{uuid.uuid4().hex}")
        final.replace(old)
    try:
        staging.replace(final)
    except BaseException as error:
        if old is not None:
            try:
                old.replace(final)
            except OSError as restore_error:
                raise OSError(
                    f"could not publish {final} ({error}) or move the old release back "
                    f"({restore_error}); it is in {old}, so rename that to {final} to recover it"
                ) from error
        raise
    if old is not None:
        # The new directory is published, so failing to delete the old one must not fail the publish.
        shutil.rmtree(old, ignore_errors=True)
