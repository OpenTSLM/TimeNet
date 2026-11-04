"""Publish a complete directory in place of another with renames."""

from pathlib import Path
import shutil
import uuid


def publish_directory(staging: Path, final: Path) -> None:
    """Move a complete staging directory to ``final``, and replace a directory that is there.

    An existing ``final`` stays in a sibling ``<name>.old-<uuid>`` directory until the new one is in
    place, and moves back if the rename fails or stops. After a hard kill between the two renames,
    rename ``<name>.old-<uuid>`` back to ``final`` to recover it.

    Args:
        staging: The complete new directory, in the same parent directory as ``final``.
        final: The directory to publish to. Its parent must exist.

    Raises:
        OSError: If the rename fails and the old directory cannot move back. The error names it.
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
