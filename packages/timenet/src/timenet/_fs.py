"""Compare directory paths and replace directories from staging."""

from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
import shutil
import uuid


def same_directory(a: Path, b: Path) -> bool:
    """Check whether two paths name the same directory.

    Existing paths match when they identify the same directory on disk.
    If either path is missing, compare resolved paths while ignoring letter case.

    Args:
        a: The first path, whether or not it exists.
        b: The second path, whether or not it exists.

    Returns:
        ``True`` if the paths match.
    """
    if a.exists() and b.exists():
        return a.samefile(b)
    return str(a.resolve()).casefold() == str(b.resolve()).casefold()


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


@contextmanager
def staged_directory(final: Path) -> Generator[Path]:
    """Yield an empty staging directory beside ``final``, and publish it as ``final`` if the block succeeds.

    The old ``final`` stays in place until the new directory is complete. The staging directory is
    removed in every case.

    Args:
        final: The destination directory.

    Yields:
        The staging directory.
    """
    final.parent.mkdir(parents=True, exist_ok=True)
    # A hard kill can leave a staging directory behind. Do not sweep other <name>.tmp-* directories:
    # a concurrent write of the same release may still use one.
    staging = final.with_name(f"{final.name}.tmp-{uuid.uuid4().hex}")
    staging.mkdir()
    try:
        yield staging
        publish_directory(staging, final)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
