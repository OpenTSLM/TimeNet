"""Compare directory paths, delete directory trees, and replace directories from staging."""

from collections.abc import Generator
from contextlib import contextmanager
import os
from pathlib import Path
import shutil
import uuid


def same_directory(a: Path, b: Path) -> bool:
    """Check whether two paths name the same directory, as the file system sees them.

    The paths compare by :func:`path_identity`, so a symlink or a case variant on a volume that ignores case
    matches, and names that do not exist compare as written.

    Args:
        a: The first path, whether or not it exists.
        b: The second path, whether or not it exists.

    Returns:
        ``True`` if the paths match.
    """
    return path_identity(a.resolve()) == path_identity(b.resolve())


def path_identity(path: Path) -> tuple[int, int, tuple[str, ...]]:
    """Identify a path by its nearest existing ancestor on disk and the names below it.

    Args:
        path: An absolute path without symlinks.

    Returns:
        The ancestor's device and inode numbers, followed by the remaining names as written.
    """
    for depth, candidate in enumerate((path, *path.parents)):
        try:
            info = candidate.stat()
        except OSError:
            continue
        missing = path.parts[len(path.parts) - depth :]
        return info.st_dev, info.st_ino, missing
    return 0, 0, path.parts  # nothing exists, not even the root


def lies_in(path: Path, folder: Path) -> bool:
    """Check whether ``path`` is ``folder`` or lies inside it.

    The check follows symlinks and respects the filesystem's case rules for existing paths.
    It compares path parts that do not exist as written.

    Args:
        path: The path to check, whether or not it exists.
        folder: The folder to check against, whether or not it exists.

    Returns:
        ``True`` if ``path`` is ``folder`` or lies inside it.
    """
    target = path_identity(folder.expanduser().resolve())
    path = path.expanduser().resolve()
    return any(path_identity(candidate) == target for candidate in (path, *path.parents))


def delete_and_measure(path: Path) -> int:
    """Delete a directory tree and return the bytes it held. It stats each file only once.

    One file pass both measures and unlinks each file. Then :func:`shutil.rmtree` removes the empty
    directory skeleton. This avoids a second walk over every file.

    An unreadable directory or a failed removal raises an ``OSError``.

    Args:
        path: The directory to delete.

    Returns:
        The total size in bytes of the files removed.
    """

    def fail(error: OSError) -> None:
        raise error  # os.walk otherwise skips a directory it cannot read, silently leaving it behind

    freed = 0
    for parent, _dirs, files in os.walk(path, topdown=False, onerror=fail):  # os.walk: Path.walk is 3.12+
        parent_dir = Path(parent)
        for name in files:
            entry = parent_dir / name
            # lstat: a symlink counts as itself, not as its (possibly huge, possibly external) target.
            freed += entry.lstat().st_size
            entry.unlink(missing_ok=True)  # a concurrent build may have removed it between walk and unlink
    shutil.rmtree(path)
    return freed


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
