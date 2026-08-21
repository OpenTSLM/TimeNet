"""On-disk filenames and layout templates for the TimeF format."""

from timenet.errors import TimeFValidationError


MANIFEST_FILE = "manifest.json"

SHARD_DIR = "time_series"
SHARD_TEMPLATE = "time_series/part-{:08d}.parquet"

TASKS_DIR = "tasks"
TASK_PART_TEMPLATE = "tasks/task={task_type}/part-{:08d}.parquet"

# Control-plane tables, each sharded into numbered parts under their own directory. part_path() caps
# the part number at PART_INDEX_DIGITS so the zero-padded names stay lexically sortable.
SAMPLES_TEMPLATE = "samples/part-{:08d}.parquet"
ANNOTATIONS_TEMPLATE = "annotations/part-{:08d}.parquet"
INDEX_TEMPLATE = "time_series_index/part-{:08d}.parquet"

# PART_INDEX_DIGITS sets how many digits pad the part or shard numbers. Lexical order matches
# numeric order only while the count fits that width, so part_path() refuses any index past the
# ceiling.
PART_INDEX_DIGITS = 8
MAX_PART_INDEX = 10**PART_INDEX_DIGITS - 1

# The writer sorts each prunable control table by this column at write time, and the reader prunes
# and bisects on it at read time. Naming the contract keeps the writer's sort and the reader's
# lookup from drifting.
INDEX_SORT_KEY = "sample_id"
ANNOTATIONS_SORT_KEY = "id"

DEFAULT_SHARD_TARGET_BYTES = 128 * 2**20
# Target size for one control-table part, measured on the in-memory Arrow table (not on disk). The
# control-table target stays separate from the values-plane target, so a maintainer can tune the two
# independently.
DEFAULT_CONTROL_SHARD_TARGET_BYTES = 128 * 2**20
DEFAULT_ROW_GROUP_TARGET_BYTES = 4 * 2**20
DEFAULT_CHUNK_MAX_BYTES = 1 * 2**20
DEFAULT_COMPRESSION = "zstd"
DEFAULT_COMPRESSION_LEVEL = 3


def check_relative_path(name: str, path: str) -> None:
    """Reject a path that would resolve outside the dataset root it is meant to stay under.

    Callers join a manifest file entry and a registry handle's ``relpath`` onto a root directory
    (or filesystem prefix) as-is. An absolute path or a ``..`` segment would escape that root instead
    of raising. This would let a crafted manifest or caller read or write outside the intended
    directory.

    Args:
        name: The name of the field to check. It appears in the error message.
        path: The path to check. It must be relative and rooted inside the dataset.

    Raises:
        TimeFValidationError: If ``path`` is absolute, or has a ``..`` segment.
    """
    if path.startswith("/") or any(segment == ".." for segment in path.split("/")):
        raise TimeFValidationError(f"{name} must stay within the dataset root, got {path!r}")


def part_path(template: str, index: int, **fields: str) -> str:
    """Render a numbered part/shard path, refusing an index that would overflow the fixed width.

    The reader visits parts in the order the manifest lists them. The zero-padded names keep an
    ``ls`` or a prefix listing in that same order. That order only holds while every number fits in
    ``PART_INDEX_DIGITS``. A ninth digit sorts before the eighth and silently breaks it. Routing
    every part name through here turns that overflow into a loud error instead.

    Args:
        template: A layout template whose part number is the ``{:08d}`` field, for example
            ``SHARD_TEMPLATE``.
        index: The zero-based part number to render.
        fields: Any remaining named fields the template needs, for example ``task_type``.

    Returns:
        The formatted version-relative path.

    Raises:
        TimeFValidationError: If ``index`` is negative or needs more than ``PART_INDEX_DIGITS`` digits.
    """
    if not 0 <= index <= MAX_PART_INDEX:
        raise TimeFValidationError(
            f"part index {index} is outside [0, {MAX_PART_INDEX}]: a wider number would break the "
            f"{PART_INDEX_DIGITS}-digit lexical ordering of {template!r}"
        )
    return template.format(index, **fields)
