"""Reading the HEARTS payloads: a restricted unpickler and a two-file cache.

A pickle has no header. Nothing about a payload, not its length, not its rate, not its axis, is
known until the whole object is rebuilt. So ``convert`` reads every file once and the writer reads
it again when it drains the loaders. Reading is narrow on purpose: :class:`_RestrictedUnpickler`
allows only the globals the release's own files ask for and refuses anything else by name. The cache
holds two payloads, so a loader can re-read its file while the resident set stays small.
"""

from collections.abc import Sequence
import functools
from pathlib import Path
import pickle  # noqa: S403 - read through _RestrictedUnpickler, never bare
from typing import Any

from timenet.errors import TimeFFormatError


_ALLOWED_GLOBALS: frozenset[tuple[str, str]] = frozenset(
    {
        ("builtins", "slice"),
        ("numpy", "dtype"),
        ("numpy", "ndarray"),
        ("numpy._core.multiarray", "_reconstruct"),
        ("numpy._core.multiarray", "scalar"),
        ("pandas._libs.arrays", "__pyx_unpickle_NDArrayBacked"),
        ("pandas._libs.internals", "_unpickle_block"),
        ("pandas._libs.tslibs.timestamps", "_unpickle_timestamp"),
        ("pandas.core.arrays.datetimes", "DatetimeArray"),
        ("pandas.core.arrays.timedeltas", "TimedeltaArray"),
        ("pandas.core.frame", "DataFrame"),
        ("pandas.core.indexes.base", "Index"),
        ("pandas.core.indexes.base", "_new_Index"),
        ("pandas.core.indexes.range", "RangeIndex"),
        ("pandas.core.internals.managers", "BlockManager"),
    }
)
"""The globals the released files ask for, recorded from every file this connector reads at the
pinned revision. This connector's README says what that measurement covered."""

_PAYLOAD_CACHE_SIZE = 2
"""How many payloads stay resident. Two let one read serve every series of a file that shares a
spec type."""


class _RestrictedUnpickler(pickle.Unpickler):  # noqa: S301 - this class is the guard S301 asks for
    """An unpickler that imports only the globals in :data:`_ALLOWED_GLOBALS`."""

    def __init__(self, handle: Any, path: str) -> None:
        super().__init__(handle)
        self._path = path

    def find_class(self, module: str, name: str) -> Any:
        """Return the requested global, or refuse it by name.

        Args:
            module: The module the pickle names.
            name: The attribute the pickle names.

        Returns:
            The resolved global.

        Raises:
            TimeFFormatError: If the pair is not in the allowlist.
        """
        if (module, name) not in _ALLOWED_GLOBALS:
            raise TimeFFormatError(
                f"HEARTS test case {self._path} asks for {module}.{name}, which this connector does "
                f"not allow. The allowlist is a record of what the pinned revision asks for, so a "
                f"file written by a different pandas or numpy build can name a pair that belongs "
                f"there. List what the file asks for with pickletools.genops, which imports "
                f"nothing, then add ({module!r}, {name!r}) to _ALLOWED_GLOBALS if it belongs there"
            )
        return super().find_class(module, name)


def require_pandas() -> Any:
    """Import pandas, with a message naming this connector's requirements file.

    Returns:
        The pandas module.

    Raises:
        ImportError: If pandas is not installed.
    """
    try:
        import pandas  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "reading HEARTS needs pandas, declared in this connector's requirements.txt. The "
            "released files are pandas 2.x pickles, so pandas 3 cannot rebuild them. Run the build "
            "without --no-isolation, or install pandas 2 yourself"
        ) from exc
    return pandas


@functools.lru_cache(maxsize=_PAYLOAD_CACHE_SIZE)
def load_payload(path: str) -> dict[str, Any]:
    """Read one test-case pickle.

    Args:
        path: The file to read.

    Returns:
        The payload dict.

    Raises:
        TimeFFormatError: If the file does not hold a dict.
    """
    require_pandas()
    with Path(path).open("rb") as handle:
        payload = _RestrictedUnpickler(handle, path).load()
    if not isinstance(payload, dict):
        raise TimeFFormatError(f"HEARTS test case {path} holds a {type(payload).__name__}, not a dict")
    return payload


def dig(payload: dict[str, Any], keys: Sequence[str]) -> Any:
    """Follow a key path into a payload.

    Args:
        payload: The payload to read.
        keys: The keys to follow, outermost first.

    Returns:
        The node at that path.

    Raises:
        TimeFFormatError: If a key is missing.
    """
    node: Any = payload
    for key in keys:
        try:
            node = node[key]
        except (KeyError, TypeError) as exc:
            raise TimeFFormatError(f"HEARTS payload has no {'.'.join(keys)}: {key!r} is missing") from exc
    return node
