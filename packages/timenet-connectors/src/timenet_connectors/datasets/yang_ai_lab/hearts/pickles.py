"""Read HEARTS pickles with an allowlist of pandas and NumPy globals.

A two-file cache shares payloads between lazy signal loaders and limits memory use.
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
        ("pandas", "DataFrame"),
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
"""Globals observed in the pinned release, plus the public pandas DataFrame alias."""

_PAYLOAD_CACHE_SIZE = 2
"""Maximum number of cached payloads."""


class _RestrictedUnpickler(pickle.Unpickler):  # noqa: S301 - this class is the guard S301 asks for
    """Allow only the globals in the pinned release."""

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
                f"HEARTS test case {self._path} requests disallowed global {module}.{name}. "
                "Inspect the file with pickletools.genops before changing _ALLOWED_GLOBALS."
            )
        return super().find_class(module, name)


def require_pandas() -> Any:
    """Import the pandas dependency.

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
    """Read and cache one test-case dictionary.

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
    """Read a nested payload value by its key path.

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
