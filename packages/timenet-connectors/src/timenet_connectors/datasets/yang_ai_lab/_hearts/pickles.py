"""Read pinned HEARTS pickles through an allowlist of pandas and NumPy globals."""

import functools
from pathlib import Path
import pickle  # noqa: S403 - every load goes through _RestrictedUnpickler, never bare pickle.load
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
"""Every global the pinned release names. A pickle can run code as it loads, so any other is refused."""


class _RestrictedUnpickler(pickle.Unpickler):  # noqa: S301 - this class is the guard S301 asks for
    """Resolve only the globals in the allowlist."""

    def find_class(self, module: str, name: str) -> Any:
        """Return the requested global, or refuse it by name.

        Returns:
            The resolved global.

        Raises:
            TimeFFormatError: If the pair is not in the allowlist.
        """
        if (module, name) not in _ALLOWED_GLOBALS:
            raise TimeFFormatError(
                f"HEARTS pickle names {module}.{name}, which this reader does not load. "
                "Inspect the file with pickletools.genops before allowing it."
            )
        return super().find_class(module, name)


@functools.lru_cache(maxsize=2)
def load_payload(path: str) -> dict[str, Any]:
    """Read one test case, keeping the two read last.

    The writer reads values sorted by spec and then by source id, so the frames of one case that
    share a spec are read together, and the cache serves them from one read of the file.

    Returns:
        The test case's dictionary.

    Raises:
        ImportError: If pandas, which the pickled frames need, is not installed.
    """
    try:
        with Path(path).open("rb") as handle:
            return _RestrictedUnpickler(handle).load()
    except ModuleNotFoundError as exc:
        raise ImportError(
            f"reading HEARTS needs {exc.name}, declared in this connector's requirements.txt. The released "
            "files are pandas 2.x pickles, so pandas 3 cannot rebuild them"
        ) from exc
