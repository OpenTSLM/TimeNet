"""Derived caches: the state a page-cache drop does not touch.

Dropping the page cache evicts file pages. It invalidates none of the caches a loader writes for
itself, and those swing a first-item cell by orders of magnitude. HuggingFace ``datasets`` writes a
memory-mapped Arrow copy once and mmaps it forever after. PyHealth writes LitData chunks under the
user cache directory. Plain functions carry ``functools.lru_cache``.

So each lane declares one of three states and this module checks it before every timed region.
``absent`` means nothing was built and the timed region pays for building it. ``prebuilt`` means the
cache was built outside the timed region, and its bytes are charged to that side's storage as Tier
C. ``forbidden`` means the lane must not have one, and finding one voids the observation.

The state reaches the cell record, and the aggregator refuses to median two states together.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
import os
from pathlib import Path
import shutil
from typing import Any, Protocol

from benchmarks.paper.errors import CacheControlError
from benchmarks.paper.record import CACHE_ABSENT, CACHE_FORBIDDEN, CACHE_PREBUILT, DERIVED_CACHE_STATES
from benchmarks.paper.storage import apparent_bytes


_UNBUILT = frozenset({CACHE_ABSENT, CACHE_FORBIDDEN})


@dataclass(frozen=True)
class DerivedCache:
    """One on-disk cache a loader writes for itself.

    Attributes:
        name: What the appendix calls it.
        roots: Directories that hold it. A cache is present when any of them holds a file.
        owner: The library that writes it, for the error message.
    """

    name: str
    roots: tuple[Path, ...]
    owner: str = ""

    def present(self) -> bool:
        """Return whether the cache exists on disk.

        An empty directory does not count: HuggingFace and PyHealth both create their roots before
        they write anything into them.

        Returns:
            True when any root holds at least one file.
        """
        return any(_holds_a_file(root) for root in self.roots)

    def bytes(self) -> int:
        """Return the bytes this cache occupies, for the Tier C charge.

        Returns:
            The apparent size of every regular file under every root.
        """
        return sum(apparent_bytes(root) for root in self.roots if root.exists())

    def clear(self) -> None:
        """Delete the cache, so the next read builds it again.

        Raises:
            CacheControlError: If a root could not be removed.
        """
        for root in self.roots:
            if not root.exists():
                continue
            try:
                shutil.rmtree(root)
            except OSError as error:
                raise CacheControlError(f"could not clear the {self.name} cache at {root}: {error}") from error

    def as_dict(self) -> dict[str, Any]:
        """Return the JSON form.

        Returns:
            A plain mapping of the name, the roots and the owner.
        """
        return {"name": self.name, "roots": [str(root) for root in self.roots], "owner": self.owner}


class Cached(Protocol):
    """A function wrapped in ``functools.lru_cache``."""

    def cache_clear(self) -> None:
        """Drop everything the wrapper is holding."""
        ...


@dataclass(frozen=True)
class DerivedCacheDeclaration:
    """What one lane says about its derived caches, and the check that holds it to it.

    Attributes:
        lane: The lane this declaration belongs to.
        state: One of ``absent``, ``prebuilt`` or ``forbidden``.
        caches: The on-disk caches the lane's loader can write.
        functions: In-process ``lru_cache`` wrappers the lane must clear before a timed region.
    """

    lane: str
    state: str
    caches: tuple[DerivedCache, ...] = ()
    functions: tuple[Cached, ...] = ()

    def __post_init__(self) -> None:
        """Refuse a state outside the vocabulary the cell record writes.

        Raises:
            CacheControlError: If the state is not one of the three.
        """
        if self.state not in DERIVED_CACHE_STATES:
            raise CacheControlError(
                f"derived cache state must be one of {list(DERIVED_CACHE_STATES)}, got {self.state!r}"
            )

    def prepare(self) -> None:
        """Put the machine into the declared state before a timed region.

        A forbidden or absent state clears every declared cache. A pre-built state leaves them
        alone: they were built outside the timed region on purpose. Either way the in-process
        ``lru_cache`` wrappers are cleared, because a warm one is never part of a declared state.

        A cache that cannot be cleared raises :class:`CacheControlError`.
        """
        if self.state in _UNBUILT:
            for cache in self.caches:
                cache.clear()
        clear_function_caches(self.functions)

    def check(self) -> None:
        """Refuse a machine that is not in the declared state.

        Raises:
            CacheControlError: If a cache is present when the state says it should not be, or
                missing when the state says it should be there.
        """
        present = tuple(cache for cache in self.caches if cache.present())
        missing = tuple(cache for cache in self.caches if not cache.present())
        if self.state in _UNBUILT and present:
            names = ", ".join(cache.name for cache in present)
            raise CacheControlError(
                f"lane {self.lane} declares derived caches {self.state}, but these are on disk: {names}"
            )
        if self.state == CACHE_PREBUILT and missing:
            names = ", ".join(cache.name for cache in missing)
            raise CacheControlError(
                f"lane {self.lane} declares its derived caches pre-built, but these are not on disk: {names}. "
                "Build them outside the timed region, or the first repetition pays for them and the rest do not"
            )

    def tier_c_bytes(self) -> int:
        """Return the bytes a pre-built cache charges to its side's storage.

        A cache that is absent or forbidden charges nothing, because none exists.

        Returns:
            The apparent size of every declared cache, or zero.
        """
        if self.state != CACHE_PREBUILT:
            return 0
        return sum(cache.bytes() for cache in self.caches)

    def as_dict(self) -> dict[str, Any]:
        """Return the JSON form.

        Returns:
            A plain mapping of the lane, the state and the declared caches.
        """
        return {"lane": self.lane, "state": self.state, "caches": [cache.as_dict() for cache in self.caches]}


def clear_function_caches(functions: Iterable[Cached]) -> None:
    """Clear every ``lru_cache`` wrapper a lane declared.

    Args:
        functions: The wrapped functions.

    Raises:
        CacheControlError: If one of them is not a cache wrapper, which means the lane declared a
            function that silently keeps its results across repetitions.
    """
    for function in functions:
        clear = getattr(function, "cache_clear", None)
        if clear is None:
            raise CacheControlError(f"{function!r} is not wrapped in a cache, so it cannot be cleared")
        clear()


def huggingface_datasets_cache() -> DerivedCache:
    """Return the HuggingFace ``datasets`` Arrow cache.

    ``load_dataset`` writes a memory-mapped Arrow copy of the corpus once and mmaps it on every
    later call, so the second read of the same dataset is a different measurement from the first.

    Returns:
        The cache, rooted where the environment points it.
    """
    roots = [Path(os.environ.get("HF_DATASETS_CACHE", "")).expanduser()] if os.environ.get("HF_DATASETS_CACHE") else []
    if not roots:
        home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")).expanduser()
        roots = [home / "datasets"]
    return DerivedCache(name="huggingface-datasets", roots=tuple(roots), owner="datasets")


def pyhealth_cache() -> DerivedCache:
    """Return the PyHealth LitData cache.

    PyHealth wraps MNE and writes LitData chunks under the user cache directory on the first pass,
    then reads those chunks instead of the EDF files.

    Returns:
        The cache, rooted where the environment points it.
    """
    root = Path(os.environ.get("PYHEALTH_CACHE", Path.home() / ".cache" / "pyhealth")).expanduser()
    return DerivedCache(name="pyhealth-litdata", roots=(root,), owner="pyhealth")


def declaration_for(lane: str, state: str, caches: Sequence[DerivedCache] = ()) -> DerivedCacheDeclaration:
    """Build a declaration for one lane.

    Returns:
        The declaration.
    """
    return DerivedCacheDeclaration(lane=lane, state=state, caches=tuple(caches))


def _holds_a_file(root: Path) -> bool:
    """Return whether a directory tree holds at least one regular file.

    Returns:
        True for a file, or for a directory with a file anywhere under it.
    """
    if root.is_file():
        return True
    if not root.is_dir():
        return False
    return any(path.is_file() for path in root.rglob("*"))
