"""A fake set of installed builder backends."""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from timenet import builders


if TYPE_CHECKING:
    import pytest


@dataclass(frozen=True)
class _FakeDistribution:
    """The distribution that declares each fake entry point."""

    name: str = "timenet-testing"
    version: str = "0"


@dataclass(frozen=True)
class _FakeEntryPoint:
    """An entry point of the fake distribution ``timenet-testing 0``."""

    backend: object
    dist: _FakeDistribution = field(default_factory=_FakeDistribution)

    def load(self) -> Callable[[], object]:
        """Return the factory of the backend."""
        return lambda: self.backend


def install_backends(monkeypatch: "pytest.MonkeyPatch", *backends: object) -> list[object]:
    """Make the given builder backends the only installed ones, for the rest of the test.

    Returns:
        The installed backends. Append a backend to this list to install it too.
    """
    installed = list(backends)

    def entry_points(*, group: str) -> list[_FakeEntryPoint]:
        return [_FakeEntryPoint(backend) for backend in installed] if group == builders.ENTRY_POINT_GROUP else []

    monkeypatch.setattr(builders, "entry_points", entry_points)
    return installed
