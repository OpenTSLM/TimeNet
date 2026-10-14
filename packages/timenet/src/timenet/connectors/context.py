"""Resolved build inputs and dependency paths shared by connectors and builders."""

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
import json
import os
from types import MappingProxyType

from timenet.errors import TimeNetBuildError
from timenet.reader import TimeFReader
from timenet.types import DatasetRef


_BUILD_PATH: ContextVar[tuple[DatasetRef, ...] | None] = ContextVar("timenet_build_path", default=None)


@dataclass(frozen=True)
class BuildContext:
    """Exact, already opened inputs available for the whole conversion and write."""

    inputs: Mapping[str, TimeFReader] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "inputs", MappingProxyType(dict(self.inputs)))


def active_build_path() -> tuple[DatasetRef, ...]:
    """Return the dependency path, including an isolated parent's path.

    Returns:
        Exact releases currently being built.

    Raises:
        TimeNetBuildError: If the subprocess build path is invalid.
    """
    active = _BUILD_PATH.get()
    if active is not None:
        return active
    try:
        return tuple(
            DatasetRef.model_validate(value) for value in json.loads(os.environ.get("TIMENET_BUILD_PATH", "[]"))
        )
    except (ValueError, TypeError) as exc:
        raise TimeNetBuildError(f"invalid TIMENET_BUILD_PATH: {exc}") from exc


@contextmanager
def build_scope(reference: DatasetRef) -> Iterator[None]:
    """Keep an exact dependency path while a connector is built.

    Yields:
        Control to the build.

    Raises:
        TimeNetBuildError: If the build closes a dependency cycle.
    """
    active = active_build_path()
    if reference in active:
        chain = " -> ".join(str(item) for item in (*active, reference))
        raise TimeNetBuildError(f"connector inputs form a cycle: {chain}")
    token = _BUILD_PATH.set((*active, reference))
    try:
        yield
    finally:
        _BUILD_PATH.reset(token)
