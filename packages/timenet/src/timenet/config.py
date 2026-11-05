"""Runtime configuration from the ``TIMENET_*`` environment variables.

An explicit argument, for example a CLI flag, wins over its variable, and the variable wins over
its default. Each variable sets one thing:

- ``TIMENET_LOCAL_REGISTRY`` (default ``~/.timenet/registry``): the store, where downloads and
  builds land.
- ``TIMENET_REMOTE_REGISTRY`` (default ``timenet://``): the registry after the store in the default
  list of sources.
- ``TIMENET_AUTO_BUILD`` (default ``true``): whether that list ends with the installed connectors.
- ``TIMENET_CACHE`` (default ``~/.cache/timenet``): files that a build or a read can make again,
  the raw downloads in ``downloads/`` and the control databases in ``control/``.
- ``TIMENET_TOKEN``: the bearer token for a remote registry.
- ``TIMENET_ISOLATION`` (default ``on``): whether a build runs in an environment of its own.
"""

import os
from pathlib import Path
from typing import Any, Literal

from pydantic import ValidationError, ValidationInfo, field_validator
from pydantic_core import ErrorDetails
from pydantic_settings import BaseSettings, SettingsConfigDict

from timenet._registry_uri import local_registry_path
from timenet.errors import TimeFValidationError


# Follow the Cache Directory Tagging Specification by Bryan Ford, from 2004.
#
# A folder is a cache if it holds a file CACHEDIR.TAG that starts with a fixed signature line.
# Many common tools write it, for example Cargo in target/, pytest in .pytest_cache, ruff, mypy
# and uv in their caches. TimeNet's own cleanup logic uses it as a safety check. This way, a
# TIMENET_CACHE that points at the wrong folder, such as your home or another tool's cache, is
# never wiped. Backup tools also skip TimeNet's caches as a side effect.
CACHE_TAG = "CACHEDIR.TAG"
"""The name of the marker file in each folder of the cache that TimeNet created."""

_CACHE_TAG_LINES = (
    "Signature: 8a477f597d28d172789f06886806bc55",
    "# This file is a cache directory tag created by TimeNet.",
    "# For information about cache directory tags, see https://bford.info/cachedir/",
)
"""The text of the marker file. The first line is the signature of the standard."""


class TimeNetSettings(BaseSettings):
    """TimeNet's settings. Reads ``TIMENET_*`` environment variables (and an optional ``.env``)."""

    model_config = SettingsConfigDict(env_prefix="TIMENET_", env_file=".env", extra="ignore")

    local_registry: str = "~/.timenet/registry"
    """The store: a local path or a ``file://`` URI (``TIMENET_LOCAL_REGISTRY``)."""
    remote_registry: str = "timenet://"
    """The registry after the store in the default list of sources (``TIMENET_REMOTE_REGISTRY``)."""
    auto_build: bool = True
    """Whether the default list of sources ends with ``InstalledConnectors()`` (``TIMENET_AUTO_BUILD``)."""
    cache: str = "~/.cache/timenet"
    """The directory of raw downloads and control databases (``TIMENET_CACHE``)."""
    token: str | None = None
    """Bearer token for a remote registry (``TIMENET_TOKEN``); ``None`` is anonymous."""
    isolation: Literal["on", "off"] = "on"
    """Whether a build runs in an environment built from the connector's requirements.

    ``"off"`` runs it in the current interpreter. The isolated child sets this to ``"off"`` in its own
    environment, which is what stops it re-execing forever.
    """

    @field_validator("local_registry", "remote_registry", "cache", mode="before")
    @classmethod
    def _not_empty(cls, value: Any, info: ValidationInfo) -> Any:
        """Turn a path into a string, and reject an empty or blank value, which names no directory.

        Args:
            value: The value from an argument, the environment or the ``.env`` file.
            info: Names the field.

        Returns:
            The value, with a path turned into a string.

        Raises:
            TimeFValidationError: If the value is empty or holds only whitespace.
        """
        if isinstance(value, os.PathLike):
            value = os.fspath(value)
        if isinstance(value, str) and not value.strip():
            name = str(info.field_name)
            raise TimeFValidationError(
                f"TIMENET_{name.upper()} must not be empty; give it a value, "
                f"or leave it unset for {cls.model_fields[name].default}."
            )
        return value

    @property
    def local_registry_dir(self) -> Path:
        """The directory of the store, with ``~`` expanded.

        Raises:
            TimeNetRegistryError: If ``local_registry`` names a remote registry.
        """  # ruff: ignore[docstring-extraneous-exception] (raised by local_registry_path)
        return local_registry_path(self.local_registry)

    @property
    def cache_dir(self) -> Path:
        """The cache directory, with ``~`` expanded."""
        return Path(self.cache).expanduser()

    @property
    def downloads_dir(self) -> Path:
        """Where a build downloads raw sources, one directory per dataset ID."""
        return self.cache_dir / "downloads"

    @property
    def control_cache_dir(self) -> Path:
        """Where verified remote control databases are cached by checksum."""
        return self.cache_dir / "control"


def settings(**overrides: Any) -> TimeNetSettings:
    """Return the settings, with each override that is not ``None`` over its variable and its default.

    Overrides other than ``None`` take precedence over environment variables and defaults.

    Args:
        **overrides: Values by field name, for example ``local_registry=...`` or ``auto_build=...``.

    Returns:
        The settings.

    Raises:
        TimeFValidationError: If a value is invalid.
    """
    given = {key: value for key, value in overrides.items() if value is not None}
    try:
        return TimeNetSettings(**given)
    except ValidationError as error:
        raise TimeFValidationError(" ".join(_sentence(item) for item in error.errors())) from error


def _sentence(item: ErrorDetails) -> str:
    """Describe one invalid value of the settings, by the name of its variable.

    Args:
        item: One error of a pydantic ``ValidationError``.

    Returns:
        A full sentence.
    """
    cause = item.get("ctx", {}).get("error")
    if isinstance(cause, TimeFValidationError):
        return str(cause)
    name = "TIMENET_" + "_".join(str(part) for part in item["loc"]).upper()
    return f"{name} is invalid: {item['msg']}, got {item['input']!r}."


def make_cache_folder(folder: Path) -> None:
    """Create a folder of the cache, and mark it as TimeNet's if this call creates it.

    The mark is a ``CACHEDIR.TAG`` file (see https://bford.info/cachedir/), and ``timenet local cache clear``
    deletes only a folder with it. A folder that exists already stays as it is.

    Args:
        folder: The folder, for example ``settings().downloads_dir``. Its parents are created too.
    """
    try:
        folder.mkdir(parents=True)
    except FileExistsError:
        return
    (folder / CACHE_TAG).write_text("\n".join(_CACHE_TAG_LINES) + "\n")


def is_cache_folder(folder: Path) -> bool:
    """Tell whether TimeNet created ``folder``: it holds the ``CACHEDIR.TAG`` file that TimeNet writes.

    A tag of another tool does not count. The check follows a symlink to its target.

    Args:
        folder: The folder.

    Returns:
        ``True`` if the first two lines of ``<folder>/CACHEDIR.TAG`` are those that TimeNet writes.
    """
    try:
        with (folder / CACHE_TAG).open(encoding="utf-8", errors="replace") as tag:
            head = [tag.readline().rstrip("\n") for _ in range(2)]
    except OSError:
        return False
    return head == list(_CACHE_TAG_LINES[:2])
