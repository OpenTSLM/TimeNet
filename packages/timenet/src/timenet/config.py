"""Load TimeNet settings from arguments and ``TIMENET_*`` environment variables.

Explicit values override environment variables, which override defaults.
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
"""The marker filename for cache folders created by TimeNet."""

_CACHE_TAG_LINES = (
    "Signature: 8a477f597d28d172789f06886806bc55",
    "# This file is a cache directory tag created by TimeNet.",
    "# For information about cache directory tags, see https://bford.info/cachedir/",
)
"""The cache marker text, starting with the standard cache tag signature."""

_FLAGS = {"local_registry": "--local-registry", "remote_registry": "--remote-registry"}
"""The CLI flag of each setting that has one, so that an error about the setting names it too."""


class TimeNetSettings(BaseSettings):
    """TimeNet settings from arguments, ``TIMENET_*`` variables and an optional ``.env`` file."""

    model_config = SettingsConfigDict(env_prefix="TIMENET_", env_file=".env", extra="ignore")

    local_registry: str = "~/.timenet/registry"
    """The local registry path or ``file://`` URI (``TIMENET_LOCAL_REGISTRY``)."""
    remote_registry: str = "timenet://"
    """The registry after the local registry in the default sources (``TIMENET_REMOTE_REGISTRY``)."""
    auto_build: bool = True
    """Whether default sources include installed connectors (``TIMENET_AUTO_BUILD``)."""
    cache: str = "~/.cache/timenet"
    """The cache directory for raw downloads and control databases (``TIMENET_CACHE``)."""
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
        """Convert paths to strings and reject blank settings.

        Args:
            value: The value to validate.
            info: Validation metadata that includes the field name.

        Returns:
            The value with any path converted to a string.

        Raises:
            TimeFValidationError: If the value is empty or contains only whitespace.
        """
        if isinstance(value, os.PathLike):
            value = os.fspath(value)
        if isinstance(value, str) and not value.strip():
            name = str(info.field_name)
            flag = f" (also as {_FLAGS[name]})" if name in _FLAGS else ""
            raise TimeFValidationError(
                f"TIMENET_{name.upper()} must not be empty{flag}; give it a value, "
                f"or leave it unset for {cls.model_fields[name].default}."
            )
        return value

    @property
    def local_registry_dir(self) -> Path:
        """The local registry directory with ``~`` expanded.

        Raises:
            TimeNetRegistryError: If the local registry path is blank, remote or uses an invalid URI.
        """  # ruff: ignore[docstring-extraneous-exception] (raised by local_registry_path)
        return local_registry_path(self.local_registry)

    @property
    def cache_dir(self) -> Path:
        """The cache directory path with ``~`` expanded."""
        return Path(self.cache).expanduser()

    @property
    def downloads_dir(self) -> Path:
        """The raw download cache directory, with a subdirectory for each dataset ID."""
        return self.cache_dir / "downloads"

    @property
    def control_cache_dir(self) -> Path:
        """Where verified remote control databases are cached by checksum."""
        return self.cache_dir / "control"


def settings(**overrides: Any) -> TimeNetSettings:
    """Load and validate the TimeNet settings.

    Overrides other than ``None`` take precedence over environment variables and defaults.

    Args:
        **overrides: Values by field name, with ``None`` values ignored.

    Returns:
        The validated settings.

    Raises:
        TimeFValidationError: If a value is invalid.
    """
    given = {key: value for key, value in overrides.items() if value is not None}
    try:
        return TimeNetSettings(**given)
    except ValidationError as error:
        raise TimeFValidationError(" ".join(_sentence(item) for item in error.errors())) from error


def _sentence(item: ErrorDetails) -> str:
    """Describe a setting error using its environment variable name.

    Args:
        item: One error from a pydantic ``ValidationError``.

    Returns:
        A sentence describing the invalid setting.
    """
    cause = item.get("ctx", {}).get("error")
    if isinstance(cause, TimeFValidationError):
        return str(cause)
    name = "TIMENET_" + "_".join(str(part) for part in item["loc"]).upper()
    return f"{name} is invalid: {item['msg']}, got {item['input']!r}."


def make_cache_folder(folder: Path) -> None:
    """Create a cache folder with TimeNet's ``CACHEDIR.TAG`` marker.

    Existing folders stay unchanged.

    Args:
        folder: The cache folder to create along with any missing parents.
    """
    try:
        folder.mkdir(parents=True)
    except FileExistsError:
        return
    (folder / CACHE_TAG).write_text("\n".join(_CACHE_TAG_LINES) + "\n")


def is_cache_folder(folder: Path) -> bool:
    """Check a folder for TimeNet's ``CACHEDIR.TAG`` marker.

    The check follows symlinks.

    Args:
        folder: The folder to check.

    Returns:
        ``True`` if the marker's first two lines match TimeNet's cache tag.
    """
    try:
        with (folder / CACHE_TAG).open(encoding="utf-8", errors="replace") as tag:
            head = [tag.readline().rstrip("\n") for _ in range(2)]
    except OSError:
        return False
    return head == list(_CACHE_TAG_LINES[:2])
