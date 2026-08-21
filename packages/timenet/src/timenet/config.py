"""Runtime configuration, resolved with pydantic-settings.

All local state lives under one home directory (default ``~/.cache/timenet``). This mirrors
HuggingFace's ``HF_HOME`` -> ``HF_DATASETS_CACHE`` / ``HF_HUB_CACHE`` hierarchy. Set ``TIMENET_HOME``
to relocate everything. Each per-area env var overrides only its own path. For any value the precedence
is explicit argument (for example a CLI flag), then environment variable, then default. Add future
configuration to this model.
"""

from pathlib import Path
from typing import Any, Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class TimeNetSettings(BaseSettings):
    """TimeNet's settings. Reads ``TIMENET_*`` environment variables (and an optional ``.env``)."""

    model_config = SettingsConfigDict(env_prefix="TIMENET_", env_file=".env", extra="ignore")

    home: Path = Path("~/.cache/timenet")
    storage: Path | None = None
    cache: Path | None = None
    registry: str | None = None
    token: str | None = None
    """Bearer token for a remote registry (``TIMENET_TOKEN``); ``None`` is anonymous."""
    isolation: Literal["on", "off"] = "on"
    """Whether a build runs in an environment built from the connector's requirements.

    ``"off"`` runs it in the current interpreter. The isolated child sets this to ``"off"`` in its own
    environment, which is what stops it re-execing forever.
    """

    @property
    def home_dir(self) -> Path:
        """The resolved home directory (analog of ``HF_HOME``)."""
        return self.home.expanduser()

    @property
    def storage_dir(self) -> Path:
        """Where the client caches loaded or downloaded datasets (analog of ``HF_DATASETS_CACHE``)."""
        return (self.storage if self.storage is not None else self.home_dir / "storage").expanduser()

    @property
    def cache_dir(self) -> Path:
        """Where the client caches raw build sources and Hub downloads (analog of ``HF_HUB_CACHE``)."""
        return (self.cache if self.cache is not None else self.home_dir / "cache").expanduser()

    @property
    def registry_path(self) -> Path:
        """The default local registry directory (``$TIMENET_REGISTRY`` is the selector, not this)."""
        return self.home_dir / "registry"


def settings(**overrides: Any) -> TimeNetSettings:
    """Build settings so explicit non-``None`` overrides win over env vars and defaults.

    This function drops a ``None`` override. A missing CLI flag then falls through to the environment,
    then the default.

    Args:
        **overrides: Field overrides (for example ``storage=...``, ``registry=...``).

    Returns:
        The resolved :class:`TimeNetSettings`.
    """
    return TimeNetSettings(**{key: value for key, value in overrides.items() if value is not None})
