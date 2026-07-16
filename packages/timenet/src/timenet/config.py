"""Runtime configuration, resolved with pydantic-settings.

All local state lives under a single home directory (default ``~/.cache/timenet``), mirroring
HuggingFace's ``HF_HOME`` -> ``HF_DATASETS_CACHE`` / ``HF_HUB_CACHE`` hierarchy. Setting ``TIMENET_HOME``
relocates everything; the per-area env vars override just their own path. Precedence for any value is
**explicit argument (e.g. a CLI flag) > environment variable > default**. This model is the single place
to add future configuration.
"""

from pathlib import Path
from typing import Any

from pydantic_settings import BaseSettings, SettingsConfigDict


class TimeNetSettings(BaseSettings):
    """TimeNet's settings. Reads ``TIMENET_*`` environment variables (and an optional ``.env``)."""

    model_config = SettingsConfigDict(env_prefix="TIMENET_", env_file=".env", extra="ignore")

    home: Path = Path("~/.cache/timenet")
    storage: Path | None = None
    cache: Path | None = None
    registry: str | None = None

    @property
    def home_dir(self) -> Path:
        """The resolved home directory (analog of ``HF_HOME``)."""
        return self.home.expanduser()

    @property
    def storage_dir(self) -> Path:
        """Where loaded/downloaded datasets are cached (analog of ``HF_DATASETS_CACHE``)."""
        return (self.storage if self.storage is not None else self.home_dir / "storage").expanduser()

    @property
    def cache_dir(self) -> Path:
        """Where curation raw sources and Hub downloads are cached (analog of ``HF_HUB_CACHE``)."""
        return (self.cache if self.cache is not None else self.home_dir / "cache").expanduser()

    @property
    def registry_path(self) -> Path:
        """The default local registry directory (``$TIMENET_REGISTRY`` is the selector, not this)."""
        return self.home_dir / "registry"


def settings(**overrides: Any) -> TimeNetSettings:
    """Build settings, letting explicit non-``None`` overrides win over env and defaults.

    ``None`` overrides are dropped so a missing CLI flag falls through to the environment, then the
    default.

    Args:
        **overrides: Field overrides (e.g. ``storage=...``, ``registry=...``).

    Returns:
        The resolved :class:`TimeNetSettings`.
    """
    return TimeNetSettings(**{key: value for key, value in overrides.items() if value is not None})
