"""TimeNet's exception hierarchy.

Every TimeNet-raised error derives from :class:`TimeNetError`. Validation and manifest errors also
derive from :class:`ValueError` so existing ``except ValueError`` handlers keep working. TimeNet raises
the builtin ``FileNotFoundError`` and ``FileExistsError`` directly for user-supplied paths.
"""


class TimeNetError(Exception):
    """Base class for all TimeNet errors."""


class TimeNetRegistryError(TimeNetError):
    """TimeNet could not load or reach a registry, or the registry could not serve a request."""


class TimeNetDatasetNotFoundError(TimeNetError):
    """A requested dataset id or version is not present in a registry or on disk."""


class TimeNetAccessError(TimeNetError):
    """A dataset needs credentialed or restricted access, so TimeNet does not host its data."""


class TimeFValidationError(TimeNetError, ValueError):
    """A dataset or its arrays violate a TimeF invariant (raised while building or writing)."""


class TimeFEditError(TimeFValidationError):
    """An edit would leave a dataset referentially inconsistent (for example, a dangling reference)."""


class TimeNetInvalidCardError(TimeNetError, ValueError):
    """A dataset card YAML is malformed or fails schema validation (raised while building)."""


class TimeNetBuildError(TimeNetError):
    """A build run failed: the connector build, or the environment it needed."""


class TimeFFormatError(TimeNetError):
    """An on-disk TimeF artifact is corrupt or uses an unsupported format version."""


class TimeNetInvalidManifestError(TimeFFormatError, ValueError):
    """A ``manifest.json`` is missing required blocks or fields, or TimeNet cannot parse it."""
