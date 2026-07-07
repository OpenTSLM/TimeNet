"""TimeNet's exception hierarchy.

Every TimeNet-raised error derives from :class:`TimeNetError`. Validation and manifest errors also
derive from :class:`ValueError` so existing ``except ValueError`` handlers keep working. Builtin
``FileNotFoundError`` / ``FileExistsError`` are still raised directly for user-supplied paths.
"""


class TimeNetError(Exception):
    """Base class for all TimeNet errors."""


class RegistryError(TimeNetError):
    """A registry could not be loaded, reached, or served a request."""


class DatasetNotFoundError(TimeNetError):
    """A requested dataset id or version is not present in a registry or on disk."""


class TimeFValidationError(TimeNetError, ValueError):
    """A dataset or its arrays violate a TimeF invariant (raised while building or writing)."""


class TimeFEditError(TimeFValidationError):
    """An edit would leave a dataset referentially inconsistent (e.g. a dangling reference)."""


class TimeFFormatError(TimeNetError):
    """An on-disk TimeF artifact is corrupt or uses an unsupported format version."""


class InvalidManifestError(TimeFFormatError, ValueError):
    """A ``manifest.json`` is missing required blocks/fields or cannot be parsed."""
