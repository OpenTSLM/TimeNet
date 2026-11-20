"""TimeNet's exception hierarchy, and the warnings it raises.

Every TimeNet-raised error derives from :class:`TimeNetError`. TimeF validation errors also derive
from :class:`ValueError`, so existing ``except ValueError`` handlers keep working. TimeNet raises
the builtin ``FileNotFoundError`` and ``FileExistsError`` directly for user-supplied paths.

Every TimeNet-raised warning derives from :class:`TimeNetWarning`.

Some errors carry facts as attributes and leave the remedy to the front end that called the lower
layer: the CLI names its flags, the SDK names its keywords. A front end sets ``hint`` to one or two
full sentences and raises the error again. ``str()`` adds the hint after the message as its own
sentence.
"""

from enum import Enum, auto
from functools import partial
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from pathlib import Path

    from timenet.types.access import Access, AccessRequirement
    from timenet.types.metadata import DatasetMetadata
    from timenet.types.references import DatasetRef
    from timenet.types.version import Version


class TimeNetError(Exception):
    """Base class for all TimeNet errors.

    String conversion appends ``hint`` to the error message.

    Attributes:
        hint: A suggested way to resolve the error, or ``None``.
    """

    hint: str | None = None

    def __str__(self) -> str:
        message = self._message()
        if self.hint is None:
            return message
        return f"{message} {self.hint}" if message.endswith((".", "?", "!")) else f"{message}. {self.hint}"

    def _message(self) -> str:
        """Return the message without the hint."""
        return super().__str__()


class TimeNetRegistryError(TimeNetError):
    """TimeNet could not load or reach a registry, or the registry could not serve a request."""


class TimeNetUnreachableError(TimeNetRegistryError):
    """A request to a registry or its storage failed on the network."""


class TimeNetNoCatalogError(TimeNetRegistryError):
    """No source of a dataset listing has a catalog."""


class TimeNetDatasetNotFoundError(TimeNetError):
    """A requested dataset id or version is not present in a registry or on disk."""


class TimeNetNotBuiltError(TimeNetDatasetNotFoundError):
    """Only an installed connector provides a release, so it has no manifest yet.

    Build the release to create its manifest, or read the connector's card for metadata.

    Attributes:
        release: The unbuilt release.
        card: The connector's card of the release.
    """

    def __init__(self, release: "DatasetRef", card: "DatasetMetadata") -> None:
        super().__init__(release, card)
        self.release = release
        self.card = card

    def _message(self) -> str:
        return f"{self.release} is not built yet: only an installed connector provides it."


class TimeNetMissingParentError(TimeNetDatasetNotFoundError):
    """A required parent release or payload source is missing.

    Attributes:
        missing: The absent release.
        needed_by: The release that needs it.
        payload_source: ``True`` if ``needed_by`` reads files of ``missing``, ``False`` for a declared build parent.
        searched: The search location.
    """

    def __init__(self, missing: "DatasetRef", needed_by: "DatasetRef", searched: str, *, payload_source: bool) -> None:
        super().__init__(missing, needed_by, searched, payload_source)
        self.missing = missing
        self.needed_by = needed_by
        self.payload_source = payload_source
        self.searched = searched

    def __reduce__(self) -> tuple[object, ...]:
        # Pickle passes the arguments by position, so the keyword-only flag travels in a partial.
        rebuild = partial(type(self), payload_source=self.payload_source)
        return rebuild, (self.missing, self.needed_by, self.searched), self.__dict__

    def _message(self) -> str:
        role = "payload source" if self.payload_source else "parent release"
        return f"{role} {self.missing} of {self.needed_by} is not in {self.searched}."


class ReleaseDifference(Enum):
    """What differs between two manifests of the same release."""

    FILES = auto()
    """The file records differ."""
    CARD = auto()
    """The file records are the same, and the rest of the manifest differs."""


class TimeNetVersionConflictError(TimeNetError):
    """A registry already holds a release with other files or another card than the copy.

    A changed release needs a new version.

    Attributes:
        release: The release that differs.
        registry: The destination registry, as text.
        registry_location: The ``location`` of the destination registry.
        source: The source registry, as text.
        difference: What differs.
        needed_by: The requested release that needs this payload source, or ``None`` if ``release`` is the request.
    """

    def __init__(  # ruff: ignore[too-many-arguments, too-many-positional-arguments] - a pickled error passes them in order
        self,
        release: "DatasetRef",
        registry: str,
        registry_location: str,
        source: str,
        difference: ReleaseDifference,
        needed_by: "DatasetRef | None" = None,
    ) -> None:
        super().__init__(release, registry, registry_location, source, difference, needed_by)
        self.release = release
        self.registry = registry
        self.registry_location = registry_location
        self.source = source
        self.needed_by = needed_by
        self.difference = difference

    def _message(self) -> str:
        held = str(self.release) if self.needed_by is None else f"{self.release}, a payload source of {self.needed_by},"
        other = "other files" if self.difference is ReleaseDifference.FILES else "another card"
        return f"{self.registry} already holds {held} with {other} than {self.source}."


class TimeNetAccessError(TimeNetError):
    """A dataset needs credentialed or restricted access, so only the user's own registry may hold its data.

    Attributes:
        dataset: The release.
        access: Its effective access level: the strictest level of its card and of the cards in its
            lineage. See :func:`~timenet.types.access.effective_access`.
        requirements: Each card, of the release itself or of an ancestor, whose access is not open,
            with its access level and its ``access_url``. A user needs access at each of them. See
            :func:`~timenet.types.access.access_requirements`.
    """

    def __init__(self, dataset: "DatasetRef", access: "Access", requirements: "tuple[AccessRequirement, ...]") -> None:
        super().__init__(dataset, access, requirements)
        self.dataset = dataset
        self.access = access
        self.requirements = requirements

    def _message(self) -> str:
        message = f"{self.dataset} is {self.access.value}."
        if self.requirements:
            pages = [f"{item.access_url} for {item.release}" for item in self.requirements]
            listed = pages[0] if len(pages) == 1 else f"{', at '.join(pages[:-1])} and at {pages[-1]}"
            message = f"{message} Get access at {listed}."
        return message


class TimeNetConnectorError(TimeNetError):
    """A connector found that one of its assumptions does not hold.

    Connector code raises this error. For example, the source data has an unknown code, a download
    lacks a file, or a parent lacks a value that the connector needs. The message says where, such
    as the file and the row. TimeNet's own checks raise their own errors.
    """


class TimeNetDownloadError(TimeNetConnectorError):
    """TimeNet could not fetch a dataset's source data."""


class TimeFValidationError(TimeNetError, ValueError):
    """A dataset or its arrays violate a TimeF invariant (raised while building or writing)."""


class TimeFNotFoundError(TimeFValidationError, LookupError):
    """An ID, parent alias, or parent release names nothing in the dataset, or the stored object has another kind."""


class TimeFEditError(TimeFValidationError):
    """An edit would leave a dataset referentially inconsistent (for example, a dangling reference)."""


class TimeFNullValuesError(TimeFValidationError):
    """Values contain nulls where the requested form, such as a NumPy array, cannot hold them."""


class TimeNetVersionUnavailableError(TimeNetDatasetNotFoundError, TimeFValidationError):
    """The installed connector cannot build the requested dataset version.

    Attributes:
        release: The requested release.
        declared: The version that the connector builds.
    """

    def __init__(self, release: "DatasetRef", declared: "Version") -> None:
        super().__init__(release, declared)
        self.release = release
        self.declared = declared

    def _message(self) -> str:
        return (
            f"connector for {self.release.dataset_id!r} builds version {self.declared}, "
            f"not the requested {self.release.version}."
        )


class TimeNetInvalidCardError(TimeNetError, ValueError):
    """A dataset card YAML is malformed or fails schema validation (raised while building)."""


class TimeNetBuildError(TimeNetError):
    """A build run failed: the connector build, or the environment it needed."""


class TimeNetIsolationError(TimeNetBuildError):
    """An isolated build cannot make its environment."""


class TimeNetWarning(UserWarning):
    """Base class for all TimeNet warnings."""


class ConnectorCardWarning(TimeNetWarning):
    """A dataset listing skips a dataset with an unreadable connector card."""


class CatalogWarning(TimeNetWarning):
    """A dataset listing skips a source without a catalog."""


class SpanOutsideWindowWarning(TimeNetWarning):
    """An annotation's span falls outside the window of the series it names, and is kept."""


class TimeFFormatError(TimeNetError):
    """An on-disk TimeF artifact is corrupt or uses an unsupported format version."""


class TimeNetFileError(TimeFFormatError):
    """A release file is missing or differs from its manifest record.

    A local copy is checked by file size. A transfer is checked by size and checksum.

    Attributes:
        release: The release that the file belongs to.
        owner: The release, payload source, or transfer that the file belongs to, as text.
        path: The relative file path, using the first missing or resized file in manifest order for local checks.
        expected: What the manifest records, for example ``"1,024 bytes"`` or ``"sha256:<hex>"``.
        actual: What the file holds, in the same form, or ``None`` if the file is missing.
    """

    def __init__(self, release: "DatasetRef", owner: str, path: str, expected: str, actual: str | None) -> None:
        super().__init__(release, owner, path, expected, actual)
        self.release = release
        self.owner = owner
        self.path = path
        self.expected = expected
        self.actual = actual

    def _message(self) -> str:
        found = "is missing" if self.actual is None else f"has {self.actual}"
        return f"{self.owner}: file {self.path!r} {found}, expected {self.expected}."


class TimeNetManifestError(TimeFFormatError):
    """A stored manifest is not a valid manifest.

    Attributes:
        where: The file of the manifest, or a description of where a registry keeps it.
        detail: The first problem that the parser found.
        path: The file of the manifest, or ``None`` for a manifest that a registry served.
    """

    def __init__(self, where: str, detail: str, path: "Path | None" = None) -> None:
        super().__init__(where, detail, path)
        self.where = where
        self.detail = detail
        self.path = path

    def _message(self) -> str:
        return f"{self.where} is not a valid manifest: {self.detail}"


class FormatVersionSubject(Enum):
    """A versioned part of a stored TimeF release. The value names it in error messages."""

    TIMEF_FORMAT = "TimeF format"
    """The format of ``manifest.json``, given by its ``timef_format_version``."""
    CONTROL_SCHEMA = "control schema"
    """The schema of ``control.duckdb``, given by the ``schema_version`` key of ``control_metadata``."""


class TimeFFormatVersionError(TimeFFormatError):
    """A stored TimeF artifact has a format version that this TimeNet does not read.

    Attributes:
        subject: The versioned part of the release.
        where: The file of the artifact, or a description of where a registry keeps it.
        found: The version that the artifact records, older than ``oldest_readable`` or newer than
            ``newest_readable``.
        oldest_readable: The oldest version that this TimeNet reads.
        newest_readable: The newest version that this TimeNet reads.
        path: The file of the artifact, or ``None`` for an artifact that a registry served.
    """

    def __init__(  # ruff: ignore[too-many-arguments, too-many-positional-arguments] - a pickled error passes them in order
        self,
        subject: FormatVersionSubject,
        where: str,
        found: int,
        oldest_readable: int,
        newest_readable: int,
        path: "Path | None" = None,
    ) -> None:
        super().__init__(subject, where, found, oldest_readable, newest_readable, path)
        self.subject = subject
        self.where = where
        self.found = found
        self.oldest_readable = oldest_readable
        self.newest_readable = newest_readable
        self.path = path

    def _message(self) -> str:
        bound = (
            f"the newest that this timenet reads is {self.newest_readable}"
            if self.found > self.newest_readable
            else f"the oldest that this timenet reads is {self.oldest_readable}"
        )
        return f"{self.where} uses {self.subject.value} version {self.found}, but {bound}."


class TimeNetOriginError(TimeFFormatError):
    """The provenance record of a release in the local registry is not valid.

    Attributes:
        release: The release of the record.
        path: The file of the record.
        detail: The first problem that the parser found.
    """

    def __init__(self, release: "DatasetRef", path: str, detail: str) -> None:
        super().__init__(release, path, detail)
        self.release = release
        self.path = path
        self.detail = detail

    def _message(self) -> str:
        return f"{self.path} is not a valid provenance record: {self.detail}."


class TimeFReaderError(TimeNetError):
    """A reader could not complete a read, e.g. its connection was closed while the read ran."""
