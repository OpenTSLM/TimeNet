"""TimeNet's exception hierarchy, and the warnings it raises.

Every TimeNet-raised error derives from :class:`TimeNetError`. TimeF validation errors also derive
from :class:`ValueError`, so existing ``except ValueError`` handlers keep working. TimeNet raises
the builtin ``FileNotFoundError`` and ``FileExistsError`` directly for user-supplied paths.

Every TimeNet-raised warning derives from :class:`TimeNetWarning`.

Some errors carry facts as attributes and leave the remedy to the front end that called the lower
layer: the CLI names its flags, the SDK names its keywords. A front end sets ``hint`` to a full
sentence and raises the error again. The message ends with a period, and the hint follows as its own
sentence.
"""

from functools import partial
from typing import TYPE_CHECKING, Literal


if TYPE_CHECKING:
    from timenet.types.access import Access, AccessRequirement
    from timenet.types.references import DatasetRef
    from timenet.types.version import Version


class TimeNetError(Exception):
    """Base class for all TimeNet errors."""


class TimeNetRegistryError(TimeNetError):
    """TimeNet could not load or reach a registry, or the registry could not serve a request."""


class TimeNetDatasetNotFoundError(TimeNetError):
    """A requested dataset id or version is not present in a registry or on disk."""


class TimeNetMissingParentError(TimeNetDatasetNotFoundError):
    """A required parent release or payload source is missing.

    Attributes:
        missing: The absent release.
        needed_by: The release that needs it.
        payload_source: ``True`` if ``needed_by`` reads files of ``missing``, ``False`` for a declared build parent.
        searched: The search location.
        hint: Advice on obtaining the release, or ``None``.
    """

    def __init__(self, missing: "DatasetRef", needed_by: "DatasetRef", searched: str, *, payload_source: bool) -> None:
        super().__init__(missing, needed_by, searched, payload_source)
        self.missing = missing
        self.needed_by = needed_by
        self.payload_source = payload_source
        self.searched = searched
        self.hint: str | None = None

    def __reduce__(self) -> tuple[object, ...]:
        # Pickle passes the arguments by position, so the keyword-only flag travels in a partial.
        rebuild = partial(type(self), payload_source=self.payload_source)
        return rebuild, (self.missing, self.needed_by, self.searched), self.__dict__

    def __str__(self) -> str:
        role = "payload source" if self.payload_source else "parent release"
        message = f"{role} {self.missing} of {self.needed_by} is not in {self.searched}."
        return message if self.hint is None else f"{message} {self.hint}"


class TimeNetVersionConflictError(TimeNetError):
    """A registry already holds a release with other files or another card than the copy.

    A changed release needs a new version.

    Attributes:
        release: The release that differs.
        registry: The destination registry, as text.
        source: The source registry, as text.
        needed_by: The requested release that needs this payload source, or ``None`` if ``release`` is the request.
        difference: ``"files"`` if the file records differ, else ``"card"`` for the rest of the manifest.
        hint: Advice to append to the error message, or ``None``.
    """

    def __init__(
        self,
        release: "DatasetRef",
        registry: str,
        source: str,
        needed_by: "DatasetRef | None" = None,
        difference: Literal["files", "card"] = "files",
    ) -> None:
        super().__init__(release, registry, source, needed_by, difference)
        self.release = release
        self.registry = registry
        self.source = source
        self.needed_by = needed_by
        self.difference = difference
        self.hint: str | None = None

    def __str__(self) -> str:
        held = str(self.release) if self.needed_by is None else f"{self.release}, a payload source of {self.needed_by},"
        other = "other files" if self.difference == "files" else "another card"
        message = (
            f"{self.registry} already holds {held} with {other} than {self.source}. "
            "A published version cannot change, so a changed release needs a new version."
        )
        return message if self.hint is None else f"{message} {self.hint}"


class TimeNetAccessError(TimeNetError):
    """A dataset needs credentialed or restricted access, so only the user's own registry may hold its data.

    Attributes:
        dataset: The release.
        access: Its effective access level: the strictest level of its card and of the cards in its
            lineage. See :func:`~timenet.types.access.effective_access`.
        requirements: Each card, of the release itself or of an ancestor, whose access is not open,
            with its access level and its ``access_url``. A user needs access at each of them. See
            :func:`~timenet.types.access.access_requirements`.
        hint: What the caller can do instead, added by the front end, or ``None``.
    """

    def __init__(self, dataset: "DatasetRef", access: "Access", requirements: "tuple[AccessRequirement, ...]") -> None:
        super().__init__(dataset, access, requirements)
        self.dataset = dataset
        self.access = access
        self.requirements = requirements
        self.hint: str | None = None

    def __str__(self) -> str:
        message = f"{self.dataset} is {self.access.value}."
        if self.requirements:
            pages = [f"{item.access_url} for {item.release}" for item in self.requirements]
            listed = pages[0] if len(pages) == 1 else f"{', at '.join(pages[:-1])} and at {pages[-1]}"
            message = f"{message} Get access at {listed}."
        return message if self.hint is None else f"{message} {self.hint}"


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
        hint: Suggested next steps to append to the error message, or ``None``.
    """

    def __init__(self, release: "DatasetRef", declared: "Version") -> None:
        super().__init__(release, declared)
        self.release = release
        self.declared = declared
        self.hint: str | None = None

    def __str__(self) -> str:
        message = (
            f"connector for {self.release.dataset_id!r} builds version {self.declared}, "
            f"not the requested {self.release.version}. "
            f"Ask for version {self.declared}, or leave out the version."
        )
        return message if self.hint is None else f"{message} {self.hint}"


class TimeNetInvalidCardError(TimeNetError, ValueError):
    """A dataset card YAML is malformed or fails schema validation (raised while building)."""


class TimeNetBuildError(TimeNetError):
    """A build run failed: the connector build, or the environment it needed."""


class TimeNetWarning(UserWarning):
    """Base class for all TimeNet warnings."""


class SpanOutsideWindowWarning(TimeNetWarning):
    """An annotation's span falls outside the window of the series it names, and is kept."""


class TimeFFormatError(TimeNetError):
    """An on-disk TimeF artifact is corrupt or uses an unsupported format version."""


class TimeNetFileError(TimeFFormatError):
    """A file of a release is missing, or it does not match its manifest record.

    A local copy is checked by file size. A transfer is checked by size and checksum.

    Attributes:
        release: The release that the file belongs to.
        owner: The release, payload source, or transfer that the file belongs to, as text.
        path: The file, relative to the release directory. For a local copy, this is the first
            missing or resized file in manifest order.
        expected: What the manifest records, for example ``"1,024 bytes"`` or ``"sha256:<hex>"``.
        actual: What the file holds, in the same form, or ``None`` if the file is missing.
        hint: How the caller can get an intact copy, added by the front end, or ``None``.
    """

    def __init__(self, release: "DatasetRef", owner: str, path: str, expected: str, actual: str | None) -> None:
        super().__init__(release, owner, path, expected, actual)
        self.release = release
        self.owner = owner
        self.path = path
        self.expected = expected
        self.actual = actual
        self.hint: str | None = None

    def __str__(self) -> str:
        found = "is missing" if self.actual is None else f"has {self.actual}"
        message = f"{self.owner}: file {self.path!r} {found}, expected {self.expected}."
        return message if self.hint is None else f"{message} {self.hint}"


class TimeFReaderError(TimeNetError):
    """A reader could not complete a read, e.g. its connection was closed while the read ran."""
