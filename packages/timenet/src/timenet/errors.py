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

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from timenet.types.access import Access, AccessRequirement
    from timenet.types.references import DatasetRef


class TimeNetError(Exception):
    """Base class for all TimeNet errors."""


class TimeNetRegistryError(TimeNetError):
    """TimeNet could not load or reach a registry, or the registry could not serve a request."""


class TimeNetDatasetNotFoundError(TimeNetError):
    """A requested dataset id or version is not present in a registry or on disk."""


class TimeNetMissingParentError(TimeNetDatasetNotFoundError):
    """A release that another release needs is absent: a declared parent of a build, or a payload source.

    Attributes:
        missing: The absent release.
        needed_by: The release that needs it.
        payload_source: Whether ``needed_by`` reads files of ``missing`` (a payload source), rather than
            declaring it as a parent to build from.
        searched: Where the release was looked for.
        hint: How the caller can obtain the release, added by the front end, or ``None``.
    """

    def __init__(self, missing: "DatasetRef", needed_by: "DatasetRef", payload_source: bool, searched: str) -> None:
        super().__init__(missing, needed_by, payload_source, searched)
        self.missing = missing
        self.needed_by = needed_by
        self.payload_source = payload_source
        self.searched = searched
        self.hint: str | None = None

    def __str__(self) -> str:
        role = "payload source" if self.payload_source else "parent release"
        message = f"{role} {self.missing} of {self.needed_by} is not in {self.searched}."
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
    """A connector found that an assumption it makes does not hold.

    Connector code raises it, for example when its source data has an unknown code, a file is missing
    from a download, or a parent lacks a value it needs. The message says where, such as the file and
    row. TimeNet's own checks raise their own errors.
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
