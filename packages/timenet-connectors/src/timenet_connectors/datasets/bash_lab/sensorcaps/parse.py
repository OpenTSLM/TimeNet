"""Read SensorCaps JSONL rows into their messages, descriptive fields, and validated sensor arrays."""

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
import json
from typing import Annotated

from pydantic import Field, FiniteFloat, JsonValue, Strict, TypeAdapter, ValidationError

from timenet.errors import TimeFValidationError
from timenet_connectors.bases.huggingface import HubFileRef


EXPORTED_SAMPLES = 60
"""Readings per sensor in every row of the pinned v2 release."""
FEATURE_HEADINGS = tuple(
    f"{sensor} {statistic}"
    for statistic in ("moving average", "FFT", "minimum", "maximum", "median", "variance")
    for sensor in ("Gyroscope", "Accelerometer")
)
"""The derived features every v2 user message carries after its sensor arrays, in source order."""
_FIELDS = frozenset({"Summary", "Accelerometer", "Gyroscope", "Location"})
_ROLES = ("system", "user", "assistant")
_XYZValues = Annotated[list[Annotated[FiniteFloat, Strict()]], Field(min_length=3, max_length=3)]
_SENSOR_VALUES = TypeAdapter(Annotated[list[_XYZValues], Field(min_length=1)])


@dataclass(frozen=True)
class RowRef:
    """Where one source row starts in its local file."""

    file: HubFileRef
    index: int
    offset: int

    @property
    def key(self) -> tuple[str, int]:
        """Return the row's partition and index within it."""
        return self.file.part, self.index

    @property
    def context(self) -> str:
        """Return the row's file, partition, and index, for error messages."""
        return f"{self.file.path} ({self.file.part}, row {self.index})"


@dataclass(frozen=True)
class ParsedRow:
    """One source row's messages, descriptive fields, and sensor arrays."""

    system: str
    caption: str
    summary: str
    location: str | None
    features: Mapping[str, str]
    """Each derived feature's original text, keyed by its heading in :data:`FEATURE_HEADINGS`."""
    values: tuple[list[list[float]], list[list[float]]]
    """The accelerometer's and then the gyroscope's readings."""


def _sensor_values(text: str, name: str) -> list[list[float]]:
    """Decode and validate one time-major numeric xyz array.

    Returns:
        Validated numeric triples in source order.

    Raises:
        TimeFValidationError: If the decoded values are empty, ragged, or nonnumeric.
    """
    try:
        # Finite published values do not imply complete raw measurements: PAMAP2's
        # missing readings appear to have been zero-filled before upstream resampling.
        return _SENSOR_VALUES.validate_json(text)
    except ValidationError as exc:
        raise TimeFValidationError(f"{name}: {exc}") from exc


def _user_sections(user: str) -> tuple[dict[str, str], dict[str, str]]:
    """Split the user message into its structured fields and its derived features.

    Returns:
        Named source fields, and each derived feature's original text keyed by its heading.

    Raises:
        TimeFValidationError: If a section is repeated, a derived feature is missing, or a line
            belongs to no known section.
    """
    sections: dict[str, str] = {}
    feature_lines: dict[str, list[str]] = {}
    current: list[str] | None = None
    for line in user.splitlines(keepends=True):
        heading, separator, value = line.partition(":")
        if separator and heading in _FIELDS:
            if heading in sections:
                raise TimeFValidationError(f"repeated {heading} section")
            sections[heading] = value.strip()
            current = None
        elif separator and heading in FEATURE_HEADINGS:
            if heading in feature_lines:
                raise TimeFValidationError(f"repeated {heading} section")
            current = feature_lines[heading] = [value]
        elif current is not None and (not line.strip() or line[0].isspace()):
            # NumPy prints a long statistic over indented continuation lines.
            current.append(line)
        else:
            raise TimeFValidationError(f"unexpected line in the user message: {line[:40]!r}")
    missing = [heading for heading in FEATURE_HEADINGS if heading not in feature_lines]
    if missing:
        raise TimeFValidationError(f"missing derived features: {missing}")
    # Keep each statistic's original text, including complex numbers and NumPy ellipses.
    features = {heading: "".join(feature_lines[heading]).strip() for heading in FEATURE_HEADINGS}
    return sections, features


def _message_contents(row: JsonValue) -> tuple[str, str, str]:
    """Validate and extract the ordered system, user, and assistant messages.

    Returns:
        The three nonempty message contents in source order.

    Raises:
        TimeFValidationError: If messages are missing, empty, or out of order.
    """
    messages = row.get("messages") if isinstance(row, dict) else None
    if not isinstance(messages, list) or len(messages) != len(_ROLES):
        raise TimeFValidationError("expected system, user, and assistant messages")
    contents: list[str] = []
    for message, role in zip(messages, _ROLES, strict=True):
        content = message.get("content") if isinstance(message, dict) else None
        if (
            not isinstance(message, dict)
            or message.get("role") != role
            or not isinstance(content, str)
            or not content.strip()
        ):
            raise TimeFValidationError(f"expected a nonempty {role} message")
        contents.append(content)
    return contents[0], contents[1], contents[2]


def _section_values(sections: dict[str, str]) -> tuple[list[list[float]], list[list[float]]]:
    """Validate structured fields and decode the paired sensor arrays.

    Returns:
        Accelerometer and gyroscope readings.

    Raises:
        TimeFValidationError: If fields are missing or invalid for the pinned release.
    """
    if not sections.get("Summary"):
        raise TimeFValidationError("missing or empty Summary section")
    for heading in ("Accelerometer", "Gyroscope"):
        if heading not in sections:
            raise TimeFValidationError(f"missing {heading} section")
    accelerometer = _sensor_values(sections["Accelerometer"], "Accelerometer")
    gyroscope = _sensor_values(sections["Gyroscope"], "Gyroscope")
    if len(accelerometer) != len(gyroscope):
        raise TimeFValidationError("Accelerometer and Gyroscope lengths differ")
    if len(accelerometer) != EXPORTED_SAMPLES:
        raise TimeFValidationError(f"expected {EXPORTED_SAMPLES} readings per sensor in the pinned v2 release")
    if "Location" in sections and not sections["Location"]:
        raise TimeFValidationError("empty Location section")
    return accelerometer, gyroscope


def parse_row(raw: bytes, context: str) -> ParsedRow:
    """Parse a source row, adding file and row context to decoding failures.

    Returns:
        Captioning instructions, targets, descriptive fields, and validated sensor arrays.

    Raises:
        TimeFValidationError: If the JSON, messages, or sensor sections are malformed.
    """
    try:
        row: JsonValue = json.loads(raw)
        system, user, caption = _message_contents(row)
        sections, features = _user_sections(user)
        values = _section_values(sections)
        return ParsedRow(
            system,
            caption,
            sections["Summary"],
            sections.get("Location"),
            features,
            values,
        )
    except (ValueError, UnicodeError, OverflowError, RecursionError) as exc:
        # Source text is untrusted; retain the narrow decoding failure and identify its exact row.
        raise TimeFValidationError(f"{context}: {exc}") from exc


def iter_rows(file: HubFileRef) -> Iterator[tuple[RowRef, ParsedRow]]:
    """Read a JSONL file one row at a time, retaining seekable binary offsets.

    Yields:
        Each lightweight row reference and its transient decoded contents.
    """
    with file.path.open("rb") as handle:
        index = 0
        while True:
            offset = handle.tell()
            raw = handle.readline()
            if not raw:
                return
            ref = RowRef(file, index, offset)
            yield ref, parse_row(raw, ref.context)
            index += 1
