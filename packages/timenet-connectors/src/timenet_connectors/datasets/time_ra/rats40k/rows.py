"""Read the published RATS40K JSON: its rows, their observation format, the walk, and the reread.

Each file is one JSON document, not JSON Lines: ``{"TSAD_train": {key: row, ...}, "TSAD_test": {...}}``.
Walk it one entry at a time and keep each entry's UTF-8 byte range, so a signal loader rereads just its
row instead of holding the decoded corpus in memory.
"""

from collections.abc import Iterator
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Annotated, Any, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    FiniteFloat,
    InstanceOf,
    Strict,
    StrictInt,
    StrictStr,
    StringConstraints,
    TypeAdapter,
    ValidationInfo,
    field_validator,
    model_validator,
)

from timenet.errors import TimeFFormatError, TimeFValidationError
from timenet.types import Split
from timenet_connectors.bases.huggingface import HubFileRef
from timenet_connectors.datasets.time_ra.rats40k.release import SourceSplit, Variant
from timenet_connectors.sources.json_stream import JsonCursor, read_entry


# Strict types: a JSON true or "1" must not pass as a number, and a blank string not as text.
Text = Annotated[StrictStr, StringConstraints(pattern=r"\S")]
Number = Annotated[FiniteFloat, Strict()]
Numbers = Annotated[list[Number], Field(min_length=1)]

# Named, nonempty, finite channels in source order. Their equal length is checked in parse_channels.
_CHANNELS = TypeAdapter(dict[Text, Numbers])


@dataclass(frozen=True)
class RowRef:
    """Where one entry lives: its file, variant, split and key, and its byte range."""

    path: Path
    variant: Variant
    split: Split
    key: str
    offset: int
    length: int

    @property
    def context(self) -> str:
        """Return where the row is, for error messages: its file, split and key."""
        return f"{self.path} ({self.split}, key {self.key})"


class Row(BaseModel):
    """A row's published fields, validated, with its observation's channel lengths but not its values.

    Each field keeps the published name as its alias, so a validation error names the source field.
    The row also carries where it was read from, which validation takes as context, since the row's
    variant decides the observation's format. Use :meth:`parse` to build one.
    """

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    ref: InstanceOf[RowRef] = Field(exclude=True)
    """Where the row was read from. It comes from the validation context, never from the JSON."""
    domain: Text = Field(alias="Source")
    """The domain and data source, such as ``Healthcare-ECG``."""
    figure_path: Text = Field(alias="FigurePath")
    thought: Text = Field(alias="Thought")
    action: Text = Field(alias="Action")
    """The category name. It may lie outside its variant's vocabulary, as published."""
    action_id: Annotated[StrictInt, Field(ge=0)] = Field(alias="ActionID")
    """The category code, as published. It doesn't always follow the prompt's numbering."""
    # Label sometimes disagrees with Action; both are kept as published.
    label: Literal["Normal", "Anomaly"] = Field(alias="Label")
    rank: list[Text] | None = Field(default=None, alias="Rank")
    """Test rows only: the GPT-4 judge's ordering of the labelling models, as parsed model text."""
    channel_lengths: dict[str, int] = Field(alias="Observation")
    """Each channel's name and length, in source order. The values stay in the file."""

    @model_validator(mode="before")
    @classmethod
    def _add_ref(cls, data: Any, info: ValidationInfo) -> Any:
        if not isinstance(data, dict):
            return data
        # The JSON must not supply the reference, so a published "ref" key is rejected, not used.
        if "ref" in data:
            raise TimeFValidationError("a published row has no 'ref' key")
        return {**data, "ref": _context_ref(info)}

    @field_validator("channel_lengths", mode="before")
    @classmethod
    def _channel_lengths(cls, observation: Any, info: ValidationInfo) -> dict[str, int]:
        channels = parse_channels(observation, _context_ref(info).variant)
        return {name: len(values) for name, values in channels.items()}

    @classmethod
    def parse(cls, value: Any, ref: RowRef) -> Self:
        """Validate a published row against its variant, and attach where it was read from.

        Returns:
            The row's fields, its channels' lengths, and its reference.

        Raises:
            TimeFValidationError: If the row is malformed, with file, split and key.
        """
        try:
            if not isinstance(value, dict):
                raise TimeFValidationError("expected a row object")
            return cls.model_validate(value, context={"ref": ref})
        # A pydantic ValidationError is a ValueError, so it gains the same context.
        except (ValueError, OverflowError, RecursionError) as exc:
            raise TimeFValidationError(f"{ref.context}: {exc}") from exc


def _context_ref(info: ValidationInfo) -> RowRef:
    ref = (info.context or {}).get("ref")
    if not isinstance(ref, RowRef):
        raise TimeFValidationError("a row is validated with its RowRef as context")
    return ref


def parse_channels(observation: Any, variant: Variant) -> dict[str, list[float]]:
    """Decode an observation: a list of numbers (uni), or a string of named arrays (multi).

    Returns:
        Nonempty, equally long channels of finite numbers, in source order.

    Raises:
        TimeFValidationError: If the observation violates its variant's format.
    """
    named = {"series": observation} if variant is Variant.UNI else _named_channels(observation)
    channels = _CHANNELS.validate_python(named)
    if not channels:
        raise TimeFValidationError("Observation has no channels")
    if len({len(values) for values in channels.values()}) != 1:
        raise TimeFValidationError("channel lengths differ")
    return channels


def _named_channels(observation: Any) -> dict[str, Any]:
    """Parse a multivariate observation, ``"name": [values]; ...``, allowing ';' inside quoted names.

    Returns:
        The decoded arrays by name, in source order.

    Raises:
        TimeFValidationError: If names or separators violate the source format.
    """
    if not isinstance(observation, str):
        raise TimeFValidationError("multivariate Observation must be a named-channel string")
    decoder = json.JSONDecoder()
    channels: dict[str, Any] = {}
    remaining = observation.strip()
    while remaining:
        name, end = decoder.raw_decode(remaining)
        remaining = remaining[end:].lstrip()
        if not isinstance(name, str) or not name.strip() or not remaining.startswith(":"):
            raise TimeFValidationError("expected a nonempty channel name followed by ':'")
        if name in channels:
            raise TimeFValidationError(f"duplicate channel {name!r}")
        remaining = remaining[1:].lstrip()
        values, end = decoder.raw_decode(remaining)
        channels[name] = values
        remaining = remaining[end:].strip()
        if remaining:
            if not remaining.startswith(";") or not remaining[1:].strip():
                raise TimeFValidationError("expected ';' between channel arrays")
            remaining = remaining[1:].lstrip()
    return channels


def iter_entries(path: Path, variant: Variant) -> Iterator[tuple[RowRef, Any]]:
    """Walk the split/key objects without retaining decoded rows.

    Yields:
        Each entry's reference and its decoded, unvalidated value, in file order.

    Raises:
        TimeFValidationError: If the document or split structure is malformed.
    """
    context = str(path)
    try:
        with JsonCursor.open(path) as cursor:
            splits: set[SourceSplit] = set()
            for name in cursor.object_keys():
                if name not in set(SourceSplit):
                    raise TimeFValidationError(f"unknown split {name!r}")
                split = SourceSplit(name)
                splits.add(split)
                context = f"{path} ({split})"
                for key in cursor.object_keys():
                    context = f"{path} ({split}, key {key})"
                    if not key.isascii() or not key.isdecimal():
                        raise TimeFValidationError(f"{split}: expected a decimal row key, got {key!r}")
                    value, offset, length = cursor.value()
                    yield RowRef(path, variant, split.timenet_split, key, offset, length), value
                    context = f"{path} ({split})"
                context = str(path)
            if splits != set(SourceSplit):
                raise TimeFValidationError(f"expected {' and '.join(SourceSplit)}")
            cursor.expect_end()
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise TimeFValidationError(f"{context}: {exc}") from exc


def read_rows(file: HubFileRef, variant: Variant) -> Iterator[tuple[RowRef, Row | None]]:
    """Validate every entry of one file, in file order.

    Yields:
        Each entry's reference, and its validated row, or None for a published null entry.

    Raises:
        TimeFValidationError: If the document or any row is malformed, with file, split and key.
    """  # noqa: DOC502 - raised by iter_entries and Row.parse
    for ref, value in iter_entries(file.path, variant):
        yield ref, None if value is None else Row.parse(value, ref)


def reread_channels(ref: RowRef) -> dict[str, list[float]]:
    """Reread an entry and decode only its observation.

    The row's other fields were validated at conversion and the tasks use those, so a reread parses
    only what it supplies. ``read_entry`` already fails if the byte range no longer holds one JSON
    value. :func:`check_channels` then compares the channels with the converted row.

    Returns:
        The channels in source order.

    Raises:
        TimeFFormatError: If the entry no longer holds a valid observation, as when the file changed.
    """
    value = read_entry(ref.path, ref.offset, ref.length)
    try:
        if not isinstance(value, dict):
            raise TimeFFormatError("expected a row object")
        return parse_channels(value.get("Observation"), ref.variant)
    except (ValueError, OverflowError, RecursionError) as exc:
        raise TimeFFormatError(f"{ref.context}: {exc}") from exc


def check_channels(channels: dict[str, list[float]], row: Row) -> None:
    """Require reread channels to have the converted row's channel names and lengths.

    Raises:
        TimeFFormatError: If they differ, as when the file changed since conversion.
    """
    if {name: len(values) for name, values in channels.items()} != row.channel_lengths:
        raise TimeFFormatError(f"{row.ref.context}: the channels differ from the converted row; the file changed")
