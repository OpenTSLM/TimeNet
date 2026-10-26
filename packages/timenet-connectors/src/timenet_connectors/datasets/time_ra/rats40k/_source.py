"""Read the published RATS40K JSON: walk its split/key entries, validate rows, and parse observations.

Each file is one JSON document, not JSON Lines: ``{"TSAD_train": {key: row, ...}, "TSAD_test": {...}}``.
Walk it one entry at a time and keep each entry's UTF-8 byte range, so a signal loader rereads just its
row instead of holding the decoded corpus in memory.
"""

from collections.abc import Iterator
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

from timenet.errors import TimeFValidationError
from timenet.types import Split
from timenet_connectors.bases.huggingface import HubFileRef
from timenet_connectors.sources.json_stream import JsonCursor, read_entry


_SPLITS = {"TSAD_train": Split.TRAIN, "TSAD_test": Split.TEST}
_TEXT_FIELDS = ("Source", "FigurePath", "Thought", "Action", "Label")


@dataclass(frozen=True)
class RowRef:
    """Where one entry lives: its file, variant, split and key, and its byte range."""

    path: Path
    variant: str
    split: Split
    key: str
    offset: int
    length: int

    @property
    def id(self) -> str:
        return f"rats40k-{self.variant}-{self.split}-{self.key}"

    @property
    def context(self) -> str:
        return f"{self.path} ({self.split}, key {self.key})"


@dataclass(frozen=True)
class Row:
    """A validated row's fields, without its observation values."""

    domain: str
    figure_path: str
    thought: str
    action: str
    action_id: int
    label: str
    rank: list[Any] | None
    channel_lengths: dict[str, int]
    """Each channel's name and length, in source order."""


def iter_entries(path: Path) -> Iterator[tuple[str, str, Any, int, int]]:
    """Walk the split/key objects without retaining decoded rows.

    Yields:
        Split, original key, decoded row, and its UTF-8 byte offset and length.

    Raises:
        TimeFValidationError: If the document or split structure is malformed.
    """
    context = str(path)
    try:
        with JsonCursor.open(path) as cursor:
            splits: set[str] = set()
            for split in cursor.object_keys():
                if split not in _SPLITS:
                    raise TimeFValidationError(f"unknown split {split!r}")
                splits.add(split)
                context = f"{path} ({split})"
                for key in cursor.object_keys():
                    context = f"{path} ({split}, key {key})"
                    if not key.isascii() or not key.isdecimal():
                        raise TimeFValidationError(f"{split}: expected a decimal row key, got {key!r}")
                    row, offset, length = cursor.value()
                    yield split, key, row, offset, length
                    context = f"{path} ({split})"
                context = str(path)
            if splits != set(_SPLITS):
                raise TimeFValidationError("expected TSAD_train and TSAD_test")
            cursor.expect_end()
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise TimeFValidationError(f"{context}: {exc}") from exc


def read_rows(file: HubFileRef) -> Iterator[tuple[RowRef, Row | None]]:
    """Validate every entry of one file, in file order.

    Yields:
        Each entry's reference, and its fields, or None for a published null entry.

    Raises:
        TimeFValidationError: If the document or any row is malformed, with file, split and key.
    """  # noqa: DOC502 - raised by iter_entries and _channels
    for split, key, row, offset, length in iter_entries(file.path):
        ref = RowRef(file.path, file.part, _SPLITS[split], key, offset, length)
        yield ref, None if row is None else _fields(row, ref)


def load_channels(ref: RowRef) -> dict[str, list[float]]:
    """Reread and validate one entry's observation.

    Returns:
        The channels in source order.
    """
    return _channels(read_entry(ref.path, ref.offset, ref.length), ref)


def _fields(row: Any, ref: RowRef) -> Row:
    """Validate a row and keep its fields, dropping the observation values.

    Returns:
        The row's fields and its channels' lengths.
    """
    channels = _channels(row, ref)
    return Row(
        domain=row["Source"],
        figure_path=row["FigurePath"],
        thought=row["Thought"],
        action=row["Action"],
        action_id=row["ActionID"],
        label=row["Label"],
        rank=row.get("Rank"),
        channel_lengths={name: len(values) for name, values in channels.items()},
    )


def _channels(row: Any, ref: RowRef) -> dict[str, list[float]]:
    """Validate a row's fields and decode its observation.

    Returns:
        Nonempty, equally long channels of finite numbers, in source order.

    Raises:
        TimeFValidationError: If the row is malformed, with file, split and key.
    """
    try:
        if not isinstance(row, dict):
            raise TimeFValidationError("expected a row object")
        for field in _TEXT_FIELDS:
            if not isinstance(row.get(field), str) or not row[field].strip():
                raise TimeFValidationError(f"missing or empty {field}")
        # Label sometimes disagrees with Action; both are kept as published.
        if row["Label"] not in {"Normal", "Anomaly"}:
            raise TimeFValidationError("Label must be Normal or Anomaly")
        if type(row.get("ActionID")) is not int or row["ActionID"] < 0:
            raise TimeFValidationError("ActionID must be a nonnegative integer")
        observation = row.get("Observation")
        channels = {"series": observation} if ref.variant == "uni" else _named_channels(observation)
        _check_channels(channels)
        return channels
    except (ValueError, OverflowError, RecursionError) as exc:
        raise TimeFValidationError(f"{ref.context}: {exc}") from exc


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


def _check_channels(channels: dict[str, Any]) -> None:
    """Require nonempty, equally long channels of finite numbers.

    Raises:
        TimeFValidationError: If a channel is empty, nonnumeric, nonfinite, or of another length.
    """
    if not channels:
        raise TimeFValidationError("Observation has no channels")
    lengths = set()
    for name, values in channels.items():
        if not isinstance(values, list) or not values:
            raise TimeFValidationError(f"channel {name!r} must be a nonempty numeric list")
        for value in values:
            if type(value) not in {int, float} or not math.isfinite(value):
                raise TimeFValidationError(f"channel {name!r} must contain finite float64 numbers")
        lengths.add(len(values))
    if len(lengths) != 1:
        raise TimeFValidationError("channel lengths differ")
