"""Validate source rows and retain checked descriptors for lazy rereads."""

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from functools import partial
import hashlib
import json
from pathlib import Path
from typing import Literal, NamedTuple

import pyarrow as pa
from pydantic import Field

from timenet.errors import TimeNetConnectorError
from timenet.types import Split
from timenet_connectors.datasets.time_ra.rats40k.vocabulary import (
    NON_SPACE,
    SPEC,
    UNI_CHANNEL,
    SourceField,
    SourceLabel,
    SourceSplit,
    Variant,
)
from timenet_connectors.sources.json_stream import JsonCursor, read_entry
from timenet_connectors.sources.reread import checked_reread
from timenet_connectors.sources.rows import NonBlankText, SourceRow, validate_row
from timenet_connectors.sources.signals import numeric_signal


@dataclass(frozen=True)
class Observation:
    """A source observation's location and immutable channel fingerprint."""

    path: Path
    variant: Variant
    split: Split
    key: str
    offset: int
    length: int
    channel_lengths: tuple[tuple[str, int], ...]
    values_digest: bytes

    @property
    def context(self) -> str:
        """The source row for diagnostics."""
        return f"{self.path} ({self.split}, key {self.key})"


@dataclass(frozen=True)
class Row:
    """Checked task fields with a separately loadable observation."""

    observation: Observation
    domain: str
    figure_path: str
    thought: str
    action: str
    action_id: int
    label: SourceLabel
    rank: tuple[str, ...] | None


class Entry(NamedTuple):
    """One entry of a file, as decoded and not yet validated."""

    split: Split
    key: str
    value: object
    offset: int
    length: int


class SourceInventory(NamedTuple):
    """One file's rows and null entries at the pinned revision."""

    row_counts: Mapping[Split, int]
    """The number of usable rows in each split, without the null entries."""
    null_keys: Mapping[Split, frozenset[str]]
    """The keys of the null entries in each split. The connector skips these entries."""

    def iter_rows(self, path: Path, variant: Variant) -> Iterator["Row"]:
        """Yield usable rows and check the inventory when the file is exhausted.

        Counts include every split, including empty ones. Only explicit null entries are skipped;
        malformed rows still fail in the source parser.
        """
        splits = [split.timef_split for split in SourceSplit]
        rows = dict.fromkeys(splits, 0)
        null_keys: dict[Split, set[str]] = {split: set() for split in splits}
        for entry in iter_entries(path):
            if (row := parse_row(path, variant, entry)) is None:
                null_keys[entry.split].add(entry.key)
            else:
                rows[entry.split] += 1
                yield row
        found = SourceInventory(rows, {split: frozenset(keys) for split, keys in null_keys.items()})
        self.check(path, found)

    def check(self, path: Path, found: "SourceInventory") -> None:
        """Compare the rows and null entries found in a file with this inventory.

        Args:
            path: The file, for the error message.
            found: What the connector found in the file, with every split.

        Raises:
            TimeNetConnectorError: If the counts or the null keys differ, so the file is not the pinned one.
        """
        if dict(found.row_counts) != dict(self.row_counts) or dict(found.null_keys) != dict(self.null_keys):
            raise TimeNetConnectorError(
                f"{path}: the rows differ from the pinned inventory: expected {self._text()}, found {found._text()}"
            )

    def _text(self) -> str:
        """Describe the inventory for an error message.

        Returns:
            The rows and null keys of each split, such as ``test: 1 rows, null keys ['7']``.
        """
        return ", ".join(
            f"{split}: {self.row_counts.get(split, 0)} rows, null keys {sorted(self.null_keys.get(split, ()))}"
            for split in sorted({*self.row_counts, *self.null_keys})
        )


# The eleven null entries are univariate test keys that the authors' labeling/rank_models.py starts as
# None and never fills.
PINNED_INVENTORY: dict[Variant, SourceInventory] = {
    Variant.UNI: SourceInventory(
        row_counts={Split.TRAIN: 30_266, Split.TEST: 6_034},
        null_keys={
            Split.TRAIN: frozenset(),
            Split.TEST: frozenset({"106", "386", "560", "875", "1632", "2081", "2581", "2644", "2721", "3584", "5558"}),
        },
    ),
    Variant.MULTI: SourceInventory(
        row_counts={Split.TRAIN: 2_728, Split.TEST: 546},
        null_keys={Split.TRAIN: frozenset(), Split.TEST: frozenset()},
    ),
}


def _split_entries(cursor: JsonCursor, split: SourceSplit) -> Iterator[Entry]:
    """Yield one split's entries with split and key context on malformed input.

    Raises:
        TimeNetConnectorError: If an entry key or value cannot be read.
    """
    context = str(split)
    try:
        for key in cursor.object_keys():
            context = f"{split}, key {key}"
            if not key.isascii() or not key.isdecimal():
                raise TimeNetConnectorError(f"expected a decimal row key, got {key!r}")
            yield Entry(split.timef_split, key, *cursor.value())
            context = str(split)
    except TimeNetConnectorError as exc:
        raise TimeNetConnectorError(f"({context}): {exc}") from exc


def iter_entries(path: Path) -> Iterator[Entry]:
    """Walk a file's ``{split: {key: row}}`` objects without keeping the decoded rows.

    Yields:
        Each entry, in file order.

    Raises:
        TimeNetConnectorError: If the document or its split structure is malformed, with the file and the
            split or key in the message.
    """
    try:
        with JsonCursor.open(path) as cursor:
            splits: set[SourceSplit] = set()
            for name in cursor.object_keys():
                if name not in set(SourceSplit):
                    raise TimeNetConnectorError(f"unknown split {name!r}")
                split = SourceSplit(name)
                splits.add(split)
                yield from _split_entries(cursor, split)
            if splits != set(SourceSplit):
                raise TimeNetConnectorError(f"expected {' and '.join(SourceSplit)}")
            cursor.expect_end()
    except (OSError, TimeNetConnectorError) as exc:
        raise TimeNetConnectorError(f"{path}: {exc}") from exc


class _ChannelValues(SourceRow):
    """One nonempty decoded channel before lossless float64 conversion."""

    values: list[int | float] = Field(min_length=1)


def _series(name: str, values: object) -> pa.Array:
    """Return a validated channel as losslessly converted float64 samples."""
    parsed = validate_row({"values": values}, _ChannelValues, location=f"channel {name!r}")
    return numeric_signal(parsed.values, name, spec=SPEC)


def _skip_space(text: str, position: int) -> int:
    """Return the position of the first character at or after ``position`` that is not whitespace."""
    match = NON_SPACE.search(text, position)
    return len(text) if match is None else match.start()


def _named_channels(observation: object) -> dict[str, object]:
    """Split a multivariate observation, ``"name": [values]; ...``, into its channels.

    Names are JSON strings, so a ``;`` inside a name is not a separator.

    Returns:
        The decoded values by channel name, in source order.

    Raises:
        TimeNetConnectorError: If the observation is not text, or its names, colons or separators are wrong.
    """
    if not isinstance(observation, str):
        raise TimeNetConnectorError("a multivariate Observation is a string of named channels")
    decoder = json.JSONDecoder()
    text = observation.strip()
    channels: dict[str, object] = {}
    position = 0
    while position < len(text):
        name, position = decoder.raw_decode(text, position)
        position = _skip_space(text, position)
        if not isinstance(name, str) or not name.strip() or text[position : position + 1] != ":":
            raise TimeNetConnectorError("expected a nonblank channel name followed by ':'")
        if name in channels:
            raise TimeNetConnectorError(f"duplicate channel {name!r}")
        channels[name], position = decoder.raw_decode(text, _skip_space(text, position + 1))
        position = _skip_space(text, position)
        if position < len(text):
            if text[position] != ";":
                raise TimeNetConnectorError("expected ';' between channels")
            position = _skip_space(text, position + 1)
            if position == len(text):
                raise TimeNetConnectorError("expected a channel after the last ';'")
    return channels


def _channels(variant: Variant, observation: object) -> dict[str, pa.Array]:
    """Decode an observation: a list of numbers (uni), or a string of named lists (multi).

    Returns:
        The channels in source order, nonempty and equally long.

    Raises:
        TimeNetConnectorError: If the observation violates its variant's format.
    """
    named = {UNI_CHANNEL: observation} if variant is Variant.UNI else _named_channels(observation)
    if not named:
        raise TimeNetConnectorError("the Observation has no channels")
    channels = {name: _series(name, values) for name, values in named.items()}
    if len({len(values) for values in channels.values()}) != 1:
        raise TimeNetConnectorError("the channels differ in length")
    return channels


def _values_digest(channels: Mapping[str, pa.Array]) -> bytes:
    """Digest the channel names and values, in order, as float64 bytes.

    Returns:
        A 16-byte digest that differs when any name, value or the order of the channels differs.
    """
    digest = hashlib.blake2b(digest_size=16)
    for name, values in channels.items():
        digest.update(name.encode("utf-8") + b"\0" + values.to_numpy().tobytes())
    return digest.digest()


class _PublishedRow(SourceRow):
    """The published metadata and encoded observation of a RATs40K row."""

    observation: object = Field(alias="Observation")
    domain: NonBlankText = Field(alias="Source")
    figure_path: NonBlankText = Field(alias="FigurePath")
    thought: NonBlankText = Field(alias="Thought")
    action: NonBlankText = Field(alias="Action")
    action_id: int = Field(alias="ActionID", ge=0)
    label: Literal["Normal", "Anomaly"] = Field(alias="Label")
    rank: list[NonBlankText] | None = Field(default=None, alias="Rank")


def parse_row(path: Path, variant: Variant, entry: Entry) -> Row | None:
    """Validate a row and retain descriptors; explicit published null entries stay skipped.

    Returns:
        The checked descriptor, or None for a published null entry.

    Raises:
        TimeNetConnectorError: If the source row or its observation is invalid.
    """
    if entry.value is None:
        return None
    context = f"{path} ({entry.split}, key {entry.key})"
    published = validate_row(entry.value, _PublishedRow, location=context)
    try:
        channels = _channels(variant, published.observation)
    except (ValueError, OverflowError, RecursionError, TimeNetConnectorError) as exc:
        raise TimeNetConnectorError(f"{context}: {exc}") from exc
    return Row(
        observation=Observation(
            path,
            variant,
            entry.split,
            entry.key,
            entry.offset,
            entry.length,
            tuple((name, len(values)) for name, values in channels.items()),
            _values_digest(channels),
        ),
        domain=published.domain,
        figure_path=published.figure_path,
        thought=published.thought,
        action=published.action,
        action_id=published.action_id,
        label=SourceLabel(published.label),
        rank=None if published.rank is None else tuple(published.rank),
    )


def _read_channels(row: Observation) -> dict[str, pa.Array]:
    """Return validated channels from the observation's recorded byte range.

    Returns:
        The channels in source order.

    Raises:
        TimeNetConnectorError: If the source row cannot be read or its channels are invalid.
    """
    value = read_entry(row.path, row.offset, row.length)
    try:
        if not isinstance(value, dict):
            raise TimeNetConnectorError("expected a row object")
        channels = _channels(row.variant, value.get(SourceField.OBSERVATION))
    except (ValueError, OverflowError, RecursionError, TimeNetConnectorError) as exc:
        raise TimeNetConnectorError(f"{row.context}: {exc}") from exc
    return channels


def load_channels(row: Observation) -> tuple[pa.Array, ...]:
    """Return every checked channel for the conversion-wide shared loader."""
    channels = checked_reread(
        partial(_read_channels, row),
        fingerprint=lambda channels: (
            tuple((name, len(values)) for name, values in channels.items()),
            _values_digest(channels),
        ),
        expected=(row.channel_lengths, row.values_digest),
        context=row.context,
    )
    return tuple(channels.values())
