"""Read PLAID's two metadata files: one entry per CSV file, keyed by the file's number.

``metadata_submetered.json`` describes one appliance per file; ``metadata_aggregated.json`` describes two
to four, each with the sample indices at which it switched on and off. Box 1 and Box 2 of the paper give
the layouts. Each entry is validated by its model, so a field the layout doesn't have fails the build.

The measurer left many fields blank, as an empty string. Reading leaves a blank field out, as the authors'
``clean_meta`` does, so a blank never passes as text.
"""

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum, unique
import json
from pathlib import Path
import re
from typing import Annotated, Any, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field, StrictStr, StringConstraints

from timenet.errors import TimeFFormatError


# A strict, non-blank string, as RATs40K and TRACE define it: a blank field must not pass as text.
Text = Annotated[StrictStr, StringConstraints(pattern=r"\S")]

_INDICES = re.compile(r"\[\s*(\d+(?:\s+\d+)*)?\s*\]")
"""An ``on`` or ``off`` field: sample indices in brackets, separated by spaces, as NumPy prints an array."""


@unique
class LoadType(StrEnum):
    """An appliance's load type, as Table 2 of the paper groups them."""

    RESISTIVE = "R"
    INDUCTIVE = "I"
    NON_LINEAR = "NL"


@unique
class EventKind(StrEnum):
    """Whether an aggregated file's event switched its appliance on or off."""

    ON = "on"
    OFF = "off"


class Appliance(BaseModel):
    """One measured appliance: its type, its load type, and the facts the measurer filled in.

    The type is written as each metadata file writes it. The two files don't always agree: the submetered
    file writes ``Soldering Iron``, the aggregated one ``Iron solder``.
    """

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    type: Text
    # JSON writes the load type as a string, which only a lax enum field accepts.
    load: LoadType = Field(strict=False)
    brand: Text | None = None
    model_number: Text | None = None
    manufacture_year: Text | None = None
    current: Text | None = None
    """The rated current, as written, such as ``1.1 A``."""
    voltage: Text | None = None
    """The rated voltage, as written, such as ``120V``."""
    wattage: Text | None = None
    """The rated power, as written, such as ``740W``."""
    notes: Text | None = None


class Header(BaseModel):
    """How a file was captured: when, with what note, and at what rate."""

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    collection_time: Text
    """A month and year, such as ``July, 2013``: which collection campaign measured the file."""
    notes: Text | None = None
    """The campaign's note. The 2013 files say they are not all well calibrated."""
    sampling_frequency: Literal["30000Hz"]


class _Entry(BaseModel):
    """What both layouts give for a file: its number, its capture, its stated length and its location."""

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    number: int = Field(ge=1)
    """The file's number, which names its CSV file and keys its entry."""
    header: Header
    length: Annotated[StrictStr, StringConstraints(pattern=r"^[0-9]+\.[0-9]{2}s$")]
    """The measurement's duration, as written, such as ``2.00s``: its rows over the rate, to hundredths."""
    location: Text
    """Where the file was measured, as written, such as ``house1`` or ``CMU lab``."""
    status: Text
    """For a submetered file, the appliance's operating mode, such as ``off-on`` or ``highcool``. For an
    aggregated file, its sequence of events, such as ``on-on-off-off``, as the measurer wrote it."""

    @property
    def length_s(self) -> float:
        """Return the stated duration in seconds."""
        return float(self.length.removesuffix("s"))

    @property
    def household(self) -> str:
        """Return the household the authors' GlobalSIP notebook groups by: the campaign, then the location.

        Locations ``house1`` to ``house9`` recur in two campaigns as different households, so the location
        alone does not name one. The paper's 65 locations are these pairs.
        """
        return f"{self.header.collection_time}_{self.location}"


class SubmeteredEntry(_Entry):
    """A submetered file's entry: one appliance switching on, measured alone."""

    appliance: Appliance


@dataclass(frozen=True)
class SwitchEvent:
    """One labelled event of an aggregated file: an appliance switching on or off at a sample."""

    kind: EventKind
    appliance_type: str
    sample_index: int
    """The sample at which the appliance switched, counted from the file's first row. The release does not
    say whether its indices count from 0 or 1; they are read as counting from 0."""


class SwitchedAppliance(BaseModel):
    """One appliance of an aggregated file, with the samples at which it switched on and off."""

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    appliance: Appliance
    on: tuple[int, ...]
    """Usually one index. The soldering iron switches on in two steps, and the paper labels both."""
    off: tuple[int, ...]
    """Usually one index. One air conditioner (file 364) has none."""


class AggregatedEntry(_Entry):
    """An aggregated file's entry: several appliances switching on and off together."""

    appliances: tuple[SwitchedAppliance, ...] = Field(min_length=1)

    def events(self) -> tuple[SwitchEvent, ...]:
        """Return the file's labelled events, in sample order, then in the entry's appliance order.

        Returns:
            One event per index of every ``on`` and ``off`` field.
        """
        events = [
            SwitchEvent(kind=kind, appliance_type=switched.appliance.type, sample_index=index)
            for switched in self.appliances
            for kind, indices in ((EventKind.ON, switched.on), (EventKind.OFF, switched.off))
            for index in indices
        ]
        return tuple(sorted(events, key=lambda event: event.sample_index))


def _drop_blanks(fields: dict[str, Any]) -> dict[str, Any]:
    """Return the fields without the ones the measurer left blank."""
    # Only an empty string is blank: a field of another type still reaches validation.
    return {key: value for key, value in fields.items() if not (isinstance(value, str) and not value)}


def _indices(field: object) -> tuple[int, ...]:
    """Parse an ``on`` or ``off`` field, such as ``[79294 265470]``.

    Returns:
        The sample indices, in the order written.

    Raises:
        ValueError: If the field is not bracketed whole numbers.
    """
    match = _INDICES.fullmatch(field) if isinstance(field, str) else None
    if match is None:
        raise ValueError(f"expected sample indices such as [37043], got {field!r}")
    return tuple(int(index) for index in (match.group(1) or "").split())


def _submetered(number: int, raw: dict[str, Any]) -> SubmeteredEntry:
    appliance = dict(raw["appliance"])
    status = appliance.pop("status")
    return SubmeteredEntry(
        number=number,
        appliance=Appliance.model_validate(_drop_blanks(appliance)),
        status=status,
        header=Header.model_validate(_drop_blanks(raw["header"])),
        length=raw["instances"]["length"],
        location=raw["location"],
    )


def _aggregated(number: int, raw: dict[str, Any]) -> AggregatedEntry:
    appliances = []
    for written in raw["appliances"]:
        fields = dict(written)
        on, off = fields.pop("on"), fields.pop("off")
        appliances.append(
            SwitchedAppliance(
                appliance=Appliance.model_validate(_drop_blanks(fields)),
                on=_indices(on),
                off=_indices(off),
            )
        )
    return AggregatedEntry(
        number=number,
        appliances=tuple(appliances),
        status=raw["instances"]["status"],
        header=Header.model_validate(_drop_blanks(raw["header"])),
        length=raw["instances"]["length"],
        location=raw["location"],
    )


EntryT = TypeVar("EntryT", bound=_Entry)


def _read(path: Path, parse: Callable[[int, dict[str, Any]], EntryT], layout: type[EntryT]) -> tuple[EntryT, ...]:
    """Read one metadata file: a JSON object of entries keyed by file number.

    Returns:
        The entries, in number order.

    Raises:
        TimeFFormatError: If the file is not a JSON object of objects, a key is not a file number, or an
            entry does not fit ``layout``.
    """
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TimeFFormatError(f"{path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise TimeFFormatError(f"{path}: expected a JSON object of entries keyed by file number")
    entries = []
    for key, fields in raw.items():
        context = f"{path.name} entry {key!r}"
        if not key.isdecimal() or not isinstance(fields, dict):
            raise TimeFFormatError(f"{context}: expected a file number keying an object")
        try:
            entries.append(parse(int(key), fields))
        # ValidationError is a ValueError, as is a malformed on or off field.
        except (KeyError, TypeError, ValueError) as exc:
            raise TimeFFormatError(f"{context} does not fit the {layout.__name__} layout: {exc!r}") from exc
    return tuple(sorted(entries, key=lambda entry: entry.number))


def read_submetered(path: Path) -> tuple[SubmeteredEntry, ...]:
    """Read ``metadata_submetered.json``.

    Returns:
        One entry per submetered file, in number order.
    """
    return _read(path, _submetered, SubmeteredEntry)


def read_aggregated(path: Path) -> tuple[AggregatedEntry, ...]:
    """Read ``metadata_aggregated.json``.

    Returns:
        One entry per aggregated file, in number order.
    """
    return _read(path, _aggregated, AggregatedEntry)
