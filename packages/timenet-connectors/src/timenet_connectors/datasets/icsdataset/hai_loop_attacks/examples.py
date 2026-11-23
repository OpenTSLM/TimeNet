"""The examples table: one row per 300-second window of one HAI file, seen through one control loop.

Each row names its file, its window, its loop and the loop's series, the label, the split, and, for
an attack, every run of flagged seconds of the loop inside the window. It holds no series value.
:class:`Example` parses and checks one row, and creates the example's tasks.
"""

from collections.abc import Iterable, Mapping
import csv
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
import gzip
from pathlib import Path
from typing import IO, Self

from timenet.dataset import Record, Signal
from timenet.errors import TimeNetConnectorError
from timenet.json import JsonMapping
from timenet.types import (
    Annotation,
    ClassificationTask,
    Label,
    LabelSet,
    LocalizationMode,
    Split,
    TemporalLocalizationTask,
    TimeInterval,
)


EXAMPLE_COLUMNS = (
    "example_id",
    "window_id",
    "version",
    "file",
    "split",
    "window_kind",
    "anchor",
    "scored",
    "start",
    "loop",
    "label",
    "attack_ids",
    "loop_uncertain",
    "attack_start_s",
    "attack_end_s",
    "attack_runs",
    "no_visible_change",
    "reacting",
    "train_use",
    "tags",
    "roles",
    "source_tags",
    "matched_from",
    "answer",
)
WINDOW_SECONDS = 300
"""Every example covers 300 rows, one per second."""
VERSIONS = ("20.07", "21.03", "22.04", "23.05")
TIME_FORMAT = "%Y-%m-%d %H:%M:%S"


class LoopState(StrEnum):
    """The label of an example: whether its loop was under attack in the window."""

    NORMAL = "normal"
    ATTACK = "attack"


class SeriesRole(StrEnum):
    """The part a series plays in its control loop."""

    TARGET = "target"
    MEASUREMENT = "measurement"
    COMMAND = "command"


class MatchedFrom(StrEnum):
    """How a version's column was matched to its 22.04 tag."""

    SAME_NAME = "same_name"
    """The column has the 22.04 name."""
    MANUAL = "manual"
    """The HAI manual names the column as that point, such as ``P3_LT01`` for ``P3_LIT01``."""
    DATA = "data"
    """The column has another name, and its behaviour in the data matches the point. A guess."""
    NAME = "name"
    """The column has a similar name only, such as ``P3_LH`` for ``P3_LH01``. A guess."""


LOOP_STATE = LabelSet(
    [
        Label(
            name=LoopState.NORMAL.value,
            description=(
                "No attack that the HAI manual assigns to this control loop is flagged in any second of the"
                " window. Another loop of the window may be under attack."
            ),
        ),
        Label(
            name=LoopState.ATTACK.value,
            description=(
                "An attack that the HAI manual assigns to this control loop is flagged in at least one second"
                " of the window: by the flag of the loop's process in 20.07 and 21.03, by HAI's attack flag in"
                " 22.04 and 23.05."
            ),
        ),
    ],
    annotations=[Annotation("name", value="hai_loop_state")],
)


@dataclass(frozen=True)
class FileKey:
    """One HAI data file: its version and its file name without extensions."""

    version: str
    stem: str


@dataclass(frozen=True)
class SeriesTag:
    """One series of a loop: its 22.04 name, its column in the version's files, and its role."""

    tag: str
    source_tag: str
    loop: str
    role: SeriesRole
    matched_from: MatchedFrom

    def signal_metadata(self) -> JsonMapping:
        """Return the metadata the series adds to its signal.

        Returns:
            The JSON-compatible metadata.
        """
        return {
            "tag": self.tag,
            "loop": self.loop,
            "role": self.role.value,
            "tag_matched_from": self.matched_from.value,
        }


@dataclass(frozen=True)
class FlaggedAttack:
    """The attack on the loop of an example. The seconds count from the window start."""

    attack_ids: tuple[str, ...]
    runs: tuple[tuple[int, int], ...]
    """Each run of flagged seconds, its first and last second, inclusive. The last run can end after
    the window."""

    @property
    def first_second(self) -> int:
        """The first flagged second."""
        return self.runs[0][0]

    @property
    def last_second(self) -> int:
        """The last flagged second, inclusive. It can lie after the end of the window."""
        return self.runs[-1][1]

    def is_cut_at_window_end(self) -> bool:
        """Return whether the attack is still flagged when the window ends.

        Returns:
            True when the last flagged second lies at or after the end of the window.
        """
        return self.last_second >= WINDOW_SECONDS

    def intervals(self, offset: int, series: tuple[Signal, ...]) -> tuple[TimeInterval, ...]:
        """Return the runs as intervals of the record, each cut at the end of the window.

        Args:
            offset: The second of the record where the window starts.
            series: The signals the intervals cover.

        Returns:
            One interval per run.
        """
        return tuple(
            TimeInterval.seconds(offset + first, offset + min(last + 1, WINDOW_SECONDS), time_series=series)
            for first, last in self.runs
        )


@dataclass(frozen=True)
class FileRecord:
    """The record of one data file, and its signals by 22.04 tag."""

    record: Record
    signals: Mapping[str, Signal]

    def offset_of(self, example: "Example") -> int:
        """Return the row of the record where an example's window starts.

        Returns:
            The row, counted from zero, which is also the second from the first row.

        Raises:
            TimeNetConnectorError: If the window does not lie inside the file.
        """
        metadata = self.record.metadata
        first = datetime.strptime(str(metadata["start_time"]), TIME_FORMAT)
        n_seconds = metadata["n_seconds"]
        offset = int((example.start - first).total_seconds())
        if not isinstance(n_seconds, int) or offset < 0 or offset + WINDOW_SECONDS > n_seconds:
            raise TimeNetConnectorError(f"{example.example_id}: the window leaves its file of {n_seconds} rows")
        return offset

    def series_of(self, example: "Example") -> tuple[Signal, ...]:
        """Return the signals of an example's loop, in the order of the table.

        Returns:
            The signals.
        """
        return tuple(self.signals[one.tag] for one in example.series)


def _parse_bool(value: str, column: str) -> bool:
    """Parse a ``True`` or ``False`` cell.

    Returns:
        The value.

    Raises:
        TimeNetConnectorError: If the cell holds anything else.
    """
    if value not in {"True", "False"}:
        raise TimeNetConnectorError(f"{column} {value!r} is not True or False")
    return value == "True"


def _parse_int(value: str, column: str) -> int:
    """Parse a whole-number cell.

    Returns:
        The value.

    Raises:
        TimeNetConnectorError: If the cell is not a whole number.
    """
    try:
        return int(value)
    except ValueError as exc:
        raise TimeNetConnectorError(f"{column} {value!r} is not a whole number") from exc


def _parse_runs(value: str) -> tuple[tuple[int, int], ...]:
    """Parse the ``attack_runs`` cell, such as ``"35-63 120-180"``.

    Returns:
        The runs, as first and last second.
    """
    runs = []
    for part in value.split():
        first, _, last = part.partition("-")
        runs.append((_parse_int(first, "attack_runs"), _parse_int(last, "attack_runs")))
    return tuple(runs)


def _parse_attack(row: Mapping[str, str], label: LoopState) -> FlaggedAttack | None:
    """Parse and check the attack cells of a row.

    An attack example names its attacks and at least one run. The runs are in order, do not touch,
    and each starts inside the window. ``attack_start_s`` and ``attack_end_s`` are the first and last
    flagged second. A normal example names no attack, no run, and -1 for both seconds.

    Returns:
        The attack, or ``None`` for a normal example.

    Raises:
        TimeNetConnectorError: If the attack cells do not fit the label or each other.
    """
    ids, runs = tuple(row["attack_ids"].split()), _parse_runs(row["attack_runs"])
    start, end = _parse_int(row["attack_start_s"], "attack_start_s"), _parse_int(row["attack_end_s"], "attack_end_s")
    if label is LoopState.NORMAL:
        if ids or runs or (start, end) != (-1, -1):
            raise TimeNetConnectorError("a normal example names an attack")
        return None
    if not ids or not runs:
        raise TimeNetConnectorError("an attack example names no attack or no flagged run")
    for (first, last), after in zip(runs, (*runs[1:], None), strict=True):
        if not 0 <= first < WINDOW_SECONDS or last < first or (after is not None and after[0] <= last + 1):
            raise TimeNetConnectorError(f"attack_runs {row['attack_runs']!r} are not separate runs inside the window")
    attack = FlaggedAttack(attack_ids=ids, runs=runs)
    if (start, end) != (attack.first_second, attack.last_second):
        raise TimeNetConnectorError("attack_start_s and attack_end_s are not the first and last flagged second")
    return attack


def _parse_series(row: Mapping[str, str]) -> tuple[SeriesTag, ...]:
    """Parse the series of a row: tags, source tags, roles and matches, one word each per series.

    Returns:
        The series.

    Raises:
        TimeNetConnectorError: If the four cells name different numbers of series, or a role or a
            match is unknown.
    """
    tags, sources, roles, matches = (row[column].split() for column in ("tags", "source_tags", "roles", "matched_from"))
    if not tags or not len(tags) == len(sources) == len(roles) == len(matches):
        raise TimeNetConnectorError("tags, source_tags, roles and matched_from must name the same one or more series")
    try:
        return tuple(
            SeriesTag(tag, source, row["loop"], SeriesRole(role), MatchedFrom(match))
            for tag, source, role, match in zip(tags, sources, roles, matches, strict=True)
        )
    except ValueError as exc:
        raise TimeNetConnectorError(f"roles {roles} or matched_from {matches}: {exc}") from exc


@dataclass(frozen=True)
class Example:
    """One row of the examples table: a window of one file seen through one control loop."""

    example_id: str
    window_id: str
    version: str
    file: str
    split: Split
    window_kind: str
    anchor: str
    scored: bool
    start: datetime
    loop: str
    label: LoopState
    attack: FlaggedAttack | None
    loop_uncertain: bool
    no_visible_change: bool
    reacting: bool
    train_use: bool
    series: tuple[SeriesTag, ...]
    rationale: str | None
    """The rule-written text that reads the series and ends with the label. ``None`` for an attack
    example whose series show no visible change."""

    @property
    def key(self) -> FileKey:
        """The data file of the example."""
        return FileKey(self.version, self.file)

    @classmethod
    def from_row(cls, row: Mapping[str, str]) -> Self:
        """Parse and check one row of the table.

        Returns:
            The example.

        Raises:
            TimeNetConnectorError: If a cell does not hold what the table promises, or two cells
                disagree.
        """
        version = row["version"]
        if version not in VERSIONS:
            raise TimeNetConnectorError(f"version {version!r} is not one of {VERSIONS}")
        try:
            label = LoopState(row["label"])
        except ValueError as exc:
            raise TimeNetConnectorError(f"label {row['label']!r} is not one of {LOOP_STATE.names}") from exc
        try:
            split = Split(row["split"])
        except ValueError as exc:
            raise TimeNetConnectorError(f"split {row['split']!r} is not one of {[one.value for one in Split]}") from exc
        try:
            # HAI prints local wall-clock times and states no time zone, so the time stays naive.
            start = datetime.strptime(row["start"], TIME_FORMAT)
        except ValueError as exc:
            raise TimeNetConnectorError(f"start {row['start']!r} is not a time like 2020-07-07 15:00:00") from exc
        window_id = f"{version}/{row['file']}/{start:%Y%m%dT%H%M%S}"
        if row["window_id"] != window_id or row["example_id"] != f"{window_id}/{row['loop']}":
            raise TimeNetConnectorError(
                f"example_id {row['example_id']!r} does not match its version, file, start and loop"
            )
        no_visible_change = _parse_bool(row["no_visible_change"], "no_visible_change")
        reacting = _parse_bool(row["reacting"], "reacting")
        train_use = _parse_bool(row["train_use"], "train_use")
        if no_visible_change and label is not LoopState.ATTACK:
            raise TimeNetConnectorError("no_visible_change is set on a normal example")
        if reacting and label is not LoopState.NORMAL:
            raise TimeNetConnectorError("reacting is set on an attack example")
        if train_use and (no_visible_change or reacting):
            raise TimeNetConnectorError("train_use is set on a no_visible_change or reacting example")
        answer = row["answer"]
        if no_visible_change and answer:
            raise TimeNetConnectorError("a no_visible_change example has an answer")
        if not no_visible_change and not answer.endswith(f"Answer: {label}"):
            raise TimeNetConnectorError(f"answer does not end with 'Answer: {label}'")
        return cls(
            example_id=row["example_id"],
            window_id=window_id,
            version=version,
            file=row["file"],
            split=split,
            window_kind=row["window_kind"],
            anchor=row["anchor"],
            scored=_parse_bool(row["scored"], "scored"),
            start=start,
            loop=row["loop"],
            label=label,
            attack=_parse_attack(row, label),
            loop_uncertain=_parse_bool(row["loop_uncertain"], "loop_uncertain"),
            no_visible_change=no_visible_change,
            reacting=reacting,
            train_use=train_use,
            series=_parse_series(row),
            rationale=answer or None,
        )

    def classification_metadata(self) -> JsonMapping:
        """Return the metadata of the classification task.

        It names the example and its series, and nothing that depends on the label: no attack, no
        window kind, and no flag that is set only on one label.

        Returns:
            The JSON-compatible metadata.
        """
        return {
            "example_id": self.example_id,
            "window_id": self.window_id,
            "version": self.version,
            "file": self.file,
            "loop": self.loop,
            "tags": [one.tag for one in self.series],
            "source_tags": [one.source_tag for one in self.series],
            "roles": [one.role.value for one in self.series],
            "tag_matched_from": [one.matched_from.value for one in self.series],
        }

    def localization_metadata(self, attack: FlaggedAttack) -> JsonMapping:
        """Return the metadata of the localization task: the attack and its window.

        Returns:
            The JSON-compatible metadata.
        """
        return {
            "example_id": self.example_id,
            "attack_ids": list(attack.attack_ids),
            "attack_start_s": attack.first_second,
            "attack_end_s": attack.last_second,
            "attack_runs": [list(run) for run in attack.runs],
            "cut_at_window_end": attack.is_cut_at_window_end(),
            "window_kind": self.window_kind,
            "anchor": self.anchor,
            "scored": self.scored,
            "loop_uncertain": self.loop_uncertain,
            "no_visible_change": self.no_visible_change,
        }

    def create_classification_task(self, file_record: FileRecord) -> ClassificationTask:
        """Create the task that asks whether the loop was under attack in the window.

        Args:
            file_record: The record of the example's file.

        Returns:
            The task, not yet added to a dataset.
        """
        offset = file_record.offset_of(self)
        return ClassificationTask(
            inputs=(file_record.record,),
            targets=(self.label.value,),
            label_set=LOOP_STATE,
            scope=TimeInterval.seconds(offset, offset + WINDOW_SECONDS, time_series=file_record.series_of(self)),
            split=self.split,
            rationale=self.rationale,
            metadata=self.classification_metadata(),
        )

    def create_localization_task(
        self, file_record: FileRecord, *, from_task: ClassificationTask
    ) -> TemporalLocalizationTask:
        """Create the task that asks for the flagged runs of an attack example.

        Args:
            file_record: The record of the example's file.
            from_task: The example's classification task.

        Returns:
            The task, not yet added to a dataset.

        Raises:
            TimeNetConnectorError: If the example is not an attack example.
        """
        if self.attack is None:
            raise TimeNetConnectorError(f"{self.example_id} is not an attack example")
        offset, series = file_record.offset_of(self), file_record.series_of(self)
        return TemporalLocalizationTask(
            inputs=(file_record.record,),
            targets=self.attack.intervals(offset, series),
            scope=from_task.scope,
            split=self.split,
            mode=LocalizationMode.SPARSE,
            from_tasks=(from_task,),
            metadata=self.localization_metadata(self.attack),
        )


def _open_text(path: Path) -> IO[str]:
    """Open the gzip-compressed table as text, keeping its line ends for the csv module.

    Returns:
        The open text stream.
    """
    return gzip.open(path, "rt", encoding="utf-8", newline="")


def read_examples(path: Path) -> list[Example]:
    """Read every example of the table, in file order.

    Returns:
        The examples.

    Raises:
        TimeNetConnectorError: If the header is not the expected one, a row is malformed, or an
            example id repeats. The message names the data row.
    """
    with _open_text(path) as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != EXAMPLE_COLUMNS:
            raise TimeNetConnectorError(f"{path.name} has columns {reader.fieldnames}, not {list(EXAMPLE_COLUMNS)}")
        examples, seen = [], set()
        for number, row in enumerate(reader, start=1):
            try:
                example = Example.from_row(row)
                if example.example_id in seen:
                    raise TimeNetConnectorError(f"example_id {example.example_id!r} repeats")
            except TimeNetConnectorError as exc:
                raise TimeNetConnectorError(f"{path.name} data row {number}: {exc}") from exc
            seen.add(example.example_id)
            examples.append(example)
    return examples


def series_by_version(examples: Iterable[Example]) -> dict[str, dict[str, SeriesTag]]:
    """Collect the series each version uses, in the order the examples first name them.

    Returns:
        For each version, its series by 22.04 tag.

    Raises:
        TimeNetConnectorError: If two examples of one version give one tag another column, loop,
            role or match, or two tags share one column.
    """
    found: dict[str, dict[str, SeriesTag]] = {}
    for example in examples:
        tags = found.setdefault(example.version, {})
        for one in example.series:
            known = tags.setdefault(one.tag, one)
            if known != one:
                raise TimeNetConnectorError(f"{example.example_id}: {one} disagrees with {known}")
    for version, tags in found.items():
        columns = [one.source_tag for one in tags.values()]
        if len(set(columns)) != len(columns):
            raise TimeNetConnectorError(f"version {version}: two tags share one source column")
    return found
