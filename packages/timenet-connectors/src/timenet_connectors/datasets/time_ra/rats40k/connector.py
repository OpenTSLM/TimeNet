"""Convert the pinned RATS40K JSON files into lazy Records and one category task per row.

Each row becomes a Record whose signals reread their entry on demand, and a ClassificationTask whose
target is the published category name. README.md explains how this follows the authors.

Metadata is provenance and never model input. An annotation is model input only for a task that
names it in ``input_annotations``.
"""

from array import array
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from enum import StrEnum, unique
from functools import cache, lru_cache, partial
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, ClassVar, NamedTuple

import pyarrow as pa

from timenet.dataset import Record, Signal, Source, TimeFDataset
from timenet.dataset.axis import OrdinalAxis
from timenet.errors import TimeFFormatError, TimeFValidationError
from timenet.json import JsonValue
from timenet.types import Annotation, ClassificationTask, Label, LabelSet, Split, Task, TimeSeriesSpec, read_value
from timenet_connectors.bases.huggingface import BaseHuggingFaceConnector, HubFileRef
from timenet_connectors.sources.json_stream import JsonCursor, read_entry


@unique
class _Variant(StrEnum):
    """The release's two files: univariate and multivariate series, each with its own prompt."""

    UNI = "uni"
    MULTI = "multi"

    @property
    def source_file(self) -> str:
        """The file's name in the Hub repository, such as ``RATs-Uni-TSImage_Reason.json``."""
        return f"RATs-{self.value.capitalize()}-TSImage_Reason.json"

    @property
    def label_set_name(self) -> str:
        """The name of this variant's label set, such as ``uni_category``."""
        return f"{self.value}_category"

    @property
    def template_key(self) -> str:
        """The key of the dataset annotation that holds this variant's full evaluation prompt."""
        return f"{self.value}_request_template"


@unique
class _SourceSplit(StrEnum):
    """A top-level key of a file: the authors' split."""

    TRAIN = "TSAD_train"
    TEST = "TSAD_test"

    @property
    def timenet_split(self) -> Split:
        """The TimeNet split of the key's rows: ``TSAD_train`` is ``train``."""
        return Split(self.value.removeprefix("TSAD_"))


@unique
class _Field(StrEnum):
    """A field of a published row."""

    OBSERVATION = "Observation"
    DOMAIN = "Source"
    FIGURE_PATH = "FigurePath"
    THOUGHT = "Thought"
    ACTION = "Action"
    ACTION_ID = "ActionID"
    LABEL = "Label"
    RANK = "Rank"
    """Published on test rows. May be null or absent."""


@unique
class _Label(StrEnum):
    """The binary label of a row."""

    NORMAL = "Normal"
    ANOMALY = "Anomaly"


@unique
class _Key(StrEnum):
    """A key of the metadata or the annotations the connector writes."""

    SOURCE_REPOSITORY = "source_repository"
    SOURCE_REVISION = "source_revision"
    SOURCE_DOMAIN = "source_domain"
    """A Record's domain, such as ``Healthcare-ECG``, and its task's only input annotation."""
    LABEL = "label"
    ACTION_ID = "action_id"
    VARIANT = "variant"
    SPLIT = "split"
    ROW_KEY = "key"
    CHANNEL_ORDER = "channel_order"
    """The Source's channels in source order, which TimeF doesn't keep on read."""
    TARGET_IN_VOCABULARY = "target_in_vocabulary"
    FIGURE_PATH = "figure_path"
    RANK = "rank"


_UNI_CHANNEL = "series"
"""The name of the one channel of a univariate row."""
_PROMPTS = Path(__file__).parent / "prompts"
# The authors' spelling, "oberservation", included.
_DOMAIN_CLAUSE = re.compile(r" The oberservation is from (?:the )?domain of \{our_source\}\.")
_OBSERVATION_LINE = "Observation: {our_observation}\n"
_ANSWER_FORMAT = "Thought: \\boxed1{{{thought}}}\nAction: \\boxed2{{{action}}}"
"""The two boxed fields that end each prompt, as a ``str.format`` template.

The prompt shows them empty. :func:`authors_answer` fills them with a task's rationale and target.
"""
_CATEGORY = re.compile(r"^\d+\. ([^:\n]+):", re.MULTILINE)
"""A category the prompt defines: ``12. Sudden Spike Anomaly: ...``."""
_NON_SPACE = re.compile(r"\S")
# The release gives neither units nor a sampling cadence.
_SPEC = TimeSeriesSpec(spec_type="rats40k_series", name="RATS40K series", unit_value=None, dtype="float64")
_AXIS = OrdinalAxis()
_LABEL_DESCRIPTION = "The release's Label, Normal or Anomaly, as published. It can disagree with the target."
_ACTION_ID_DESCRIPTION = "The release's ActionID, as published; it doesn't always follow the prompt's numbering."
_OUTSIDE_PROMPT_DESCRIPTION = "A published target name that is not one of this variant's prompt categories."


@dataclass(frozen=True)
class _Prompts:
    """One variant's prompts, read from ``prompts/<variant>.txt``."""

    template: str
    """The authors' full evaluation prompt, a ``str.format`` template with ``{our_source}`` and
    ``{our_observation}``."""
    task_prompt: str
    """The template without the domain clause and the ``Observation:`` line, which a task's input
    annotation and Record supply."""
    categories: tuple[str, ...]
    """The category names the prompt defines, in the authors' code order."""


@cache
def _prompts(variant: _Variant) -> _Prompts:
    """Read a variant's prompt file and derive the task prompt and the category names from it.

    Returns:
        The variant's prompts.

    Raises:
        TimeFValidationError: If the prompt defines no category, or does not end with the empty answer
            fields of :data:`_ANSWER_FORMAT`.
    """
    template = (_PROMPTS / f"{variant.value}.txt").read_text(encoding="utf-8")
    # ``format`` resolves the template's ``{{}}`` escapes, and fails if a placeholder is left.
    task_prompt = _DOMAIN_CLAUSE.sub("", template).replace(_OBSERVATION_LINE, "").format().rstrip("\n")
    categories = tuple(_CATEGORY.findall(task_prompt))
    if not categories:
        raise TimeFValidationError(f"{variant.value} prompt defines no category")
    # authors_answer fills these fields, so the answer has the format that the prompt asks for.
    if not task_prompt.endswith(_ANSWER_FORMAT.format(thought="", action="")):
        raise TimeFValidationError(f"{variant.value} prompt does not end with the answer fields Thought and Action")
    return _Prompts(template, task_prompt, categories)


def _label_set(variant: _Variant, targets: Iterable[str]) -> LabelSet:
    """Build a variant's label set: its prompt categories, then the other names its targets use.

    The prompt's categories come first, in the authors' code order, so a category's index in the set
    is its code. Some published targets are not prompt categories, such as the multivariate names of
    univariate test rows. They follow in sorted order, so that every row keeps its task.

    Args:
        variant: The variant whose prompt lists the categories.
        targets: The published target names of the variant's rows.

    Returns:
        The label set.
    """
    categories = _prompts(variant).categories
    others = sorted(set(targets) - set(categories))
    return LabelSet(
        name=variant.label_set_name,
        labels=(
            *(Label(name=name) for name in categories),
            *(Label(name=name, description=_OUTSIDE_PROMPT_DESCRIPTION) for name in others),
        ),
    )


@dataclass(frozen=True, eq=False)
class _Row:
    """A published row, validated, with its observation's channel lengths and digest but not its values.

    ``eq=False`` makes a row hash by identity, so a cache can key on it.
    """

    path: Path
    variant: _Variant
    split: Split
    key: str
    """The row's key in its file. Keys repeat across splits, so variant, split and key name the row."""
    offset: int
    length: int
    """The byte range of the row in its file, which the signal loaders reread."""
    domain: str
    figure_path: str
    thought: str
    action: str
    """The category name. It may not be one of its variant's prompt categories, as published."""
    action_id: int
    label: _Label
    rank: list[str] | None
    channel_lengths: dict[str, int]
    """Each channel's length, in source order."""
    values_digest: bytes
    """The digest of the channel names and values, to tell a reread of the same values from another."""

    @property
    def context(self) -> str:
        """Where the row is, for error messages: its file, split and key."""
        return f"{self.path} ({self.split}, key {self.key})"


class _Entry(NamedTuple):
    """One entry of a file, as decoded and not yet validated."""

    split: Split
    key: str
    value: Any
    offset: int
    length: int


class SourceInventory(NamedTuple):
    """One file's rows and null entries at the pinned revision."""

    rows: Mapping[Split, int]
    """The number of usable rows in each split, without the null entries."""
    null_keys: Mapping[Split, frozenset[str]]
    """The keys of the null entries in each split. The connector skips these entries."""

    def check(self, path: Path, found: "SourceInventory") -> None:
        """Compare the rows and null entries found in a file with this inventory.

        Args:
            path: The file, for the error message.
            found: What the connector found in the file, with every split.

        Raises:
            TimeFValidationError: If the counts or the null keys differ, so the file is not the pinned one.
        """
        if dict(found.rows) != dict(self.rows) or dict(found.null_keys) != dict(self.null_keys):
            raise TimeFValidationError(
                f"{path}: the rows differ from the pinned inventory: expected {self._text()}, found {found._text()}"
            )

    def _text(self) -> str:
        """Describe the inventory for an error message.

        Returns:
            The rows and null keys of each split, such as ``test: 1 rows, null keys ['7']``.
        """
        return ", ".join(
            f"{split}: {self.rows.get(split, 0)} rows, null keys {sorted(self.null_keys.get(split, ()))}"
            for split in sorted({*self.rows, *self.null_keys})
        )


# The eleven null entries are univariate test keys that the authors' labeling/rank_models.py starts as
# None and never fills.
_PINNED_INVENTORY: dict[str, SourceInventory] = {
    _Variant.UNI: SourceInventory(
        rows={Split.TRAIN: 30_266, Split.TEST: 6_034},
        null_keys={
            Split.TRAIN: frozenset(),
            Split.TEST: frozenset({"106", "386", "560", "875", "1632", "2081", "2581", "2644", "2721", "3584", "5558"}),
        },
    ),
    _Variant.MULTI: SourceInventory(
        rows={Split.TRAIN: 2_728, Split.TEST: 546},
        null_keys={Split.TRAIN: frozenset(), Split.TEST: frozenset()},
    ),
}


class _Converted(NamedTuple):
    """A row with the Record it became."""

    row: _Row
    record: Record


def _iter_entries(path: Path) -> Iterator[_Entry]:
    """Walk a file's ``{split: {key: row}}`` objects without keeping the decoded rows.

    Yields:
        Each entry, in file order.

    Raises:
        TimeFValidationError: If the document or its split structure is malformed, with the file and the
            split or key in the message.
    """
    context = str(path)
    try:
        with JsonCursor.open(path) as cursor:
            splits: set[_SourceSplit] = set()
            for name in cursor.object_keys():
                if name not in set(_SourceSplit):
                    raise TimeFValidationError(f"unknown split {name!r}")
                split = _SourceSplit(name)
                splits.add(split)
                context = f"{path} ({split})"
                for key in cursor.object_keys():
                    context = f"{path} ({split}, key {key})"
                    if not key.isascii() or not key.isdecimal():
                        raise TimeFValidationError(f"expected a decimal row key, got {key!r}")
                    yield _Entry(split.timenet_split, key, *cursor.value())
                    context = f"{path} ({split})"
                context = str(path)
            if splits != set(_SourceSplit):
                raise TimeFValidationError(f"expected {' and '.join(_SourceSplit)}")
            cursor.expect_end()
    except TimeFValidationError as exc:
        raise TimeFValidationError(f"{context}: {exc}") from exc


def _is_finite_number(value: Any) -> bool:
    """Tell whether a JSON value is a finite number, not a boolean.

    Returns:
        Whether the value is a finite int or float.
    """
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _series(name: str, values: Any) -> list[float]:
    """Check one channel: a nonempty list of finite numbers.

    Returns:
        The values as floats.

    Raises:
        ValueError: If the channel is not a nonempty list of finite numbers.
    """
    if not isinstance(values, list) or not values or not all(_is_finite_number(value) for value in values):
        raise ValueError(f"channel {name!r} must be a nonempty list of finite numbers")
    return [float(value) for value in values]


def _skip_space(text: str, position: int) -> int:
    """Return the position of the first character at or after ``position`` that is not whitespace."""
    match = _NON_SPACE.search(text, position)
    return len(text) if match is None else match.start()


def _named_channels(observation: Any) -> dict[str, Any]:
    """Split a multivariate observation, ``"name": [values]; ...``, into its channels.

    Names are JSON strings, so a ``;`` inside a name is not a separator.

    Returns:
        The decoded values by channel name, in source order.

    Raises:
        ValueError: If the observation is not text, or its names, colons or separators are wrong.
    """
    if not isinstance(observation, str):
        raise ValueError("a multivariate Observation is a string of named channels")
    decoder = json.JSONDecoder()
    text = observation.strip()
    channels: dict[str, Any] = {}
    position = 0
    while position < len(text):
        name, position = decoder.raw_decode(text, position)
        position = _skip_space(text, position)
        if not isinstance(name, str) or not name.strip() or text[position : position + 1] != ":":
            raise ValueError("expected a nonblank channel name followed by ':'")
        if name in channels:
            raise ValueError(f"duplicate channel {name!r}")
        channels[name], position = decoder.raw_decode(text, _skip_space(text, position + 1))
        position = _skip_space(text, position)
        if position < len(text):
            if text[position] != ";":
                raise ValueError("expected ';' between channels")
            position = _skip_space(text, position + 1)
            if position == len(text):
                raise ValueError("expected a channel after the last ';'")
    return channels


def _channels(variant: _Variant, observation: Any) -> dict[str, list[float]]:
    """Decode an observation: a list of numbers (uni), or a string of named lists (multi).

    Returns:
        The channels in source order, nonempty and equally long.

    Raises:
        ValueError: If the observation violates its variant's format.
    """
    named = {_UNI_CHANNEL: observation} if variant is _Variant.UNI else _named_channels(observation)
    if not named:
        raise ValueError("the Observation has no channels")
    channels = {name: _series(name, values) for name, values in named.items()}
    if len({len(values) for values in channels.values()}) != 1:
        raise ValueError("the channels differ in length")
    return channels


def _values_digest(channels: Mapping[str, list[float]]) -> bytes:
    """Digest the channel names and values, in order, as float64 bytes.

    Returns:
        A 16-byte digest that differs when any name, value or the order of the channels differs.
    """
    digest = hashlib.blake2b(digest_size=16)
    for name, values in channels.items():
        digest.update(name.encode("utf-8") + b"\0" + array("d", values).tobytes())
    return digest.digest()


def _text(field: _Field, value: Any) -> str:
    """Check a published field that is nonblank text.

    Returns:
        The text.

    Raises:
        ValueError: If the value is not nonblank text.
    """
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be nonblank text, got {value!r}")
    return value


def _action_id(value: Any) -> int:
    """Check a published ``ActionID``, a nonnegative integer.

    Returns:
        The code.

    Raises:
        ValueError: If the value is not a nonnegative integer. A JSON boolean is one in Python, so it is
            rejected on its own.
    """
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{_Field.ACTION_ID} must be a nonnegative integer, got {value!r}")
    return value


def _rank(value: Any) -> list[str] | None:
    """Check a published ``Rank``, which test rows may carry: null, or a list of model names.

    Returns:
        The names, or ``None`` for null.

    Raises:
        ValueError: If the value is neither null nor a list of nonblank text.
    """
    if value is None:
        return None
    if not isinstance(value, list):
        raise ValueError(f"{_Field.RANK} must be a list, got {value!r}")
    return [_text(_Field.RANK, name) for name in value]


def _parse_row(path: Path, variant: _Variant, entry: _Entry) -> _Row | None:
    """Validate one entry against its variant.

    Returns:
        The row, or ``None`` for a published null entry, which the connector skips.

    Raises:
        TimeFValidationError: If the entry is not an object, lacks a field, has another field, or holds a
            bad value, with the file, split and key.
    """  # ruff: ignore[docstring-missing-exception] - the ValueErrors are caught and re-raised as TimeFValidationError
    if entry.value is None:
        return None
    context = f"{path} ({entry.split}, key {entry.key})"
    try:
        published = entry.value
        if not isinstance(published, dict):
            raise ValueError("expected a row object")
        required = {field for field in _Field if field is not _Field.RANK}
        if missing := required - published.keys():
            raise ValueError(f"missing fields {sorted(missing)}")
        if unknown := published.keys() - set(_Field):
            raise ValueError(f"unknown fields {sorted(unknown)}")
        label = published[_Field.LABEL]
        if not isinstance(label, str) or label not in set(_Label):
            raise ValueError(f"{_Field.LABEL} must be one of {[label.value for label in _Label]}")
        channels = _channels(variant, published[_Field.OBSERVATION])
        return _Row(
            path=path,
            variant=variant,
            split=entry.split,
            key=entry.key,
            offset=entry.offset,
            length=entry.length,
            domain=_text(_Field.DOMAIN, published[_Field.DOMAIN]),
            figure_path=_text(_Field.FIGURE_PATH, published[_Field.FIGURE_PATH]),
            thought=_text(_Field.THOUGHT, published[_Field.THOUGHT]),
            action=_text(_Field.ACTION, published[_Field.ACTION]),
            action_id=_action_id(published[_Field.ACTION_ID]),
            label=_Label(label),
            rank=_rank(published.get(_Field.RANK)),
            channel_lengths={name: len(values) for name, values in channels.items()},
            values_digest=_values_digest(channels),
        )
    # An integer too long for a float raises OverflowError, and deep nesting RecursionError.
    except (ValueError, OverflowError, RecursionError) as exc:
        raise TimeFValidationError(f"{context}: {exc}") from exc


def _read_channels(row: _Row) -> dict[str, list[float]]:
    """Reread a row's channels from its file and check them against the converted row.

    Returns:
        The channels in source order.

    Raises:
        TimeFFormatError: If the range no longer holds the converted row's channels, as when the file changed.
    """  # ruff: ignore[docstring-missing-exception] - the ValueErrors are caught and re-raised as TimeFFormatError
    value = read_entry(row.path, row.offset, row.length)
    try:
        if not isinstance(value, dict):
            raise ValueError("expected a row object")
        channels = _channels(row.variant, value.get(_Field.OBSERVATION))
    except (ValueError, OverflowError, RecursionError) as exc:
        raise TimeFFormatError(f"{row.context}: {exc}; the file changed since conversion") from exc
    lengths = {name: len(values) for name, values in channels.items()}
    if lengths != row.channel_lengths or _values_digest(channels) != row.values_digest:
        raise TimeFFormatError(f"{row.context}: the channels differ from the converted row; the file changed")
    return channels


class _ChannelLoader:
    """Load one channel of a row lazily, rereading each row once for all its channels.

    A channel loads only if its row's channel names, lengths and values digest equal those kept at
    conversion, so a changed file fails instead of writing other values.
    """

    def __init__(self) -> None:
        # A Record's signals load one after another, so one cached row suffices.
        self._reread = lru_cache(maxsize=1)(_read_channels)

    def __call__(self, row: _Row, name: str) -> pa.Array:
        """Return one channel of a row as float64 values.

        Returns:
            The channel's values.
        """
        return pa.array(self._reread(row)[name], type=pa.float64())


def _record(row: _Row, loader: _ChannelLoader) -> Record:
    """Build a row's Record: one Source with a lazy signal per channel, and the row's domain annotation.

    Returns:
        The Record, whose values stay in the source file until read.
    """
    signals = tuple(
        Signal.from_loader(name=name, spec=_SPEC, time_axis=_AXIS, n_values=length, loader=partial(loader, row, name))
        for name, length in row.channel_lengths.items()
    )
    source = Source(
        name=row.domain,
        signals=signals,
        # TimeF does not keep signal order on read, so record the source's channel order.
        metadata={_Key.CHANNEL_ORDER: list(row.channel_lengths)},
    )
    # Provenance only: the task reads the domain, so it alone is an annotation.
    metadata: dict[str, JsonValue] = {
        _Key.VARIANT: row.variant.value,
        _Key.SPLIT: row.split.value,
        _Key.ROW_KEY: row.key,
    }
    record = Record(sources=(source,), metadata=metadata)
    record.annotate(Annotation(key=_Key.SOURCE_DOMAIN, value=row.domain))
    return record


def _task(row: _Row, record: Record, label_set: LabelSet) -> ClassificationTask:
    """Build a row's task: name the series' anomaly category, as the authors' prompt asks.

    The authors ask one question per row and score detection, category and reasoning from one answer. So
    one task carries all three: the name as target, the ``Thought`` as rationale, and ``Label`` and
    ``ActionID`` as task annotations, which are never model input.

    Args:
        row: The published row.
        record: The row's Record.
        label_set: The label set of the row's variant.

    Returns:
        The classification task.
    """
    metadata: dict[str, Any] = {
        # False for the univariate test rows labelled with multivariate names.
        _Key.TARGET_IN_VOCABULARY: row.action in _prompts(row.variant).categories,
        # In univariate test, a .pdf path marks the rows where gpt-4o ranks first, so it stays off the
        # input Record.
        _Key.FIGURE_PATH: row.figure_path,
    }
    if row.rank is not None:
        metadata[_Key.RANK] = row.rank
    task = ClassificationTask(
        inputs=(record,),
        prompt=_prompts(row.variant).task_prompt,
        targets=(row.action,),
        rationale=row.thought,
        split=row.split,
        input_annotations=tuple(a for a in record.annotations if a.key == _Key.SOURCE_DOMAIN),
        label_set=label_set,
        metadata=metadata,
    )
    # The authors score detection and category codes against these.
    task.annotate(Annotation(key=_Key.LABEL, value=row.label.value, description=_LABEL_DESCRIPTION))
    task.annotate(Annotation(key=_Key.ACTION_ID, value=row.action_id, description=_ACTION_ID_DESCRIPTION))
    return task


def authors_answer(task: Task) -> str:
    r"""Format a task's rationale and target as the answer that the prompt asks for.

    The prompt ends with the empty fields ``Thought: \boxed1{}`` and ``Action: \boxed2{}``. This fills
    them with the rationale and the category name. It is also the target that the authors fine-tune on
    (``train/train_sft_parser.py`` at Time-RA ``200b244``).

    Args:
        task: A RATS40K task, with its rationale and one target.

    Returns:
        ``Thought: \boxed1{<rationale>}``, a newline, and ``Action: \boxed2{<target>}``.

    Raises:
        TimeFValidationError: If the task has no rationale or not exactly one target.
    """
    if task.rationale is None or task.targets is None or len(task.targets) != 1:
        raise TimeFValidationError("a RATS40K task needs a rationale and exactly one target")
    (target,) = task.targets
    return _ANSWER_FORMAT.format(thought=read_value(task.rationale), action=read_value(target))


class RATS40KConnector(BaseHuggingFaceConnector):
    """Build RATS40K from the pinned, gated JSON files of its Hugging Face release."""

    HF_REPO = "Time-RA/RATs40K"
    SOURCE_REVISION = "dc968d6a88f396215be50023c93c3928e06096df"
    SOURCE_FILES: ClassVar[Mapping[str, str]] = {variant.value: variant.source_file for variant in _Variant}
    SOURCE_INVENTORY: ClassVar[Mapping[str, SourceInventory]] = _PINNED_INVENTORY
    """Each file's rows and null entries at :attr:`SOURCE_REVISION`, by variant."""

    def convert(self, raw_refs: list[HubFileRef]) -> TimeFDataset:
        """Convert each file into one lazy Record per usable row, and a stream of one task per Record.

        Each file is read once up front to check every row and keep its small fields. The series values
        stay in the files until written, so the files must stay available until then. Only explicit null
        entries are skipped. Each file's rows and null entries must match :attr:`SOURCE_INVENTORY`.

        Args:
            raw_refs: The downloaded files, one per variant.

        Returns:
            A dataset with the provenance and prompt annotations, the Records, and a stream of every task.

        Raises:
            TimeFValidationError: If a file's structure or a row is malformed, or a file's rows differ from
                its inventory.
        """  # ruff: ignore[docstring-extraneous-exception] - raised by the helpers it calls
        loader = _ChannelLoader()
        variants = [_Variant(ref.part) for ref in raw_refs]
        converted: list[_Converted] = []
        for ref, variant in zip(raw_refs, variants, strict=True):
            splits = [split.timenet_split for split in _SourceSplit]
            rows = dict.fromkeys(splits, 0)
            null_keys: dict[Split, set[str]] = {split: set() for split in splits}
            for entry in _iter_entries(ref.path):
                if (row := _parse_row(ref.path, variant, entry)) is None:
                    null_keys[entry.split].add(entry.key)
                else:
                    rows[entry.split] += 1
                    converted.append(_Converted(row, _record(row, loader)))
            found = SourceInventory(rows, {split: frozenset(keys) for split, keys in null_keys.items()})
            self.SOURCE_INVENTORY[variant].check(ref.path, found)

        dataset = TimeFDataset(metadata=self.metadata())
        dataset.annotate(Annotation(key=_Key.SOURCE_REPOSITORY, value=self.HF_REPO))
        dataset.annotate(Annotation(key=_Key.SOURCE_REVISION, value=self.SOURCE_REVISION))
        for _, record in converted:
            dataset.add_record(record=record)
        for variant in variants:
            dataset.annotate(Annotation(key=variant.template_key, value=_prompts(variant).template))
        label_sets = {
            variant: _label_set(variant, (row.action for row, _ in converted if row.variant is variant))
            for variant in variants
        }

        def tasks() -> Iterator[ClassificationTask]:
            for row, record in converted:
                yield _task(row, record, label_sets[row.variant])

        dataset.set_task_stream((ClassificationTask,), tasks)
        return dataset


CONNECTOR = RATS40KConnector
