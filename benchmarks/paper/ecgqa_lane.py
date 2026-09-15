"""Both sides of the ECG-QA CoT row: PTB-XL WFDB records beside the CoT CSVs, and the TimeF version.

The canonical item is one QA row with the 12-lead recording it asks about. There are 231,543 rows
over 16,024 recordings, and the rows carry no locality: 231,528 of the 231,543 rows sit next to a
row about a different recording. So a full read decodes each recording about 14 times, and the
declared rule that a representation holds at most one decoded recording at a time neither costs nor
saves this lane anything in canonical order.

Both sides live here because the item does. The Original lane cuts it out of the CoT CSVs and the
WFDB records; the TimeF lane cuts it out of the task partitions and the records table. They build
the delivered value through the same :class:`QaRow`, so the two cannot drift into delivering
different items while still reporting the same count.

The release ships a loader and this lane replaces it, which the methodology declares. OpenTSLM's
``ECGQACoTQADataset.preload_ecg_data`` fills a class-level dict with ``wfdb.rdrecord(...).p_signal``
for every referenced recording before it formats an item: 16,024 x 5,000 x 12 float64 is 7.69 GB,
which no resident-set cap in this campaign admits. What replaces it is the two libraries that
release and PTB-XL's own ``example_physionet.py`` already use, wfdb for the signals and pandas or
the ``csv`` module for the tables.

Both variants deliver float32 millivolts, the dtype the connector stores. wfdb reads the same int16
samples either way; what differs is whether it converts them through float64 first. Delivering
wfdb's native float64 would double this lane's value plane against a TimeF side that holds float32,
which measures a dtype rather than a format.

The CoT ``prompt`` column names a two-way candidate pair per question ("This question has one of
two possible answers"), and the gold answer is always one of the two. There are 1,029 distinct
unordered pairs over the 231,543 rows. The Original lane parses the pair out of the prompt; the
connector stores it as an ``answer_options`` annotation, one row per distinct pair, and the TimeF
lane reads it back from there. Both sort it, because the same two answers appear in both orders
across the corpus and the pair is a set of options rather than an order.
"""

from __future__ import annotations

from array import array
from collections.abc import Generator, Iterator, Mapping, Sequence
import csv
from dataclasses import dataclass, field
import importlib
import json
import math
from pathlib import Path
import re
from typing import Any, Protocol

import numpy as np

from benchmarks.paper.errors import LaneError
from benchmarks.paper.lanes import Buffer, DeliveredItem, array_buffers, array_frame, as_buffer
from benchmarks.paper.record import AS_SHIPPED, LAZY_CAPABLE, NATIVE, ORIGINAL, PANDAS, TIMEF, TORCH


DATASET = "physionet/ecg-qa-cot"
"""The dataset id both representations of this row carry."""

SPLIT_FILES: tuple[tuple[str, str], ...] = (
    ("train", "ecg_qa_cot_train.csv"),
    ("validation", "ecg_qa_cot_val.csv"),
    ("test", "ecg_qa_cot_test.csv"),
)
"""The CoT splits, in the order the connector streams them, which is the canonical order."""

COT_DIRNAME = "ecg_qa_cot"
TEMPLATE_ANSWERS_CSV = "answers_for_each_template.csv"
RECORDS_DIRNAME = "records500"
PTBXL_MARKER = "ptbxl_database.csv"

N_CANDIDATES = 2
"""Candidates a CoT question offers. Every one of the 231,543 rows offers exactly two."""

DEFAULT_CONTEXT = "12-lead ECG recording."
"""What the connector puts in for a row whose clinical context is empty."""

LIBRARIES = {AS_SHIPPED: "wfdb+pandas", LAZY_CAPABLE: "wfdb+csv"}
"""What reads the tables in each variant. wfdb reads the signals in both."""

_CANDIDATE_BLOCK = re.compile(r"This question has one of two possible answers:\n((?:-[^\n]*\n?)+)")
_ITEM_ID = re.compile(r"ecgqa-(?P<split>[a-z]+)-(?P<index>\d+)\Z")
_SPLIT_RANK = {split: rank for rank, (split, _) in enumerate(SPLIT_FILES)}


@dataclass(frozen=True)
class RawLayout:
    """Where the four kinds of raw file live under the download cache.

    Attributes:
        records_root: The PTB-XL ``records500`` directory holding the ``.dat`` / ``.hea`` pairs.
        answers_csv: The per-template answer options.
        split_csvs: One ``(split, path)`` pair per CoT split, in canonical order.
    """

    records_root: Path
    answers_csv: Path
    split_csvs: tuple[tuple[str, Path], ...]

    def record_base(self, ecg_id: int) -> Path:
        """Return a recording's path without its extension.

        Returns:
            The path under the ``records500/<bucket>/`` layout PTB-XL ships.
        """
        return self.records_root / f"{ecg_id // 1000 * 1000:05d}" / f"{ecg_id:05d}_hr"

    def signal_files(self, ecg_ids: Sequence[int]) -> Iterator[Path]:
        """Walk the header and data file of every referenced recording.

        Yields:
            One ``.hea`` and one ``.dat`` path per distinct recording, in id order.
        """
        for ecg_id in sorted(set(ecg_ids)):
            base = self.record_base(ecg_id)
            yield base.with_suffix(".hea")
            yield base.with_suffix(".dat")


def resolve_layout(root: Path) -> RawLayout:
    """Find the raw files under a download cache directory.

    Args:
        root: The cache directory the connector downloaded into.

    Returns:
        The resolved layout.

    Raises:
        LaneError: If a file the lane reads is missing.
    """
    records_root = _ptbxl_root(root) / RECORDS_DIRNAME
    answers_csv = root / TEMPLATE_ANSWERS_CSV
    split_csvs = tuple((split, root / COT_DIRNAME / name) for split, name in SPLIT_FILES)
    for path in (records_root, answers_csv, *(path for _, path in split_csvs)):
        if not path.exists():
            raise LaneError(f"the ECG-QA CoT cache at {root} is missing {path}")
    return RawLayout(records_root=records_root, answers_csv=answers_csv, split_csvs=split_csvs)


def tier_a_bytes(layout: RawLayout, ecg_ids: Sequence[int]) -> int:
    """Return the Tier A bytes of the raw representation.

    Tier A is what the items are read out of: the header and data file of every *referenced*
    recording, the three CoT CSVs and the template answers CSV. PTB-XL ships 21,799 recordings and
    the CoT rows reference 16,024 of them, so a plain tree scan would charge this side 5,775
    recordings no item ever opens.

    Args:
        layout: The resolved raw layout.
        ecg_ids: The recording each canonical item asks about. Duplicates are counted once.

    Returns:
        The byte sum.

    Raises:
        LaneError: If a file is missing.
    """
    tables = [layout.answers_csv, *(path for _, path in layout.split_csvs)]
    try:
        return sum(path.stat().st_size for path in (*layout.signal_files(ecg_ids), *tables))
    except OSError as error:
        raise LaneError(f"could not size the ECG-QA CoT raw files: {error}") from error


def candidate_pair(prompt: str) -> tuple[str, str]:
    """Return the two candidate answers a CoT prompt offers, sorted.

    Sorted, because the pair is a set of options rather than an order: the same two answers appear
    in both orders across the corpus, and sorting collapses the 2,056 ordered pairs to the 1,029
    distinct pairs the questions actually draw from.

    Args:
        prompt: The rendered prompt from the CoT CSV.

    Returns:
        The two candidates in sorted order.

    Raises:
        LaneError: If the prompt names no candidate block, or names other than two candidates.
    """
    match = _CANDIDATE_BLOCK.search(prompt)
    if match is None:
        raise LaneError("a CoT prompt names no candidate answers, so its question is not a two-way choice")
    options = tuple(line[1:].strip() for line in match.group(1).strip().split("\n"))
    if len(options) != N_CANDIDATES:
        raise LaneError(f"a CoT prompt offers {len(options)} candidates, expected {N_CANDIDATES}: {options}")
    first, second = sorted(options)
    return first, second


@dataclass(frozen=True)
class QaRow:
    """One CoT row, in the fields both representations of the item carry.

    Attributes:
        item_id: The task id the connector writes, and the identity the two sides share.
        ecg_id: The PTB-XL recording the question asks about.
        question: The question text.
        answer: The gold answer, always one of the two candidates.
        question_type: The ECG-QA question type.
        template_id: The ECG-QA template the question was rendered from.
        clinical_context: The recording's clinical context.
        candidates: The two candidate answers, sorted.
        rationale: The chain-of-thought target.
    """

    item_id: str
    ecg_id: int
    question: str
    answer: str
    question_type: str
    template_id: int
    clinical_context: str
    candidates: tuple[str, str]
    rationale: str

    @property
    def record_id(self) -> str:
        """Return the TimeF record id of the recording this row asks about.

        Returns:
            The connector's ``ptbxl-<ecg id>`` id.
        """
        return f"ptbxl-{self.ecg_id}"

    def as_mapping(self) -> dict[str, Any]:
        """Return the row as the columns the pandas consumer builds a frame from.

        Returns:
            One entry per field, with the candidate pair spread over two columns.
        """
        return {
            "item_id": self.item_id,
            "record_id": self.record_id,
            "question": self.question,
            "answer": self.answer,
            "question_type": self.question_type,
            "template_id": self.template_id,
            "clinical_context": self.clinical_context,
            "candidate_a": self.candidates[0],
            "candidate_b": self.candidates[1],
            "rationale": self.rationale,
        }

    def text(self) -> str:
        """Return the same fields as one string, for a consumer with no frame to put them in.

        Returns:
            The values of :meth:`as_mapping`, unit-separated.
        """
        return "\x1f".join(str(value) for value in self.as_mapping().values())


def qa_row(item_id: str, row: Mapping[str, Any]) -> QaRow:
    """Read one raw CoT row into the fields both representations carry.

    Args:
        item_id: The task id of this ordinal.
        row: The raw row, keyed by CSV column. Values arrive as strings from the ``csv`` module and
            as parsed objects from pandas, so every field is normalized here rather than at the
            reader, and the two loader variants deliver the same item.

    Returns:
        The parsed row.

    Raises:
        LaneError: If a column is missing or cannot be read.
    """
    try:
        return QaRow(
            item_id=item_id,
            ecg_id=_ecg_id(row["ecg_id"]),
            question=_text(row["question"]),
            answer=_text(row["answer"]),
            question_type=_text(row["question_type"]),
            template_id=int(float(row["template_id"])),
            clinical_context=_text(row["clinical_context"]) or DEFAULT_CONTEXT,
            candidates=candidate_pair(_text(row["prompt"])),
            rationale=_text(row["rationale"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise LaneError(f"CoT row {item_id} could not be read: {error}") from error


@dataclass(frozen=True)
class Recording:
    """One decoded PTB-XL recording.

    Attributes:
        ecg_id: The recording id.
        leads: One contiguous float32 array of millivolts per lead.
        signals: The lead names, in the order the header lists them.
        sampling_hz: The rate the header declares.
    """

    ecg_id: int
    leads: tuple[np.ndarray, ...]
    signals: tuple[str, ...]
    sampling_hz: int


@dataclass
class SignalReader:
    """Decodes PTB-XL recordings with wfdb, holding at most one at a time.

    One decoded recording is the declared rule for this row, and it is the only reason a lane over
    231,543 questions about 16,024 recordings has a bounded resident set.

    Attributes:
        layout: Where the recordings live.
        return_res: Bit width wfdb converts to. 64 is its default and what the release calls; 32
            skips the float64 intermediate and lands on the dtype this lane delivers.
        held: The recording currently decoded, if any.
    """

    layout: RawLayout
    return_res: int = 64
    held: Recording | None = None

    def recording(self, ecg_id: int) -> Recording:
        """Return one recording, decoding it unless it is the one already held.

        Args:
            ecg_id: The recording to read.

        Returns:
            The decoded recording, as float32 millivolts.

        Raises:
            LaneError: If wfdb cannot read the record.
        """
        if self.held is not None and self.held.ecg_id == ecg_id:
            return self.held
        # The lane's own reader, imported once per lane rather than at module import.
        import wfdb  # noqa: PLC0415

        base = self.layout.record_base(ecg_id)
        try:
            signal, fields = wfdb.rdsamp(str(base), return_res=self.return_res)
        except (OSError, ValueError) as error:
            raise LaneError(f"wfdb could not read record {base}: {error}") from error
        self.held = Recording(
            ecg_id=ecg_id,
            leads=tuple(np.ascontiguousarray(signal[:, lead], dtype=np.float32) for lead in range(signal.shape[1])),
            signals=tuple(str(name) for name in fields["sig_name"]),
            sampling_hz=int(fields["fs"]),
        )
        return self.held

    def close(self) -> None:
        """Drop the recording being held."""
        self.held = None


class RowSource(Protocol):
    """Where a handle reads its CoT rows. The two loader variants differ in this and nothing else.

    A source is addressed by task id rather than by the lane's ordinal, so a lane over part of the
    corpus reads the rows its ids name rather than the rows at those positions.
    """

    def row(self, item_id: str) -> Mapping[str, Any]:
        """Return one raw CoT row."""
        ...

    def close(self) -> None:
        """Release whatever the source opened."""
        ...


@dataclass
class EagerRows:
    """The whole corpus of CoT rows, parsed into one pandas table before the first item.

    This is the ``as_shipped`` shape. The release reads each split with ``pandas.read_csv`` and
    walks the frame row by row, and PTB-XL's own example does the same with its metadata table. The
    cost lands where the first-item column can see it: 650 MB of CSV is parsed before ordinal zero.

    The splits stay three frames rather than one. Concatenating them would copy 2 GB of parsed
    strings to buy nothing: a task id already names its split.

    Attributes:
        frames: One parsed table per split, in canonical order.
    """

    frames: tuple[Any, ...]

    @classmethod
    def load(cls, layout: RawLayout) -> EagerRows:
        """Read every split with pandas.

        Returns:
            The loaded source.

        Raises:
            LaneError: If pandas is missing or a split cannot be parsed.
        """
        # The lane's own consumer, imported once per lane rather than at module import.
        import pandas as pd  # noqa: PLC0415

        try:
            return cls(frames=tuple(pd.read_csv(path) for _, path in layout.split_csvs))
        except (OSError, ValueError) as error:
            raise LaneError(f"pandas could not read the CoT splits: {error}") from error

    def row(self, item_id: str) -> Mapping[str, Any]:
        """Return one row of the tables.

        Returns:
            The row, keyed by column.

        Raises:
            LaneError: If the id names a row the tables do not hold.
        """
        split, index = row_position(item_id)
        try:
            return self.frames[split].iloc[index]
        except IndexError as error:
            raise LaneError(f"{item_id} is past the end of its split") from error

    def close(self) -> None:
        """Drop the tables."""
        self.frames = ()


@dataclass
class StreamRows:
    """CoT rows read one at a time, with a byte offset per row built only when one is needed.

    This is the ``lazy_capable`` shape. Reaching the first item costs one row, not one corpus. A
    pass that walks a split in order rides the cursor and never builds an offset index. Only a pass
    that jumps, which is the block-shuffled one, pays one scan of that split for an ``array`` of
    offsets: 1.3 MB for the 159,313 train rows, against the roughly 2 GB the parsed table costs.

    Attributes:
        split_csvs: One ``(split, path)`` pair per split, in canonical order.
        headers: The column names of each split.
    """

    split_csvs: tuple[tuple[str, Path], ...]
    headers: tuple[tuple[str, ...], ...]
    _next: tuple[int, int] | None = None
    _rows: Generator[Mapping[str, Any], None, None] | None = None
    _offsets: list[array[int] | None] = field(default_factory=list)

    @classmethod
    def open(cls, layout: RawLayout) -> StreamRows:
        """Open the splits and read their header lines.

        Args:
            layout: Where the CoT CSVs live.

        Returns:
            The opened source.

        Raises:
            LaneError: If a split cannot be read.
        """
        try:
            headers = tuple(_header_of(path) for _, path in layout.split_csvs)
        except OSError as error:
            raise LaneError(f"could not read a CoT split header: {error}") from error
        return cls(split_csvs=layout.split_csvs, headers=headers, _offsets=[None] * len(headers))

    def row(self, item_id: str) -> Mapping[str, Any]:
        """Return one CoT row, advancing the cursor or repositioning it.

        Returns:
            The row, keyed by column.

        Raises:
            LaneError: If the id names a row its split does not hold.
        """
        position = row_position(item_id)
        if self._rows is None or position != self._next:
            self._rows = self._rows_from(position)
        row = next(self._rows, None)
        if row is None:
            raise LaneError(f"{item_id} is past the end of its split")
        self._next = (position[0], position[1] + 1)
        return row

    def close(self) -> None:
        """Close the open split and drop the offset indexes."""
        if self._rows is not None:
            self._rows.close()
            self._rows = None
        self._offsets = [None] * len(self.headers)

    def _rows_from(self, position: tuple[int, int]) -> Generator[Mapping[str, Any], None, None]:
        """Walk one split's rows from one row to the end of that split.

        Yields:
            One raw row per record.
        """
        split, index = position
        offset = 0 if index == 0 else self._offset_of(split, index)
        yield from _iter_rows(self.split_csvs[split][1], self.headers[split], offset)

    def _offset_of(self, split: int, index: int) -> int:
        """Return the byte offset a row starts at, scanning its split once if that is not known.

        Returns:
            The offset inside the split.

        Raises:
            LaneError: If the split holds no such row.
        """
        offsets = self._offsets[split]
        if offsets is None:
            offsets = array("q", _row_offsets(self.split_csvs[split][1]))
            self._offsets[split] = offsets
        try:
            return offsets[index]
        except IndexError as error:
            raise LaneError(f"{self.split_csvs[split][0]} holds no row {index}") from error


@dataclass
class EcgQaCotHandle:
    """An open ECG-QA CoT lane, delivering QA rows with their recordings as one consumer's items.

    Attributes:
        rows: Where the CoT rows come from, which is what the loader variant chooses.
        signals: The wfdb reader, holding one decoded recording.
        item_ids: The canonical enumeration, one task id per ordinal.
        consumer: ``pandas`` or ``torch``.
        recordings: Per delivered ordinal, the recording the item read. Filled as items go out and
            read back after the clock stops, so a group switch here is a re-decode.
    """

    rows: RowSource
    signals: SignalReader
    item_ids: tuple[str, ...]
    consumer: str
    recordings: dict[int, str] = field(default_factory=dict)

    def n_items(self) -> int:
        """Return how many canonical items this lane holds.

        Returns:
            The item count.
        """
        return len(self.item_ids)

    def deliver(self, ordinals: Sequence[int]) -> Iterator[DeliveredItem]:
        """Read one request's worth of QA rows and their recordings.

        Args:
            ordinals: Canonical ordinals to read. They come back in the order asked.

        Yields:
            One item per ordinal.

        Raises:
            LaneError: If an ordinal is outside the enumeration.
        """
        for ordinal in ordinals:
            try:
                item_id = self.item_ids[ordinal]
            except IndexError as error:
                raise LaneError(f"a request asked for an ordinal outside the {len(self.item_ids)} items") from error
            row = qa_row(item_id, self.rows.row(item_id))
            recording = self.signals.recording(row.ecg_id)
            self.recordings[ordinal] = row.record_id
            yield DeliveredItem(ordinal=ordinal, item_id=item_id, buffers=self._buffers(row, recording))

    def group_of(self, ordinal: int) -> object:
        """Return the recording an item was read out of.

        A group is a source file on the Original side, and here the file that matters is the
        recording: a switch is one more ``.dat`` opened and decoded.

        Returns:
            The record id of the recording, or ``None`` when the item was not delivered.
        """
        return self.recordings.get(ordinal)

    def close(self) -> None:
        """Release the table and the recording being held."""
        self.rows.close()
        self.signals.close()

    def _buffers(self, row: QaRow, recording: Recording) -> tuple[Buffer, ...]:
        """Convert one QA row and its recording into the consumer's type.

        Returns:
            One buffer per value plane, which the checksum walks inside the timed region.

        Raises:
            LaneError: If the consumer is not one this lane implements.
        """
        if self.consumer == PANDAS:
            qa = qa_frame(row)
            return (
                *array_buffers(ecg_frame(recording)),
                *(as_buffer(qa[name].to_numpy()) for name in qa.columns),
            )
        if self.consumer == TORCH:
            return (*(as_buffer(tensor.numpy()) for tensor in lead_tensors(recording)), as_buffer(row.text()))
        raise LaneError(f"the ECG-QA CoT lane has no {self.consumer!r} consumer; it implements {PANDAS} and {TORCH}")


@dataclass(frozen=True)
class EcgQaCotLane:
    """One Original lane over the raw ECG-QA CoT files.

    Attributes:
        consumer: ``pandas`` or ``torch``.
        root: The download cache directory holding PTB-XL and the CoT CSVs.
        item_ids: The canonical enumeration, one task id per ordinal.
        tier_a_bytes: Tier A bytes of the raw files, which sizes the blocks.
        loader_variant: ``as_shipped`` or ``lazy_capable``.
        name: The lane name that reaches the cell record.
    """

    consumer: str
    root: Path
    item_ids: tuple[str, ...]
    tier_a_bytes: int
    loader_variant: str = LAZY_CAPABLE
    name: str = ""
    dataset: str = DATASET
    representation: str = ORIGINAL

    def __post_init__(self) -> None:
        """Refuse a lane that cannot be measured.

        Raises:
            LaneError: If the consumer or the variant is unknown, the enumeration is empty, or Tier
                A bytes are not positive.
        """
        if self.consumer not in {PANDAS, TORCH}:
            raise LaneError(f"the ECG-QA CoT lane implements {PANDAS} and {TORCH}, got {self.consumer!r}")
        if self.loader_variant not in LIBRARIES:
            raise LaneError(f"the loader variant must be one of {sorted(LIBRARIES)}, got {self.loader_variant!r}")
        if not self.item_ids:
            raise LaneError("the ECG-QA CoT lane has no items to enumerate")
        if self.tier_a_bytes < 1:
            raise LaneError("the ECG-QA CoT lane needs positive Tier A bytes to size its blocks")
        if not self.name:
            object.__setattr__(self, "name", f"{DATASET}/{ORIGINAL}/{self.consumer}/{LIBRARIES[self.loader_variant]}")

    def preload(self) -> None:
        """Import everything this lane reads with, without opening a file.

        wfdb reads the signals in both variants, ``as_shipped`` reads its tables with pandas, and
        the consumer decides the delivered type. All of them are imported on first use in the normal
        path, which is right for a timed read but wrong for the memory baseline. This pulls them
        forward so the baseline can be taken after them.
        """
        modules = {"wfdb", "timenet.dataset.axis"}
        if self.loader_variant == AS_SHIPPED:
            modules.add("pandas")
        if self.consumer == PANDAS:
            modules.update(("pandas", "timenet.pandas"))
        else:
            modules.add("torch")
        for module in sorted(modules):
            importlib.import_module(module)

    def open(self) -> EcgQaCotHandle:
        """Open the raw files.

        What this costs is the whole point of the two variants: ``as_shipped`` parses 650 MB of CSV
        here, and ``lazy_capable`` reads three header lines.

        Returns:
            The open handle.
        """
        layout = resolve_layout(self.root)
        eager = self.loader_variant == AS_SHIPPED
        rows: RowSource = EagerRows.load(layout) if eager else StreamRows.open(layout)
        signals = SignalReader(layout=layout, return_res=64 if eager else 32)
        return EcgQaCotHandle(rows=rows, signals=signals, item_ids=self.item_ids, consumer=self.consumer)


def ecg_frame(recording: Recording) -> Any:
    """Return one recording as the array-cell frame the TimeF pandas consumer builds.

    One row, one column per lead, each cell that lead's whole float32 array. The shape and the
    column order are ``timenet.pandas.record_array_frame``'s, so the two sides of this row hand the
    checksum the same frame rather than two shapes.

    Args:
        recording: The decoded recording.

    Returns:
        A one-row frame: ``record_id``, ``time_axis``, then one column per lead.
    """
    # Lane dependencies, imported once per lane rather than at module import.
    from timenet.dataset.axis import RegularAxis  # noqa: PLC0415

    axis = RegularAxis.from_rate_hz(recording.sampling_hz)
    return array_frame(
        f"ptbxl-{recording.ecg_id}",
        [(signal, axis, values) for signal, values in zip(recording.signals, recording.leads, strict=True)],
    )


def qa_frame(row: QaRow) -> Any:
    """Return one QA row as a one-row frame.

    Args:
        row: The parsed CoT row.

    Returns:
        A frame of the fields both representations carry, including the candidate pair.
    """
    import pandas as pd  # noqa: PLC0415 - the lane's own consumer

    return pd.DataFrame([row.as_mapping()])


def lead_tensors(recording: Recording) -> tuple[Any, ...]:
    """Return one recording as one float32 tensor per lead.

    The torch consumer ends here: the values go from wfdb's numpy array into tensors and are never
    put in a frame. The QA text has no tensor form without a tokenizer, which is a modelling choice
    and not this lane's, so it rides along as text the way the TimeF item dict carries its task.

    Args:
        recording: The decoded recording.

    Returns:
        One tensor per lead, in header order.
    """
    import torch  # noqa: PLC0415 - the lane's own consumer

    return tuple(torch.from_numpy(lead) for lead in recording.leads)


TASK_GLOB = "tasks/task=answer/part-*.parquet"
"""Where the connector's QA rows live. It writes one task type, so one partition series holds them."""

ANNOTATIONS_GLOB = "annotations/part-*.parquet"
"""Where the deduped question metadata lives, including the contexts and the candidate pairs."""

TASK_COLUMNS = ("id", "record_ids", "prompt", "target", "rationale", "input_annotation_ids")
"""The task columns an item reads. The partition carries three more that no field of the item needs."""

METADATA_KEYS = frozenset({"clinical_context", "answer_options"})
"""The annotation keys an item has to look up. The other two state their value inside their own id."""

RECORD_PREFIX = "ptbxl-"
QTYPE_PREFIX = "ecgqa-qtype-"
TEMPLATE_PREFIX = "ecgqa-template-"
TEMPLATE_OPTIONS_PREFIX = "ecgqa-template-options-"
CONTEXT_PREFIX = "ecgqa-context-"
OPTIONS_PREFIX = "ecgqa-options-"


@dataclass
class TaskRows:
    """The QA rows of a TimeF version, read one task row group at a time.

    The connector streams the rows in canonical order and the writer stores them in the order it
    receives them, so a pass in canonical order walks the partitions forward and decodes each row
    group once. Reaching a row by its id needs a directory of which group holds it, and this builds
    that directory the same way: group by group, reading the ``id`` column alone, and only as far as
    the ids asked for so far. The first item therefore reads one group's ids rather than the
    corpus's, and a whole pass reads each group's ids once.

    The reader's own :meth:`~timenet.reader.TimeFReader.tasks_for_records` is the wrong door here.
    It looks tasks up by record through a task index that names the first and last row group holding
    one of that record's tasks. This connector interleaves the recordings, so one recording's 14
    questions are spread over the whole table and that span covers nearly every group. One lookup
    would decode the whole task table, and 231,543 of them would decode it 231,543 times.

    Attributes:
        root: The committed version directory.
    """

    root: Path
    _codec: Any = None
    _groups: tuple[tuple[str, int], ...] = ()
    _files: dict[str, Any] = field(default_factory=dict)
    _index: dict[str, tuple[int, int]] = field(default_factory=dict)
    _scanned: int = 0
    _held: int = -1
    _table: Any = None

    def row(self, item_id: str) -> Mapping[str, Any]:
        """Return one QA row, decoding its row group unless it is the one already held.

        Args:
            item_id: The task id of this ordinal.

        Returns:
            The row keyed by task column, with the id columns decoded to strings.
        """
        position, offset = self._locate(item_id)
        if position != self._held:
            part, ordinal = self._groups[position]
            self._table = self._file(part).read_row_group(ordinal, columns=list(TASK_COLUMNS))
            self._held = position
        row = self._table.slice(offset, 1).to_pylist()[0]
        return {
            "id": self._codec.decode("task_id", row["id"]),
            "record_ids": self._codec.decode_list("record_id", row["record_ids"]),
            "prompt": row["prompt"],
            "target": row["target"],
            "rationale": row["rationale"],
            "input_annotation_ids": self._codec.decode_list("annotation_id", row["input_annotation_ids"]),
        }

    def task_ids(self) -> tuple[str, ...]:
        """Return every QA row id, in stored order, reading the ``id`` column and nothing else.

        Returns:
            The task ids, which is the canonical enumeration of this representation.
        """
        groups = self._directory()
        return tuple(
            self._codec.decode("task_id", stored) for part, ordinal in groups for stored in self._ids_of(part, ordinal)
        )

    def close(self) -> None:
        """Close the open partitions and drop the directory, the id index and the group held."""
        for handle in self._files.values():
            handle.close()
        self._files.clear()
        self._groups = ()
        self._index = {}
        self._scanned = 0
        self._held = -1
        self._table = None

    def _locate(self, item_id: str) -> tuple[int, int]:
        """Return which row group holds a task id and where in it, scanning forward for a new id.

        Args:
            item_id: The task id to find.

        Returns:
            The group's position in file order, and the row's offset inside that group.

        Raises:
            LaneError: If no partition of this version holds the id.
        """
        groups = self._directory()
        while item_id not in self._index and self._scanned < len(groups):
            part, ordinal = groups[self._scanned]
            for offset, stored in enumerate(self._ids_of(part, ordinal)):
                self._index.setdefault(self._codec.decode("task_id", stored), (self._scanned, offset))
            self._scanned += 1
        try:
            return self._index[item_id]
        except KeyError:
            raise LaneError(f"no task partition under {self.root} holds the QA row {item_id!r}") from None

    def _directory(self) -> tuple[tuple[str, int], ...]:
        """Return one entry per task row group, in file order, read off the footers on first use.

        Returns:
            The ``(partition, row group)`` pairs.

        Raises:
            LaneError: If the version holds no answer-task partition.
        """
        if self._groups:
            return self._groups
        # A lane dependency, imported here rather than at module import.
        from timenet.format.schemas import IdCodec  # noqa: PLC0415

        parts = sorted(self.root.glob(TASK_GLOB))
        if not parts:
            raise LaneError(f"{self.root} holds no {TASK_GLOB} partition, so it carries no QA rows")
        schema = self._file(str(parts[0])).schema_arrow
        self._codec = IdCodec.from_uuid16(
            logical
            for logical, column in (
                ("task_id", "id"),
                ("record_id", "record_ids"),
                ("annotation_id", "input_annotation_ids"),
            )
            if _is_uuid16(schema.field(column).type)
        )
        self._groups = tuple(
            (str(part), ordinal) for part in parts for ordinal in range(self._file(str(part)).metadata.num_row_groups)
        )
        return self._groups

    def _ids_of(self, part: str, ordinal: int) -> list[Any]:
        """Return one row group's stored task ids, in row order.

        Returns:
            The stored ids, still in their on-disk form.
        """
        return self._file(part).read_row_group(ordinal, columns=["id"]).column("id").to_pylist()

    def _file(self, part: str) -> Any:
        """Return the open handle for one task partition, opening it on first use.

        Returns:
            The cached handle. :meth:`close` closes it.
        """
        handle = self._files.get(part)
        if handle is None:
            # A lane dependency, imported here rather than at module import.
            import pyarrow.parquet as pq  # noqa: PLC0415

            handle = pq.ParquetFile(part)
            self._files[part] = handle
        return handle


@dataclass
class MetadataAnnotations:
    """The deduped question metadata a QA task references, read once and held.

    The connector stores the clinical context and the candidate pair as annotations, one row per
    distinct value rather than one per question, so this table holds thousands of rows against
    231,543 questions. It is read whole on the first item that needs it, and every later item is a
    dict hit.

    Attributes:
        root: The committed version directory.
    """

    root: Path
    _values: dict[str, Any] | None = None

    def value(self, annotation_id: str) -> Any:
        """Return the value of one annotation.

        Args:
            annotation_id: The id a task references.

        Returns:
            The decoded value: a string for a context, a list for a candidate pair.

        Raises:
            LaneError: If this version holds no such annotation.
        """
        values = self._load()
        try:
            return values[annotation_id]
        except KeyError:
            raise LaneError(f"{self.root} holds no annotation {annotation_id!r}") from None

    def close(self) -> None:
        """Drop the table."""
        self._values = None

    def _load(self) -> dict[str, Any]:
        """Read the annotations this row's items look up, on first use.

        Returns:
            Per annotation id, its decoded value.

        Raises:
            LaneError: If the version holds no annotations table.
        """
        if self._values is not None:
            return self._values
        # Lane dependencies, imported here rather than at module import.
        import pyarrow.dataset as pads  # noqa: PLC0415

        from timenet.format.schemas import IdCodec  # noqa: PLC0415

        parts = sorted(self.root.glob(ANNOTATIONS_GLOB))
        if not parts:
            raise LaneError(f"{self.root} holds no {ANNOTATIONS_GLOB} table, so its QA rows carry no metadata")
        scan = pads.dataset([str(part) for part in parts], format="parquet")
        table = scan.to_table(columns=["id", "key", "value"])
        codec = IdCodec.from_uuid16({"annotation_id"} if _is_uuid16(scan.schema.field("id").type) else set())
        self._values = {
            codec.decode("annotation_id", stored): json.loads(value)
            for stored, key, value in zip(
                table.column("id").to_pylist(),
                table.column("key").to_pylist(),
                table.column("value").to_pylist(),
                strict=True,
            )
            if key in METADATA_KEYS and value is not None
        }
        return self._values


def timef_qa_row(row: Mapping[str, Any], metadata: MetadataAnnotations) -> QaRow:
    """Read one TimeF task row into the fields both representations of the item carry.

    The task holds the question, the answer and the rationale itself and references the rest: the
    connector stores the question type, the template, the clinical context and the candidate pair as
    value-deduped annotations. Two of those four ids state their own value, so only the context and
    the pair are looked up. The template's whole vocabulary is skipped, because the item carries the
    pair the question offers rather than the label space it draws from.

    The pair comes back sorted, which is what the Original side delivers. The connector stores it in
    the order the prompt lists it, and the same two answers appear in both orders across the corpus.

    Args:
        row: One decoded task row, from :meth:`TaskRows.row`.
        metadata: Where the referenced annotations are read.

    Returns:
        The parsed row, in the same fields the Original side builds.

    Raises:
        LaneError: If the task names other than one recording, offers other than two candidates, or
            references none of a field the item carries.
    """
    record_ids = tuple(row["record_ids"])
    if len(record_ids) != 1:
        raise LaneError(f"the QA row {row['id']!r} names {len(record_ids)} records, expected one recording")
    question_type = ""
    template_id: int | None = None
    clinical_context = ""
    candidates: tuple[str, str] | None = None
    for annotation_id in row["input_annotation_ids"]:
        if annotation_id.startswith(TEMPLATE_OPTIONS_PREFIX):
            continue
        if annotation_id.startswith(QTYPE_PREFIX):
            question_type = annotation_id[len(QTYPE_PREFIX) :]
        elif annotation_id.startswith(TEMPLATE_PREFIX):
            template_id = _template_id(annotation_id)
        elif annotation_id.startswith(CONTEXT_PREFIX):
            clinical_context = str(metadata.value(annotation_id))
        elif annotation_id.startswith(OPTIONS_PREFIX):
            candidates = _candidates(row["id"], metadata.value(annotation_id))
    if not question_type or template_id is None or not clinical_context or candidates is None:
        raise LaneError(
            f"the QA row {row['id']!r} references {sorted(row['input_annotation_ids'])}, which is not the "
            "question type, the template, the clinical context and the candidate pair"
        )
    return QaRow(
        item_id=str(row["id"]),
        ecg_id=_ecg_id_of(str(record_ids[0])),
        question=str(row["prompt"]),
        answer=str(row["target"]),
        question_type=question_type,
        template_id=template_id,
        clinical_context=clinical_context,
        candidates=candidates,
        rationale=str(row["rationale"]),
    )


@dataclass
class EcgQaCotTimeFHandle:
    """An open TimeF ECG-QA CoT lane, delivering the QA rows the Original side delivers.

    Attributes:
        reader: The open reader. It resolves nothing until an item asks for it.
        tasks: Where the QA rows come from.
        metadata: Where the annotations a QA row references come from.
        item_ids: The canonical enumeration, one task id per ordinal.
        consumer: ``pandas`` or ``torch``.
        recordings: Per delivered ordinal, the record the item read. Filled as items go out and read
            back after the clock stops, so a group switch here is a re-decode.
        held: The record currently decoded, if any.
    """

    reader: Any
    tasks: TaskRows
    metadata: MetadataAnnotations
    item_ids: tuple[str, ...]
    consumer: str
    recordings: dict[int, str] = field(default_factory=dict)
    held: Any = None

    def n_items(self) -> int:
        """Return how many canonical items this lane holds.

        Returns:
            The item count.
        """
        return len(self.item_ids)

    def deliver(self, ordinals: Sequence[int]) -> Iterator[DeliveredItem]:
        """Read one request's worth of QA rows and the recordings they ask about.

        Args:
            ordinals: Canonical ordinals to read. They come back in the order asked.

        Yields:
            One item per ordinal.

        Raises:
            LaneError: If an ordinal is outside the enumeration.
        """
        for ordinal in ordinals:
            try:
                item_id = self.item_ids[ordinal]
            except IndexError as error:
                raise LaneError(f"a request asked for an ordinal outside the {len(self.item_ids)} items") from error
            row = timef_qa_row(self.tasks.row(item_id), self.metadata)
            record = self._record(row.record_id)
            self.recordings[ordinal] = row.record_id
            yield DeliveredItem(ordinal=ordinal, item_id=item_id, buffers=self._buffers(row, record))

    def group_of(self, ordinal: int) -> object:
        """Return the record an item was read out of.

        The Original side names the recording here, because a switch there is one more ``.dat``
        decoded. This side names the same thing, so the two switch counts compare.

        Returns:
            The record id, or ``None`` when the item was not delivered.
        """
        return self.recordings.get(ordinal)

    def close(self) -> None:
        """Close the reader and the task partitions, and drop the record being held."""
        self.tasks.close()
        self.metadata.close()
        self.held = None
        self.reader.close()

    def _record(self, record_id: str) -> Any:
        """Return one record, reading it unless it is the one already held.

        One decoded recording at a time is the declared rule for this row, and the Original side
        holds one too, so over a pass the two sides decode the same recordings the same number of
        times.

        Args:
            record_id: The record the QA row asks about.

        Returns:
            The record, with its series still unread.

        Raises:
            LaneError: If the version holds no such record.
        """
        if self.held is not None and self.held.record_id == record_id:
            return self.held
        try:
            record = next(iter(self.reader.iter_records([record_id])), None)
        except (OSError, ValueError) as error:
            raise LaneError(f"the TimeF version could not read record {record_id!r}: {error}") from error
        if record is None:
            raise LaneError(f"the TimeF version holds no record {record_id!r}")
        self.held = record
        return record

    def _buffers(self, row: QaRow, record: Any) -> tuple[Buffer, ...]:
        """Convert one QA row and its record into the consumer's type.

        Returns:
            One buffer per value plane, in the order the Original side delivers them.

        Raises:
            LaneError: If the consumer is not one this lane implements.
        """
        if self.consumer == PANDAS:
            # The lane's own consumer, imported once per lane rather than at module import.
            from timenet.pandas import record_array_frame  # noqa: PLC0415

            qa = qa_frame(row)
            return (
                *array_buffers(record_array_frame(record)),
                *(as_buffer(qa[name].to_numpy()) for name in qa.columns),
            )
        if self.consumer == TORCH:
            from timenet.torch import record_item  # noqa: PLC0415 - as above

            return (*(as_buffer(tensor.numpy()) for tensor in record_item(record)["series"]), as_buffer(row.text()))
        raise LaneError(f"the ECG-QA CoT lane has no {self.consumer!r} consumer; it implements {PANDAS} and {TORCH}")


@dataclass(frozen=True)
class EcgQaCotTimeFLane:
    """One TimeF lane over the ECG-QA CoT artifact, whose item is a QA row rather than a record.

    The generic TimeF lane delivers one item per record, which is what the four record-carried rows
    want. This row's item is one of the 231,543 questions and the artifact holds 16,024 recordings,
    so a lane that read the enumeration as record ids would ask for 231,543 records that are not
    there. This one resolves an item to the recording plus the one task row that asks about it, and
    delivers what the Original side delivers.

    Attributes:
        consumer: ``pandas`` or ``torch``.
        root: The committed version directory.
        item_ids: The canonical enumeration, one task id per ordinal.
        tier_a_bytes: Tier A bytes of this artifact, which sizes the blocks.
        name: The lane name that reaches the cell record.
    """

    consumer: str
    root: Path
    item_ids: tuple[str, ...]
    tier_a_bytes: int
    name: str = ""
    dataset: str = DATASET
    representation: str = TIMEF
    loader_variant: str = NATIVE

    def __post_init__(self) -> None:
        """Refuse a lane that cannot be measured.

        Raises:
            LaneError: If the consumer is unknown, the enumeration is empty, or Tier A bytes are not
                positive.
        """
        if self.consumer not in {PANDAS, TORCH}:
            raise LaneError(f"the ECG-QA CoT lane implements {PANDAS} and {TORCH}, got {self.consumer!r}")
        if not self.item_ids:
            raise LaneError("the ECG-QA CoT lane has no items to enumerate")
        if self.tier_a_bytes < 1:
            raise LaneError("the ECG-QA CoT lane needs positive Tier A bytes to size its blocks")
        if not self.name:
            object.__setattr__(self, "name", f"{DATASET}/{TIMEF}/{self.consumer}")

    def preload(self) -> None:
        """Import the reader, the task partition reader and this lane's consumer, opening nothing.

        Raises:
            LaneError: If the consumer is not one this lane implements.
        """
        if self.consumer not in {PANDAS, TORCH}:
            raise LaneError(
                f"the ECG-QA CoT lane has no {self.consumer!r} consumer; it implements {PANDAS} and {TORCH}"
            )
        for module in ("pyarrow.parquet", "timenet.reader", "timenet.registry", f"timenet.{self.consumer}"):
            importlib.import_module(module)

    def open(self) -> EcgQaCotTimeFHandle:
        """Open the version.

        Nothing decodes here. The reader holds the parsed manifest, the task partitions are not even
        opened, and the annotations are read on the first item that references one. So the first
        item costs one task row group, one annotations table and one record, not the corpus.

        Returns:
            The open handle.

        Raises:
            LaneError: If the version directory cannot be opened.
        """
        # Imported per lane, not at module import: the harness stays importable without a build.
        from timenet.reader import TimeFReader  # noqa: PLC0415
        from timenet.registry import DatasetVersion  # noqa: PLC0415

        try:
            reader = TimeFReader(DatasetVersion.open_local(self.root))
        except (OSError, ValueError) as error:
            raise LaneError(f"lane {self.name} could not open {self.root}: {error}") from error
        return EcgQaCotTimeFHandle(
            reader=reader,
            tasks=TaskRows(root=self.root),
            metadata=MetadataAnnotations(root=self.root),
            item_ids=self.item_ids,
            consumer=self.consumer,
        )


def task_ids_of(root: Path) -> tuple[str, ...]:
    """Return a TimeF version's QA row ids in stored order, which is its canonical enumeration.

    The generic :func:`~benchmarks.paper.timef_lane.record_ids_of` enumerates records, and this row
    carries its item on a task instead. It reads the ``id`` column of the task partitions and
    nothing else, so it is cheap enough to run once per campaign to build the item registry.

    Args:
        root: The committed version directory.

    Returns:
        The task ids, in stored order.

    Raises:
        LaneError: If the version cannot be read.
    """
    rows = TaskRows(root=root)
    try:
        return rows.task_ids()
    except OSError as error:
        raise LaneError(f"could not enumerate the QA rows of {root}: {error}") from error
    finally:
        rows.close()


def item_ids_of(layout: RawLayout) -> tuple[str, ...]:
    """Walk the CoT splits and return the canonical enumeration.

    The ids are the connector's task ids, ``ecgqa-<split>-<row>``, so the two representations of
    this row can be gated against each other.

    Args:
        layout: The resolved raw layout.

    Returns:
        One id per canonical item, in canonical order.
    """
    return tuple(item_id for item_id, _ in enumerate_rows(layout))


def enumerate_rows(layout: RawLayout) -> Iterator[tuple[str, int]]:
    """Walk the CoT splits and yield each row's item id and recording.

    Args:
        layout: The resolved raw layout.

    Yields:
        One ``(item id, ecg id)`` pair per canonical item, in canonical order.

    Raises:
        LaneError: If a split cannot be read.
    """
    for split, path in layout.split_csvs:
        try:
            with path.open(newline="", encoding="utf-8") as handle:
                for index, row in enumerate(csv.DictReader(handle)):
                    yield f"ecgqa-{split}-{index}", _ecg_id(row["ecg_id"])
        except (OSError, KeyError, ValueError) as error:
            raise LaneError(f"could not enumerate {path}: {error}") from error


def row_position(item_id: str) -> tuple[int, int]:
    """Return which split holds a task id's row, and which row of that split it is.

    The connector's task id is ``ecgqa-<split>-<row>``, so it addresses the raw table on its own.
    That is what lets a lane over part of the corpus read the right rows, and what lets the
    streaming source tell a step forward from a jump.

    Args:
        item_id: A CoT task id.

    Returns:
        The split's position in canonical order, and the row's position in that split.

    Raises:
        LaneError: If the id is not a CoT task id.
    """
    match = _ITEM_ID.match(item_id)
    if match is None or match["split"] not in _SPLIT_RANK:
        raise LaneError(f"{item_id!r} is not an ECG-QA CoT task id")
    return _SPLIT_RANK[match["split"]], int(match["index"])


def _header_of(path: Path) -> tuple[str, ...]:
    """Return a CSV's column names.

    Returns:
        The header row.
    """
    with path.open(newline="", encoding="utf-8") as handle:
        return tuple(next(csv.reader(handle)))


def _iter_rows(path: Path, header: Sequence[str], offset: int) -> Iterator[Mapping[str, Any]]:
    """Walk a CSV's rows from one byte offset to the end of the file.

    Args:
        path: The CSV.
        header: Its column names, which an offset past the header cannot read for itself.
        offset: Where to start. Zero starts at the header line and skips it.

    Yields:
        One row per record, keyed by column.
    """
    with path.open("rb") as handle:
        handle.seek(offset)
        reader = csv.reader(_decoded_lines(handle))
        if offset == 0:
            next(reader, None)
        for values in reader:
            yield dict(zip(header, values, strict=False))


def _row_offsets(path: Path) -> Iterator[int]:
    """Walk a CSV and yield the byte offset each record starts at.

    A CoT record spans many physical lines: its prompt, its rationale and its clinical context all
    hold newlines inside quotes. So the offsets come from the ``csv`` reader itself, counting the
    bytes it consumed, rather than from a scan that would have to re-implement quoting.

    Args:
        path: The CSV.

    Yields:
        One offset per record, in file order.
    """
    with path.open("rb") as handle:
        consumed = 0

        def lines() -> Iterator[str]:
            nonlocal consumed
            for raw in handle:
                consumed += len(raw)
                yield raw.decode("utf-8")

        reader = csv.reader(lines())
        next(reader, None)
        start = consumed
        for _ in reader:
            yield start
            start = consumed


def _decoded_lines(handle: Any) -> Iterator[str]:
    """Decode a binary file's lines for the csv reader.

    Returns:
        One decoded line per physical line.
    """
    return (raw.decode("utf-8") for raw in handle)


def _ecg_id(raw: Any) -> int:
    """Parse a PTB-XL ecg_id that arrives as ``123`` or ``"[123]"``.

    Returns:
        The recording id.
    """
    return int(str(raw).strip().strip("[]").strip())


def _ecg_id_of(record_id: str) -> int:
    """Return the PTB-XL recording a TimeF record id names.

    Returns:
        The recording id.

    Raises:
        LaneError: If the id is not the connector's ``ptbxl-<ecg id>``.
    """
    try:
        return int(record_id.removeprefix(RECORD_PREFIX))
    except ValueError as error:
        raise LaneError(f"{record_id!r} is not an ECG-QA CoT record id") from error


def _template_id(annotation_id: str) -> int:
    """Return the template a ``ecgqa-template-<id>`` annotation id names.

    Returns:
        The template id.

    Raises:
        LaneError: If the id does not end in a template number.
    """
    try:
        return int(annotation_id[len(TEMPLATE_PREFIX) :])
    except ValueError as error:
        raise LaneError(f"{annotation_id!r} names no ECG-QA CoT template") from error


def _candidates(item_id: object, value: Any) -> tuple[str, str]:
    """Return an ``answer_options`` annotation as the sorted candidate pair the item carries.

    Returns:
        The two candidates, sorted.

    Raises:
        LaneError: If the annotation does not hold two candidates.
    """
    options = tuple(sorted(str(option) for option in value))
    if len(options) != N_CANDIDATES:
        raise LaneError(f"the QA row {item_id!r} offers {len(options)} candidates, expected {N_CANDIDATES}: {options}")
    first, second = options
    return first, second


def _is_uuid16(arrow_type: Any) -> bool:
    """Return whether a column stores its ids as 16 raw bytes rather than as text.

    A list column states it on its element type, so this reads through one level of list.

    Returns:
        True when the stored ids are ``uuid16``.
    """
    # A lane dependency, imported here rather than at module import.
    from timenet.format.schemas import UUID16  # noqa: PLC0415

    value_type = getattr(arrow_type, "value_type", None)
    return (arrow_type if value_type is None else value_type) == UUID16


def _text(value: Any) -> str:
    """Return a CSV field as text, whatever the reader made of it.

    pandas turns an empty field into a float nan and the ``csv`` module into an empty string. Both
    mean the same thing here, so both become an empty string.

    Returns:
        The field as text, empty when it holds nothing.
    """
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value)


def _ptbxl_root(root: Path) -> Path:
    """Find the extracted PTB-XL directory under a cache root.

    Returns:
        The directory holding ``ptbxl_database.csv``.

    Raises:
        LaneError: If no extracted PTB-XL tree is there.
    """
    for candidate in sorted(root.glob("*")):
        if (candidate / PTBXL_MARKER).is_file():
            return candidate
    raise LaneError(f"no extracted PTB-XL tree under {root}: nothing holds {PTBXL_MARKER}")
