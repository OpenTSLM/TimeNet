"""Convert pinned RATS40K JSON files into lazy Signals and one diagnosis task per example.

The source has no converted parquet revision. Download original JSON through the shared Hub helper,
then index each row by byte offset. Only one decoded observation is cached across channel loaders.
Diagnostic targets and optional ranking provenance are streamed from the original files.
"""

from collections import Counter
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from functools import partial
import json
import math
from pathlib import Path
import re
from typing import Any, ClassVar

import pyarrow as pa

from timenet.dataset import Record, Signal, Source, TimeFDataset
from timenet.dataset.axis import OrdinalAxis
from timenet.errors import TimeFValidationError
from timenet.types import Annotation, ClassificationTask, Split, TimeSeriesSpec
from timenet_connectors.bases.huggingface import BaseHubSnapshotConnector, HubFileRef
from timenet_connectors.datasets.time_ra.rats40k._json import iter_entries
from timenet_connectors.sources.json_stream import read_entry


_VARIANTS = ("uni", "multi")
# Counts exclude the eleven published null entries; require their exact keys separately.
SPLIT_ROWS = {"uni": {"train": 30266, "test": 6034}, "multi": {"train": 2728, "test": 546}}
NULL_KEYS = frozenset({"106", "386", "560", "875", "1632", "2081", "2581", "2644", "2721", "3584", "5558"})
_SPEC = TimeSeriesSpec(spec_type="rats40k_series", name="RATS40K series", unit_value=None, dtype="float64")
_CATEGORY_SCHEMAS = {variant: f"rats40k-{variant}-category-vocabulary" for variant in _VARIANTS}
# The authors' evaluation prompts without the domain sentence and Observation line: the Record carries
# the series and the source_domain input annotation carries the domain. Every task of a variant shares one.
_PROMPTS = {
    variant: (Path(__file__).parent / "prompts" / f"{variant}.txt").read_text(encoding="utf-8").rstrip("\n")
    for variant in _VARIANTS
}
# The category names each prompt defines ("12. Sudden Spike Anomaly: ..."), in the authors' code order.
_CATEGORY_NAMES = {
    variant: tuple(re.findall(r"^\d+\. ([^:\n]+):", prompt, re.MULTILINE)) for variant, prompt in _PROMPTS.items()
}


@dataclass(frozen=True)
class _RowRef:
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


def _named_channels(observation: Any) -> dict[str, Any]:
    """Parse named arrays, allowing semicolons inside quoted channel names.

    Returns:
        Safely decoded channel values in source order.

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


def _channels(observation: Any, variant: str) -> dict[str, list[float]]:
    """Validate a published numeric list or named arrays.

    Returns:
        Nonempty, equally sized channels with finite float64-representable values.

    Raises:
        TimeFValidationError: If an observation violates the source format.
    """
    channels = {"series": observation} if variant == "uni" else _named_channels(observation)
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
    return channels


def _parse_row(row: Any, ref: _RowRef) -> dict[str, list[float]]:
    """Validate a row's fields and decode its observation, naming the source location on error.

    Returns:
        The observation's validated channels.

    Raises:
        TimeFValidationError: If the row is malformed, with file/split/key context.
    """
    try:
        if not isinstance(row, dict):
            raise TimeFValidationError("expected a row object")
        for field in ("Source", "FigurePath", "Thought", "Action", "Label"):
            if not isinstance(row.get(field), str) or not row[field].strip():
                raise TimeFValidationError(f"missing or empty {field}")
        # Label sometimes disagrees with Action; both are kept as published.
        if row["Label"] not in {"Normal", "Anomaly"}:
            raise TimeFValidationError("Label must be Normal or Anomaly")
        if type(row.get("ActionID")) is not int or row["ActionID"] < 0:
            raise TimeFValidationError("ActionID must be a nonnegative integer")
        return _channels(row.get("Observation"), ref.variant)
    except (ValueError, OverflowError, RecursionError) as exc:
        raise TimeFValidationError(f"{ref.context}: {exc}") from exc


def _iter_rows(file: HubFileRef) -> Iterator[tuple[_RowRef, Any]]:
    """Traverse local entries, retaining seekable byte ranges and original identities.

    Yields:
        Lightweight references and transient source rows, including null entries.
    """
    for split, key, row, offset, length in iter_entries(file.path):
        yield _RowRef(file.path, file.part, Split(split.removeprefix("TSAD_")), key, offset, length), row


class _ValueLoader:
    """Load values lazily, reusing one decoded row across its channels to avoid repeated reads."""

    def __init__(self) -> None:
        self.cached: tuple[_RowRef, dict[str, list[float]]] | None = None

    def __call__(self, ref: _RowRef, name: str) -> pa.Array:
        cached = self.cached
        if cached is None or cached[0] != ref:
            channels = _parse_row(read_entry(ref.path, ref.offset, ref.length), ref)
            self.cached = ref, channels
        else:
            channels = cached[1]
        return pa.array(channels[name], type=pa.float64())


def _make_record(ref: _RowRef, row: dict[str, Any], channels: dict[str, list[float]], loader: _ValueLoader) -> Record:
    """Build a record with lazy named channels, provenance metadata, and its domain annotation.

    Returns:
        One independent record, preserving source channel order.
    """
    signals = tuple(
        Signal.from_loader(
            id=f"{ref.id}-c{index}",
            name=name,
            spec=_SPEC,
            time_axis=OrdinalAxis(),
            n_values=len(values),
            loader=partial(loader, ref, name),
        )
        for index, (name, values) in enumerate(channels.items())
    )
    record = Record(
        record_id=ref.id,
        sources=(
            Source(
                id=f"{ref.id}-source",
                name=row["Source"],
                signals=signals,
                # TimeF does not keep signal order on read, so record the source's channel order.
                metadata={"channel_order": list(channels)},
            ),
        ),
        # Provenance only; tasks read the domain, so it alone is an annotation.
        metadata={"source_key": ref.key, "source_variant": ref.variant, "figure_path": row["FigurePath"]},
    )
    record.annotate(Annotation(key="source_domain", value=row["Source"], id=f"{ref.id}-source_domain"))
    return record


def _iter_tasks(files: tuple[HubFileRef, ...], records: Mapping[str, Record]) -> Iterator[ClassificationTask]:
    """Stream the published category of each row, without exposing targets as inputs.

    Yields:
        One category task per row, with the Thought as rationale and the variant's evaluation prompt.
    """
    # Each pass rereads both JSON files, observations included; repeated iteration repeats that work.
    for file in files:
        for ref, row in _iter_rows(file):
            if row is None:
                continue
            record = records[ref.id]
            yield ClassificationTask(
                id=f"{ref.id}-category",
                inputs=(record,),
                prompt=_PROMPTS[ref.variant],
                # The prompt asks for a category name, so the published name is the target.
                targets=(row["Action"],),
                rationale=row["Thought"],
                split=ref.split,
                input_annotations=tuple(a for a in record.annotations if a.key == "source_domain"),
                target_schema=_CATEGORY_SCHEMAS[ref.variant],
                # Label is the source benchmark's label; it disagrees with the Action in some rows.
                metadata={
                    "ActionID": row["ActionID"],
                    "Label": row["Label"],
                    # False for the univariate test rows that carry multivariate category names.
                    "target_in_vocabulary": row["Action"] in _CATEGORY_NAMES[ref.variant],
                    # Test rows add Rank: the GPT-4 judge's ordering of the four labelling models. The
                    # published Thought and Action come from Rank[0], so it identifies each test label's
                    # author and separates clean labels (gpt-4o) from those with multivariate names.
                    **({"Rank": row["Rank"]} if "Rank" in row else {}),
                },
            )


class RATS40KConnector(BaseHubSnapshotConnector):
    """Build RATS40K from pinned gated JSON files instead of the absent parquet ref."""

    HF_REPO = "Time-RA/RATs40K"
    SOURCE_REVISION = "dc968d6a88f396215be50023c93c3928e06096df"
    SOURCE_FILES: ClassVar = {"uni": "RATs-Uni-TSImage_Reason.json", "multi": "RATs-Multi-TSImage_Reason.json"}

    def convert(self, raw_refs: list[HubFileRef]) -> TimeFDataset:
        """Index source rows into lazy records and a re-iterable task stream.

        Returns:
            A populated dataset with original split membership and provenance.

        Raises:
            TimeFValidationError: If references, rows, or the pinned inventory are invalid.
        """
        dataset = TimeFDataset(metadata=self.metadata())
        dataset.annotate(Annotation(key="source_repository", value=self.HF_REPO, id="rats40k-source_repository"))
        dataset.annotate(Annotation(key="source_revision", value=self.SOURCE_REVISION, id="rats40k-source_revision"))
        records: dict[str, Record] = {}
        loader = _ValueLoader()
        for file in raw_refs:
            variant = file.part
            # Tally rows per split and collect the (split, key) of each null entry, to check the
            # file against the pinned inventory once it is read.
            counts: Counter[str] = Counter()
            nulls: set[tuple[str, str]] = set()
            for ref, row in _iter_rows(file):
                if row is None:
                    nulls.add((ref.split.value, ref.key))
                    continue
                # Parsing here validates the row up front; the record keeps only lazy loaders, and
                # _iter_tasks finds it again by ref.id when the task stream rereads the file.
                record = _make_record(ref, row, _parse_row(row, ref), loader)
                dataset.add_record(record=record)
                records[ref.id] = record
                counts[ref.split.value] += 1
            # The source publishes eleven null entries, all in the univariate test split. Any other
            # count or null key means the file is not the pinned revision.
            expected_nulls = {("test", key) for key in NULL_KEYS} if variant == "uni" else set()
            if counts != SPLIT_ROWS[variant] or nulls != expected_nulls:
                raise TimeFValidationError(f"{file.path}: pinned row counts or null entry keys differ")
            dataset.annotate(
                Annotation(
                    key=f"{variant}_category_vocabulary",
                    # The prompt's names; the index is the authors' code, which ActionID may not follow.
                    value=list(_CATEGORY_NAMES[variant]),
                    id=_CATEGORY_SCHEMAS[variant],
                )
            )
        dataset.set_task_stream([ClassificationTask], partial(_iter_tasks, tuple(raw_refs), records))
        return dataset


CONNECTOR = RATS40KConnector
