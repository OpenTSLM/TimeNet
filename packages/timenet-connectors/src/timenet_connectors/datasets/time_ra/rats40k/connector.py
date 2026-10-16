"""Convert pinned RATS40K JSON files into lazy Signals and three tasks per example.

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
from typing import Any, NotRequired, TypedDict

import pyarrow as pa

from timenet.connectors import BaseConnector
from timenet.dataset import Record, Signal, Source, TimeFDataset
from timenet.dataset.axis import OrdinalAxis
from timenet.errors import TimeFValidationError
from timenet.types import Annotation, AnswerTask, ClassificationTask, Split, TimeSeriesSpec
from timenet_connectors.datasets.time_ra.rats40k._json import iter_entries
from timenet_connectors.sources.huggingface_hub import hub_snapshot


SOURCE_REVISION = "dc968d6a88f396215be50023c93c3928e06096df"
SOURCE_FILES = {"uni": "RATs-Uni-TSImage_Reason.json", "multi": "RATs-Multi-TSImage_Reason.json"}
# Counts exclude the eleven published null entries; require their exact keys separately.
SOURCE_ROWS = {"uni": {"train": 30266, "test": 6034}, "multi": {"train": 2728, "test": 546}}
NULL_KEYS = frozenset({"106", "386", "560", "875", "1632", "2081", "2581", "2644", "2721", "3584", "5558"})
_SPEC = TimeSeriesSpec(spec_type="rats40k_series", name="RATS40K series", unit_value=None, dtype="float64")
_BINARY_SCHEMA = "rats40k-label-vocabulary"
_ACTION_SCHEMAS = {variant: f"rats40k-{variant}-action-vocabulary" for variant in SOURCE_FILES}
_ROW_FIELDS = frozenset({"Observation", "Source", "FigurePath", "Thought", "Action", "ActionID", "Label"})
# The authors' evaluation prompts without the domain sentence and Observation line: the Record carries
# the series and the source_domain input annotation carries the domain. Every task of a variant shares one.
_PROMPTS = {
    variant: (Path(__file__).parent / "prompts" / f"{variant}.txt").read_text(encoding="utf-8").rstrip("\n")
    for variant in SOURCE_FILES
}
_PROMPT_SOURCE = "https://github.com/yyysjz1997/Time-RA/blob/200b244f75b134348fad39c9ba140d9b2f1e1279/prompts"
_PROMPT_FILES = {"uni": "prompt_llama_anoclf_reason.py", "multi": "prompt_llama_multi_ano.py"}


class _FileRef(TypedDict):
    """Local source file and its optional pinned-inventory check."""

    path: str
    variant: str
    check_inventory: NotRequired[bool]


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


def _validate_row(row: Any, ref: _RowRef) -> None:
    """Validate source context and supervision independently of signal decoding.

    Raises:
        TimeFValidationError: If the row is malformed, with file/split/key context.
    """
    try:
        if not isinstance(row, dict):
            raise TimeFValidationError("expected a row object")
        for field in ("Source", "FigurePath", "Thought", "Action", "Label"):
            if not isinstance(row.get(field), str) or not row[field].strip():
                raise TimeFValidationError(f"missing or empty {field}")
        # Published labels sometimes disagree with Action/ActionID. Preserve independent targets.
        if row["Label"] not in {"Normal", "Anomaly"}:
            raise TimeFValidationError("Label must be Normal or Anomaly")
        if type(row.get("ActionID")) is not int or row["ActionID"] < 0:
            raise TimeFValidationError("ActionID must be a nonnegative integer")
    except (ValueError, OverflowError, RecursionError) as exc:
        raise TimeFValidationError(f"{ref.context}: {exc}") from exc


def _decode_observation(observation: Any, ref: _RowRef) -> dict[str, list[float]]:
    """Decode and validate signal values with their source location.

    Returns:
        The observation's validated channels.

    Raises:
        TimeFValidationError: If the observation is malformed, with file/split/key context.
    """
    try:
        return _channels(observation, ref.variant)
    except (ValueError, OverflowError, RecursionError) as exc:
        raise TimeFValidationError(f"{ref.context}: {exc}") from exc


def _iter_rows(file: _FileRef) -> Iterator[tuple[_RowRef, Any]]:
    """Traverse local entries, retaining seekable byte ranges and original identities.

    Yields:
        Lightweight references and transient source rows, including null entries.
    """
    path = Path(file["path"])
    for split, key, row, offset, length in iter_entries(path):
        yield _RowRef(path, file["variant"], Split(split.removeprefix("TSAD_")), key, offset, length), row


class _ValueLoader:
    """Load values lazily, reusing one decoded row across its channels to avoid repeated reads."""

    def __init__(self) -> None:
        self.cached: tuple[_RowRef, dict[str, list[float]]] | None = None

    def __call__(self, ref: _RowRef, name: str) -> pa.Array:
        cached = self.cached
        if cached is None or cached[0] != ref:
            with ref.path.open("rb") as handle:
                handle.seek(ref.offset)
                row = json.loads(handle.read(ref.length))
            channels = _decode_observation(row.get("Observation"), ref)
            self.cached = ref, channels
        else:
            channels = cached[1]
        return pa.array(channels[name], type=pa.float64())


def _make_record(ref: _RowRef, row: dict[str, Any], channels: dict[str, list[float]], loader: _ValueLoader) -> Record:
    """Build a record with lazy named channels and small provenance annotations.

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
                metadata={"channel_order": list(channels)},
            ),
        ),
    )
    for key, value in {
        "source_file": SOURCE_FILES[ref.variant],
        "source_key": ref.key,
        "source_variant": ref.variant,
        "source_split": f"TSAD_{ref.split}",
        "source_domain": row["Source"],
        "figure_path": row["FigurePath"],
    }.items():
        record.annotate(Annotation(key=key, value=value, id=f"{ref.id}-{key}"))
    return record


def _iter_tasks(
    files: tuple[_FileRef, ...], records: Mapping[str, Record], prompts: Mapping[str, Annotation]
) -> Iterator[ClassificationTask | AnswerTask]:
    """Stream original labels and complete diagnostic text, without exposing targets as inputs.

    Yields:
        Binary and variant-specific category labels with their Thought as rationale, and complete
        diagnostic answers, all referencing the variant's shared prompt annotation.
    """
    # Each pass rereads both JSON files, observations included; repeated iteration repeats that work.
    for file in files:
        for ref, row in _iter_rows(file):
            if row is None:
                continue
            record = records[ref.id]
            # One dataset-level occurrence per variant: the writer stores the prompt text once and each
            # task only a reference, where Task.prompt would store it once per task.
            context = (
                prompts[ref.variant],
                *(annotation for annotation in record.annotations if annotation.key == "source_domain"),
            )
            # The Thought was written for the Action, so it contradicts Label where the two disagree.
            yield ClassificationTask(
                id=f"{ref.id}-detection",
                inputs=(record,),
                targets=(row["Label"],),
                rationale=row["Thought"],
                split=ref.split,
                input_annotations=context,
                target_schema=_BINARY_SCHEMA,
            )
            yield ClassificationTask(
                id=f"{ref.id}-type",
                inputs=(record,),
                targets=(str(row["ActionID"]),),
                # The authors pair each category with its Thought; the AnswerTask below scores it alone.
                rationale=row["Thought"],
                split=ref.split,
                input_annotations=context,
                target_schema=_ACTION_SCHEMAS[ref.variant],
                metadata={
                    "Action": row["Action"],
                    **{key: value for key, value in row.items() if key not in _ROW_FIELDS},
                },
            )
            yield AnswerTask(
                id=f"{ref.id}-reasoning",
                inputs=(record,),
                targets=(row["Thought"],),
                split=ref.split,
                input_annotations=context,
            )


class RATS40KConnector(BaseConnector[_FileRef]):
    """Build RATS40K from pinned gated JSON files instead of the absent parquet ref."""

    HF_REPO = "Time-RA/RATs40K"

    def download(self, cache_dir: Path) -> list[_FileRef]:
        """Download original files using existing Hub authentication.

        Returns:
            File-reference dictionaries containing the local JSON path and source variant.

        Raises:
            ImportError: If the connector's optional Hub dependency is missing.
        """
        try:
            snapshot = hub_snapshot(self.HF_REPO, SOURCE_REVISION, cache_dir, tuple(SOURCE_FILES.values()))
        except ImportError as exc:
            raise ImportError(
                "RATS40K needs huggingface_hub from requirements.txt; "
                "run the build without --no-isolation or install the requirements with uv"
            ) from exc
        return [
            {"path": str(snapshot / filename), "variant": variant, "check_inventory": True}
            for variant, filename in SOURCE_FILES.items()
        ]

    def convert(self, raw_refs: list[_FileRef]) -> TimeFDataset:
        """Index source rows into lazy records and a re-iterable three-task stream.

        Returns:
            A populated dataset with original split membership and provenance.

        Raises:
            TimeFValidationError: If references, rows, or the pinned inventory are invalid.
        """
        if not raw_refs or any(ref.get("variant") not in SOURCE_FILES or not ref.get("path") for ref in raw_refs):
            raise TimeFValidationError("RATS40K requires known variants and local file paths")
        if len({ref["variant"] for ref in raw_refs}) != len(raw_refs):
            raise TimeFValidationError("RATS40K file variants must be distinct")
        dataset = TimeFDataset(metadata=self.metadata())
        for key, value in (("source_repository", self.HF_REPO), ("source_revision", SOURCE_REVISION)):
            dataset.annotate(Annotation(key=key, value=value, id=f"rats40k-{key}"))
        records: dict[str, Record] = {}
        loader = _ValueLoader()
        skipped = 0
        disagreements = 0
        for file in raw_refs:
            names: dict[tuple[str, int], set[str]] = {}
            counts: Counter[str] = Counter()
            nulls: set[tuple[str, str]] = set()
            for ref, row in _iter_rows(file):
                if row is None:
                    nulls.add((ref.split.value, ref.key))
                    skipped += 1
                    continue
                _validate_row(row, ref)
                channels = _decode_observation(row.get("Observation"), ref)
                record = _make_record(ref, row, channels, loader)
                dataset.add_record(record=record)
                records[ref.id] = record
                counts[ref.split.value] += 1
                names.setdefault((ref.split.value, row["ActionID"]), set()).add(row["Action"])
                disagreements += (row["Label"] == "Normal") != (row["Action"] == "Normal Sequence")
            if not counts:
                raise TimeFValidationError(f"{file['path']}: no usable rows")
            if file.get("check_inventory"):
                expected_nulls = {("test", key) for key in NULL_KEYS} if file["variant"] == "uni" else set()
                if counts != SOURCE_ROWS[file["variant"]] or nulls != expected_nulls:
                    raise TimeFValidationError(f"{file['path']}: pinned row counts or null entry keys differ")
            variant = file["variant"]
            dataset.annotate(
                Annotation(
                    key=f"{variant}_action_vocabulary",
                    value=[str(action_id) for action_id in sorted({action_id for _, action_id in names})],
                    metadata={
                        "observed_names": [
                            {"split": split, "action_id": action_id, "actions": sorted(action_names)}
                            for (split, action_id), action_names in sorted(names.items())
                        ],
                    },
                    id=_ACTION_SCHEMAS[variant],
                )
            )
        dataset.annotate(Annotation(key="skipped_null_entries", value=skipped, id="rats40k-skipped-null-entries"))
        dataset.annotate(
            Annotation(
                key="label_action_disagreements",
                value=disagreements,
                id="rats40k-label-action-disagreements",
            )
        )
        dataset.annotate(Annotation(key="label_vocabulary", value=["Normal", "Anomaly"], id=_BINARY_SCHEMA))
        # Tasks must reference the attached occurrence that annotate() returns, not the Annotation itself.
        prompts = {
            ref["variant"]: dataset.annotate(
                Annotation(
                    key="prompt",
                    value=_PROMPTS[ref["variant"]],
                    description="Instruction to give the model with the input series and source_domain.",
                    source=f"{_PROMPT_SOURCE}/{_PROMPT_FILES[ref['variant']]}",
                    id=f"rats40k-{ref['variant']}-prompt",
                )
            )
            for ref in raw_refs
        }
        dataset.set_task_stream(
            [ClassificationTask, AnswerTask], partial(_iter_tasks, tuple(raw_refs), records, prompts)
        )
        return dataset


CONNECTOR = RATS40KConnector
