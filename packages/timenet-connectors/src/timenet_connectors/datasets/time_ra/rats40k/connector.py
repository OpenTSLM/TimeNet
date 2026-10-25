"""Convert the pinned RATS40K JSON files into lazy Records and one category task per row.

Each row becomes a Record whose signals reread their entry on demand, and a ClassificationTask whose
target is the published category name. README.md explains how this follows the authors.
"""

from collections import Counter
from collections.abc import Mapping
from functools import partial
from typing import Any, ClassVar

import pyarrow as pa

from timenet.dataset import Record, Signal, Source, TimeFDataset
from timenet.dataset.axis import OrdinalAxis
from timenet.errors import TimeFValidationError
from timenet.types import Annotation, ClassificationTask, TimeSeriesSpec
from timenet_connectors.bases.huggingface import BaseHuggingFaceConnector, HubFileRef
from timenet_connectors.datasets.time_ra.rats40k._source import Row, RowRef, load_channels, read_rows
from timenet_connectors.datasets.time_ra.rats40k.request import CATEGORY_NAMES, TASK_PROMPTS, TEMPLATES


# Counts exclude the eleven published null entries, all univariate test keys, pinned separately.
SPLIT_ROWS = {"uni": {"train": 30266, "test": 6034}, "multi": {"train": 2728, "test": 546}}
NULL_KEYS = frozenset({"106", "386", "560", "875", "1632", "2081", "2581", "2644", "2721", "3584", "5558"})
_SPEC = TimeSeriesSpec(spec_type="rats40k_series", name="RATS40K series", unit_value=None, dtype="float64")


def _category_schema(variant: str) -> str:
    return f"rats40k-{variant}-category-vocabulary"


def _load_channel(ref: RowRef, name: str) -> pa.Array:
    return pa.array(load_channels(ref)[name], type=pa.float64())


def _record(ref: RowRef, row: Row) -> Record:
    """Build a Record with one lazy signal per channel, its provenance, and its domain annotation.

    Returns:
        The row's Record.
    """
    signals = tuple(
        Signal.from_loader(
            id=f"{ref.id}-c{index}",
            name=name,
            spec=_SPEC,
            time_axis=OrdinalAxis(),
            n_values=length,
            loader=partial(_load_channel, ref, name),
        )
        for index, (name, length) in enumerate(row.channel_lengths.items())
    )
    record = Record(
        record_id=ref.id,
        sources=(
            Source(
                id=f"{ref.id}-source",
                name=row.domain,
                signals=signals,
                # TimeF does not keep signal order on read, so record the source's channel order.
                metadata={"channel_order": list(row.channel_lengths)},
            ),
        ),
        # Provenance only. The task reads the domain, so it alone is an annotation.
        metadata={"source_key": ref.key, "source_variant": ref.variant, "figure_path": row.figure_path},
    )
    record.annotate(Annotation(key="source_domain", value=row.domain, id=f"{ref.id}-source_domain"))
    return record


def _task(ref: RowRef, row: Row, record: Record) -> ClassificationTask:
    """Ask for the row's category name, as the authors' evaluation prompt does.

    Returns:
        The row's task, with the published name as target and the Thought as rationale.
    """
    metadata: dict[str, Any] = {
        "ActionID": row.action_id,
        "Label": row.label,
        # False for the univariate test rows labelled with multivariate names.
        "target_in_vocabulary": row.action in CATEGORY_NAMES[ref.variant],
    }
    if row.rank is not None:
        metadata["Rank"] = row.rank
    return ClassificationTask(
        id=f"{ref.id}-category",
        inputs=(record,),
        prompt=TASK_PROMPTS[ref.variant],
        targets=(row.action,),
        rationale=row.thought,
        split=ref.split,
        input_annotations=tuple(a for a in record.annotations if a.key == "source_domain"),
        target_schema=_category_schema(ref.variant),
        metadata=metadata,
    )


def _check_inventory(file: HubFileRef, counts: Counter[str], nulls: set[tuple[str, str]]) -> None:
    """Require the pinned revision's row count per split and its null entry keys.

    Raises:
        TimeFValidationError: If the file isn't the pinned revision's.
    """
    expected_nulls = {("test", key) for key in NULL_KEYS} if file.part == "uni" else set()
    if counts != SPLIT_ROWS[file.part] or nulls != expected_nulls:
        raise TimeFValidationError(f"{file.path}: pinned row counts or null entry keys differ")


def _variant_annotations(variant: str) -> tuple[Annotation, ...]:
    return (
        # The prompt's names: a name's index is the authors' code, which ActionID may not follow.
        Annotation(
            key=f"{variant}_category_vocabulary",
            value=list(CATEGORY_NAMES[variant]),
            id=_category_schema(variant),
        ),
        Annotation(
            key=f"{variant}_request_template",
            value=TEMPLATES[variant],
            id=f"rats40k-{variant}-request-template",
        ),
    )


class RATS40KConnector(BaseHuggingFaceConnector):
    """Build RATS40K from the pinned, gated JSON files of its Hugging Face release."""

    HF_REPO = "Time-RA/RATs40K"
    SOURCE_REVISION = "dc968d6a88f396215be50023c93c3928e06096df"
    SOURCE_FILES: ClassVar[Mapping[str, str]] = {
        "uni": "RATs-Uni-TSImage_Reason.json",
        "multi": "RATs-Multi-TSImage_Reason.json",
    }

    def convert(self, raw_refs: list[HubFileRef]) -> TimeFDataset:
        """Read each file once into lazy Records, one task per row, and the variant's annotations.

        Returns:
            The dataset, with the source's splits and provenance.

        Raises:
            TimeFValidationError: If the references, a row, or the pinned inventory are invalid.
        """  # noqa: DOC502 - raised by new_dataset, read_rows and _check_inventory
        dataset = self.new_dataset(raw_refs, "rats40k")
        for file in raw_refs:
            counts: Counter[str] = Counter()
            nulls: set[tuple[str, str]] = set()
            tasks: list[ClassificationTask] = []
            for ref, row in read_rows(file):
                if row is None:
                    nulls.add((ref.split.value, ref.key))
                    continue
                record = _record(ref, row)
                dataset.add_record(record=record)
                tasks.append(_task(ref, row, record))
                counts[ref.split.value] += 1
            _check_inventory(file, counts, nulls)
            # One batch: each add_task call checks its id against every registered task.
            dataset.add_tasks(tasks=tasks)
            for annotation in _variant_annotations(file.part):
                dataset.annotate(annotation)
        return dataset


CONNECTOR = RATS40KConnector
