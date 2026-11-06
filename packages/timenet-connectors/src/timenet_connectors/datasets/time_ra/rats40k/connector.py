"""Orchestrate RATS40K conversion and repeatable tasks."""

from collections.abc import Iterator, Mapping
from typing import ClassVar, NamedTuple

from timenet.dataset import Record, TimeFDataset
from timenet.types import Annotation, ClassificationTask
from timenet.types.shared_loading import SharedChannelLoader
from timenet_connectors.bases.huggingface import BaseHuggingFaceConnector, HubFileRef
from timenet_connectors.datasets.time_ra.rats40k.records import build_record
from timenet_connectors.datasets.time_ra.rats40k.source import (
    PINNED_INVENTORY,
    Row,
    SourceInventory,
    load_channels,
)
from timenet_connectors.datasets.time_ra.rats40k.tasks import build_label_set, build_task, load_prompts
from timenet_connectors.datasets.time_ra.rats40k.vocabulary import AnnotationKey, Variant


class _Converted(NamedTuple):
    """A row with the Record it became."""

    row: Row
    record: Record


class RATS40KConnector(BaseHuggingFaceConnector):
    """Build RATS40K from the pinned, gated JSON files of its Hugging Face release."""

    HF_REPO = "Time-RA/RATs40K"
    SOURCE_REVISION = "dc968d6a88f396215be50023c93c3928e06096df"
    SOURCE_FILES: ClassVar[Mapping[str, str]] = {variant.value: variant.source_file for variant in Variant}
    SOURCE_INVENTORY: ClassVar[Mapping[Variant, SourceInventory]] = PINNED_INVENTORY
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
            TimeNetConnectorError: If a file's structure or a row is malformed, or a file's rows differ from
                its inventory.
        """  # ruff: ignore[docstring-extraneous-exception] - raised by the helpers it calls
        loader = SharedChannelLoader(load_channels)
        variants = [Variant(ref.part) for ref in raw_refs]
        converted: list[_Converted] = []
        for ref, variant in zip(raw_refs, variants, strict=True):
            for row in self.SOURCE_INVENTORY[variant].iter_rows(ref.path, variant):
                converted.append(_Converted(row, build_record(row, loader)))

        dataset = TimeFDataset(metadata=self.metadata())
        dataset.add_annotations(
            {AnnotationKey.SOURCE_REPOSITORY: self.HF_REPO, AnnotationKey.SOURCE_REVISION: self.SOURCE_REVISION}
        )
        for _, record in converted:
            dataset.add_record(record=record)
        for variant in variants:
            dataset.annotate(Annotation(key=variant.template_key, value=load_prompts(variant).template))
        label_sets = {
            variant: build_label_set(
                variant, (row.action for row, _ in converted if row.observation.variant is variant)
            )
            for variant in variants
        }

        def tasks() -> Iterator[ClassificationTask]:
            for row, record in converted:
                yield build_task(row, record, label_sets[row.observation.variant])

        dataset.set_task_stream((ClassificationTask,), tasks)
        return dataset


CONNECTOR = RATS40KConnector
