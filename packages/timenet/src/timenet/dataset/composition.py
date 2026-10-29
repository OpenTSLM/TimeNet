"""Copy selected parent Records into a child dataset, sharing their stored values."""

from collections.abc import Iterable
from copy import deepcopy
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, TypeAlias

from timenet.dataset.record import Record
from timenet.dataset.source import Source
from timenet.dataset.time_series import Signal
from timenet.errors import TimeFValidationError
from timenet.types import Annotation
from timenet.types.ids import EntityId, check_id
from timenet.types.load_plan import LoadPlan
from timenet.types.object_kind import ObjectKind
from timenet.types.parent_dataset_card import ParentDatasetCard
from timenet.types.reference_keys import ParentAlias
from timenet.types.references import DatasetRef


if TYPE_CHECKING:
    from timenet.dataset.dataset import TimeFDataset
    from timenet.reader import TimeFReader


ImportMap: TypeAlias = dict[tuple[ParentAlias, ObjectKind, EntityId], EntityId]
"""The child ID of each imported Record and Signal, keyed by parent alias, kind, and parent ID.

The map is the only record of what a dataset imported. A copy gets the next ID of its kind in the
child, so no rule derives its ID from the parent ID. An import looks up a parent Record here. Then
it finds the copy by its child ID among the Records of the dataset. If the dataset no longer holds
the copy, the import copies the Record again. The map holds plain IDs, and ``dict(...)`` copies it.
The writer stores each entry whose copy the dataset holds in the ``imported_objects`` table. A
dataset that a reader reads back from storage gets these entries back.
"""


@dataclass(frozen=True)
class ImportedDataset:
    """The child copies of the Records that one ``import_from`` call selected, by their parent IDs.

    Each copy is one that the call made or one that the child held already.
    """

    alias: ParentAlias
    """The parent alias the objects were imported under."""
    parent: DatasetRef
    """The exact parent release."""
    records: dict[EntityId, Record] = field(default_factory=dict)
    """The copy of each Record selected with ``records=``, keyed by its parent ID."""


class _Importer:
    """Copy parent Records into one child under one parent alias."""

    def __init__(self, child: "TimeFDataset", alias: ParentAlias) -> None:
        self.child = child
        self.alias = alias
        self.records: dict[EntityId, Record] = {}
        """The child copy of each parent Record that this call found or copied, by parent ID."""

    def import_record(self, original: Record) -> Record:
        """Copy a parent Record whole, or find the copy of it that the child holds.

        A new copy and its Signals get their IDs when the child registers the copy. If the child
        holds a copy already, the call returns that copy and changes nothing.

        Args:
            original: The parent Record, as the parent's reader built it.

        Returns:
            The child's copy, registered and indexed.
        """
        child_id = self.child._imports.get((self.alias, ObjectKind.RECORD, original.id))
        target = None if child_id is None else self.child._records_by_id.get(child_id)
        if target is None:
            target = Record(
                # Time origins are frozen, like axes, so the copy shares the time origin of the parent
                # Record. Thus copies of Records that share a time origin also share it in the child.
                start_time=original.start_time,
                time_span=original.time_span,
                metadata=deepcopy(original.metadata),
            )
            copied = _RecordCopy(self, original, target)
            copied.merge()
            self.child.add_record(record=target)
            imports = self.child._imports
            imports[self.alias, ObjectKind.RECORD, original.id] = target.id
            for parent_id, signal in copied.held.items():
                imports[self.alias, ObjectKind.SIGNAL, parent_id] = signal.id
        self.records[original.id] = target
        return target

    def copy_signal(self, original: Signal) -> Signal:
        """Copy a parent Signal without its ID, with its axis and the handles to its stored values.

        Returns:
            The copy, with copies of its annotations. The child gives the copy an ID when it indexes
            the copy.
        """
        return Signal.from_loader(
            spec=original.spec,
            time_axis=original.time_axis,
            name=original.name,
            loader=original._values,
            n_values=original.n_values,
            time_offsets_loader=original._time_offsets,
            annotations=tuple(self.copy_annotation(annotation) for annotation in original.annotations),
            metadata=deepcopy(original.metadata),
        )

    @staticmethod
    def copy_annotation(original: Annotation) -> Annotation:
        """Copy a parent annotation occurrence as a value, with the handle of a stored value.

        The writer stores copies of equal content as one content row.

        Returns:
            The copy.
        """
        return replace(
            original,
            value=deepcopy(original.value),
            metadata=deepcopy(original.metadata),
            occurrence_metadata=deepcopy(original.occurrence_metadata),
        )


class _RecordCopy:
    """Bring the Sources, Signals, and annotations of one parent Record into a new copy of it."""

    def __init__(self, importer: _Importer, original: Record, target: Record) -> None:
        self.importer = importer
        self.original = original
        self.target = target
        self.held: dict[EntityId, Signal] = {}
        """The copy of each parent Signal that the Record's copy holds, by parent ID."""

    def merge(self) -> None:
        """Copy the Sources and Signals into the copy of the Record, then the annotations of the Record."""
        self._merge_sources(self.original.sources, self.target)
        self._attach_annotations(self.original, self.target)

    def _merge_sources(self, originals: Iterable[Source], target: Record | Source) -> None:
        """Put a copy of each parent Source below ``target``, with everything below the Source."""
        added: list[Source] = []
        for original in originals:
            copy = Source(name=original.name, metadata=deepcopy(original.metadata))
            self._merge_source(original, copy)
            added.append(copy)
        if added:
            target.sources = (*target.sources, *added)

    def _merge_source(self, original: Source, target: Source) -> None:
        """Copy the descendants of a parent Source into its copy, then the annotations of the Source."""
        self._merge_sources(original.sources, target)
        signals = []
        for signal in original.signals:
            self.held[signal.id] = self.importer.copy_signal(signal)
            signals.append(self.held[signal.id])
        if signals:
            target.signals = (*target.signals, *signals)
        self._attach_annotations(original, target)

    def _attach_annotations(self, original: Record | Source, target: Record | Source) -> None:
        """Attach copies of the annotations of a parent Record or Source to its copy.

        Args:
            original: The parent Record or Source.
            target: Its copy.
        """
        if original.annotations:
            copies = (self.importer.copy_annotation(annotation) for annotation in original.annotations)
            target.annotations = (*target.annotations, *copies)


def _parent_ids(selection: str, ids: Iterable[EntityId]) -> tuple[EntityId, ...]:
    """Return the parent IDs that a selection names, in order, each once.

    Returns:
        The IDs.

    Raises:
        TimeFValidationError: If ``ids`` is a string, because a string selects one ID per
            character, or if ``ids`` is not a collection of positive int64 IDs.
    """
    if isinstance(ids, str | bytes):
        raise TimeFValidationError(f"{selection}= takes a collection of parent IDs, not the string {ids!r}")
    try:
        selected = tuple(ids)
    except TypeError as exc:
        raise TimeFValidationError(f"{selection}= takes a collection of parent IDs, got {type(ids).__name__}") from exc
    return tuple(dict.fromkeys(check_id(identity, "parent id") for identity in selected))


def _resolve_alias(child: "TimeFDataset", parent: DatasetRef, alias: ParentAlias | None) -> ParentAlias:
    """Return the parent alias under which the card of the child declares ``parent``.

    Returns:
        The alias.

    Raises:
        TimeFValidationError: If the ``metadata.parents`` of the child or ``alias`` does not declare
            ``parent``, if the card declares ``parent`` under several aliases and ``alias`` is not
            given, if the child imports ``parent`` under another alias already, or if ``alias``
            imports another release already.
    """
    dataset_id = child.metadata.dataset_id
    declared = sorted(name for name, ref in child.metadata.parents.items() if ref == parent)
    if not declared:
        raise TimeFValidationError(
            f"{dataset_id} does not declare {parent} as a parent; declare it in metadata.parents to import from it"
        )
    if alias is None:
        if len(declared) > 1:
            raise TimeFValidationError(
                f"{dataset_id} declares {parent} under the parent aliases {declared}; pass alias= to choose one"
            )
        alias = declared[0]
    elif alias not in declared:
        raise TimeFValidationError(
            f"parent alias {alias!r} of {dataset_id} does not declare {parent}, which it declares as {declared}"
        )
    for other, card in child.parent_cards.items():
        if card.source == parent and other != alias:
            raise TimeFValidationError(f"{dataset_id} already imports {parent} under the parent alias {other!r}")
    card = child.parent_cards.get(alias)
    if card is not None and card.source != parent:
        raise TimeFValidationError(
            f"parent alias {alias!r} of {dataset_id} already imports {card.source}, not {parent}"
        )
    return alias


def import_from(
    child: "TimeFDataset",
    parent: "TimeFReader",
    *,
    alias: ParentAlias | None = None,
    records: Iterable[EntityId] = (),
    load: LoadPlan = LoadPlan.ALL_LAZY,
) -> ImportedDataset:
    """Copy the selected parent Records into ``child``, as :meth:`TimeFDataset.import_from` describes.

    Returns:
        The child copies of the selected Records, by their parent IDs.

    Raises:
        TimeFValidationError: If the parent is not a declared parent under ``alias``, the alias
            imports another release already, or the selection is not a collection of parent IDs.
        TimeFNotFoundError: If a selected ID names no parent object. The child does not change.
    """  # noqa: DOC502 - raised by the helpers and the parent's reader
    record_ids = _parent_ids("records", records)
    release = DatasetRef(dataset_id=parent.metadata.dataset_id, version=parent.metadata.dataset_version)
    alias = _resolve_alias(child, release, alias)
    originals = parent.get_many(Record, record_ids, load=load) if record_ids else ()
    # Every read succeeded. Record the parent first, so that the copies below always have its card,
    # also when a later copy fails.
    card = ParentDatasetCard(metadata=parent.metadata)
    child.parent_cards[alias] = card
    for ancestor in (*parent._manifest.lineage, card):
        child.lineage[str(ancestor.source)] = ancestor
    importer = _Importer(child, alias)
    for original in originals:
        importer.import_record(original)
    return ImportedDataset(
        alias=alias,
        parent=release,
        records={identity: importer.records[identity] for identity in record_ids},
    )
