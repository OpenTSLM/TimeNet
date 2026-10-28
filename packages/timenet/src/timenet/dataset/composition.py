"""Copy selected parent Records into a child dataset, sharing their stored values."""

from collections.abc import Iterable
from copy import deepcopy
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, TypeAlias

from timenet.dataset.record import Record
from timenet.dataset.source import Source
from timenet.dataset.time_series import Signal
from timenet.errors import TimeFValidationError
from timenet.types import Access, Annotation, DatasetMetadata
from timenet.types.ids import EntityId, check_id
from timenet.types.object_kind import ObjectKind
from timenet.types.parent_dataset_card import ParentDatasetCard
from timenet.types.reference_keys import ParentAlias
from timenet.types.references import DatasetRef


if TYPE_CHECKING:
    from timenet.dataset.dataset import TimeFDataset
    from timenet.reader import TimeFReader


ImportMap: TypeAlias = dict[tuple[ParentAlias, ObjectKind, EntityId], EntityId]
"""The child ID of each imported Record and Signal, keyed by parent alias, kind, and parent ID.

It is the one record of what a dataset imported. A copy's ID is the next ID of its kind in the child,
so nothing derives it from the parent's: an import looks a parent Record up here and then finds the
copy by its child ID among the dataset's own Records, and a copy the dataset no longer holds is
copied again. The map holds plain IDs and copies with ``dict(...)``. The writer stores the entries
whose copy the dataset still holds in the ``imported_objects`` table.
"""

_STRICTNESS = {Access.OPEN: 0, Access.CREDENTIALED: 1, Access.RESTRICTED: 2}


@dataclass(frozen=True)
class ImportedDataset:
    """The child copies of the Records that one ``import_from`` call selected, by their parent IDs.

    Each copy is one the call made or one the child already held.
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
        """The child copy of each parent Record this call found or copied, by parent ID."""

    def import_record(self, original: Record) -> Record:
        """Copy a parent Record whole, or find the child's copy of it.

        A new copy gets its ID, and its Signals theirs, when the child registers it. When the child
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
                # Origins are frozen and shared explicitly, like axes, so the copy keeps its parent's:
                # copies of Records that share a clock share it as well.
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
        """Copy a parent Signal without an ID, keeping its axis and the handles to its stored values.

        Returns:
            The copy, with copies of its annotations. The child gives it an ID when it indexes it.
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
        """Copy a parent annotation occurrence as a value, keeping a stored value's handle.

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
        """Bring the Sources and Signals into the Record's copy, then the Record's annotations."""
        self._merge_sources(self.original.sources, self.target)
        self._attach_annotations(self.original, self.target)

    def _merge_sources(self, originals: Iterable[Source], target: Record | Source) -> None:
        """Bring a copy of each parent Source below ``target``, with everything below the Source."""
        added: list[Source] = []
        for original in originals:
            copy = Source(name=original.name, metadata=deepcopy(original.metadata))
            self._merge_source(original, copy)
            added.append(copy)
        if added:
            target.sources = (*target.sources, *added)

    def _merge_source(self, original: Source, target: Source) -> None:
        """Bring the descendants of a parent Source into its copy, then the Source's annotations."""
        self._merge_sources(original.sources, target)
        signals = []
        for signal in original.signals:
            self.held[signal.id] = self.importer.copy_signal(signal)
            signals.append(self.held[signal.id])
        if signals:
            target.signals = (*target.signals, *signals)
        self._attach_annotations(original, target)

    def _attach_annotations(self, original: Record | Source, target: Record | Source) -> None:
        """Attach copies of the parent annotations of a Record or Source to its copy.

        Args:
            original: The parent Record or Source.
            target: Its copy.
        """
        if original.annotations:
            copies = (self.importer.copy_annotation(annotation) for annotation in original.annotations)
            target.annotations = (*target.annotations, *copies)


def _parent_ids(selection: str, ids: Iterable[EntityId]) -> tuple[EntityId, ...]:
    """Return the parent IDs a selection names, in order, each once.

    Returns:
        The IDs.

    Raises:
        TimeFValidationError: If ``ids`` is a string, which would select one ID per character, or is
            not a collection of positive int64 IDs.
    """
    if isinstance(ids, str | bytes):
        raise TimeFValidationError(f"{selection}= takes a collection of parent IDs, not the string {ids!r}")
    try:
        selected = tuple(ids)
    except TypeError as exc:
        raise TimeFValidationError(f"{selection}= takes a collection of parent IDs, got {type(ids).__name__}") from exc
    return tuple(dict.fromkeys(check_id(identity, "parent id") for identity in selected))


def _resolve_alias(child: "TimeFDataset", parent: DatasetRef, alias: ParentAlias | None) -> ParentAlias:
    """Return the parent alias under which the child's card declares ``parent``.

    Returns:
        The alias.

    Raises:
        TimeFValidationError: If the child's ``metadata.parents`` does not declare ``parent``, ``alias``
            does not declare it, the card declares it under several aliases and ``alias``
            is omitted, the child already imports it under another alias, or ``alias`` already
            imports another release.
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


def _inherit_access(child: "TimeFDataset", parent: DatasetMetadata) -> None:
    """Raise the child's access requirement to the parent's when the parent's is stricter.

    The child has one ``access_url``: its own, or that of the first parent that made it stricter. A
    parent that is no stricter keeps its access requirement and page in its parent card.
    """
    if _STRICTNESS[parent.access] > _STRICTNESS[child.metadata.access]:
        child._metadata = child.metadata.model_copy(update={"access": parent.access, "access_url": parent.access_url})


def import_from(
    child: "TimeFDataset",
    parent: "TimeFReader",
    *,
    alias: ParentAlias | None = None,
    records: Iterable[EntityId] = (),
) -> ImportedDataset:
    """Copy the selected parent Records into ``child``, as :meth:`TimeFDataset.import_from` describes.

    Returns:
        The child copies of the selected Records, by their parent IDs.

    Raises:
        TimeFValidationError: If the parent is not a declared parent under ``alias``, the alias
            imports another release already, or the selection is not a collection of parent IDs.
        TimeFNotFoundError: If a selected ID names no parent object. The child is left unchanged.
    """  # noqa: DOC502 - raised by the helpers and the parent's reader
    record_ids = _parent_ids("records", records)
    release = DatasetRef(dataset_id=parent.metadata.dataset_id, version=parent.metadata.dataset_version)
    alias = _resolve_alias(child, release, alias)
    originals = parent.get_many(Record, record_ids) if record_ids else ()
    # Every read succeeded. Record the parent first, so the copies below always have their card,
    # even if a later one fails.
    card = ParentDatasetCard(metadata=parent.metadata)
    child.parent_cards[alias] = card
    for ancestor in (*parent._manifest.lineage, card):
        child.lineage[str(ancestor.source)] = ancestor
    _inherit_access(child, parent.metadata)
    importer = _Importer(child, alias)
    for original in originals:
        importer.import_record(original)
    return ImportedDataset(
        alias=alias,
        parent=release,
        records={identity: importer.records[identity] for identity in record_ids},
    )
