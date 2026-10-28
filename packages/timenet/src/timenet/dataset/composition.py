"""Copy selected parent Records and Signals into a child dataset, sharing their stored values."""

from collections.abc import Iterable, Iterator
from copy import deepcopy
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, TypeAlias

from timenet.dataset.record import Record
from timenet.dataset.source import Source
from timenet.dataset.span_validation import check_span_within_window
from timenet.dataset.time_series import Signal
from timenet.errors import TimeFValidationError
from timenet.types import Access, Annotation, DatasetMetadata, TimeInterval
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

It is the one record of what a dataset imported. A copy's ID is the next ID of its kind in the child,
so nothing derives it from the parent's: an import looks a parent object up here and then finds the
copy by its child ID among the dataset's own objects, and a copy the dataset no longer holds is
copied again. The map holds plain IDs and copies with ``dict(...)``. The writer stores the entries
whose copy the dataset holds in the ``imported_objects`` table, and a dataset read back from storage
gets them back.
"""

_STRICTNESS = {Access.OPEN: 0, Access.CREDENTIALED: 1, Access.RESTRICTED: 2}


@dataclass(frozen=True)
class ImportedDataset:
    """The child copies of the objects that one ``import_from`` call selected, by their parent IDs.

    Each copy is one the call made or one the child already held. The view lists only what the call
    selected: a Record the call copied to hold a selected Signal is in the dataset, not in ``records``.
    """

    alias: ParentAlias
    """The parent alias the objects were imported under."""
    parent: DatasetRef
    """The exact parent release."""
    records: dict[EntityId, Record] = field(default_factory=dict)
    """The copy of each Record selected with ``records=``, keyed by its parent ID."""
    signals: dict[EntityId, Signal] = field(default_factory=dict)
    """The copy of each Signal selected with ``signals=``, keyed by its parent ID."""


def _signal_paths(record: Record) -> Iterator[tuple[tuple[Source, ...], Signal]]:
    """Yield each Signal of a Record with the Sources above it, from its root Source down.

    Yields:
        The Sources from the root to the Signal's own Source, and the Signal.
    """
    pending: list[tuple[tuple[Source, ...], Source]] = [((source,), source) for source in record.sources]
    while pending:
        path, source = pending.pop()
        for signal in source.signals:
            yield path, signal
        pending.extend(((*path, child), child) for child in source.sources)


def _pair_sources(original: Record, copy: Record, held: dict[EntityId, Signal]) -> dict[int, Source]:
    """Match the parent Sources to their copies, through the copies of the Signals below them.

    Args:
        original: The parent Record.
        copy: The child's copy of it.
        held: The child copy of each parent Signal the child holds, by parent ID.

    Returns:
        The copy of each parent Source the child holds, keyed by the ``id()`` of the parent Source.
    """
    copied_paths = {id(signal): path for path, signal in _signal_paths(copy)}
    pairs: dict[int, Source] = {}
    for path, signal in _signal_paths(original):
        held_signal = held.get(signal.id)
        copied_path = None if held_signal is None else copied_paths.get(id(held_signal))
        if copied_path is not None and len(copied_path) == len(path):
            pairs.update(zip(map(id, path), copied_path, strict=True))
    return pairs


def _fits(annotation: Annotation, signals: tuple[Signal, ...], owner: str, time_span: TimeInterval | None) -> bool:
    """Return whether an annotation's span lies inside the windows of the given Signals.

    The rule is the one ``annotate`` applies, without its warning: an annotation without a span
    fits, and a Record's ``time_span``, when it has one, replaces the Signals' windows.

    Args:
        annotation: The annotation to place.
        signals: The Signals its owner holds.
        owner: The owner, named for the check's message.
        time_span: The owning Record's session span, or ``None`` for a Source.
    """
    if annotation.span is None:
        return True
    try:
        check_span_within_window(
            f"annotation {annotation.key!r}", annotation.span, signals, owner, time_span, warn_when_outside=False
        )
    except TimeFValidationError:
        return False
    return True


def _hierarchy(record: Record) -> list[tuple[Record | Source, dict[str, tuple]]]:
    """Return the fields that an extension changes, for each container of a Record.

    Returns:
        Each container with its ``sources``, ``annotations``, and, for a Source, ``signals``.
    """
    return [
        (container, {name: getattr(container, name) for name in names})
        for container, names in (
            (record, ("sources", "annotations")),
            *((source, ("sources", "signals", "annotations")) for source in record.walk_sources()),
        )
    ]


class _Importer:
    """Copy parent Records into one child under one parent alias, extending the copies the child holds."""

    def __init__(self, child: "TimeFDataset", alias: ParentAlias, release: DatasetRef) -> None:
        self.child = child
        self.alias = alias
        self.release = release
        """The exact parent release."""
        self.records: dict[EntityId, Record] = {}
        """The child copy of each parent Record this call found or copied, by parent ID."""
        self.signals: dict[EntityId, Signal] = {}
        """The child copy of each parent Signal this call found or copied, by parent ID."""

    def import_record(self, original: Record, wanted: frozenset[EntityId] | None) -> Record:
        """Copy a parent Record, or extend the child's copy of it, with the wanted Signals.

        A new copy gets its ID, and its Signals theirs, when the child registers it; a Signal that
        extends a copy gets its ID when the child indexes the copy again. If extending a copy fails,
        its Sources, Signals, and annotations are restored as they were, and the dataset's indexes
        and ID counters are unchanged.

        Args:
            original: The parent Record, as the parent's reader built it.
            wanted: The parent IDs of the Signals to bring, or ``None`` for the whole Record.

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
            copied = _RecordCopy(self, original, target, wanted)
            copied.merge(new=True)
            self.child.add_record(record=target)
        else:
            saved = _hierarchy(target)
            try:
                copied = _RecordCopy(self, original, target, wanted)
                copied.merge(new=False)
                self.child._reindex_record(target)
            except BaseException:
                for container, fields in saved:
                    for name, value in fields.items():
                        setattr(container, name, value)
                raise
        self.records[original.id] = target
        self.signals.update(copied.held)
        imports = self.child._imports
        imports[self.alias, ObjectKind.RECORD, original.id] = target.id
        for parent_id, signal in copied.held.items():
            imports[self.alias, ObjectKind.SIGNAL, parent_id] = signal.id
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
    """Bring the wanted Sources and Signals of one parent Record into the child's copy of it.

    The copy's held Signals come from its own hierarchy, through the import map, so a Signal the
    child removed from the copy counts as not held, whatever the dataset's index says.
    """

    def __init__(
        self, importer: _Importer, original: Record, target: Record, wanted: frozenset[EntityId] | None
    ) -> None:
        self.importer = importer
        self.original = original
        self.target = target
        self.wanted = wanted
        """The parent IDs of the Signals to bring, or ``None`` for the whole Record."""
        imports = importer.child._imports
        keys = {signal.id: (importer.alias, ObjectKind.SIGNAL, signal.id) for signal in original.signals}
        # A new copy holds no Signal yet; a registered one holds Signals with child IDs.
        own = {signal.id: signal for signal in target.signals}
        self.held: dict[EntityId, Signal] = {
            parent_id: own[child_id]
            for parent_id, key in keys.items()
            if (child_id := imports.get(key)) is not None and child_id in own
        }
        """The copy of each parent Signal that the Record's copy holds, by parent ID."""
        self.held_before = frozenset(self.held)
        """The parent IDs of the Signals whose copy the Record's copy held when the call began."""
        self.imported = {id(signal) for signal in self.held.values()}
        """The ``id()`` of each held copy, which tells imported Signals from those the child added."""
        self.pairs = _pair_sources(original, target, self.held)
        """The copy of each parent Source the Record's copy holds, keyed by the ``id()`` of the parent Source."""

    def merge(self, *, new: bool) -> None:
        """Bring the wanted Sources and Signals into the Record's copy, then the annotations they place."""
        if self._merge_sources(self.original.sources, self.target) or new:
            self._attach_annotations(self.original, self.target, new=new)

    def _wants(self, signal: Signal) -> bool:
        """Return whether the call brings a parent Signal of which the copy holds no copy yet."""
        return signal.id not in self.held and (self.wanted is None or signal.id in self.wanted)

    def _brings(self, original: Source) -> bool:
        """Return whether a parent Source without a copy has anything for this call to bring.

        A Source that holds no Signal at any depth comes with its whole Record only.
        """
        signals = tuple(original.walk_signals())
        return any(self._wants(signal) for signal in signals) if signals else self.wanted is None

    def _merge_sources(self, originals: Iterable[Source], target: Record | Source) -> bool:
        """Bring the parent Sources below ``target``: extend the copies it holds, and copy the others.

        A copy is found through the copies of the Signals below it. A parent Source that none of
        them finds is matched by name among the copies below ``target`` that hold no imported
        Signal and are not matched yet; Signals the child added to a copy do not count. When
        several parent Sources, or several copies, share the name, the name can't tell them apart:
        the copies stay as they are if the call brings no Signal below the Source, and the call
        raises otherwise.

        Returns:
            Whether ``target`` gained a Signal at any depth.

        Raises:
            TimeFValidationError: If the call brings a Signal below a parent Source whose copy the
                name can't tell apart from a sibling's.
        """
        originals = tuple(originals)
        grew = False
        added: list[Source] = []
        unmatched = [
            source
            for source in target.sources
            if not any(id(signal) in self.imported for signal in source.walk_signals())
        ]
        unpaired = [original.name for original in originals if id(original) not in self.pairs]
        for original in originals:
            copy = self.pairs.get(id(original))
            if copy is None:
                candidates = [source for source in unmatched if source.name == original.name]
                if len(candidates) > 1 or (candidates and unpaired.count(original.name) > 1):
                    if not any(self._wants(signal) for signal in original.walk_signals()):
                        continue
                    raise TimeFValidationError(
                        f"an import cannot tell apart the Sources named {original.name!r} of Record "
                        f"{self.original.id!r} of {self.importer.release}: no imported Signal identifies "
                        "their copies, and the name does not"
                    )
                copy = candidates[0] if candidates else None
                if copy is not None:
                    unmatched.remove(copy)
            if copy is not None:
                grew = self._merge_source(original, copy, new=False) or grew
            elif self._brings(original):
                copy = Source(name=original.name, metadata=deepcopy(original.metadata))
                grew = self._merge_source(original, copy, new=True) or grew
                added.append(copy)
        if added:
            target.sources = (*target.sources, *added)
        return grew

    def _merge_source(self, original: Source, target: Source, *, new: bool) -> bool:
        """Bring the wanted descendants of a parent Source into its copy, then the annotations they place.

        Returns:
            Whether ``target`` gained a Signal at any depth.
        """
        grew = self._merge_sources(original.sources, target)
        signals = []
        for signal in original.signals:
            if self._wants(signal):
                self.held[signal.id] = self.importer.copy_signal(signal)
                signals.append(self.held[signal.id])
        if signals:
            target.signals = (*target.signals, *signals)
            grew = True
        if new or grew:
            self._attach_annotations(original, target, new=new)
        return grew

    def _attach_annotations(self, original: Record | Source, target: Record | Source, *, new: bool) -> None:
        """Attach copies of the parent annotations that the call's new Signal copies place.

        Placement is judged on the parent's Signals below ``original``: the copy places an
        annotation when it holds a copy of each of them, or when the annotation's span lies inside
        the windows of those it holds a copy of. A new copy gets every annotation it places. An
        existing copy gets those it places now and did not place with the Signals it held when the
        call began, so its other annotations stay as the child left them, deleted ones included.

        Args:
            original: The parent Record or Source.
            target: Its copy.
            new: Whether the call created the copy.
        """
        signals = tuple(original.walk_signals())
        after = tuple(signal for signal in signals if signal.id in self.held)
        before = () if new else tuple(signal for signal in signals if signal.id in self.held_before)
        if isinstance(original, Record):
            time_span, owner = original.time_span, original._label
        else:
            time_span, owner = None, f"source {original.name!r}"

        def places(annotation: Annotation, held: tuple[Signal, ...]) -> bool:
            return len(held) == len(signals) or _fits(annotation, held, owner, time_span)

        placed = [
            annotation
            for annotation in original.annotations
            if places(annotation, after) and (new or not places(annotation, before))
        ]
        if placed:
            copies = (self.importer.copy_annotation(annotation) for annotation in placed)
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


def import_from(  # noqa: PLR0913 - one argument per selection
    child: "TimeFDataset",
    parent: "TimeFReader",
    *,
    alias: ParentAlias | None = None,
    records: Iterable[EntityId] = (),
    signals: Iterable[EntityId] = (),
    load: LoadPlan = LoadPlan.ALL_LAZY,
) -> ImportedDataset:
    """Copy the selected parent Records and Signals into ``child``; see :meth:`TimeFDataset.import_from`.

    Returns:
        The child copies of the selected objects, by their parent IDs.

    Raises:
        TimeFValidationError: If the parent is not a declared parent under ``alias``, the alias
            imports another release already, or a selection is not a collection of parent IDs.
        TimeFNotFoundError: If a selected ID names no parent object. The child is left unchanged.
    """  # noqa: DOC502 - raised by the helpers and the parent's reader
    record_ids = _parent_ids("records", records)
    signal_ids = _parent_ids("signals", signals)
    release = DatasetRef(dataset_id=parent.metadata.dataset_id, version=parent.metadata.dataset_version)
    alias = _resolve_alias(child, release, alias)
    # The parent IDs of the Signals to bring from each Record, or None for the whole Record.
    wanted: dict[EntityId, set[EntityId] | None] = dict.fromkeys(record_ids)
    holders = parent._record_ids_of_signals(signal_ids) if signal_ids else {}
    for signal_id in signal_ids:
        partial = wanted.setdefault(holders[signal_id], set())
        if partial is not None:
            partial.add(signal_id)
    originals = parent.get_many(Record, tuple(wanted), load=load) if wanted else ()
    # Every read succeeded. Record the parent first, so the copies below always have their card,
    # even if a later one fails.
    card = ParentDatasetCard(metadata=parent.metadata)
    child.parent_cards[alias] = card
    for ancestor in (*parent._manifest.lineage, card):
        child.lineage[str(ancestor.source)] = ancestor
    _inherit_access(child, parent.metadata)
    importer = _Importer(child, alias, release)
    for original in originals:
        partial = wanted[original.id]
        importer.import_record(original, None if partial is None else frozenset(partial))
    return ImportedDataset(
        alias=alias,
        parent=release,
        records={identity: importer.records[identity] for identity in record_ids},
        signals={identity: importer.signals[identity] for identity in signal_ids},
    )
