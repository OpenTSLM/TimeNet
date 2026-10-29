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
from timenet.types import Annotation, TimeInterval
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
child, so no rule derives its ID from the parent ID. An import looks up a parent object here. Then
it finds the copy by its child ID among the objects of the dataset. If the dataset no longer holds
the copy, the import copies the object again. The map holds plain IDs, and ``dict(...)`` copies it.
The writer stores each entry whose copy the dataset holds in the ``imported_objects`` table. A
dataset that a reader reads back from storage gets these entries back.
"""


@dataclass(frozen=True)
class ImportedDataset:
    """The child copies of the objects that one ``import_from`` call selected, by their parent IDs.

    Each copy is one that the call made or one that the child held already. The view lists only what
    the call selected. For example, a Record that the call copied to hold a selected Signal is in the
    dataset, but not in ``records``.
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
        held: The child copy of each parent Signal that the child holds, by parent ID.

    Returns:
        The copy of each parent Source that the child holds, keyed by the ``id()`` of the parent Source.
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
    """Return whether the span of an annotation lies inside the windows of the given Signals.

    The window of a Signal is the time range that its values cover. The rule is the rule of
    ``annotate``, without its warning. An annotation without a span fits. If a Record has a
    ``time_span``, this session span replaces the windows of the Signals.

    Args:
        annotation: The annotation to check.
        signals: The Signals that the owner of the annotation holds.
        owner: The owner, named in the message of the check.
        time_span: The session span of the Record that owns the annotation, or ``None`` for a Source.
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
    """Return the fields that an import changes when it adds to the copy of a Record.

    The import restores these fields when it fails.

    Returns:
        The Record and each of its Sources, with its ``sources``, its ``annotations``, and, for a
        Source, its ``signals``.
    """
    return [
        (container, {name: getattr(container, name) for name in names})
        for container, names in (
            (record, ("sources", "annotations")),
            *((source, ("sources", "signals", "annotations")) for source in record.walk_sources()),
        )
    ]


class _Importer:
    """Copy parent Records into one child under one parent alias, and add to the copies that the child holds."""

    def __init__(self, child: "TimeFDataset", alias: ParentAlias) -> None:
        self.child = child
        self.alias = alias
        self.records: dict[EntityId, Record] = {}
        """The child copy of each parent Record that this call found or copied, by parent ID."""
        self.signals: dict[EntityId, Signal] = {}
        """The child copy of each parent Signal that this call found or copied, by parent ID."""

    def import_record(self, original: Record, wanted: frozenset[EntityId] | None) -> Record:
        """Copy a parent Record with the wanted Signals, or add them to the copy that the child holds.

        A new copy and its Signals get their IDs when the child registers the copy. A Signal that the
        call adds to a copy gets its ID when the child indexes the copy again. If the call fails to
        add to a copy, it restores the Sources, Signals, and annotations of the copy. The indexes
        and the ID counters of the dataset then do not change.

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
                # Time origins are frozen, like axes, so the copy shares the time origin of the parent
                # Record. Thus copies of Records that share a time origin also share it in the child.
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
    """Copy the wanted Sources and Signals of one parent Record into the copy of the Record in the child.

    The call finds the Signals that the copy holds in the hierarchy of the copy, through the import
    map. So a Signal that the child removed from the copy counts as missing, whatever the index of
    the dataset says.
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
        # A new copy holds no Signal yet. A registered copy holds Signals with child IDs.
        own = {signal.id: signal for signal in target.signals}
        self.held: dict[EntityId, Signal] = {
            parent_id: own[child_id]
            for parent_id, key in keys.items()
            if (child_id := imports.get(key)) is not None and child_id in own
        }
        """The copy of each parent Signal that the copy of the Record holds, by parent ID."""
        self.held_before = frozenset(self.held)
        """The parent IDs of the Signals that had a copy in the copy of the Record when the call began."""
        self.imported = {id(signal) for signal in self.held.values()}
        """The ``id()`` of each Signal copy in ``held``, which tells the imported Signals apart from the
        Signals that the child added."""
        self.pairs = _pair_sources(original, target, self.held)
        """The copy of each parent Source that the copy of the Record holds, keyed by the ``id()`` of the parent Source."""

    def merge(self, *, new: bool) -> None:
        """Copy the wanted Sources and Signals into the copy of the Record, then the annotations that they bring."""
        if self._merge_sources(self.original.sources, self.target) or new:
            self._attach_annotations(self.original, self.target, new=new)

    def _wants(self, signal: Signal) -> bool:
        """Return whether the call wants a parent Signal, and the copy of the Record lacks it."""
        return signal.id not in self.held and (self.wanted is None or signal.id in self.wanted)

    def _brings(self, original: Source) -> bool:
        """Return whether the call copies anything below a parent Source that has no copy yet.

        A Source without a Signal at any depth comes only with its whole Record.
        """
        signals = tuple(original.walk_signals())
        return any(self._wants(signal) for signal in signals) if signals else self.wanted is None

    def _merge_sources(self, originals: Iterable[Source], target: Record | Source) -> bool:
        """Copy the parent Sources below ``target``, or add to the copies of them that ``target`` holds.

        The call finds the copy of a parent Source through the copies of the Signals below it. If
        none of them finds a copy, the call looks for the name of the Source among the copies below
        ``target`` that hold no imported Signal. A Signal that the child added to a copy does not
        count. Sibling Sources have distinct names, so the name finds one copy at most.

        Returns:
            Whether ``target`` gained a Signal at any depth.
        """
        grew = False
        added: list[Source] = []
        unmatched = {
            source.name: source
            for source in target.sources
            if not any(id(signal) in self.imported for signal in source.walk_signals())
        }
        for original in originals:
            copy = self.pairs.get(id(original))
            if copy is None:
                copy = unmatched.pop(original.name, None)
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
        """Copy the wanted descendants of a parent Source into its copy, then the annotations that they bring.

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
        """Attach copies of the parent annotations that the new Signal copies of the call bring.

        The rules look at the Signals below ``original`` and at their copies. The copy of
        ``original`` keeps an annotation of ``original`` in these cases:

        - The copy holds a copy of every Signal below ``original``.
        - The annotation has no span.
        - The span of the annotation lies inside the windows of the Signals that have a copy. For a
          Record with a ``time_span``, the session span replaces the windows.

        A new copy gets every annotation that the rules keep. A copy that the child held before the
        call gets only the annotations that the rules keep now and did not keep when the call began.
        So its other annotations stay as the child left them. This includes the annotations that the
        child deleted.

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


def import_from(  # noqa: PLR0913 - one argument per selection
    child: "TimeFDataset",
    parent: "TimeFReader",
    *,
    alias: ParentAlias | None = None,
    records: Iterable[EntityId] = (),
    signals: Iterable[EntityId] = (),
    load: LoadPlan = LoadPlan.ALL_LAZY,
) -> ImportedDataset:
    """Copy the selected parent Records and Signals into ``child``, as :meth:`TimeFDataset.import_from` describes.

    Returns:
        The child copies of the selected objects, by their parent IDs.

    Raises:
        TimeFValidationError: If the parent is not a declared parent under ``alias``, the alias
            imports another release already, or a selection is not a collection of parent IDs.
        TimeFNotFoundError: If a selected ID names no parent object. The child does not change.
    """  # noqa: DOC502 - raised by the helpers and the parent's reader
    record_ids = _parent_ids("records", records)
    signal_ids = _parent_ids("signals", signals)
    release = DatasetRef(dataset_id=parent.metadata.dataset_id, version=parent.metadata.dataset_version)
    alias = _resolve_alias(child, release, alias)
    # The parent IDs of the Signals to copy from each Record, or None for the whole Record.
    wanted: dict[EntityId, set[EntityId] | None] = dict.fromkeys(record_ids)
    holders = parent._record_ids_of_signals(signal_ids) if signal_ids else {}
    for signal_id in signal_ids:
        partial = wanted.setdefault(holders[signal_id], set())
        if partial is not None:
            partial.add(signal_id)
    originals = parent.get_many(Record, tuple(wanted), load=load) if wanted else ()
    # Every read succeeded. Record the parent first, so that the copies below always have its card,
    # also when a later copy fails.
    card = ParentDatasetCard(metadata=parent.metadata)
    child.parent_cards[alias] = card
    for ancestor in (*parent._manifest.lineage, card):
        child.lineage[str(ancestor.source)] = ancestor
    importer = _Importer(child, alias)
    for original in originals:
        partial = wanted[original.id]
        importer.import_record(original, None if partial is None else frozenset(partial))
    return ImportedDataset(
        alias=alias,
        parent=release,
        records={identity: importer.records[identity] for identity in record_ids},
        signals={identity: importer.signals[identity] for identity in signal_ids},
    )
