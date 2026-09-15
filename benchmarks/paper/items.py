"""The canonical item, and the enumerator that walks it.

The canonical item is the smallest unit a task consumes, fixed per dataset before anything runs. It
is what an items/s rate counts, what a block is measured in and what parity compares. A
representation that stores a coarser unit pays the cost of cutting it.

Both representations of a row must yield the *same* items, not merely the same number of them.
Counts that disagree make the rate columns meaningless, so the registry gates on them, and the
enumerators emit stable item ids so the two sides can be compared rather than trusted.

The enumerator is the layer the draft skipped, and it is where four of six rows live. A count is a
number in a table; an enumerator is what a lane iterates, what a block sampler permutes, and what
proves the two sides hold the same corpus.

A ``task`` carrier is bounded by :data:`TASK_CARRIER_LIMIT`, but only against a reader that cannot
stream tasks. The bound is a property of the SDK, not a property of the corpus, so
:func:`reader_streams_tasks` decides whether it applies instead of a number someone has to remember
to update.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass, fields
from typing import Any

from benchmarks.paper.errors import ItemRegistryError


RECORD = "record"
ANNOTATION = "annotation"
TASK = "task"
CARRIERS = (RECORD, ANNOTATION, TASK)

TASK_CARRIER_LIMIT = 50_000
"""Largest item count a task carrier may declare against a reader that cannot stream tasks.

A reader that has to decode every task partition to reach one record makes the first-item cell
O(n_tasks). This bounds how much of that a row may buy. It does not apply to a reader that can
stream, which :func:`reader_streams_tasks` reports.
"""


def reader_streams_tasks() -> bool:
    """Return whether the installed SDK reaches one record's tasks without decoding the rest.

    Two pieces have to be present. ``TimeFReader.tasks_for_records`` walks the task partitions one
    row group at a time and stops at the group that completes the record's ids. The ``task_index``
    file group records which task row groups hold a given record's tasks, which is the only way a
    streamed write can be looked up at all, because a streamed write stores no ``Record.task_ids``.

    Returns:
        True when both are present.
    """
    try:
        from timenet.manifest.files import ManifestFiles  # noqa: PLC0415 - probed, not used
        from timenet.reader import TimeFReader  # noqa: PLC0415 - same
    except ImportError:
        return False
    indexed = any(field.name == "task_index" for field in fields(ManifestFiles))
    return indexed and callable(getattr(TimeFReader, "tasks_for_records", None))


@dataclass(frozen=True)
class ItemRef:
    """One canonical item, addressed in a way the lane can act on.

    The ``item_id`` is the identity the two representations share. It is not a path, an index or a
    row number, because those differ across representations of the same corpus.

    Attributes:
        item_id: Identity of the item, equal on both sides of the row.
        ordinal: Position in the declared enumeration order, counting from zero.
        record_id: The TimeF record this item sits on, when it sits on one.
        start_step: First temporal step of the item, for an item cut out of a longer series.
        stop_step: Last temporal step, exclusive.
        label: The annotation value the item carries, when the carrier is an annotation.
    """

    item_id: str
    ordinal: int
    record_id: str = ""
    start_step: int | None = None
    stop_step: int | None = None
    label: str = ""

    def __post_init__(self) -> None:
        """Refuse an item that cannot be addressed.

        Raises:
            ItemRegistryError: If the id is empty, the ordinal is negative, or the step range is
                empty or reversed.
        """
        if not self.item_id.strip():
            raise ItemRegistryError("an item needs an id that both representations can agree on")
        if self.ordinal < 0:
            raise ItemRegistryError(f"item {self.item_id} has ordinal {self.ordinal}")
        if self.start_step is not None and self.stop_step is not None and self.stop_step <= self.start_step:
            raise ItemRegistryError(
                f"item {self.item_id} covers steps [{self.start_step}, {self.stop_step}), which is empty"
            )

    def as_dict(self) -> dict[str, Any]:
        """Return the JSON form.

        Returns:
            A plain mapping of every field.
        """
        return {
            "item_id": self.item_id,
            "ordinal": self.ordinal,
            "record_id": self.record_id,
            "start_step": self.start_step,
            "stop_step": self.stop_step,
            "label": self.label,
        }


Enumerate = Callable[[], Iterator[ItemRef]]
"""What an entry supplies: a fresh iterator over the items, in the declared enumeration order."""


@dataclass(frozen=True)
class ItemSpec:
    """What one representation of one dataset declares about its items.

    Attributes:
        dataset: Dataset id.
        representation: ``original``, ``timef`` or ``control``.
        carrier: Which TimeF object carries the item: ``record``, ``annotation`` or ``task``.
        n_items: How many canonical items this representation holds.
        citation: File and line in an external, citable loader that defines the item.
        enumerate_items: Returns a fresh iterator over the items.
        order: One sentence naming the declared enumeration order.
    """

    dataset: str
    representation: str
    carrier: str
    n_items: int
    citation: str
    enumerate_items: Enumerate
    order: str = ""

    def __post_init__(self) -> None:
        """Refuse a declaration the campaign cannot stand behind.

        Raises:
            ItemRegistryError: If the carrier is unknown, the count is not positive, the citation is
                missing, or a task carrier declares more items than this reader can reach cheaply.
        """
        if self.carrier not in CARRIERS:
            raise ItemRegistryError(f"carrier must be one of {list(CARRIERS)}, got {self.carrier!r}")
        if self.n_items < 1:
            raise ItemRegistryError(f"{self.dataset}/{self.representation} declares {self.n_items} items")
        if not self.citation.strip():
            raise ItemRegistryError(
                f"{self.dataset}/{self.representation} must cite the external loader line that defines its item"
            )
        if self.carrier == TASK and self.n_items > TASK_CARRIER_LIMIT and not reader_streams_tasks():
            raise ItemRegistryError(
                f"{self.dataset}/{self.representation} carries {self.n_items} items on tasks, above the "
                f"{TASK_CARRIER_LIMIT} limit that applies to a reader without a streaming task read. "
                "This SDK has no TimeFReader.tasks_for_records or no task index, so it decodes every "
                "task partition to reach one record and the first-item cell would be O(n_tasks)"
            )

    def items(self) -> Iterator[ItemRef]:
        """Walk this representation's items.

        Returns:
            An iterator over one reference per canonical item, in the declared order.
        """
        return self.enumerate_items()

    def item_ids(self) -> tuple[str, ...]:
        """Walk the items and collect their ids.

        Returns:
            The ids, in enumeration order.

        Raises:
            ItemRegistryError: If the walk does not yield exactly ``n_items``, or the ordinals are
                not 0, 1, 2 and so on.
        """
        ids: list[str] = []
        for expected, ref in enumerate(self.items()):
            if ref.ordinal != expected:
                raise ItemRegistryError(
                    f"{self.dataset}/{self.representation} enumerated {ref.item_id} at ordinal "
                    f"{ref.ordinal}, expected {expected}"
                )
            ids.append(ref.item_id)
        if len(ids) != self.n_items:
            raise ItemRegistryError(
                f"{self.dataset}/{self.representation} declares {self.n_items} items but enumerated {len(ids)}"
            )
        duplicates = len(ids) - len(set(ids))
        if duplicates:
            raise ItemRegistryError(
                f"{self.dataset}/{self.representation} enumerated {duplicates} duplicate item id(s)"
            )
        return tuple(ids)


class ItemRegistry:
    """The item declarations of a campaign, and the checks that hold them together."""

    def __init__(self) -> None:
        self._specs: dict[tuple[str, str], ItemSpec] = {}

    def register(self, spec: ItemSpec) -> ItemSpec:
        """Add one declaration.

        Args:
            spec: The declaration.

        Returns:
            The same spec, so a caller can register and keep it in one line.

        Raises:
            ItemRegistryError: If that dataset and representation is already registered.
        """
        key = (spec.dataset, spec.representation)
        if key in self._specs:
            raise ItemRegistryError(f"{spec.dataset}/{spec.representation} is already registered")
        self._specs[key] = spec
        return spec

    def spec(self, dataset: str, representation: str) -> ItemSpec:
        """Return one declaration.

        Returns:
            The spec.

        Raises:
            ItemRegistryError: If nothing is registered for that pair.
        """
        try:
            return self._specs[dataset, representation]
        except KeyError:
            raise ItemRegistryError(f"nothing is registered for {dataset}/{representation}") from None

    def representations(self, dataset: str) -> tuple[str, ...]:
        """Return every representation registered for a dataset.

        Returns:
            The representation names, sorted.
        """
        return tuple(sorted(rep for ds, rep in self._specs if ds == dataset))

    def datasets(self) -> tuple[str, ...]:
        """Return every dataset that has at least one declaration.

        Returns:
            The dataset ids, sorted.
        """
        return tuple(sorted({dataset for dataset, _ in self._specs}))

    def enumerate_items(self, dataset: str, representation: str) -> Iterator[ItemRef]:
        """Walk one representation's items.

        Returns:
            An iterator over one reference per canonical item, in the declared order.
        """
        return self.spec(dataset, representation).items()

    def check_counts(self, dataset: str) -> int:
        """Refuse a row whose representations disagree on how many items they hold.

        This is the cheap check, run before every campaign. It compares declarations, not walks.

        Returns:
            The item count the row agrees on.

        Raises:
            ItemRegistryError: If the dataset has no declaration, or two of them disagree.
        """
        specs = [spec for (ds, _), spec in sorted(self._specs.items()) if ds == dataset]
        if not specs:
            raise ItemRegistryError(f"{dataset} has no item declaration")
        counts = {spec.n_items: spec.representation for spec in specs}
        if len(counts) > 1:
            listed = ", ".join(f"{count} ({where})" for count, where in sorted(counts.items()))
            raise ItemRegistryError(
                f"{dataset} declares more than one item count: {listed}. The rate columns count items, "
                "so two counts make them incomparable"
            )
        return specs[0].n_items

    def check_items(self, dataset: str) -> tuple[str, ...]:
        """Walk every representation of a row and refuse any that holds different items.

        This is the expensive check. It runs once per campaign, not once per observation, and it is
        what turns "both sides report 1,005" into "both sides hold the same 1,005".

        Returns:
            The item ids, taken from the first representation in sorted order.

        Raises:
            ItemRegistryError: If two representations hold different items.
        """
        self.check_counts(dataset)
        walked: dict[str, tuple[str, ...]] = {}
        for representation in self.representations(dataset):
            walked[representation] = self.spec(dataset, representation).item_ids()
        names = list(walked)
        first = names[0]
        for other in names[1:]:
            missing = set(walked[first]) - set(walked[other])
            extra = set(walked[other]) - set(walked[first])
            if missing or extra:
                raise ItemRegistryError(
                    f"{dataset}: {first} and {other} hold different items. "
                    f"{len(missing)} only in {first} (for example {_example(missing)}), "
                    f"{len(extra)} only in {other} (for example {_example(extra)})"
                )
        return walked[first]


def record_items(record_ids: Iterable[str]) -> Enumerate:
    """Return an enumerator for a record-carried item.

    One item per TimeF record, in stored order. That is the carrier for four of the six rows, and
    it is the one whose first item is O(1) on the TimeF side.

    Args:
        record_ids: The record ids, in the declared enumeration order.

    Returns:
        A callable that yields one item per record.
    """
    ordered = tuple(record_ids)

    def walk() -> Iterator[ItemRef]:
        for ordinal, record_id in enumerate(ordered):
            yield ItemRef(item_id=record_id, ordinal=ordinal, record_id=record_id)

    return walk


def task_items(task_ids: Iterable[str], record_of: Sequence[str] | None = None) -> Enumerate:
    """Return an enumerator for a task-carried item.

    Args:
        task_ids: The task ids, in the declared enumeration order.
        record_of: The record each task sits on, in the same order. Optional.

    Returns:
        A callable that yields one item per task.

    Raises:
        ItemRegistryError: If ``record_of`` is given and is a different length.
    """
    ordered = tuple(task_ids)
    owners = tuple(record_of) if record_of is not None else ()
    if owners and len(owners) != len(ordered):
        raise ItemRegistryError(f"{len(ordered)} tasks but {len(owners)} owning records")

    def walk() -> Iterator[ItemRef]:
        for ordinal, task_id in enumerate(ordered):
            yield ItemRef(
                item_id=task_id,
                ordinal=ordinal,
                record_id=owners[ordinal] if owners else "",
            )

    return walk


@dataclass(frozen=True)
class EpochGrid:
    """How a run-length annotation is cut into fixed-length items.

    Sleep-EDF stores 197 whole-night records and scores them in 30-second epochs. The epochs are not
    stored one per row: the annotation is run-length, one interval per stretch of one stage. So the
    grid is built from the annotations rather than read off the artifact, and an epoch that runs
    past the signals it scores is dropped on both sides.

    Attributes:
        key: The annotation key that carries the label.
        epoch_us: Length of one epoch, in microseconds.
        keep_labels: Labels that produce an item. Everything else is dropped.
    """

    key: str
    epoch_us: int
    keep_labels: frozenset[str]

    def __post_init__(self) -> None:
        """Refuse a grid that cannot cut anything.

        Raises:
            ItemRegistryError: If the epoch length is not positive, or no label is kept.
        """
        if self.epoch_us < 1:
            raise ItemRegistryError(f"epoch length must be positive microseconds, got {self.epoch_us}")
        if not self.keep_labels:
            raise ItemRegistryError("an epoch grid that keeps no label produces no item")


@dataclass(frozen=True)
class ScoredRun:
    """One run-length annotation on one record.

    Attributes:
        record_id: The record the run sits on.
        label: The annotation value over the run.
        start_us: Start of the run on the record timeline.
        end_us: End of the run, exclusive.
    """

    record_id: str
    label: str
    start_us: int
    end_us: int


def epoch_grid_items(runs: Iterable[ScoredRun], grid: EpochGrid, signal_end_us: dict[str, int]) -> Enumerate:
    """Return an enumerator that cuts run-length annotations into fixed-length epochs.

    An epoch is emitted when its label is kept and it fits entirely inside the record's signals. An
    epoch that runs past its own signals is dropped, which is a decision that must apply to both
    representations or the counts diverge.

    Args:
        runs: The run-length annotations, in the declared order.
        grid: How to cut them.
        signal_end_us: Per record, where the signals stop. An epoch past this is dropped.

    Returns:
        A callable that yields one item per kept epoch.
    """
    ordered = tuple(runs)

    def walk() -> Iterator[ItemRef]:
        ordinal = 0
        for run in ordered:
            if run.label not in grid.keep_labels:
                continue
            limit = min(run.end_us, signal_end_us.get(run.record_id, run.end_us))
            start = run.start_us
            while start + grid.epoch_us <= limit:
                step = start // grid.epoch_us
                yield ItemRef(
                    item_id=f"{run.record_id}#{step:09d}",
                    ordinal=ordinal,
                    record_id=run.record_id,
                    start_step=start,
                    stop_step=start + grid.epoch_us,
                    label=run.label,
                )
                ordinal += 1
                start += grid.epoch_us

    return walk


def _example(ids: set[str]) -> str:
    """Return one id from a set, for an error message.

    Returns:
        The lowest id, or a dash for an empty set.
    """
    return min(ids) if ids else "-"
