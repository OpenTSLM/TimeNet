"""Choose the final id of every in-memory id that one write stores.

Objects carry provisional ids from :func:`~timenet.types.new_id` (at or above ``2**52``) and keep
them. The writer never changes the caller's objects. It plans a final id for each in-memory id and
translates every id it stores, including references, through that plan.

Per kind, the plan keeps every id below ``2**52`` and gives the provisional ids the next free ids in
storage order. The first new id is one more than the largest kept id of that kind, or the watermark
(the largest id that the base version's lineage or the dataset's last write stored), whichever is
larger. A first write therefore numbers each kind ``1..N``.

A dataset remembers the ids its last successful write assigned (:class:`IdHistory`), and raises its
watermarks to the largest id that write stored. When you write it again, a provisional id that the
last write stored keeps its final id, and an annotation content or axis equal to one the last write
stored takes its id. New objects get ids above the watermark. An object you removed leaves a gap,
and its id is not given to another object. An unchanged dataset therefore writes the same catalog
again, and an edit of a stored version keeps the stored ids. A task stream keeps its ids only if it
yields the same task objects on each pass.

Annotation contents and regular or ordinal axes are deduplicated by value: provisional contents with
equal payloads share one final content id, and provisional axes with equal definitions share one
final axis id. Irregular axes, and every other kind, are never merged. Ids that a caller copies into
JSON metadata are not translated.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from hashlib import blake2b
import json
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal

from timenet.dataset.axis import IrregularAxis, TimeAxis
from timenet.dataset.storage_order import ID_KINDS, iter_hierarchy_occurrences, iter_record_sources
from timenet.errors import TimeFValidationError
from timenet.format.annotation_codec import encode_annotation_value
from timenet.json import canonical_json
from timenet.types import Annotation, Task
from timenet.types.ids import FINAL_ID_LIMIT, is_provisional


if TYPE_CHECKING:
    from timenet.dataset.dataset import TimeFDataset


IdKind = Literal["record", "source", "signal", "axis", "content", "occurrence", "task"]

_FIELDS: dict[str, str] = {
    "record": "records",
    "source": "sources",
    "signal": "signals",
    "axis": "axes",
    "content": "contents",
    "occurrence": "occurrences",
    "task": "tasks",
}


@dataclass(frozen=True)
class FinalIds:
    """The stored id of every in-memory id that one write stored, by kind.

    Kept ids map to themselves. Several provisional content or axis ids can map to one stored id,
    because the writer stores equal annotation contents and equal regular or ordinal axes once. An
    id the write did not store is absent.
    """

    records: Mapping[int, int]
    sources: Mapping[int, int]
    signals: Mapping[int, int]
    axes: Mapping[int, int]
    contents: Mapping[int, int]
    occurrences: Mapping[int, int]
    tasks: Mapping[int, int]

    def __getitem__(self, kind: IdKind) -> Mapping[int, int]:
        """Return the mapping of one id kind.

        Args:
            kind: The id kind, for example ``"record"``.

        Returns:
            Stored id by in-memory id.
        """
        return getattr(self, _FIELDS[kind])


def content_digest(annotation: Annotation) -> bytes:
    """Return a digest of the reusable content of one annotation, as the writer stores it.

    The digest covers the name, the typed value, the unit, the description, and the metadata. It
    uses the stored value columns, so ``True``, ``1``, and ``1.0`` differ. A float compares by its
    exact bits, so every NaN is equal. ``-0.0`` counts as ``0.0``, because the float column does not
    keep the sign of zero. A zero inside the metadata keeps its sign, because the JSON text does.

    Args:
        annotation: Any occurrence of the content.

    Returns:
        A 16-byte digest.
    """
    value = encode_annotation_value(annotation.value, label=f"annotation {annotation.id!r}")
    float_value = value["float_value"]
    fields = [
        annotation.name,
        value["value_kind"],
        value["text_value"],
        value["integer_value"],
        None if float_value is None else float.hex(float_value + 0.0),
        value["boolean_value"],
        value["text_list_value"],
        annotation.unit,
        annotation.description,
        canonical_json(annotation.metadata),
    ]
    return blake2b(json.dumps(fields).encode(), digest_size=16).digest()


@dataclass(frozen=True)
class IdHistory:
    """The ids that the last successful write of an in-memory dataset assigned.

    The dataset keeps it, so its next write gives the same objects and values the same ids.
    """

    final_ids: FinalIds
    """The stored id of every in-memory id that the write stored."""
    contents: Mapping[bytes, int]
    """The stored content id by :func:`content_digest`."""
    axes: Mapping[object, int]
    """The stored axis id by :func:`axis_key`."""


def axis_key(axis: TimeAxis) -> object:
    """Return the value under which the writer stores one axis once.

    Returns:
        The axis itself for a regular or ordinal axis, which compares by definition. The axis id for
        an irregular axis, whose offsets are too large to compare.
    """
    if isinstance(axis, IrregularAxis):
        return ("irregular", axis.id)
    return axis


def _collision(kind: str, final: int) -> str:
    """Describe two objects of one kind that get the same final id.

    Returns:
        The error message.
    """
    return (
        f"two different {kind} objects get the final id {final!r}: an explicit id equals an id that an "
        f"earlier write of this dataset gave to another {kind}; leave the id provisional"
    )


class IdAssigner:
    """Assign and look up final ids during one write. Internal to the writer."""

    def __init__(self, watermarks: Mapping[str, int], history: IdHistory | None = None) -> None:
        self._history = history
        self._maps: dict[str, dict[int, int]] = {kind: {} for kind in ID_KINDS}
        self._watermarks = dict(watermarks)
        self._start: dict[str, int] = {}
        self._next: dict[str, int] = {}
        self._content_by_digest: dict[bytes, int] = {}
        self._kept_digests: dict[int, bytes] = {}
        self._axis_by_key: dict[object, int] = {}
        self._axis_by_id: dict[int, TimeAxis] = {}
        self._stream_explicit: bool | None = None

    def get(self, kind: IdKind, value: int) -> int | None:
        """Return the final id of an in-memory id, or ``None`` if the write does not store it.

        Returns:
            The final id, or ``None``.
        """
        return self._maps[kind].get(value)

    def final(self, kind: IdKind, value: int) -> int:
        """Return the final id of an in-memory id the plan walked.

        Returns:
            The final id.
        """
        return self._maps[kind][value]

    def freeze(self) -> FinalIds:
        """Return the read-only mapping of everything assigned so far.

        Returns:
            The final ids by kind.
        """
        return FinalIds(**{_FIELDS[kind]: MappingProxyType(mapping) for kind, mapping in self._maps.items()})

    def history(self) -> IdHistory:
        """Return what the dataset keeps for its next write.

        Returns:
            The final ids, and the content and axis ids by value.
        """
        return IdHistory(
            final_ids=self.freeze(),
            contents=MappingProxyType(self._content_by_digest),
            axes=MappingProxyType(self._axis_by_key),
        )

    def watermarks(self) -> dict[str, int]:
        """Return, per kind, the largest id that this write or an earlier one stored.

        Returns:
            The watermark by kind.
        """
        return {
            kind: max(self._watermarks.get(kind, 0), max(self._maps[kind].values(), default=0)) for kind in ID_KINDS
        }

    def check_unique(self) -> None:
        """Reject two different objects of one kind that get the same final id.

        This happens only when an explicit id equals an id that an earlier write of the dataset
        gave to a provisional id.

        Raises:
            TimeFValidationError: If two objects of one kind get the same final id.
        """
        for kind in ("record", "source", "signal", "occurrence", "task"):
            seen: set[int] = set()
            for final in self._maps[kind].values():
                if final in seen:
                    raise TimeFValidationError(_collision(kind, final))
                seen.add(final)
        for digest, final in self._content_by_digest.items():
            if self._kept_digests.get(final, digest) != digest:
                raise TimeFValidationError(_collision("content", final))
        for key, final in self._axis_by_key.items():
            kept = self._axis_by_id.get(final)
            if kept is not None and axis_key(kept) != key:
                raise TimeFValidationError(_collision("axis", final))

    def _earlier_or_take(self, kind: IdKind, value: int, take: Callable[[str], int]) -> int:
        """Return the final id the last write gave a provisional id, or the next free one.

        Returns:
            The final id.
        """
        earlier = None if self._history is None else self._history.final_ids[kind].get(value)
        return take(kind) if earlier is None else earlier

    def _take(self, kind: str) -> int:
        """Return the next free final id of ``kind``.

        Returns:
            The id.

        Raises:
            TimeFValidationError: If the id would reach ``2**52``.
        """
        value = self._next[kind]
        if value >= FINAL_ID_LIMIT:
            raise TimeFValidationError(f"a streamed task needs a new {kind} id, but {kind} ids would reach 2**52")
        self._next[kind] = value + 1
        return value

    def _keep_content(self, annotation: Annotation, digest: bytes) -> None:
        """Register a kept content id, or check it against its earlier payload.

        Raises:
            TimeFValidationError: If the id already carries a different payload.
        """
        known = self._kept_digests.get(annotation.id)
        if known is None:
            self._maps["content"][annotation.id] = annotation.id
            self._kept_digests[annotation.id] = digest
            self._content_by_digest.setdefault(digest, annotation.id)
        elif known != digest:
            raise TimeFValidationError(f"annotation content id {annotation.id!r} is reused with different content")

    def _assign_content(self, annotation: Annotation, digest: bytes) -> None:
        """Give a provisional content id the final id of its payload.

        Raises:
            TimeFValidationError: If the id already carries a different payload.
        """
        contents = self._maps["content"]
        final = contents.get(annotation.id)
        if final is not None:
            if self._content_by_digest.get(digest) != final:
                raise TimeFValidationError(f"annotation content id {annotation.id!r} is reused with different content")
            return
        final = self._content_by_digest.get(digest)
        if final is None and self._history is not None:
            final = self._history.contents.get(digest)
        if final is None:
            final = self._next_planned("content")
        self._content_by_digest[digest] = final
        contents[annotation.id] = final

    def _plan(self, ordered: Mapping[str, list[int]], axes: list[TimeAxis], occurrences: list[Annotation]) -> None:
        """Assign the final ids of every planned object, before anything is written.

        Raises:
            TimeFValidationError: If a kind runs out of final ids.
        """
        self._check_planned(ordered, axes)
        digests = [content_digest(annotation) for annotation in occurrences]
        largest = {
            kind: max((value for value in ordered[kind] if not is_provisional(value)), default=0) for kind in ID_KINDS
        }
        largest["axis"] = max((axis.id for axis in axes if not is_provisional(axis.id)), default=0)
        largest["content"] = max(
            (annotation.id for annotation in occurrences if not is_provisional(annotation.id)), default=0
        )
        for kind in ID_KINDS:
            self._start[kind] = self._next[kind] = max(largest[kind], self._watermarks.get(kind, 0)) + 1

        planned: tuple[IdKind, ...] = ("record", "source", "signal", "occurrence", "task")
        for kind in planned:
            mapping = self._maps[kind]
            for value in ordered[kind]:
                if value not in mapping:
                    mapping[value] = (
                        self._earlier_or_take(kind, value, self._next_planned) if is_provisional(value) else value
                    )
        self._plan_shared(axes, occurrences, digests)

        for kind in ID_KINDS:
            if self._next[kind] > FINAL_ID_LIMIT:
                raise TimeFValidationError(
                    f"cannot give {self._next[kind] - self._start[kind]} new {kind} ids final values: the "
                    f"largest kept {kind} id is {self._start[kind] - 1}, so they would reach 2**52"
                )

    def _plan_shared(self, axes: list[TimeAxis], occurrences: list[Annotation], digests: list[bytes]) -> None:
        """Assign the final ids of axes and contents, which equal values share.

        Kept ids are registered first, so a provisional axis or content equal to a kept one takes the
        kept id.
        """
        for axis in axes:
            if not is_provisional(axis.id):
                self._maps["axis"][axis.id] = axis.id
                self._axis_by_key.setdefault(axis_key(axis), axis.id)
        for annotation, digest in zip(occurrences, digests, strict=True):
            if not is_provisional(annotation.id):
                self._keep_content(annotation, digest)
        for axis in axes:
            if axis.id not in self._maps["axis"]:
                key = axis_key(axis)
                if key not in self._axis_by_key:
                    earlier = None if self._history is None else self._history.axes.get(key)
                    self._axis_by_key[key] = self._next_planned("axis") if earlier is None else earlier
                self._maps["axis"][axis.id] = self._axis_by_key[key]
        for annotation, digest in zip(occurrences, digests, strict=True):
            if is_provisional(annotation.id):
                self._assign_content(annotation, digest)

    def _check_planned(self, ordered: Mapping[str, list[int]], axes: list[TimeAxis]) -> None:
        """Reject repeated task and occurrence ids, and one axis id with two definitions.

        Raises:
            TimeFValidationError: If a task or occurrence id repeats, or an axis id is reused with a
                different definition.
        """
        for kind, name in (("task", "task"), ("occurrence", "annotation occurrence")):
            seen: set[int] = set()
            for value in ordered[kind]:
                if value in seen:
                    raise TimeFValidationError(f"{name} id {value!r} is not unique")
                seen.add(value)
        for axis in axes:
            if self._axis_by_id.setdefault(axis.id, axis) != axis:
                raise TimeFValidationError(f"axis id {axis.id!r} is reused with different definitions")

    def _next_planned(self, kind: str) -> int:
        """Return the next final id of ``kind`` during planning. Overflow is checked afterwards.

        Returns:
            The id.
        """
        value = self._next[kind]
        self._next[kind] = value + 1
        return value

    def stream_task(self, task: Task) -> None:
        """Assign the final ids of one streamed task and of the occurrences and contents it owns.

        A stream cannot be scanned ahead, so its tasks are numbered in stream order. Every task in a
        stream carries a provisional id, or every task an explicit id below ``2**52``.

        Args:
            task: The streamed task. It is not changed.

        Raises:
            TimeFValidationError: If the task or one of its occurrences repeats an earlier one, the
                stream mixes explicit and provisional task ids, an explicit occurrence or content id
                may collide with an id the writer assigned, a content id carries two payloads, or a
                kind runs out of final ids.
        """
        explicit = not is_provisional(task.id)
        if self._stream_explicit is None:
            self._stream_explicit = explicit
        elif explicit != self._stream_explicit:
            raise TimeFValidationError(
                f"task {task.id!r}: a task stream must give every task an explicit id below 2**52, or none"
            )
        tasks = self._maps["task"]
        if task.id in tasks:
            raise TimeFValidationError(f"task id {task.id!r} is not unique")
        tasks[task.id] = task.id if explicit else self._earlier_or_take("task", task.id, self._take)
        occurrences = self._maps["occurrence"]
        for annotation in task.annotations:
            occurrence_id = annotation.occurrence_id
            if occurrence_id is not None:
                if occurrence_id in occurrences:
                    raise TimeFValidationError(f"annotation occurrence id {occurrence_id!r} is not unique")
                self._check_stream_explicit(task, "occurrence", occurrence_id)
                occurrences[occurrence_id] = (
                    self._earlier_or_take("occurrence", occurrence_id, self._take)
                    if is_provisional(occurrence_id)
                    else occurrence_id
                )
            digest = content_digest(annotation)
            if is_provisional(annotation.id):
                self._assign_content(annotation, digest)
                if self._next["content"] > FINAL_ID_LIMIT:
                    raise TimeFValidationError(
                        "a streamed task needs a new content id, but content ids would reach 2**52"
                    )
            else:
                if annotation.id not in self._maps["content"]:
                    self._check_stream_explicit(task, "content", annotation.id)
                self._keep_content(annotation, digest)

    def _check_stream_explicit(self, task: Task, kind: str, value: int) -> None:
        """Keep an explicit id that first appears in a stream, unless the writer may have assigned it.

        Later ids of the kind are assigned above it.

        Raises:
            TimeFValidationError: If ``value`` is in the range of ids the writer already assigned.
        """
        if is_provisional(value):
            return
        if value >= self._next[kind]:
            self._next[kind] = value + 1
        elif value >= self._start[kind]:
            raise TimeFValidationError(
                f"streamed task {task.id!r} has the explicit {kind} id {value!r}, which the writer may "
                f"already have given to another {kind}; leave the id provisional"
            )


def plan_final_ids(dataset: "TimeFDataset", *, with_tasks: bool | None = None) -> IdAssigner:
    """Plan the final id of every object of a dataset, without changing anything.

    Tasks left out of the plan are numbered later, as they pass, by :meth:`IdAssigner.stream_task`.
    The plan starts from the ids of the dataset's last write, if it has one.

    Args:
        dataset: The dataset to write.
        with_tasks: Whether to plan the dataset's tasks. ``None`` plans them unless they stream.

    Returns:
        The assigner holding the plan.

    Raises:
        TimeFValidationError: If one id carries two payloads or definitions, an occurrence or task
            id repeats, a kind runs out of final ids, or a source hierarchy is malformed.
    """  # noqa: DOC502 - raised by IdAssigner._plan and iter_record_sources
    ordered: dict[str, list[int]] = {kind: [] for kind in ID_KINDS}
    axes: list[TimeAxis] = []
    for record in dataset.records:
        ordered["record"].append(record.id)
        for source in iter_record_sources(record):
            ordered["source"].append(source.id)
            for signal in source.signals:
                ordered["signal"].append(signal.id)
                axes.append(signal.time_axis)
    occurrences = [annotation for _, _, annotation in iter_hierarchy_occurrences(dataset)]
    if with_tasks is None:
        with_tasks = not dataset.has_task_stream
    for task in dataset.iter_tasks() if with_tasks else ():
        ordered["task"].append(task.id)
        occurrences.extend(task.annotations)
    ordered["occurrence"] = [
        annotation.occurrence_id for annotation in occurrences if annotation.occurrence_id is not None
    ]
    assigner = IdAssigner(dataset._id_watermarks, dataset._id_history)
    assigner._plan(ordered, axes, occurrences)
    return assigner
