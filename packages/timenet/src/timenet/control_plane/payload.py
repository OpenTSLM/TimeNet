"""How a typed task maps to rows: which table holds each payload field, and under what name.

Every task carries the same frame, whatever its class: its id, the records it names, the tasks it
derives from, its prompt, its scope, the annotations it takes as input or gives as its answer, and
its rationale. The writer and the reader handle the frame directly. What differs per class is the
payload. The control plane stores a payload as rows keyed by field name rather than as a column per
field, so something has to say which table a given field's value belongs in. This module declares
that, one entry per built-in task class.

This is knowledge about the task classes, not about the database. The tables it names are defined in
:mod:`timenet.control_plane.schema`, and :mod:`timenet.control_plane.checks` turns the same
declaration into the queries that refuse a task whose stored payload is not the one its class
declares. :func:`task_payload` compares the declaration against the live dataclass on every read, so
a field added to a task class fails loudly rather than being dropped on write.
"""

from dataclasses import MISSING, dataclass, fields
from enum import StrEnum, unique
from typing import Final

from timenet.errors import TimeFValidationError
from timenet.types import TASKS, TaskType


@unique
class PayloadKind(StrEnum):
    """How one task payload field is stored, which says which table holds its value."""

    TEXT = "text"
    """A string, in ``task_fields.text_value``. A ``StrEnum`` payload stores as its value."""
    NUMBER = "number"
    """A float, in ``task_fields.double_value``."""
    RECORD_REF = "record_ref"
    """One record id or a tuple of them, in ``task_refs`` with ``ref_kind = 'record'``."""
    TIME_SERIES_REF = "time_series_ref"
    """One series id or a tuple of them, in ``task_refs`` with ``ref_kind = 'time_series'``."""
    SPAN = "span"
    """One span or a tuple of them, in ``task_spans``."""


@dataclass(frozen=True)
class PayloadField:
    """One field of a task's type-specific payload, and where the control plane keeps it."""

    name: str
    """The dataclass field's name, stored verbatim in the ``field`` column."""
    kind: PayloadKind
    """Which table holds the value."""
    is_list: bool = False
    """Whether the field holds a tuple rather than a single value. Only a ref or a span field may.

    ``None`` and ``()`` mean different things: a localization target of ``None`` says the answer is
    stored by reference, and ``()`` says the task looked and found nothing. Zero element rows cannot
    tell the two apart. So every payload field that is not ``None`` gets one ``task_fields`` row,
    whatever its kind, and that row alone says the field is set.
    """

    @property
    def stores_elements(self) -> bool:
        """Whether the field's value lives in ``task_refs`` or ``task_spans`` rather than in ``task_fields``."""
        return self.kind is not PayloadKind.TEXT and self.kind is not PayloadKind.NUMBER


TASK_PAYLOAD: Final[dict[TaskType, tuple[PayloadField, ...]]] = {
    TaskType.CLASSIFICATION: (
        PayloadField("target", PayloadKind.TEXT),
        PayloadField("target_schema", PayloadKind.TEXT),
    ),
    TaskType.ANSWER: (PayloadField("target", PayloadKind.TEXT),),
    TaskType.SCALAR_PREDICTION: (
        PayloadField("target", PayloadKind.NUMBER),
        PayloadField("unit", PayloadKind.TEXT),
        PayloadField("target_name", PayloadKind.TEXT),
    ),
    TaskType.TEMPORAL_LOCALIZATION: (
        PayloadField("target", PayloadKind.SPAN, is_list=True),
        PayloadField("mode", PayloadKind.TEXT),
    ),
    TaskType.FORECASTING: (
        PayloadField("context_record_ids", PayloadKind.RECORD_REF, is_list=True),
        PayloadField("target_record_id", PayloadKind.RECORD_REF),
        PayloadField("target_span", PayloadKind.SPAN),
    ),
    TaskType.TS_EDITING: (
        PayloadField("source_record_id", PayloadKind.RECORD_REF),
        PayloadField("target_record_id", PayloadKind.RECORD_REF),
    ),
    TaskType.TS_GENERATION: (PayloadField("target_record_id", PayloadKind.RECORD_REF),),
    TaskType.TS_CORRESPONDENCE: (
        PayloadField("candidate_record_ids", PayloadKind.RECORD_REF, is_list=True),
        PayloadField("target", PayloadKind.RECORD_REF, is_list=True),
        PayloadField("target_time_series_ids", PayloadKind.TIME_SERIES_REF, is_list=True),
    ),
}
"""The payload fields of each task type, beyond the frame every task shares.

The writer and the reader handle the frame (``id``, ``record_ids``, ``from_task_ids``, ``prompt``,
``scope``, the annotation id tuples and ``rationale``) directly, so it is not listed here.
:func:`task_payload` compares this declaration against the live dataclass, so a field added to a
task type fails loudly rather than being dropped on write.
"""

_TASK_FRAME: Final = frozenset(
    {
        "id",
        "record_ids",
        "prompt",
        "scope",
        "input_annotation_ids",
        "target_annotation_ids",
        "rationale",
        "from_tasks",
    }
)
"""The base :class:`~timenet.types.Task` fields, which every task type stores the same way."""


def task_payload(task_type: TaskType) -> tuple[PayloadField, ...]:
    """Return one task type's payload declaration, compared against its dataclass.

    Args:
        task_type: The task type whose payload to describe.

    Returns:
        The payload fields, in declaration order.

    Raises:
        TimeFValidationError: If the task dataclass and this declaration have drifted apart, or a
            scalar-valued field is declared as a list.
    """
    declared = TASK_PAYLOAD[task_type]
    cls = TASKS[task_type]
    expected = {field.name for field in fields(cls)} - _TASK_FRAME
    if cls.answer_is_record:
        # The answer is a produced series, found through a payload record id, so `target` stays None.
        expected.discard("target")
    actual = {field.name for field in declared}
    if expected != actual:
        raise TimeFValidationError(
            f"{cls.__name__} payload fields {sorted(expected)} do not match the control-plane "
            f"declaration {sorted(actual)}"
        )
    scalar_lists = sorted(field.name for field in declared if field.is_list and not field.stores_elements)
    if scalar_lists:
        raise TimeFValidationError(
            f"{cls.__name__} declares {scalar_lists} as list-valued {PayloadKind.TEXT.value} or "
            f"{PayloadKind.NUMBER.value} fields, which task_fields stores one value per row"
        )
    return declared


def text_answer(task_type: TaskType) -> PayloadField | None:
    """Return the payload field a task type answers with in free text, if it has one.

    That answer is an item of the task rather than a named field of its payload, so it is stored in
    ``task_items`` beside the records the task names, not in ``task_fields``.

    Args:
        task_type: The task type to describe.

    Returns:
        The field, or ``None`` when the type answers with a number, a span, a reference or a record.
    """
    answers = TASKS[task_type].answer_fields
    for declared in task_payload(task_type):
        if declared.kind is PayloadKind.TEXT and declared.name in answers:
            return declared
    return None


def required_payload_fields(task_type: TaskType) -> tuple[str, ...]:
    """Return the payload fields a task type cannot leave unset.

    A field the dataclass gives no default is one its constructor demands, so a stored task without
    it cannot be rebuilt. The list comes from the dataclass rather than from a second declaration,
    so it stays in step with the class.

    Args:
        task_type: The task type to describe.

    Returns:
        The names of its required payload fields, in declaration order.
    """
    defaults = {field.name: field for field in fields(TASKS[task_type])}
    return tuple(
        declared.name
        for declared in task_payload(task_type)
        if defaults[declared.name].default is MISSING and defaults[declared.name].default_factory is MISSING
    )
