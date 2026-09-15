"""Payload: a typed task holds what its class declares, and nothing else.

Main enforced the typing with one table per task type. Each field had its own column, and the
database refused a row that did not fit. An entity-attribute-value payload buys a stable set of
tables at the cost of that refusal. These checks put the refusal back.

The rules come from :mod:`timenet.control_plane.payload`, which says which table holds each field
of each task type. A field added to a task class therefore changes these checks with it. Each check
compares the stored rows against the declaration in one pass, and joins on a rendered ``VALUES``
list rather than scanning the payload tables one time per task type.
"""

from collections.abc import Sequence
from typing import Final

from timenet.control_plane.checks._check import Check
from timenet.control_plane.payload import (
    TASK_PAYLOAD,
    PayloadKind,
    required_payload_fields,
    task_payload,
    text_answer,
)


def _rows(rows: Sequence[tuple[str, ...]]) -> str:
    """Render a declaration as a SQL ``VALUES`` list.

    Args:
        rows: The tuples to render. Every value comes from this package's own declarations or from a
            task dataclass, never from input.

    Returns:
        The rendered list, ready to follow ``VALUES``.
    """
    return ", ".join("(" + ", ".join(f"'{value}'" for value in row) + ")" for row in rows)


def _typed_task_payload() -> tuple[Check, ...]:
    """Return the checks that each task's stored payload is the one its class declares.

    Returns:
        The named checks. A type that declares no reference or no required field drops the check
        that would have no rows to compare against.
    """
    field_rows: list[tuple[str, ...]] = []
    ref_rows: list[tuple[str, ...]] = []
    span_rows: list[tuple[str, ...]] = []
    required_rows: list[tuple[str, ...]] = []
    answering_types: list[str] = []
    for task_type in TASK_PAYLOAD:
        answer = text_answer(task_type)
        if answer is not None:
            answering_types.append(str(task_type))
        # Every task may carry a scope, stored in task_spans under its own field name.
        span_rows.append((str(task_type), "scope"))
        for name in required_payload_fields(task_type):
            required_rows.append((str(task_type), name))
        for declared in task_payload(task_type):
            if declared is answer:
                continue
            kind = "element" if declared.stores_elements else declared.kind.value
            field_rows.append((str(task_type), declared.name, kind))
            if declared.kind is PayloadKind.SPAN:
                span_rows.append((str(task_type), declared.name))
            elif declared.kind is PayloadKind.RECORD_REF:
                ref_rows.append((str(task_type), declared.name, "record"))
            elif declared.kind is PayloadKind.TIME_SERIES_REF:
                ref_rows.append((str(task_type), declared.name, "time_series"))

    declared_fields = [(task_type, name) for task_type, name, _ in field_rows]
    answering = ", ".join(f"'{name}'" for name in answering_types)
    checks = [
        Check(
            name="task_fields_declared",
            invariant="A task holds only the payload fields that its type declares.",
            sql=(
                f"SELECT t.external_id FROM task_fields f JOIN tasks t ON t.task_id = f.task_id "  # noqa: S608
                f"ANTI JOIN (VALUES {_rows(declared_fields)}) AS d(task_type, field) "
                f"ON d.task_type = t.task_type AND d.field = f.field"
            ),
            remedy="Remove the field from the task, or add it to the task class and to TASK_PAYLOAD.",
        ),
        Check(
            name="task_fields_column_matches_kind",
            invariant="Each payload field is in the column that its declared kind names.",
            sql=(
                f"SELECT t.external_id FROM task_fields f JOIN tasks t ON t.task_id = f.task_id "  # noqa: S608
                f"JOIN (VALUES {_rows(field_rows)}) AS d(task_type, field, kind) "
                f"ON d.task_type = t.task_type AND d.field = f.field WHERE "
                f"(d.kind = '{PayloadKind.TEXT.value}' AND (f.text_value IS NULL OR f.double_value IS NOT NULL)) OR "
                f"(d.kind = '{PayloadKind.NUMBER.value}' AND (f.double_value IS NULL OR f.text_value IS NOT NULL)) OR "
                f"(d.kind = 'element' AND (f.text_value IS NOT NULL OR f.double_value IS NOT NULL))"
            ),
            remedy="Give the field the type its class declares. A ref or span field keeps its value elsewhere.",
        ),
        Check(
            name="task_spans_declared",
            invariant="A task holds only the payload spans that its type declares, and its scope.",
            sql=(
                f"SELECT t.external_id FROM task_spans s JOIN tasks t ON t.task_id = s.task_id "  # noqa: S608
                f"ANTI JOIN (VALUES {_rows(span_rows)}) AS d(task_type, field) "
                f"ON d.task_type = t.task_type AND d.field = s.field"
            ),
            remedy="Remove the span from the task, or add it to the task class and to TASK_PAYLOAD.",
        ),
        Check(
            name="task_items_text_answer_declared",
            invariant="Only a task type that answers in free text holds a text answer.",
            sql=(
                f"SELECT t.external_id FROM task_items i JOIN tasks t ON t.task_id = i.task_id "  # noqa: S608
                f"WHERE i.role = 'target' AND i.item_type = 'text' AND t.task_type NOT IN ({answering})"
            ),
            remedy="Put the answer in the field that the task class declares for it.",
        ),
        Check(
            name="task_items_one_text_answer",
            invariant="A task holds one text answer at most.",
            sql=(
                "SELECT task_id FROM task_items WHERE role = 'target' AND item_type = 'text' "
                "GROUP BY task_id HAVING count(*) > 1"
            ),
            remedy="Give the task one answer.",
        ),
    ]
    if ref_rows:
        checks.append(
            Check(
                name="task_refs_declared",
                invariant="A task references a record or a series only through a field that its type declares.",
                sql=(
                    f"SELECT t.external_id FROM task_refs p JOIN tasks t ON t.task_id = p.task_id "  # noqa: S608
                    f"ANTI JOIN (VALUES {_rows(ref_rows)}) AS d(task_type, field, ref_kind) "
                    f"ON d.task_type = t.task_type AND d.field = p.field AND d.ref_kind = p.ref_kind"
                ),
                remedy="Correct the name of the field, or reference the kind of entity the class declares.",
            )
        )
    if required_rows:
        checks.append(
            Check(
                name="task_fields_required_present",
                invariant="A task holds every payload field that its type requires.",
                sql=(
                    f"SELECT t.external_id FROM tasks t "  # noqa: S608
                    f"JOIN (VALUES {_rows(required_rows)}) AS d(task_type, field) ON d.task_type = t.task_type "
                    f"ANTI JOIN task_fields f ON f.task_id = t.task_id AND f.field = d.field"
                ),
                remedy="Set the field on the task. The reader cannot rebuild the task without it.",
            )
        )
    return tuple(checks)


PAYLOAD: Final = _typed_task_payload()
"""A stored task can be given back to the class it came from."""
