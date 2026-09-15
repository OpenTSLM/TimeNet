"""Shape: a row does not contradict itself.

Each rule here looks at one row and reads no other table. A column group that must be all set or
all unset, a discriminator that must agree with the column it selects, an interval that must end
after it starts. Where a check joins, the join only fetches the id that the message reports.

DuckDB can express some of this as a ``CHECK`` constraint, and the schema uses one where the rule
fits in a column list (``role``, ``item_type``, ``ref_kind``, ``backend``). A constraint that spans
columns costs a test per inserted row, and the load inserts by the million. These run one time
against the finished load instead.
"""

from typing import Final

from timenet.control_plane.checks._check import Check


SHAPE: Final = (
    Check(
        name="specs_data_source_complete",
        invariant="A spec declares a data source with both a type and a name, or with neither.",
        sql="SELECT spec_type FROM specs WHERE (data_source_type IS NULL) <> (data_source_name IS NULL)",
        remedy="Set the type and the name of the data source together, or leave both unset.",
    ),
    Check(
        name="task_items_value_matches_type",
        invariant="A text item of a task holds text and no record. A record item holds no text.",
        sql=(
            "SELECT t.external_id FROM task_items i JOIN tasks t ON t.task_id = i.task_id "
            "WHERE (i.item_type = 'text' AND (i.text_value IS NULL OR i.record_id IS NOT NULL)) "
            "OR (i.item_type = 'record' AND i.text_value IS NOT NULL)"
        ),
        remedy="Give the item text or a record, and set item_type to the one it holds.",
    ),
    Check(
        name="spans_interval_ordered",
        invariant="An interval span ends after it starts.",
        sql=(
            "SELECT task_id FROM task_spans WHERE end_at IS NOT NULL AND end_at <= start_at "
            "UNION ALL "
            "SELECT annotation_id FROM annotations WHERE span_end_us IS NOT NULL AND span_end_us <= span_start_us"
        ),
        remedy="Correct the span. Put its end after its start, or remove the end to make it a point.",
    ),
    Check(
        name="annotations_span_start_present",
        invariant="An annotation that has a span end also has a span start.",
        sql="SELECT annotation_id FROM annotations WHERE span_start_us IS NULL AND span_end_us IS NOT NULL",
        remedy="Give the annotation a span start, or remove its span end.",
    ),
)
"""A row holds together on its own, with no other table read."""
