"""Build the input records selected by CGMacros benchmark cases."""

from typing import Any, cast

from timenet.dataset import Source
from timenet.types import ForecastingTask, TimeInterval
from timenet_connectors.datasets.yang_ai_lab.hearts_cgmacros.release import COLUMN_SPECS, HELD_OUT_READINGS, PHOTOGRAPHS
from timenet_connectors.datasets.yang_ai_lab.hearts_core.cases import Case, moment_us
from timenet_connectors.datasets.yang_ai_lab.hearts_core.records import Built, FrameLayout, frame_record
from timenet_connectors.datasets.yang_ai_lab.hearts_core.release import US_PER_MINUTE
from timenet_connectors.sources.images import image_signal


_LAYOUT = FrameLayout(COLUMN_SPECS, ("Timestamp", "Time (min)"), ("timestamp_min",))


def case_records(case: Case) -> tuple[Built, ...]:
    """Build CGMacros windows, combining reference and forecast frames when required.

    Returns:
        Input records with their source origins.
    """
    inputs: list[Built] = []
    entries = cast("tuple[str | tuple[str, ...], ...]", case.definition.inputs)
    for entry in entries:
        frames: dict[tuple[str, ...], Any] = {}
        if isinstance(entry, tuple):
            name = None
            frames.update(((key,), case.payload[key]) for key in entry)
        else:
            name = entry if len(entries) > 1 else None
            frames[entry,] = case.payload[entry]
        subject = case.payload.get(f"{name}_subject", case.payload.get("subject_id"))
        built = frame_record(case, name, frames, _LAYOUT, subject=subject)
        if case.task == "meal_img_classification":
            built.record.sources = (*built.record.sources, _images_source(case, built.record.id))
        inputs.append(built)
    if case.definition.task_type is ForecastingTask:
        record, origin_us = inputs[0]
        meal_us = moment_us(case.payload["meal_time"]) - origin_us
        record.time_span = TimeInterval.micros(0, meal_us + HELD_OUT_READINGS * US_PER_MINUTE)
    return tuple(inputs)


def _images_source(case: Case, record_id: str) -> Source:
    """Build the Source holding the four meal photographs as image Signals.

    Returns:
        The image source.
    """
    mapping = case.payload["image_mapping"]
    source_id = f"{record_id}-image_mapping"
    signals = tuple(image_signal(mapping[name], signal_id=f"{source_id}-{name}", name=name) for name in PHOTOGRAPHS)
    return Source(id=source_id, name="image_mapping", signals=signals)
