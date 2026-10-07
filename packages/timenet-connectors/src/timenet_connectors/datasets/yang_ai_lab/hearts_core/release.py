"""Common definitions for the pinned HEARTS release."""

from collections.abc import Mapping
from pathlib import Path
from string import Template
from typing import Annotated

import numpy as np
from pydantic import BeforeValidator, StrictBool, TypeAdapter
from pydantic.dataclasses import dataclass
import yaml

from timenet.types import Annotation, InputModality, Task, TimeSeriesSpec, ureg


REPO = "yang-ai-lab/HEARTS"

REVISION = "7c18df521ae36cbc6b61e17782f1ac08dc378ea1"

ANSWER_KEY = "GT"

US_PER_MINUTE = 60_000_000

BooleanValue = Annotated[
    StrictBool, BeforeValidator(lambda value: value.item() if isinstance(value, np.bool_) else value)
]
"""A strict boolean that also accepts NumPy booleans from source pickles."""

AUDIO = TimeSeriesSpec(
    spec_type="audio",
    name="Audio waveform",
    unit_value=ureg.dimensionless,
    dtype="float32",
    modality=InputModality.AUDIO,
)

DESCRIPTIONS: dict[str, str] = {
    "answer_options": "The closed set of answers the reference harness accepts for this task.",
    "subject_id": "The subject identifier, qualified by the HEARTS corpus directory.",
    "recording_start_local": (
        "The local date and time of the first sample, exactly as the source states it. The source names no time zone."
    ),
}


@dataclass(frozen=True)
class Waveform:
    """The key path and sampling rate of a benchmark waveform."""

    keys: tuple[str, ...]
    rate: tuple[str, ...] | int


@dataclass(frozen=True)
class TaskDef:
    """A task's input keys, answer vocabulary, and scalar fields."""

    task_type: type[Task]
    inputs: tuple[str | tuple[str, ...] | Waveform, ...] = ()
    options: tuple[str, ...] = ()
    target_name: str | None = None
    fields: tuple[str, ...] = ()


def load_prompts(path: Path) -> dict[str, Template]:
    """Read a connector's prompt templates.

    Returns:
        Templates keyed by task directory.
    """
    texts = TypeAdapter(dict[str, str]).validate_python(yaml.safe_load(path.read_text(encoding="utf-8")))
    return {directory: Template(text) for directory, text in texts.items()}


def vocabularies(tasks: Mapping[str, TaskDef]) -> dict[str, Annotation]:
    """Build the reusable classification vocabularies.

    Returns:
        Answer-option annotations keyed by task directory.
    """
    return {
        directory: Annotation(
            key="answer_options",
            value=list(definition.options),
            description=DESCRIPTIONS["answer_options"],
            id=f"hearts-options-{directory.partition('/')[2]}",
        )
        for directory, definition in tasks.items()
        if definition.options
    }
