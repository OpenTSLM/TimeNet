"""Task definitions and prompts for HEARTS vctk."""

from pathlib import Path

from timenet.types import ClassificationTask
from timenet_connectors.datasets.yang_ai_lab.hearts_core.release import TaskDef, Waveform, load_prompts, vocabularies


# The reference harness reads these waveforms at 16 kHz.
_VCTK = Waveform(("waveform",), rate=16_000)

TASKS: dict[str, TaskDef] = {
    "vctk/waveform_temporal_direction_detection": TaskDef(
        ClassificationTask, inputs=(_VCTK,), options=("forward", "reversed")
    ),
}

PROMPTS = load_prompts(Path(__file__).with_name("prompts.yaml"))
VOCABULARIES = vocabularies(TASKS)
