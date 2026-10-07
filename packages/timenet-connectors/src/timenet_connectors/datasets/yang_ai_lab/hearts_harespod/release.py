"""Task definitions and prompts for HEARTS harespod."""

from pathlib import Path

from timenet.types import AnswerTask, TimeSeriesSpec, TSCorrespondenceTask, ureg
from timenet_connectors.datasets.yang_ai_lab.hearts_core.release import TaskDef, load_prompts


RESPIRATION_NORM = TimeSeriesSpec(
    spec_type="respiration_norm", name="Respiration (min-max scaled)", unit_value=ureg.dimensionless, dtype="float64"
)

SPO2_NORM = TimeSeriesSpec(
    spec_type="spo2_norm", name="Oxygen saturation (min-max scaled)", unit_value=ureg.dimensionless, dtype="float64"
)

HEART_RATE_NORM = TimeSeriesSpec(
    spec_type="heart_rate_norm", name="Heart rate (min-max scaled)", unit_value=ureg.dimensionless, dtype="float64"
)

_RANKING = TaskDef(AnswerTask, inputs=("segment_dfs",))

COLUMN_SPECS = {"rsp": RESPIRATION_NORM, "spo": SPO2_NORM, "hr": HEART_RATE_NORM}
CANDIDATE_KEYS = {"hr_resp_pairing": "hr_dfs", "spo2_resp_pairing": "spo_dfs"}

TASKS: dict[str, TaskDef] = {
    "harespod/altitude_ranking_respiration": _RANKING,
    "harespod/altitude_ranking_spo2": _RANKING,
    "harespod/hr_resp_pairing": TaskDef(TSCorrespondenceTask, inputs=("respiration_dfs",)),
    "harespod/spo2_resp_pairing": TaskDef(TSCorrespondenceTask, inputs=("respiration_dfs",)),
}

PROMPTS = load_prompts(Path(__file__).with_name("prompts.yaml"))
